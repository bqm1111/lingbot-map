#!/usr/bin/env python
"""End-to-end inference cost of the RELEASED OccAny checkpoint, in its own configuration.

This runs OccAny's own ``extract_output_occany.py`` unmodified, under its own interpreter and
its own environment, with the preset that produced the predictions our accuracy comparison
scores -- the same checkpoint, the same image size, the same novel-view generation settings
(``-rot 60 -vpi 10 -fwd 3 -seed_trans 2``), the same confidence thresholds, the same internal
generation batch. Nothing is substituted and no rendering setting is reduced.

Instrumentation is by monkeypatching the functions their script imports, applied BEFORE the
script is executed with ``runpy``, so their file on disk is never touched:

    torch.utils.data.DataLoader                      restrict to our fixed anchors, repeat the
                                                     list for warm-up + timed passes, and stamp
                                                     the clock when an item is handed over
    occany.utils.helpers.create_voxel_prediction     stamp the clock when the SECOND call of an
                                                     anchor returns (that call is the combined
                                                     reconstruction+generation grid, i.e. the
                                                     prediction our evaluator reads), after a
                                                     device-to-host copy that forces a sync
    occany.must3r_inference.inference_occany_gen     stage boundary (``--mode stages`` only)
    occany.utils.inference_helper.build_intrinsics_from_focal
                                                     stage boundary (``--mode stages`` only)

Timing boundary
    start  the collated item is in host memory: decoded, resized, normalised RGB plus
           calibration. Their ``DataLoader`` does that in worker processes, exactly as in the
           run that produced their accuracy, so data loading is outside the clock for them in
           the same way image decode is outside it for us.
    end    the combined occupancy grid as a host numpy array.

Known overcount, disclosed rather than hidden: between those two stamps their script also
converts the input images to uint8 for the output file, voxelises the reconstruction-only grid
and copies it to the host, and copies the ground-truth voxel label. None of that is needed to
produce the compared prediction. ``--mode stages`` measures the whole post-network segment that
contains it, so a reader can bound it.

    PROJECT=/media/SSD1/occany_causal/occany_shadow SCRATCH=/media/SSD1/occany_causal \\
    CUDA_VISIBLE_DEVICES=3 <occany_env>/bin/python bench_occany.py \\
        --preset K_5FRAMES_GEOM --anchors anchors_semantickitti.json --mode e2e
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import shlex
import subprocess
import sys
import time

import numpy as np

PROJECT = os.environ.get("PROJECT", "/media/SSD1/occany_causal/occany_shadow")
for p in ["third_party", "third_party/dust3r", "third_party/croco/models/curope",
          "third_party/Grounded-SAM-2", "third_party/Grounded-SAM-2/grounding_dino",
          "third_party/sam3", "third_party/Depth-Anything-3/src", ""]:
    d = os.path.join(PROJECT, p)
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

#: the two geometry presets, copied verbatim from OccAny's own sh/exp_lists/occany.sh
PRESETS = {
    "K_5FRAMES_GEOM": "--exp_name OccAny --image_size 512 --batch_gen_view 12 --dataset kitti "
                      "--setting 5frames --gen -rot 60 -vpi 10 -fwd 3 -seed_trans 2 "
                      "--boxes_folder resized_1216_box5_text5_DINOB --box_conf_thres 0.15 "
                      "--recon_conf_thres 2.5 --gen_conf_thres 2.5",
    "K_1FRAME_GEOM": "--exp_name OccAny --image_size 512 --batch_gen_view 16 --dataset kitti "
                     "--setting 1frame --gen -rot 60 -vpi 50 -fwd 1 -seed_trans 2 "
                     "--boxes_folder resized_1216_box5_text5_DINOB "
                     "--recon_conf_thres 2.0 --gen_conf_thres 2.0",
}
STATE = {"start": None, "rows": [], "calls": 0, "stage": {}, "params": None,
         "gen_shapes": None, "mode": "e2e", "reset_after": 0, "n_done": 0,
         "t_tp": [], "mark": None}


def _patch(mode: str, frame_names, n_warmup: int):
    import torch
    import torch.utils.data as TUD
    from occany.datasets import eval_helper as EH
    from occany.utils import helpers as HE
    from occany.utils import inference_helper as IH
    from occany import must3r_inference as MI

    STATE["mode"] = mode
    STATE["reset_after"] = n_warmup
    real_prepare = EH.prepare_eval_setting

    def prepare(*a, **k):
        ds, collate, recon_idx = real_prepare(*a, **k)
        have = set(ds.frame_names)
        sel = [f for f in frame_names if f in have]
        assert len(sel) == len(frame_names), \
            f"{len(frame_names) - len(sel)} benchmark anchors are absent from OccAny's dataset"
        ds.frame_names = sel
        STATE["recon_view_idx"] = list(recon_idx)
        STATE["output_resolution"] = list(ds.output_resolution)
        STATE["voxel_size"] = float(ds.voxel_size)
        return ds, collate, recon_idx
    EH.prepare_eval_setting = prepare

    RealLoader = TUD.DataLoader

    class BenchLoader(RealLoader):
        def __iter__(self):
            for item in super().__iter__():
                if STATE["n_done"] == STATE["reset_after"]:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                STATE["calls"] = 0
                STATE["stage"] = {}
                STATE["t_tp"] = []
                STATE["mark"] = None
                STATE["start"] = time.perf_counter()
                yield item
    TUD.DataLoader = BenchLoader

    real_voxel = HE.create_voxel_prediction

    def voxel(*a, **k):
        out = real_voxel(*a, **k)
        STATE["calls"] += 1
        if STATE["mode"] == "stages" and STATE["calls"] == 1 and STATE["t_tp"]:
            _r = out.detach().to("cpu", copy=True).numpy()
            torch.cuda.synchronize()
            STATE["stage"]["voxelise_recon_only_NOT_REQUIRED"] = \
                time.perf_counter() - STATE["t_tp"][0]
        if STATE["calls"] == 2:                      # the combined recon+gen grid
            _ = out.detach().to("cpu", copy=True).numpy()
            torch.cuda.synchronize()
            t = time.perf_counter()
            if STATE["mode"] == "stages" and len(STATE["t_tp"]) >= 2:
                STATE["stage"]["voxelise_combined"] = t - STATE["t_tp"][1]
            STATE["rows"].append({"index": STATE["n_done"],
                                  "e2e_s": t - STATE["start"],
                                  "stage_s": dict(STATE["stage"]),
                                  "n_occupied": int((_ != 0).sum())})
            STATE["n_done"] += 1
        return out
    HE.create_voxel_prediction = voxel

    real_count = IH.count_unique_parameters

    def count(mods, *a, **k):
        r = real_count(mods, *a, **k)
        STATE["params"] = {"total": int(r[0]), "requires_grad": int(r[1])}
        return r
    IH.count_unique_parameters = count

    if mode == "stages":
        real_infer = MI.inference_occany_gen

        def infer(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = real_infer(*a, **k)
            torch.cuda.synchronize()
            STATE["stage"]["network"] = time.perf_counter() - t0
            if STATE["gen_shapes"] is None:
                try:
                    STATE["gen_shapes"] = {
                        "recon_pts3d": list(out[0]["pts3d"].shape),
                        "gen_pts3d": list(out[1]["pts3d"].shape) if out[1] is not None else None}
                except Exception as e:
                    STATE["gen_shapes"] = f"unavailable: {e}"
            return out
        MI.inference_occany_gen = infer

        real_K = IH.build_intrinsics_from_focal

        def bK(*a, **k):
            r = real_K(*a, **k)
            torch.cuda.synchronize()
            STATE["mark"] = time.perf_counter()
            STATE["stage"]["assemble_outputs"] = STATE["mark"] - STATE["start"] \
                - STATE["stage"].get("network", 0.0)
            return r
        IH.build_intrinsics_from_focal = bK

        # the two voxelisation legs: leg 1 builds the reconstruction-only grid, which the
        # compared prediction never uses, so its cost is separated instead of carried
        real_tp = HE.transform_points_torch

        def tp(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            STATE["t_tp"].append(t)
            if len(STATE["t_tp"]) == 1 and STATE.get("mark"):
                STATE["stage"]["dict_tail_and_recon_mask"] = t - STATE["mark"]
            return real_tp(*a, **k)
        HE.transform_points_torch = tp


def preprocess_only(anchors_file: str, limit=None) -> dict:
    """OccAny's REQUIRED image preprocessing, measured on the main thread.

    Their ``__getitem__`` also reads the ground-truth voxel label, reads LiDAR and projects it
    to a GT depth map, and builds GroundingDINO / SAM2 / SAM3 tensors. None of that is needed to
    produce a geometry prediction, so timing the whole item would overstate their cost. What is
    required is ``crop_resize_if_necessary`` to the output resolution and ``ImgNorm``; that pair
    is what this measures, per anchor, over the anchor's five frames, excluding image decode
    (measured separately on the same files).
    """
    from PIL import Image
    from occany.datasets.kitti import ImgNorm
    from occany.datasets.eval_helper import prepare_eval_setting
    try:
        from occany.datasets.kitti import crop_resize_if_necessary
    except ImportError:
        from dust3r.datasets.base.base_stereo_view_dataset import crop_resize_if_necessary
    ds, _c, _r = prepare_eval_setting(
        dataset="kitti", setting="5frames", boxes_folder="resized_1216_box5_text5_DINOB",
        image_size=512, novel_view_rgb_path=None, process_id=0, num_worlds=1, split="val",
        sam3_resolution=None, base_model="must3r",
        kitti_root=os.environ.get("KITTI_ROOT", "/media/welf/SSD2/MINH_DATASETS/kitti"),
        nuscenes_root=os.environ.get("NUSCENES_ROOT",
                                     "/media/SSD1/MINH_DATASETS/occany_nuscenes_root"))
    anchors = json.load(open(anchors_file))["anchors"][:limit]
    recon = list(_r)
    dec, pre = [], []
    for name in anchors:
        v = ds.videos[name]
        paths = [v["image_paths"][i] for i in recon]
        K = v["P"][0:3, 0:3]
        t = time.perf_counter()
        imgs = [Image.open(q).convert("RGB") for q in paths]
        dec.append(time.perf_counter() - t)
        t = time.perf_counter()
        for im in imgs:
            ph = np.zeros((im.height, im.width), dtype=np.float32)
            small, _d, _k = crop_resize_if_necessary(im, ph, K, ds.output_resolution)
            ImgNorm(np.array(small))
        pre.append(time.perf_counter() - t)
    f = lambda v: {"n": len(v), "mean_ms": float(np.mean(v) * 1e3),
                   "median_ms": float(np.median(v) * 1e3),
                   "p95_ms": float(np.percentile(v, 95) * 1e3)}
    return {"method": "occany", "measurement": "required image preprocessing, per anchor "
                                               f"({len(recon)} frames)",
            "n_frames_per_anchor": len(recon), "anchors": anchors,
            "output_resolution": list(ds.output_resolution),
            "decode_excluded_by_boundary": f(dec), "preprocess": f(pre),
            "excluded_from_this_measurement": ["GT voxel label read", "LiDAR read and GT-depth "
                                               "projection", "GroundingDINO / SAM2 / SAM3 "
                                               "transforms", "box files"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="K_5FRAMES_GEOM", choices=sorted(PRESETS))
    ap.add_argument("--anchors", required=True, help="JSON written by runtime/config.py")
    ap.add_argument("--mode", default="e2e", choices=["e2e", "stages"])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", "/media/SSD1/occany_causal"))
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.preprocess_only:
        r = preprocess_only(a.anchors, a.limit)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1, default=float)
        print("\nBENCH_SUMMARY " + json.dumps(r, default=float))
        print(f"-> {a.out}")
        return 0

    rec = json.load(open(a.anchors))
    anchors = rec["anchors"][:a.limit]
    order = anchors[:a.warmup] + anchors * a.passes
    _patch(a.mode, order, a.warmup)

    import torch
    bench_out = os.path.join(a.scratch, "bench_runtime")
    argv = ["extract_output_occany.py",
            "--output_dir", os.path.join(bench_out, "ssc_voxel_pred"),
            "--vis_output_dir", os.path.join(bench_out, "ssc_output"),
            "--kitti_root", os.environ.get("KITTI_ROOT", "/media/welf/SSD2/MINH_DATASETS/kitti"),
            "--nuscenes_root", os.environ.get(
                "NUSCENES_ROOT", "/media/SSD1/MINH_DATASETS/occany_nuscenes_root"),
            "--model", "occany_must3r", "--silent",
            "--vis_interval", "1000000000",
            "--use_render_output", "--key_to_get_pts3d", "pts3d",
            "--world", "1", "--pid", "0"] + shlex.split(PRESETS[a.preset])
    sys.argv = argv
    os.chdir(PROJECT)
    t0 = time.perf_counter()
    runpy.run_path(os.path.join(PROJECT, "extract_output_occany.py"), run_name="__main__")
    wall = time.perf_counter() - t0

    rows = STATE["rows"]
    assert len(rows) == len(order), f"{len(rows)} timed items, expected {len(order)}"
    timed = rows[a.warmup:]
    lat = np.asarray([r["e2e_s"] for r in timed], np.float64)
    per_pass = [float(np.mean(lat[i * len(anchors):(i + 1) * len(anchors)]) * 1e3)
                for i in range(a.passes)]
    ms = np.sort(lat) * 1e3

    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"

    p = torch.cuda.get_device_properties(0)
    stages = {}
    for k in ("network", "assemble_outputs", "dict_tail_and_recon_mask",
              "voxelise_recon_only_NOT_REQUIRED", "voxelise_combined"):
        v = [r["stage_s"][k] for r in timed if k in r["stage_s"]]
        if v:
            stages[k + "_mean_ms"] = float(np.mean(v) * 1e3)
    if "network_mean_ms" in stages:
        mean_e2e = float(np.mean([r["e2e_s"] for r in timed]) * 1e3)
        stages["post_network_segment_mean_ms"] = mean_e2e - stages["network_mean_ms"]
        if "voxelise_recon_only_NOT_REQUIRED_mean_ms" in stages:
            stages["e2e_less_recon_only_voxelisation_mean_ms"] = \
                mean_e2e - stages["voxelise_recon_only_NOT_REQUIRED_mean_ms"]

    out = {
        "method": "occany", "preset": a.preset, "preset_args": PRESETS[a.preset],
        "dataset": rec["dataset"], "mode": a.mode, "n_anchors": len(anchors),
        "passes": a.passes, "warmup": a.warmup, "anchors": anchors,
        "latency_per_completed_occupancy_grid": {
            "n": int(ms.size), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "min_ms": float(ms.min()),
            "max_ms": float(ms.max()),
            "std_ms": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
            "predictions_per_second": float(1e3 / ms.mean())},
        "per_pass_mean_ms": per_pass,
        "stages_mean_ms": stages or None,
        "peak_memory": {"allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "reserved_bytes": int(torch.cuda.max_memory_reserved())},
        "parameters": {"counted_by_their_own_count_unique_parameters": STATE["params"],
                       "note": "OccAny's own counter over encoder + decoder + raymap_encoder "
                               "+ gen_decoder; the released checkpoint is fully trained, so "
                               "total and trained coincide for the modules it ships"},
        "configuration": {"recon_view_idx": STATE.get("recon_view_idx"),
                          "output_resolution": STATE.get("output_resolution"),
                          "voxel_size": STATE.get("voxel_size"),
                          "generated_view_shapes": STATE["gen_shapes"],
                          "checkpoint": os.path.join(PROJECT, "checkpoints", "occany.pth"),
                          "project": PROJECT},
        "wall_clock_including_model_load_s": wall,
        "machine": {"gpu_name": p.name, "gpu_total_memory_bytes": int(p.total_memory),
                    "gpu_capability": f"{p.major}.{p.minor}",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "python": sys.version.split()[0], "executable": sys.executable,
                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "platform": platform.platform(),
                    "driver": sh("nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"),
                    "loadavg": os.getloadavg(), "cpu_count": os.cpu_count()},
        "boundary": {"start": "collated host item: decoded, resized, normalised RGB + calibration",
                     "end": "combined reconstruction+generation occupancy grid, host numpy array",
                     "excluded": ["their DataLoader workers' decode/resize/normalise, LiDAR and "
                                  "GT-depth projection, GT voxel label read, box files",
                                  "pickle serialisation of the output", "visualisation",
                                  "model construction and checkpoint loading"],
                     "disclosed_overcount": "the timed region also contains the uint8 input-image "
                                            "conversion, the reconstruction-only voxelisation and "
                                            "its host copy, and the GT voxel-label host copy"},
        "per_item": timed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("\nBENCH_SUMMARY " + json.dumps(
        {k: out[k] for k in ("latency_per_completed_occupancy_grid", "per_pass_mean_ms",
                             "stages_mean_ms", "peak_memory", "parameters")}, default=float))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
