#!/usr/bin/env python
"""End-to-end inference cost of OUR complete causal pipeline, per completed occupancy grid.

Nothing here is a cached shortcut. Every quantity the deployed system predicts at inference
time is predicted inside the timed region: the LingBotMap streaming forward over the clip, the
five MoGe-2 forwards, the metric scale, the hard map, the CompletionUNet forward and the
read-out onto the evaluation grid. The only things read before the clock starts are the decoded
RGB frames, the camera calibration, and the calibrated horizontal FOV -- a deterministic
function of that calibration and the processed lattice width, carrying no prediction.

Timing boundary
    start  decoded RGB frames and calibration in host memory
    end    the completed occupancy grid for the evaluation grid, as a host numpy array

Excluded, and why: ground-truth voxel labels and LiDAR (evaluation inputs, not inference
inputs), scoring, visualisation, writing the prediction file, and model construction /
checkpoint loading (amortised, not steady state). Image DECODE is excluded by the stated
boundary but measured anyway, by timing ``PIL.Image.open(...).convert("RGB")`` on the same
files, so the preprocessing stage can be reported net of it.

Two measurement modes, because per-stage synchronisation distorts the total:
    ``--mode e2e``     one ``cuda.synchronize`` at each end. This is the reported latency.
    ``--mode stages``  a synchronisation at every stage boundary. Breakdown only.

    PYTHONPATH=$PWD python tools/method_vs_data/runtime/bench_ours.py \\
        --dataset semantickitti --frames 5 --mode e2e --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                           # noqa: E402
from core.datasets import grids as CG                                                  # noqa: E402
from core.model.net import apply_residual                                               # noqa: E402
from gates.gate7b import replay, scale as SC                                            # noqa: E402
from tools.method_vs_data.runtime import config as RC                                   # noqa: E402
from tools.occany_comparison import config as OC                                        # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels     # noqa: E402
from tools.occany_comparison.moge_extra import resolve_moge_path                         # noqa: E402
from tools.semantic_completion.frozen_geometry import load_frozen_geometry               # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                             # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction                # noqa: E402
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any                    # noqa: E402

STAGES = ("preprocess", "lingbot", "moge", "scale", "mapper", "map_query", "unet", "postproc")


class Clock:
    """Per-stage accumulator. In ``e2e`` mode it never synchronises or records."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.t: Dict[str, float] = {}
        self._k = None
        self._t0 = 0.0

    def __call__(self, key):
        self._k = key
        return self

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.enabled:
            torch.cuda.synchronize()
            self.t[self._k] = self.t.get(self._k, 0.0) + (time.perf_counter() - self._t0)
        return False


class Pipeline:
    """The deployed geometry pipeline, built once, then callable per anchor."""

    def __init__(self, dataset: str, device: str, frames: int):
        from gates.scale_gate.config import load_config
        from moge_gauge.calibrated import CalibratedMoGe
        self.dataset, self.frames = dataset, frames
        self.dev = torch.device(device)
        torch.cuda.set_device(self.dev)
        self.fine = GridSpec.of(CG.PREDICTION_GRID[dataset])
        self.native = GridSpec.of(CG.EVAL_GRID[dataset])
        self.ratio = int(round(self.native.voxel_size / self.fine.voxel_size))
        self.lingbot, self.lingbot_info = replay.build_model(self.dev)
        g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
        self.moge = CalibratedMoGe(self.dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                                   revision=g5.moge.hf_revision)
        self.net = load_frozen_geometry(ZC.GEOMETRY_SEED, self.dev)
        self.grid_obj = CG.PREDICTION_GRID[dataset]

    def params(self) -> dict:
        def count(m):
            ps = list(m.parameters())
            return (int(sum(p.numel() for p in ps)),
                    int(sum(p.numel() for p in ps if p.requires_grad)))
        moge_mod = next((getattr(self.moge, a) for a in ("model", "moge", "_model")
                         if getattr(self.moge, a, None) is not None
                         and isinstance(getattr(self.moge, a), torch.nn.Module)), None)
        out = {"lingbot_map": count(self.lingbot), "completion_unet": count(self.net),
               "moge2_vitl": count(moge_mod) if moge_mod is not None else None}
        tot = sum(v[0] for v in out.values() if v)
        return {"per_module_total_and_requires_grad": out,
                "total_inference_parameters": tot,
                "task_specific_trained_parameters": out["completion_unet"][0],
                "note": "requires_grad is False everywhere at inference; the trained-parameter "
                        "count is the CompletionUNet, the only module this work optimised. "
                        "LingBotMap and MoGe-2 are frozen pretrained dependencies and are NOT "
                        "free: they are in the total and in every latency figure."}

    def inputs(self, sample) -> dict:
        """Everything allowed before the clock: decoded frames, calibration, FOV."""
        causal = OC.PROTOCOLS["causal"]
        idx = sample.window(causal)
        if self.frames == 1:
            idx = [idx[causal.anchor_slot]]
        paths = [sample.frame_paths[j] for j in idx]
        keys = [sample.frame_keys[j] for j in idx]
        fovs = []
        for k in keys:
            with np.load(resolve_moge_path(self.dataset, k)) as z:
                fovs.append(float(z["fov_x_deg"]))
        return {"paths": paths, "keys": keys, "fovs": fovs,
                "T": np.asarray(sample.T_cam_to_grid_ours, np.float64),
                "anchor_slot": len(idx) - 1}

    @torch.no_grad()
    def run(self, inp: dict, clock: Clock) -> np.ndarray:
        with clock("preprocess"):
            images = replay.load_images(inp["paths"])
        with clock("lingbot"):
            self.lingbot.clean_kv_cache()
            assert replay.frame_index_trace(self.lingbot) == 0
            out = replay.replay_segment(self.lingbot, images, 1, self.dev)
        with clock("moge"):
            md, mm = [], []
            for q in range(len(inp["paths"])):
                o = self.moge.infer_calibrated(images[q].numpy(), fov_x_deg=inp["fovs"][q])
                md.append(np.asarray(o.depth_z, np.float32))
                mm.append(np.asarray(o.mask, bool))
        with clock("scale"):
            d, c = out["pred_depth"].astype(np.float32), out["pred_depth_conf"].astype(np.float32)
            log_s = np.asarray([SC.frame_candidate(md[q], mm[q], d[q], c[q])["log_s"]
                                for q in range(len(md))], np.float64)
            s = float(np.exp(np.nanmedian(log_s))) if np.isfinite(log_s).any() else float("nan")
            clip = WindowClip(inp["anchor_slot"], d, c,
                              out["pred_pose_c2w"].astype(np.float64),
                              out["pred_K"].astype(np.float64), s)
        with clock("mapper"):
            mapper = build_mapper(clip, self.dev)
        with clock("map_query"):
            S, F_ = hard_channels(mapper, self.grid_obj, inp["T"])
        del mapper
        with clock("unet"):
            x = torch.stack([S.reshape(self.fine.dims),
                             torch.zeros(self.fine.dims, dtype=torch.bool, device=self.dev),
                             F_.reshape(self.fine.dims)]).float()[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res, _ = self.net(x)
            r = res[:, 0].float()
            occ = apply_residual(torch.zeros_like(r), r).reshape(1, -1) > 0
        with clock("postproc"):
            grid = reduce_any(occ[0], self.fine, self.ratio).cpu().numpy()
        return grid


def bench(dataset: str, frames: int, mode: str, device: str, limit=None, warmup=RC.WARMUP,
          passes=RC.PASSES) -> dict:
    pipe = Pipeline(dataset, device, frames)
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    anchors = [a for a in RC.read_anchors(dataset) if a in by_sid][:limit]
    inputs = [pipe.inputs(by_sid[a]) for a in anchors]

    # image decode, on the same files, so preprocess can be reported net of it
    from PIL import Image
    dec = []
    for inp in inputs:
        t = time.perf_counter()
        for p in inp["paths"]:
            Image.open(p).convert("RGB")
        dec.append(time.perf_counter() - t)

    stage_mode = mode == "stages"
    for inp in inputs[:warmup]:
        pipe.run(inp, Clock(False))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(pipe.dev)

    per_pass: List[List[float]] = []
    stage_tot: Dict[str, float] = {}
    for _ in range(passes):
        lat = []
        for inp in inputs:
            clk = Clock(stage_mode)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe.run(inp, clk)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            for k, v in clk.t.items():
                stage_tot[k] = stage_tot.get(k, 0.0) + v
        per_pass.append(lat)

    flat = [x for p in per_pass for x in p]
    n = len(flat)
    rec = {"method": "ours", "dataset": dataset, "input_frames": frames, "mode": mode,
           "device": device, "n_anchors": len(anchors), "passes": passes, "warmup": warmup,
           "anchors": anchors,
           "latency_per_completed_occupancy_grid": RC.summarise(flat),
           "per_pass_mean_ms": [float(np.mean(p) * 1e3) for p in per_pass],
           "image_decode_excluded_by_boundary": RC.summarise(dec),
           "peak_memory": {"allocated_bytes": int(torch.cuda.max_memory_allocated(pipe.dev)),
                           "reserved_bytes": int(torch.cuda.max_memory_reserved(pipe.dev)),
                           "note": "this process holds LingBotMap + MoGe-2 + CompletionUNet "
                                   "resident; see --memory-only for an isolated figure"},
           "parameters": pipe.params(),
           "configuration": {"lingbot_checkpoint": replay.CHECKPOINT,
                             "lingbot_load": pipe.lingbot_info,
                             "lingbot_resolution": replay.INFERENCE_RESOLUTION,
                             "lingbot_autocast": replay.AUTOCAST_DTYPE,
                             "moge_repo": "Ruicheng/moge-2-vitl",
                             "moge_revision": pipe.moge.provenance().get("hf_revision"),
                             "completion_seed": ZC.GEOMETRY_SEED,
                             "prediction_grid": pipe.fine.name,
                             "prediction_dims": list(pipe.fine.dims),
                             "prediction_voxel_m": pipe.fine.voxel_size,
                             "eval_grid": pipe.native.name,
                             "eval_dims": list(pipe.native.dims),
                             "eval_voxel_m": pipe.native.voxel_size,
                             "unet_precision": "bfloat16 autocast, fp32 residual",
                             "batch": "one evaluation anchor",
                             "real_input_frames": frames,
                             "generated_views": 0,
                             "future_frames_required": 0},
           "machine": RC.machine(device),
           "stages_mean_ms": ({k: stage_tot[k] / n * 1e3 for k in STAGES if k in stage_tot}
                              if stage_mode else None),
           "boundary": {"start": "decoded RGB frames + calibration + calibrated FOV in host memory",
                        "end": "completed occupancy grid on the evaluation grid, host numpy array",
                        "excluded": ["image decode", "ground-truth labels and LiDAR", "scoring",
                                     "visualisation", "prediction serialisation",
                                     "model construction and checkpoint loading"]}}
    return rec


def verify(dataset: str, frames: int, device: str, n: int) -> dict:
    """Do the timed predictions agree with the cached canonical predictions?"""
    pipe = Pipeline(dataset, device, frames)
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    anchors = [a for a in RC.read_anchors(dataset) if a in by_sid][:n]
    from tools.observability_split import runs as R
    from tools.zero_shot_eval.practical.protocol import DATASET_SPECS
    spec = DATASET_SPECS[dataset]
    runs = {r.key: r for r in R.RUNS[dataset]}
    lk = "occany_1f_geom" if "occany_1f_geom" in runs else sorted(runs)[0]

    def sc_iou(pred, gt_occ, valid):
        tp = int((pred & gt_occ & valid).sum())
        fp = int((pred & ~gt_occ & valid).sum())
        fn = int((~pred & gt_occ & valid).sum())
        return 100.0 * tp / max(tp + fp + fn, 1), tp, fp, fn

    rows = []
    for a in anchors:
        g = pipe.run(pipe.inputs(by_sid[a]), Clock(False))
        ref = CanonicalPrediction.load(ZC.prediction_path(dataset, a)).occupied
        label, _ = R.load_occany(runs[lk], a, pipe.native.dims)
        valid = label != 255
        gt_occ = valid & (label != spec.empty_class)
        it, _a, _b, _c = sc_iou(g, gt_occ, valid)
        ic, _a, _b, _c = sc_iou(ref, gt_occ, valid)
        inter = int((g & ref).sum()); union = int((g | ref).sum())
        rows.append({"sample_id": a, "identical": bool(np.array_equal(g, ref)),
                     "agreement_iou_timed_vs_cached": 100.0 * inter / max(union, 1),
                     "sc_iou_timed_vs_gt": it, "sc_iou_cached_vs_gt": ic,
                     "sc_iou_difference": it - ic,
                     "n_timed_occupied": int(g.sum()), "n_cached_occupied": int(ref.sum()),
                     "n_disagreeing_voxels": int((g != ref).sum()),
                     "n_voxels": int(len(g))})
    diffs = [r["sc_iou_difference"] for r in rows]
    return {"method": "ours", "dataset": dataset, "input_frames": frames,
            "n_checked": len(rows), "rows": rows,
            "all_identical": all(r["identical"] for r in rows),
            "min_agreement_iou": min(r["agreement_iou_timed_vs_cached"] for r in rows),
            "mean_agreement_iou": float(np.mean(
                [r["agreement_iou_timed_vs_cached"] for r in rows])),
            "mean_disagreeing_voxel_fraction": float(np.mean(
                [r["n_disagreeing_voxels"] / r["n_voxels"] for r in rows])),
            "sc_iou_difference_mean": float(np.mean(diffs)),
            "sc_iou_difference_max_abs": float(np.max(np.abs(diffs))),
            "interpretation": "the timed pipeline recomputes LingBotMap, MoGe-2 and the scale "
                              "instead of reading the cached clip, so agreement is not required "
                              "to be bit-exact; what matters is that the SC IoU of the timed "
                              "prediction matches the cached one this runtime is paired with"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="semantickitti", choices=["semantickitti", "occ3d"])
    ap.add_argument("--frames", type=int, default=5, choices=[1, 5])
    ap.add_argument("--mode", default="e2e", choices=["e2e", "stages"])
    ap.add_argument("--device", default=RC.BENCH_DEVICE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--passes", type=int, default=RC.PASSES)
    ap.add_argument("--warmup", type=int, default=RC.WARMUP)
    ap.add_argument("--verify", type=int, default=0, help="verify N anchors and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.verify:
        r = verify(a.dataset, a.frames, a.device, a.verify)
        tag = f"verify_ours_{a.dataset}_{a.frames}f"
    else:
        r = bench(a.dataset, a.frames, a.mode, a.device, a.limit, a.warmup, a.passes)
        tag = f"ours_{a.dataset}_{a.frames}f_{a.mode}"
    out = a.out or os.path.join(RC.ARTIFACT_DIR, f"{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("latency_per_completed_occupancy_grid", "stages_mean_ms",
                               "peak_memory", "all_identical", "min_iou_vs_cached",
                               "per_pass_mean_ms")}, indent=1, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
