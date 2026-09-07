#!/usr/bin/env python
"""Gate 5.2 step 3 — the four deployable per-clip scales.

    C0   frozen constant s0 = 27.3665
    C3   frozen Gate-2 clip head, scalar a_clip only (r_shape discarded)
    A    frozen MoGe-2, FOV inferred from RGB          (the Gate-5 gauge)
    B    frozen MoGe-2, calibrated FOV supplied        (the Gate-5.1 G51-B gauge)

Full image for both MoGe variants: Gate 5.1 discarded the aspect crop as a metric-gauge
component. No occupancy label, ``.invalid`` mask, LiDAR file, extrinsic, ego pose or
camera height is opened anywhere in this tool; B receives exactly one extra scalar.

    python tools/gate5_2/cache_scales.py --variant B
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from moge_gauge.calibrated import CalibratedMoGe
from moge_gauge.calibration import calibrated_fov_x_deg
from moge_gauge.estimator import estimate_clip_scale
from sscbench_kitti360.adapter import SEQUENCE, parse_calibration
from gates.voxel_gate.c3 import clip_scale as c3_clip_scale, load_head

VARIANTS = ("C0C3", "A", "B")


def clips(cfg):
    """Yield each cached clip with the RGB LingBot itself consumed."""
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    recs = read_manifest(os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))
    root = os.path.join(cfg.dataset.kitti360_root, "data_2d_raw", SEQUENCE)
    for r in recs:
        p = os.path.join(cfg.lingbot.cache_root, r["clip_id"] + ".npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=False)
        rgb = load_and_preprocess_images(
            [os.path.join(root, q) for q in r["image_paths"]], mode="crop",
            image_size=int(cfg.lingbot.inference_resolution),
            patch_size=int(cfg.lingbot.patch_size)).numpy()
        yield {"clip_id": r["clip_id"], "block": int(r["block"]), "rgb": rgb,
               "dep": d["pred_depth"].astype(np.float32),
               "conf": d["pred_depth_conf"].astype(np.float32),
               "K_processed": d["K_processed"], "proc_hw": tuple(int(x) for x in d["proc_hw"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    dcfg = load_config("configs/depth_gate/refine.yaml")
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    est = g5.estimator
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    s0 = float(dcfg.scale.constant)
    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)

    calib = parse_calibration(os.path.join(cfg.dataset.root, "calibration"))
    pre = Preprocess.build(calib.native_hw, int(cfg.lingbot.inference_resolution),
                           int(cfg.lingbot.patch_size))
    Kp = pre.scale_intrinsics(calib.K)
    fov_cal = calibrated_fov_x_deg(float(Kp[0, 0]), int(pre.proc_hw[1]))
    print(f"native {calib.native_hw} K fx={calib.K[0,0]:.4f} -> processed {pre.proc_hw} "
          f"fx={Kp[0,0]:.4f}  aspect {pre.proc_hw[1]/pre.proc_hw[0]:.4f}  "
          f"calibrated fov_x {fov_cal:.4f} deg")

    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    os.makedirs(art, exist_ok=True)
    moge = prov = head = hck = None
    if a.variant in ("A", "B"):
        moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                              revision=g5.moge.hf_revision)
        prov = moge.provenance()
        assert prov["hf_revision"] == g5.moge.hf_revision == \
            "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "MoGe revision drifted from Gate 5"
        cache_dir = os.path.join(cfg.moge.cache_root, a.variant)
        os.makedirs(cache_dir, exist_ok=True)
    else:
        head, hck = load_head(os.path.join(REPO_ROOT, dcfg.experiment.output_dir,
                                           "runs", "depth_cnn"), dev)

    rows, frames, t0, n_fail = [], [], time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for i, c in enumerate(clips(cfg)):
        if a.limit and i >= a.limit:
            break
        assert tuple(c["proc_hw"]) == tuple(pre.proc_hw) == tuple(c["rgb"].shape[-2:]), \
            f"{c['clip_id']}: lattice mismatch {c['proc_hw']} {pre.proc_hw} {c['rgb'].shape}"
        row = {"clip_id": c["clip_id"], "block": c["block"], "variant": a.variant}
        if a.variant == "C0C3":
            ac, s_c3, sup = c3_clip_scale(head, hck, c["dep"], c["conf"], s0,
                                          conf_thr, dmin, dmax, None, dev)
            row.update({"s_c0": s0, "s_c3": float(s_c3), "a_clip": float(ac),
                        "n_support_px": int(np.asarray(sup).sum()), "ok": 1})
        else:
            T = c["rgb"].shape[0]
            md = np.empty(c["dep"].shape, np.float32)
            mm = np.empty(c["dep"].shape, bool)
            fovs = []
            for t in range(T):
                o = moge.infer_calibrated(c["rgb"][t],
                                          fov_x_deg=fov_cal if a.variant == "B" else None)
                assert o.depth_z.shape == c["dep"][t].shape, (
                    f"{c['clip_id']} f{t}: MoGe {o.depth_z.shape} vs LingBot "
                    f"{c['dep'][t].shape} -- pixel correspondence broken")
                md[t], mm[t] = o.depth_z, o.mask
                fovs.append(o.fov_x_deg)
            r = estimate_clip_scale(md, mm, c["dep"], c["conf"], float(est.conf_threshold),
                                    float(est.min_depth_m), float(est.max_depth_m),
                                    int(est.min_valid_pixels_per_clip))
            n_fail += (not r["ok"])
            np.savez_compressed(os.path.join(cache_dir, c["clip_id"] + ".npz"),
                                moge_depth=md.astype(np.float16),
                                moge_mask=np.packbits(mm.reshape(-1)),
                                mask_shape=np.asarray(mm.shape, np.int32),
                                fov_x_deg=np.asarray(fovs, np.float32))
            row.update({"s_moge": r["s_moge"], "log_s_moge": r["log_s_moge"],
                        "ok": int(r["ok"]), "n_valid_total": r["n_valid_total"],
                        "mad_log_clip": r["mad_log_clip"],
                        "per_frame_log_scale_std": r["per_frame_log_scale_std"],
                        "per_frame_scale_ratio_max_min":
                            r["per_frame_scale_ratio_max_min"],
                        "fov_supplied_deg": fov_cal if a.variant == "B" else float("nan"),
                        "fov_calibrated_deg": fov_cal,
                        "fov_moge_out_deg": float(np.median(fovs)),
                        "fov_moge_out_std": float(np.std(fovs))})
            for pf in r["per_frame"]:
                frames.append({"clip_id": c["clip_id"], "variant": a.variant, **pf,
                               "fov_moge_out_deg": fovs[pf["frame"]]})
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(art, f"scales_{a.variant}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if frames:
        with open(os.path.join(art, f"scale_frames_{a.variant}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(frames[0])); w.writeheader()
            w.writerows(frames)
    key = "s_c3" if a.variant == "C0C3" else "s_moge"
    S = np.array([r[key] for r in rows if r["ok"]], float)
    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    summary = {
        "variant": a.variant, "dataset": "sscbench_kitti360", "sequence": SEQUENCE,
        "n_clips": len(rows), "n_ok": int(sum(r["ok"] for r in rows)), "n_failed": n_fail,
        "failed_clip_ids": [r["clip_id"] for r in rows if not r["ok"]],
        "native_hw": list(calib.native_hw), "proc_hw": list(pre.proc_hw),
        "K_native": calib.K.tolist(), "K_processed": Kp.tolist(),
        "pixel_aspect_anisotropy": float(pre.sy / pre.sx),
        "processed_aspect_ratio": float(pre.proc_hw[1] / pre.proc_hw[0]),
        "calibrated_fov_x_deg": fov_cal,
        "crop_applied": False,
        f"{key}": {"median": float(np.median(S)),
                   **{f"p{int(p*100):02d}": float(np.quantile(S, p))
                      for p in (0.05, 0.25, 0.75, 0.95)}},
        "elapsed_s": time.time() - t0, "peak_gpu_gib": peak}
    if a.variant in ("A", "B"):
        summary["moge"] = prov
        summary["fov_moge_out_deg_median"] = float(
            np.median([r["fov_moge_out_deg"] for r in rows]))
        summary["fov_difference_deg_median"] = float(
            np.median([r["fov_moge_out_deg"] - fov_cal for r in rows]))
        summary["mad_log_clip_median"] = float(
            np.nanmedian([r["mad_log_clip"] for r in rows if r["ok"]]))
        summary["per_frame_log_scale_std_median"] = float(
            np.nanmedian([r["per_frame_log_scale_std"] for r in rows if r["ok"]]))
    write_json(os.path.join(art, f"scales_{a.variant}.json"), summary)
    print(f"{a.variant}: {len(rows)} clips, ok {int(sum(r['ok'] for r in rows))}, "
          f"failed {n_fail}, {key} median {np.median(S):.4f}, "
          f"{time.time()-t0:.0f}s, peak {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
