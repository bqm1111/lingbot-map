#!/usr/bin/env python
"""Gate 5.1 step 1 — the four predeclared variants' per-clip scales.

    G51-A  full lattice, MoGe infers FOV        (exact Gate-5 reproduction)
    G51-B  full lattice, calibrated FOV
    G51-C  aspect-safe crop, MoGe infers FOV
    G51-D  aspect-safe crop, calibrated FOV     (PRIMARY)

No occupancy label, camera mask, LiDAR, extrinsic, pose or camera height is opened here.
Only RGB and -- for B and D -- one scalar calibrated horizontal FOV in degrees.

    python tools/gate5_1/cache_scales.py --dataset kitti --variant G51-D
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gate5")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from moge_gauge.calibrated import CalibratedMoGe
from moge_gauge.calibration import calibrated_fov_x_deg, crop_for
from moge_gauge.estimator import estimate_clip_scale
from cache_moge_scale import kitti_clips, occ3d_clips                        # noqa: E402

VARIANTS = ("G51-A", "G51-B", "G51-C", "G51-D")


def native_K_and_hw(dataset, clip, g4):
    """Calibrated native intrinsics + native image size for this clip's camera."""
    if dataset == "occ3d":
        d = np.load(os.path.join(g4.data.cache_root, clip["clip_id"] + ".npz"))
        return d["K_native"][0].astype(np.float64), tuple(int(x) for x in d["native_hw"])
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    d = np.load(os.path.join(REPO_ROOT, scfg.cache.root, "lingbot",
                             clip["clip_id"] + ".npz"))
    return d["gt_K_native"].astype(np.float64), tuple(int(x) for x in d["orig_hw"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_1/calibrated_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    g4 = load_config(cfg.experiment.gate4_config)
    est = g5.estimator
    spec = dict(cfg.variants)[a.variant]
    use_crop, use_fov = bool(spec["crop"]), (spec["fov"] == "calibrated")
    dev = torch.device(g5.moge.device if torch.cuda.is_available() else "cpu")

    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == g5.moge.hf_revision == \
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    os.makedirs(art, exist_ok=True)
    print(f"{a.variant} [{a.dataset}]  crop={use_crop}  calibrated_fov={use_fov}")

    gen = occ3d_clips(cfg if False else g5, g4) if a.dataset == "occ3d" else kitti_clips(g5)
    rows, frames, t0, n_fail = [], [], time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    crop_logged = None
    for i, c in enumerate(gen):
        if a.limit and i >= a.limit:
            break
        Kn, orig_hw = native_K_and_hw(a.dataset, c, g4)
        cs, Kp, proc_hw = crop_for(Kn, orig_hw, int(g4.frozen_values.lingbot_image_size),
                                   int(cfg.crop.patch), float(cfg.crop.max_aspect_ratio))
        assert tuple(proc_hw) == tuple(c["rgb"].shape[-2:]), (
            f"processed lattice {proc_hw} != cached RGB {c['rgb'].shape[-2:]}")
        rgb, dep, conf = c["rgb"], c["dep"], c["conf"]
        if use_crop and not cs.identity:
            rgb, dep, conf = cs.apply(rgb), cs.apply(dep), cs.apply(conf)
            w_t, fx_t = cs.width, float(Kp[0, 0])
        else:
            w_t, fx_t = int(proc_hw[1]), float(Kp[0, 0])
        fov_cal = calibrated_fov_x_deg(fx_t, w_t)
        if crop_logged is None:
            crop_logged = {"crop": cs.to_dict(), "K_processed": Kp.tolist(),
                           "proc_hw": [int(proc_hw[0]), int(proc_hw[1])],
                           "teacher_width": w_t, "teacher_height": int(rgb.shape[-2]),
                           "calibrated_fov_x_deg": fov_cal, "crop_applied": bool(
                               use_crop and not cs.identity)}
            print(f"  teacher lattice {w_t}x{rgb.shape[-2]}  AR "
                  f"{w_t/rgb.shape[-2]:.4f}  calibrated fov_x {fov_cal:.3f} deg  "
                  f"crop {'x0=%d..%d' % (cs.x0, cs.x1) if use_crop and not cs.identity else 'identity'}")

        T = rgb.shape[0]
        md = np.empty(dep.shape, np.float32)
        mm = np.empty(dep.shape, bool)
        fovs_out = []
        for t in range(T):
            o = moge.infer_calibrated(rgb[t], fov_x_deg=fov_cal if use_fov else None)
            assert o.depth_z.shape == dep[t].shape, (
                f"{c['clip_id']} f{t}: MoGe {o.depth_z.shape} vs LingBot {dep[t].shape}")
            md[t], mm[t] = o.depth_z, o.mask
            fovs_out.append(o.fov_x_deg)
        r = estimate_clip_scale(md, mm, dep, conf, float(est.conf_threshold),
                                float(est.min_depth_m), float(est.max_depth_m),
                                int(est.min_valid_pixels_per_clip))
        if not r["ok"]:
            n_fail += 1
        rows.append({"clip_id": c["clip_id"], "group": c["group"], "variant": a.variant,
                     "s_moge": r["s_moge"], "ok": int(r["ok"]),
                     "n_valid_total": r["n_valid_total"], "mad_log_clip": r["mad_log_clip"],
                     "per_frame_log_scale_std": r["per_frame_log_scale_std"],
                     "fov_supplied_deg": fov_cal if use_fov else float("nan"),
                     "fov_calibrated_deg": fov_cal,
                     "fov_moge_out_deg": float(np.median(fovs_out)),
                     "teacher_w": w_t, "teacher_h": int(rgb.shape[-2]),
                     "crop_x0": cs.x0 if (use_crop and not cs.identity) else 0,
                     "crop_x1": cs.x1 if (use_crop and not cs.identity) else int(proc_hw[1])})
        for pf in r["per_frame"]:
            frames.append({"clip_id": c["clip_id"], "variant": a.variant, **pf})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}  {time.time()-t0:.0f}s", flush=True)

    tag = f"{a.variant}_{a.dataset}"
    with open(os.path.join(art, f"scales_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, f"scale_frames_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    ok = [r for r in rows if r["ok"]]
    S = np.array([r["s_moge"] for r in ok])
    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, f"scales_{tag}.json"), {
        "variant": a.variant, "dataset": a.dataset, "moge": prov,
        "crop_and_calibration": crop_logged,
        "uses_crop": use_crop, "uses_calibrated_fov": use_fov,
        "n_clips": len(rows), "n_ok": len(ok), "n_failed": n_fail,
        "failed_clip_ids": [r["clip_id"] for r in rows if not r["ok"]],
        "s_moge": {"median": float(np.median(S)),
                   **{f"p{int(p*100):02d}": float(np.quantile(S, p))
                      for p in (0.05, 0.25, 0.75, 0.95)}},
        "fov_calibrated_deg": float(np.median([r["fov_calibrated_deg"] for r in rows])),
        "fov_moge_out_deg": float(np.median([r["fov_moge_out_deg"] for r in rows])),
        "mad_log_clip_median": float(np.nanmedian([r["mad_log_clip"] for r in ok])),
        "per_frame_log_scale_std_median": float(
            np.nanmedian([r["per_frame_log_scale_std"] for r in ok])),
        "elapsed_s": time.time() - t0, "peak_gpu_gib": peak})
    print(f"  {len(rows)} clips ok={len(ok)} failed={n_fail}  "
          f"s median {np.median(S):.3f}  {time.time()-t0:.0f}s  peak {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
