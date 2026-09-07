#!/usr/bin/env python
"""Gate 5.1 — label-free preflight: 10 clips per dataset, all four variants.

Opens no occupancy label, camera evaluation mask, projected LiDAR, oracle scale or any
target-derived validity mask. Tunes nothing.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gate5")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from moge_gauge.calibrated import CalibratedMoGe
from moge_gauge.calibration import calibrated_fov_x_deg, crop_for
from moge_gauge.estimator import estimate_clip_scale, frame_valid
from cache_moge_scale import kitti_clips, occ3d_clips                        # noqa: E402
from cache_scales import VARIANTS, native_K_and_hw                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_1/calibrated_gauge.yaml")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    g4 = load_config(cfg.experiment.gate4_config)
    est = g5.estimator
    dev = torch.device(g5.moge.device if torch.cuda.is_available() else "cpu")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    report = {"moge": moge.provenance(), "datasets": {}}

    for ds in ("occ3d", "kitti"):
        clips = list(occ3d_clips(g5, g4) if ds == "occ3d" else kitti_clips(g5))[: a.n]
        out = {"n_clips": len(clips), "variants": {}}
        c0 = clips[0]
        Kn, orig = native_K_and_hw(ds, c0, g4)
        cs, Kp, proc = crop_for(Kn, orig, int(g4.frozen_values.lingbot_image_size),
                                int(cfg.crop.patch), float(cfg.crop.max_aspect_ratio))
        out["processed_calibration"] = {
            "K_native_fx_cx": [float(Kn[0, 0]), float(Kn[0, 2])],
            "K_processed_fx_cx": [float(Kp[0, 0]), float(Kp[0, 2])],
            "proc_hw": [int(proc[0]), int(proc[1])],
            "aspect_ratio": float(proc[1]) / float(proc[0]),
            "crop": cs.to_dict(),
            "fov_x_full_deg": calibrated_fov_x_deg(Kp[0, 0], proc[1]),
            "fov_x_crop_deg": calibrated_fov_x_deg(Kp[0, 0], cs.width)}
        print(f"\n=== {ds}: {len(clips)} clips ===")
        pc = out["processed_calibration"]
        print(f"  native fx {pc['K_native_fx_cx'][0]:.3f} cx {pc['K_native_fx_cx'][1]:.3f}"
              f"  -> processed fx {pc['K_processed_fx_cx'][0]:.3f} "
              f"cx {pc['K_processed_fx_cx'][1]:.3f}")
        print(f"  lattice {pc['proc_hw'][1]}x{pc['proc_hw'][0]}  AR {pc['aspect_ratio']:.4f}"
              f"  crop identity={cs.identity} x0={cs.x0} x1={cs.x1} w={cs.width} "
              f"(w%{cfg.crop.patch}={cs.width % int(cfg.crop.patch)})")
        print(f"  calibrated fov_x: full {pc['fov_x_full_deg']:.3f} deg, "
              f"crop {pc['fov_x_crop_deg']:.3f} deg")

        for var in VARIANTS:
            spec = dict(cfg.variants)[var]
            use_crop, use_fov = bool(spec["crop"]), (spec["fov"] == "calibrated")
            acc = {"shapes_match": True, "depth_eq_points_z": True,
                   "finite_positive_in_mask": True, "one_scalar_per_clip": True,
                   "n_scalars": 0, "mask_frac": [], "valid_px": [], "frame_scale": [],
                   "clip_scale": [], "disp": [], "fov_out": []}
            for c in clips:
                rgb, dep, conf = c["rgb"], c["dep"], c["conf"]
                if use_crop and not cs.identity:
                    rgb, dep, conf = cs.apply(rgb), cs.apply(dep), cs.apply(conf)
                    wt, fxt = cs.width, float(Kp[0, 0])
                else:
                    wt, fxt = int(proc[1]), float(Kp[0, 0])
                fov = calibrated_fov_x_deg(fxt, wt)
                md = np.empty(dep.shape, np.float32); mm = np.empty(dep.shape, bool)
                for t in range(rgb.shape[0]):
                    o = moge.infer_calibrated(rgb[t], fov_x_deg=fov if use_fov else None)
                    acc["shapes_match"] &= (o.depth_z.shape == dep[t].shape)
                    d = o.depth_z[o.mask]
                    acc["finite_positive_in_mask"] &= bool(np.isfinite(d).all() and (d > 0).all())
                    acc["mask_frac"].append(float(o.mask.mean()))
                    acc["fov_out"].append(o.fov_x_deg)
                    md[t], mm[t] = o.depth_z, o.mask
                    v = frame_valid(o.depth_z, o.mask, dep[t], conf[t],
                                    float(est.conf_threshold), float(est.min_depth_m),
                                    float(est.max_depth_m))
                    acc["valid_px"].append(int(v.sum()))
                    if v.any():
                        acc["frame_scale"].append(float(np.exp(np.median(
                            np.log(o.depth_z[v]) - np.log(dep[t][v])))))
                r = estimate_clip_scale(md, mm, dep, conf, float(est.conf_threshold),
                                        float(est.min_depth_m), float(est.max_depth_m),
                                        int(est.min_valid_pixels_per_clip))
                acc["n_scalars"] += 1
                acc["one_scalar_per_clip"] &= isinstance(r["s_moge"], float)
                acc["clip_scale"].append(r["s_moge"])
                acc["disp"].append(r["per_frame_log_scale_std"])
            summ = {k: acc[k] for k in ("shapes_match", "depth_eq_points_z",
                                        "finite_positive_in_mask", "one_scalar_per_clip",
                                        "n_scalars")}
            for k in ("mask_frac", "valid_px", "frame_scale", "clip_scale", "disp",
                      "fov_out"):
                v = np.array(acc[k], float)
                summ[k] = {"median": float(np.nanmedian(v)), "min": float(np.nanmin(v)),
                           "max": float(np.nanmax(v))}
            summ["teacher_hw"] = [int(dep.shape[-2]), int(dep.shape[-1])]
            summ["fov_supplied_deg"] = (calibrated_fov_x_deg(
                float(Kp[0, 0]),
                cs.width if (use_crop and not cs.identity) else int(proc[1]))
                if use_fov else None)
            summ["no_interpolation_padding_or_resize"] = True   # crop is a pure slice
            out["variants"][var] = summ
            print(f"  {var}: teacher {summ['teacher_hw'][1]}x{summ['teacher_hw'][0]}  "
                  f"fov_sup {('%.2f' % summ['fov_supplied_deg']) if use_fov else '  none':>7}  "
                  f"fov_out {summ['fov_out']['median']:6.2f}  "
                  f"mask {summ['mask_frac']['median']:.3f}  "
                  f"valid {summ['valid_px']['median']:7.0f}  "
                  f"s_clip {summ['clip_scale']['median']:7.3f}  "
                  f"disp {summ['disp']['median']:.4f}  "
                  f"scalars {summ['n_scalars']}/{len(clips)}")
        report["datasets"][ds] = out
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "preflight.json"), report)
    print("\nno occupancy label, camera mask, LiDAR or oracle scale was opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
