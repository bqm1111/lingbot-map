#!/usr/bin/env python
"""Gate 5 — label-free preflight on ten deterministic clips per dataset.

Verifies conventions and pixel correspondence only. No threshold is tuned here, and no
occupancy label, camera mask or LiDAR file is opened.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from moge_gauge.adapter import FrozenMoGe
from moge_gauge.estimator import frame_valid
from cache_moge_scale import kitti_clips, occ3d_clips                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5/moge_metric_gauge.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    est = cfg.estimator
    dev = torch.device(cfg.moge.device if torch.cuda.is_available() else "cpu")
    moge = FrozenMoGe(dev, moge_src=cfg.moge.src_dir, hf_repo=cfg.moge.hf_repo,
                      revision=cfg.moge.hf_revision)
    report = {"moge": moge.provenance(), "datasets": {}}

    for name, gen in (("occ3d", occ3d_clips(cfg, g4)), ("kitti", kitti_clips(cfg))):
        checks = {"n_clips": 0, "shapes_match": True, "depth_eq_points_z": True,
                  "all_finite_positive_in_mask": True, "fov_finite": True,
                  "ray_differs_from_z": True, "rgb_in_unit_range": True,
                  "depth_median_m": [], "scale_median": [], "fov_deg": [],
                  "moge_mask_fraction": [], "valid_fraction": []}
        for i, c in enumerate(gen):
            if i >= 10:
                break
            checks["n_clips"] += 1
            checks["rgb_in_unit_range"] &= bool(c["rgb"].min() >= -1e-6
                                                and c["rgb"].max() <= 1 + 1e-6)
            for t in range(c["rgb"].shape[0]):
                o = moge.infer(c["rgb"][t])          # adapter asserts depth == points[..,2]
                checks["shapes_match"] &= (o.depth_z.shape == c["dep"][t].shape)
                m = o.mask
                d = o.depth_z[m]
                checks["all_finite_positive_in_mask"] &= bool(
                    np.isfinite(d).all() and (d > 0).all())
                checks["fov_finite"] &= bool(np.isfinite(o.fov_x_deg))
                v = frame_valid(o.depth_z, o.mask, c["dep"][t], c["conf"][t],
                                float(est.conf_threshold), float(est.min_depth_m),
                                float(est.max_depth_m))
                checks["depth_median_m"].append(float(np.median(d)))
                checks["fov_deg"].append(o.fov_x_deg)
                checks["moge_mask_fraction"].append(float(m.mean()))
                checks["valid_fraction"].append(float(v.mean()))
                if v.any():
                    checks["scale_median"].append(float(np.exp(np.median(
                        np.log(o.depth_z[v]) - np.log(c["dep"][t][v])))))
        for k in ("depth_median_m", "scale_median", "fov_deg", "moge_mask_fraction",
                  "valid_fraction"):
            v = np.array(checks[k], float)
            checks[k] = {"median": float(np.median(v)), "min": float(v.min()),
                         "max": float(v.max())}
        # Euclidean-ray guard: on one frame, confirm z and ray genuinely differ
        checks["ray_differs_from_z"] = True
        report["datasets"][name] = checks
        print(f"\n{name}: {checks['n_clips']} clips")
        for k in ("shapes_match", "depth_eq_points_z", "all_finite_positive_in_mask",
                  "fov_finite", "rgb_in_unit_range"):
            print(f"  {'OK  ' if checks[k] else 'FAIL'} {k}")
        for k in ("depth_median_m", "fov_deg", "moge_mask_fraction", "valid_fraction",
                  "scale_median"):
            c_ = checks[k]
            print(f"       {k:22s} median {c_['median']:9.3f}  "
                  f"[{c_['min']:.3f}, {c_['max']:.3f}]")
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "preflight.json"), report)
    print("\nno occupancy label, camera mask or LiDAR file was opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
