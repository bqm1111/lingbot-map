#!/usr/bin/env python
"""Phase 1.2 -- controlled test of the focal-length hypothesis.

The previous study measured, across 28 TartanAir chunks:

    predicted focal / true focal          = 1.064 +- 0.018   (~5.9% too long)
    depth scale / trajectory Sim(3) scale = 0.953 +- 0.029
    1 / focal ratio                       = 0.940 +- 0.016

and observed that the second and third are close. This script asks whether that
association is *causal* by varying **only the intrinsics used for back-projection**
and measuring unaligned point-cloud geometry against ground truth.

Three arms, everything else held fixed:
    A  predicted intrinsics          (what the pipeline does today)
    B  known dataset intrinsics      (the ground-truth K on the model's grid)
    C  predicted intrinsics with the focal scaled by 1/r, r = fx_pred/fx_gt

Note what cannot move: per-pixel depth AbsRel is independent of K, so it is
reported once, not per arm. If the hypothesis holds, arms B and C should improve
point-cloud accuracy and completeness over arm A, and by a similar amount.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot import metrics as M
from prompted_lingbot.runner import DepthEvalCache, load_cached


def arm_intrinsics(seq, arm: str) -> np.ndarray:
    K_pred = seq.pred_K[0].copy()
    K_gt = seq.gt_K.copy()
    if arm == "predicted":
        return K_pred
    if arm == "dataset":
        return K_gt
    if arm == "focal_corrected":
        K = K_pred.copy()
        r = float(K_pred[0, 0] / K_gt[0, 0])
        K[0, 0] /= r
        K[1, 1] /= r
        return K
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-depth", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = None
    if args.split_file and args.split:
        names = set(json.load(open(args.split_file))[args.split])

    rows = []
    for f in sorted(os.listdir(args.cache_dir)):
        if not f.endswith(".npz"):
            continue
        if names is not None and f[:-4] not in names:
            continue
        seq = load_cached(os.path.join(args.cache_dir, f))
        if not seq.has_gt_depth:
            continue
        dc = DepthEvalCache.build(seq, seed=args.seed)

        # One scale for every arm: the per-sequence oracle depth scale. Holding it
        # fixed is what makes this a controlled comparison of intrinsics alone.
        s = float(np.median(dc.oracle_scales()))
        corr_d = seq.pred_depth.astype(np.float32) * np.float32(s)
        gt_d = seq.gt_depth.astype(np.float32)
        gt_v = seq.gt_depth_valid

        depth_m = dc.metrics(np.full(seq.n_frames, s))
        gt_pts = M.build_point_cloud(gt_d, gt_v, seq.gt_K, seq.gt_pose_c2w,
                                     seed=args.seed, max_depth=args.max_depth)

        r = float(seq.pred_K[0, 0, 0] / seq.gt_K[0, 0])
        row = {"sequence": seq.name, "depth_scale": s,
               "focal_ratio_pred_over_gt": r, "abs_rel": depth_m.get("abs_rel"),
               "rmse": depth_m.get("rmse")}
        for arm in ("predicted", "dataset", "focal_corrected"):
            K = arm_intrinsics(seq, arm)
            pred_pts = M.build_point_cloud(corr_d, gt_v, K, seq.gt_pose_c2w,
                                           seed=args.seed, max_depth=args.max_depth)
            pm = M.point_cloud_metrics(pred_pts, gt_pts)
            row[f"{arm}_fx"] = float(K[0, 0])
            for k in ("acc_mean_m", "comp_mean_m", "chamfer_m", "f1_at_0.25", "f1_at_0.5"):
                row[f"{arm}_{k}"] = pm.get(k)
        rows.append(row)
        print(f"{seq.name[:44]:44s} r={r:.4f} absrel={row['abs_rel']:.4f} "
              f"chamfer A/B/C = {row['predicted_chamfer_m']:.3f} / "
              f"{row['dataset_chamfer_m']:.3f} / {row['focal_corrected_chamfer_m']:.3f}",
              flush=True)

    if not rows:
        raise SystemExit("no sequences with ground-truth depth found")

    def agg(key):
        v = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v)}

    summary = {"n_sequences": len(rows),
               "focal_ratio_pred_over_gt": agg("focal_ratio_pred_over_gt"),
               "abs_rel_any_arm": agg("abs_rel")}
    for arm in ("predicted", "dataset", "focal_corrected"):
        summary[arm] = {k: agg(f"{arm}_{k}") for k in
                        ("acc_mean_m", "comp_mean_m", "chamfer_m", "f1_at_0.25", "f1_at_0.5")}
    # paired contrasts against arm A
    for arm in ("dataset", "focal_corrected"):
        d = np.array([r[f"{arm}_chamfer_m"] - r["predicted_chamfer_m"] for r in rows])
        summary[f"{arm}_minus_predicted_chamfer"] = {
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "win_rate": float((d < 0).mean()), "n": len(d)}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.output, "w"), indent=1)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
