#!/usr/bin/env python
"""Phase 2: external depth vs LingbotMap depth, both against projected LiDAR.

Evaluated on **exactly the same valid pixels** for both models, so the comparison
is paired. The LiDAR used here is evaluation ground truth only -- it never enters
any prediction (asserted by tests/prompted_lingbot/test_depth_substitution.py).

LingbotMap depth is monocular and scale-free, so it is given its best possible
break: a per-frame ORACLE median scale. Any remaining gap is depth *shape*, not
scale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, sparse_depth_from_velodyne
from prompted_lingbot.runner import load_cached


def metrics(pred, gt):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    d = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(d) / g)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "sq_rel": float(np.mean(d ** 2 / g)),
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25 ** 2)),
        "delta_3": float(np.mean(ratio < 1.25 ** 3)),
        "median_bias": float(np.median(p / g)),
        "n": int(p.size),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="data/kitti/dataset")
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--lingbot-cache", required=True)
    ap.add_argument("--external-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames-per-chunk", type=int, default=20)
    ap.add_argument("--conf-threshold", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=80.0)
    args = ap.parse_args()

    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence)
    lb_man = json.load(open(os.path.join(args.lingbot_cache, "manifest.json")))
    ex_man = json.load(open(os.path.join(args.external_cache, "manifest.json")))
    names = sorted(set(lb_man["sequences"]) & set(ex_man["sequences"]))

    RANGES = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80)]
    acc = {"external": [], "lingbot_oracle_scale": []}
    by_range = {k: {f"{a}_{b}m": [] for a, b in RANGES} for k in acc}
    valid_fracs = []

    for name in names:
        lb = lb_man["sequences"][name]
        seq = load_cached(os.path.join(args.lingbot_cache, f"{name}.npz"), lb)
        ext = np.load(os.path.join(args.external_cache, f"{name}.npz"))["external_depth"]
        start = int(lb.get("extra", {}).get("source_start", 0))
        Ht, Wt = lb["cached_depth_hw"]
        step = max(1, seq.n_frames // args.max_frames_per_chunk)
        for t in range(0, seq.n_frames, step)[: args.max_frames_per_chunk]:
            gt, gv = sparse_depth_from_velodyne(args.dataset_root, args.sequence, start + t,
                                                spec.calib, tuple(lb["source_image_hw"]),
                                                (Ht, Wt), depth_stride=int(lb["depth_stride"]),
                                                max_depth=args.max_depth)
            if gt is None or not gv.any():
                continue
            e = np.asarray(ext[t], np.float64)
            l = seq.pred_depth[t].astype(np.float64)
            c = seq.pred_depth_conf[t].astype(np.float64)
            m = gv & (e > 0.5) & (l > 1e-6) & (c >= args.conf_threshold) & (gt > 0.5)
            if m.sum() < 200:
                continue
            valid_fracs.append(float(m.mean()))
            s = float(np.median(gt[m] / l[m]))          # oracle per-frame scale
            preds = {"external": e[m], "lingbot_oracle_scale": l[m] * s}
            for k, p in preds.items():
                acc[k].append(metrics(p, gt[m]))
                for a, b in RANGES:
                    r = m & (gt >= a) & (gt < b)
                    if r.sum() >= 50:
                        pr = e[r] if k == "external" else l[r] * s
                        by_range[k][f"{a}_{b}m"].append(
                            float(np.mean(np.abs(pr - gt[r]) / gt[r])))

    def agg(rows):
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    out = {"n_frames": len(acc["external"]),
           "valid_pixel_fraction": float(np.mean(valid_fracs)) if valid_fracs else 0.0,
           "external_model": ex_man.get("model"),
           "note": ("LingbotMap is given a per-frame ORACLE median scale; external "
                    "depth is used exactly as the model emits it."),
           "metrics": {k: agg(v) for k, v in acc.items() if v},
           "abs_rel_by_range": {k: {r: (float(np.mean(v)) if v else None)
                                    for r, v in d.items()} for k, d in by_range.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
