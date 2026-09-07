#!/usr/bin/env python
"""Protocol ceiling: GT LiDAR depth + GT poses through the identical Gate-0 pipeline.

    python tools/scale_gate/eval_ceiling.py --config <cfg> --split val

Absolute occupancy IoU is uninterpretable on this benchmark: the ground truth is
accumulated from multi-pass LiDAR including *future* frames, while the protocol observes
one forward camera over a short causal clip. This computes the denominator -- what the
same clips, grid, voxeliser and metric yield when LingBot's depth and poses are replaced
by ground truth -- so every method can be reported as a fraction of what is achievable
rather than as a bare number.
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)


def fuse_gt_clip(D, gt_c2w: np.ndarray) -> np.ndarray:
    """Unproject the sparse metric LiDAR depth of every frame into the last camera."""
    dep, val = D["depth"].astype(np.float64), D["valid"]
    K = D["K_processed"].astype(np.float64)
    T, H, W = dep.shape
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    anchor = gt_c2w[T - 1]
    chunks = []
    for f in range(T):
        m = val[f] & (dep[f] > 0)
        if not m.any():
            continue
        d = dep[f][m]
        p = np.stack([(u[m] - K[0, 2]) * d / K[0, 0],
                      (v[m] - K[1, 2]) * d / K[1, 1], d], axis=-1)
        if f != T - 1:
            p = apply_transform(np.linalg.inv(anchor) @ gt_c2w[f], p)
        chunks.append(p)
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache = os.path.join(REPO_ROOT, cfg.cache.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    recs = read_manifest(os.path.join(out_dir, "manifests", f"{a.split}.jsonl"))

    specs, rows = {}, []
    for rec in recs:
        cid, seq = rec["clip_id"], rec["sequence"]
        dp = os.path.join(cache, "lidar_depth", f"{cid}.npz")
        lp = os.path.join(cache, "lingbot", f"{cid}.npz")
        if not (os.path.exists(dp) and os.path.exists(lp)):
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        target, valid = spec.target(int(rec["frame_ids"][-1]))
        if target is None:
            continue
        pts = fuse_gt_clip(np.load(dp, allow_pickle=False),
                           np.load(lp, allow_pickle=False)["gt_pose_c2w"].astype(np.float64))
        pg = apply_transform(spec.cam_to_velo, pts) if len(pts) else pts
        sc = binary_occupancy_scores(
            occupancy_from_points(pg, G, cfg.voxel.min_points_per_voxel), target, G, valid=valid)
        rows.append({"clip_id": cid, "sequence": seq, "method": "ceiling_gt_lidar_gt_pose",
                     "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                     "n_points": int(pts.shape[0])})

    iou = np.array([r["iou"] for r in rows])
    ceiling = float(iou.mean())
    summary = {"provenance": collect_provenance(cfg, "eval_ceiling"), "split": a.split,
               "n_clips": len(rows), "ceiling_iou_mean": ceiling,
               "ceiling_iou_median": float(np.median(iou)),
               "precision_mean": float(np.mean([r["precision"] for r in rows])),
               "recall_mean": float(np.mean([r["recall"] for r in rows])),
               "note": "GT LiDAR depth + GT poses through the identical clips, grid, "
                       "voxeliser and metric; the denominator for every other method"}

    # Express every evaluated method as a fraction of this ceiling.
    frac = {}
    per = os.path.join(out_dir, "oracle_metrics_per_clip.csv")
    if os.path.exists(per):
        by = {}
        for r in csv.DictReader(open(per)):
            by.setdefault(r["method"], []).append(float(r["iou"]))
        for m, v in by.items():
            frac[m] = {"iou": float(np.mean(v)), "pct_of_ceiling": 100 * float(np.mean(v)) / ceiling}
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("learned_summary_") and f.endswith(".json"):
            d = json.load(open(os.path.join(out_dir, f)))
            frac[d["model"]] = {"iou": d["iou_mean"],
                                "pct_of_ceiling": 100 * d["iou_mean"] / ceiling}
    summary["methods_as_fraction_of_ceiling"] = frac

    with open(os.path.join(out_dir, "ceiling_metrics_per_clip.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(os.path.join(out_dir, "ceiling_summary.json"), summary)

    print(f"ceiling (GT LiDAR + GT poses, {len(rows)} clips): IoU {ceiling:.4f}  "
          f"P {summary['precision_mean']:.3f}  R {summary['recall_mean']:.3f}\n")
    for m, d in sorted(frac.items(), key=lambda kv: kv[1]["iou"]):
        print(f"  {m:24s} {d['iou']:.4f}  = {d['pct_of_ceiling']:5.1f}% of ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
