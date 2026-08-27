#!/usr/bin/env python
"""Visual sanity checks for the occupancy pipeline.

Overlays ground-truth occupied voxels, predicted occupied voxels, false
positives, false negatives, and the causal camera trajectory with frusta.  Run
this BEFORE trusting any benchmark number -- a coordinate, axis-order or scale
error is obvious here and invisible in an IoU column.

    python scripts/visualize_lingbot_occupancy.py \
      --dataset-root data/kitti/dataset --sequence 08 \
      --cache-dir outputs/prompted_lingbot/cache_semkitti08 \
      --output-dir outputs/prompted_lingbot/occupancy_geometry/semantickitti/vis \
      --frames 100 1000 2500 --history 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.conventions import Sim3, umeyama_sim3
from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform, relative_c2w
from prompted_lingbot.occ_eval import OccPointConfig, points_in_anchor_camera
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
from prompted_lingbot.occupancy import binary_occupancy_scores, occupancy_from_points
from prompted_lingbot.runner import load_cached


def frustum(pose_c2w, cam_to_grid, length=12.0, half=6.0):
    """Camera frustum edges in grid coordinates.

    The corners are already in metres, so they must NOT be multiplied by the
    metric scale -- that is applied to the *model's* depth, not to a figure
    annotation drawn in world units.
    """
    pts = np.array([[0.0, 0.0, 0.0],
                    [-half, -half * 0.4, length], [half, -half * 0.4, length],
                    [half, half * 0.4, length], [-half, half * 0.4, length]])
    pts = apply_transform(cam_to_grid, pts)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return pts, edges


def panel(ax, gt, pred, valid, title, axis="z"):
    """Bird's-eye (axis='z') or side (axis='y') categorical overlay."""
    ax_i = {"z": 2, "y": 1}[axis]
    g = gt.any(axis=ax_i)
    p = pred.any(axis=ax_i)
    v = valid.any(axis=ax_i)
    img = np.zeros(g.shape + (3,))
    img[v & ~g & ~p] = (0.97, 0.97, 0.97)     # evaluated, both empty
    img[g & p] = (0.13, 0.55, 0.20)           # true positive
    img[~g & p] = (0.85, 0.33, 0.10)          # false positive
    img[g & ~p] = (0.20, 0.40, 0.80)          # false negative
    img[~v] = (0.80, 0.80, 0.80)              # not evaluated
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x voxel (forward)")
    ax.set_ylabel({"z": "y voxel (left)", "y": "z voxel (up)"}[axis])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 1000, 2500])
    ap.add_argument("--history", type=int, default=20)
    ap.add_argument("--conf-threshold", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=60.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence)
    manifest = json.load(open(os.path.join(args.cache_dir, "manifest.json")))
    cfg = OccPointConfig(conf_threshold=args.conf_threshold, max_depth=args.max_depth)

    entries = sorted(manifest["sequences"].items(),
                     key=lambda kv: kv[1]["extra"].get("source_start", 0))
    written = []
    for gframe in args.frames:
        chunk = None
        for name, meta in entries:
            start = int(meta["extra"].get("source_start", 0))
            if start <= gframe < start + int(meta["num_frames"]):
                chunk, chunk_start = name, start
                break
        if chunk is None:
            print(f"frame {gframe}: not in the cache, skipped")
            continue
        seq = load_cached(os.path.join(args.cache_dir, f"{chunk}.npz"),
                          manifest["sequences"][chunk])
        t = gframe - chunk_start
        target, valid = spec.target(gframe)
        if target is None:
            print(f"frame {gframe}: no ground-truth voxels, skipped")
            continue

        S = umeyama_sim3(seq.pred_pose_c2w[:, :3, 3], seq.gt_pose_c2w[:, :3, 3])
        corr = Sim3(S.s, np.eye(3), np.zeros(3))
        pts_cam = points_in_anchor_camera(seq, t, args.history, corr.s, cfg)
        pred = occupancy_from_points(apply_transform(spec.cam_to_velo, pts_cam), G)
        gt = (target != G.empty_class) & valid
        sc = binary_occupancy_scores(pred, target, G, valid=valid)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        panel(axes[0], gt, pred, valid,
              f"BEV (collapsed over z) — frame {gframe}, history {args.history}", axis="z")
        panel(axes[1], gt, pred, valid, "Side view (collapsed over y)", axis="y")

        # causal camera trajectory + frusta, in grid voxel coordinates
        lo = max(0, t - args.history + 1)
        pose_t = np.eye(4); pose_t[:3, :4] = seq.pred_pose_c2w[t]
        traj = []
        for f in range(lo, t + 1):
            pose_f = np.eye(4); pose_f[:3, :4] = seq.pred_pose_c2w[f]
            c = apply_transform(relative_c2w(pose_f, pose_t), np.zeros((1, 3)))[0] * corr.s
            traj.append(apply_transform(spec.cam_to_velo, c[None])[0])
        traj = (np.array(traj) - np.array(G.origin)) / G.voxel_size
        axes[0].plot(traj[:, 0], traj[:, 1], "-", color="black", lw=1.6, label="causal trajectory")
        axes[0].plot(traj[-1, 0], traj[-1, 1], "*", color="black", ms=14)
        fp, edges = frustum(np.eye(4), spec.cam_to_velo)
        fp = (fp - np.array(G.origin)) / G.voxel_size
        for a, b in edges:
            axes[0].plot(fp[[a, b], 0], fp[[a, b], 1], "-", color="black", lw=0.8, alpha=0.7)

        handles = [Patch(color=(0.13, 0.55, 0.20), label="true positive"),
                   Patch(color=(0.85, 0.33, 0.10), label="false positive"),
                   Patch(color=(0.20, 0.40, 0.80), label="false negative"),
                   Patch(color=(0.80, 0.80, 0.80), label="not evaluated (invalid)")]
        axes[0].legend(handles=handles, fontsize=7, loc="upper right")
        fig.suptitle(f"seq {args.sequence} frame {gframe} — IoU {sc['iou']:.4f}  "
                     f"P {sc['precision']:.3f}  R {sc['recall']:.3f}  "
                     f"scale {corr.s:.2f}  points {pts_cam.shape[0]:,}", fontsize=11)
        fig.tight_layout()
        out = os.path.join(args.output_dir,
                           f"occ_{args.sequence}_{gframe:06d}_h{args.history}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out)
        print(f"frame {gframe}: IoU {sc['iou']:.4f} P {sc['precision']:.3f} "
              f"R {sc['recall']:.3f} -> {out}", flush=True)

    print(f"\nwrote {len(written)} visualisations to {args.output_dir}")


if __name__ == "__main__":
    main()
