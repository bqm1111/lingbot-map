#!/usr/bin/env python
"""Pipeline validation + the surface-observation ceiling.

Runs the **ground-truth LiDAR scan** through the identical voxelizer, frame chain,
grid and metric used for LingbotMap predictions.

Two things come out of it:

1. **A validation.** A perfect metric sensor must score near-perfect *precision*.
   If the grid origin, axis order or camera-to-LiDAR transform were wrong,
   precision would collapse. This is the check that makes the predicted numbers
   trustworthy.
2. **The real ceiling.** SSC targets score occupied *volume*; any sensor that
   observes *surfaces* -- LiDAR included -- cannot fill the interior of objects or
   the space behind the first return. This measures how far that alone gets you,
   which is the right reference for a surface-predicting model, and is much lower
   than 1.0.

    python scripts/lidar_occupancy_ceiling.py --dataset-root data/kitti/dataset \
      --sequence 08 --history-lengths 1 5 20 100 \
      --output outputs/prompted_lingbot/occupancy_geometry/semantickitti/lidar_ceiling.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, load_semantickitti_target
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, accumulate_scores, binary_occupancy_scores, occupancy_from_points,
)


def lidar_poses(root: str, sequence: str, Tr: np.ndarray) -> np.ndarray:
    """KITTI ``poses.txt`` are CAMERA poses; LiDAR poses are ``Tr^-1 @ pose @ Tr``."""
    cam = np.loadtxt(os.path.join(root, "poses", f"{sequence}.txt")).reshape(-1, 3, 4)
    P = np.tile(np.eye(4), (len(cam), 1, 1))
    P[:, :3, :4] = cam
    Tr_inv = np.linalg.inv(Tr)
    return np.einsum("ij,njk,kl->nil", Tr_inv, P, Tr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--history-lengths", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--eval-stride", type=int, default=5)
    ap.add_argument("--max-anchors", type=int, default=160)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence)
    P_velo = lidar_poses(args.dataset_root, args.sequence, spec.calib["Tr"])
    velo_dir = os.path.join(args.dataset_root, "sequences", args.sequence, "velodyne")
    n = len(P_velo)
    frames = [f for f in range(0, n, args.eval_stride)]
    step = max(1, len(frames) // args.max_anchors)
    frames = frames[::step][: args.max_anchors]

    def scan(f):
        return np.fromfile(os.path.join(velo_dir, f"{f:06d}.bin"),
                           dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

    out = {"sequence": args.sequence, "n_anchor_frames": 0, "history": {}}
    for hist in args.history_lengths:
        rows = []
        for t in frames:
            target, valid = load_semantickitti_target(args.dataset_root, args.sequence, t)
            if target is None:
                continue
            chunks = []
            for f in range(max(0, t - hist + 1), t + 1):
                pts = scan(f)
                if f != t:
                    T = np.linalg.inv(P_velo[t]) @ P_velo[f]
                    pts = pts @ T[:3, :3].T + T[:3, 3]
                chunks.append(pts)
            vol = occupancy_from_points(np.concatenate(chunks, 0), G)
            rows.append(binary_occupancy_scores(vol, target, G, valid=valid))
        a = accumulate_scores(rows)
        out["history"][str(hist)] = a
        out["n_anchor_frames"] = a["n_frames"]
        print(f"GT LiDAR history {hist:4d}: IoU {a['iou']:.4f}  P {a['precision']:.3f}  "
              f"R {a['recall']:.3f}  (n={a['n_frames']})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
