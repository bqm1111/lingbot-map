#!/usr/bin/env python
"""Export LingBot vs GT-LiDAR occupancy volumes as compact 3D voxel data.

Renders the *same* anchor frame two ways, on the identical 256x256x32 / 0.2 m
SemanticKITTI SSC grid and the identical binary metric:

``lingbot``  causal prediction -- LingBot depth, predicted poses, running metric
             scale anchor (the configuration that scores 0.087 / 0.084).
``lidar``    the ceiling -- ground-truth LiDAR scans accumulated over the same
             causal history with ground-truth LiDAR poses.

Both are compared against the same ``(target, valid)`` voxels, so every voxel is
labelled TP / FP / FN exactly as ``binary_occupancy_scores`` counts it.  Output is a
JSON payload of int16 voxel coordinates + a class byte, ready for a browser renderer.

    python scripts/export_occupancy_3d.py --dataset-root data/kitti/dataset \
      --cache-dir outputs/prompted_lingbot/cache_semkitti08 \
      --output outputs/prompted_lingbot/occupancy_geometry/semantickitti/vis3d/frame_data.json \
      --frames 100 1000 --history 20
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.conventions import Sim3, umeyama_sim3
from prompted_lingbot.occ_datasets import (
    SemanticKittiOccSpec, apply_transform, load_semantickitti_target, relative_c2w,
)
from prompted_lingbot.occ_eval import OccPointConfig, points_in_anchor_camera
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
from prompted_lingbot.occupancy import binary_occupancy_scores, occupancy_from_points
from prompted_lingbot.runner import load_cached

# Voxel classes as rendered.
TP, FP, FN = 0, 1, 2


def lidar_poses(root: str, sequence: str, Tr: np.ndarray) -> np.ndarray:
    """KITTI ``poses.txt`` are CAMERA poses; LiDAR poses are ``Tr^-1 @ pose @ Tr``."""
    cam = np.loadtxt(os.path.join(root, "poses", f"{sequence}.txt")).reshape(-1, 3, 4)
    P = np.tile(np.eye(4), (len(cam), 1, 1))
    P[:, :3, :4] = cam
    Tr_inv = np.linalg.inv(Tr)
    return np.einsum("ij,njk,kl->nil", Tr_inv, P, Tr)


def classify(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, max_voxels: int,
             rng: np.random.Generator) -> dict:
    """Label every evaluated voxel TP/FP/FN and subsample each class independently."""
    cls = {
        TP: np.argwhere(valid & gt & pred),
        FP: np.argwhere(valid & ~gt & pred),
        FN: np.argwhere(valid & gt & ~pred),
    }
    out_xyz, out_c, kept = [], [], {}
    for c, idx in cls.items():
        n = len(idx)
        # Subsample per class so the rarest class stays visible.
        if n > max_voxels:
            idx = idx[rng.choice(n, max_voxels, replace=False)]
        kept[c] = {"total": int(n), "shown": int(len(idx))}
        out_xyz.append(idx.astype(np.int16))
        out_c.append(np.full(len(idx), c, dtype=np.uint8))
    xyz = np.concatenate(out_xyz, 0) if out_xyz else np.zeros((0, 3), np.int16)
    return {
        "xyz_b64": base64.b64encode(np.ascontiguousarray(xyz).tobytes()).decode(),
        "cls_b64": base64.b64encode(np.concatenate(out_c).tobytes()).decode(),
        "counts": {str(k): v for k, v in kept.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--history", type=int, default=20)
    ap.add_argument("--conf-threshold", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=60.0)
    ap.add_argument("--max-voxels-per-class", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    spec = SemanticKittiOccSpec.build(a.dataset_root, a.sequence)
    manifest = json.load(open(os.path.join(a.cache_dir, "manifest.json")))
    cfg = OccPointConfig(conf_threshold=a.conf_threshold, max_depth=a.max_depth)
    P_velo = lidar_poses(a.dataset_root, a.sequence, spec.calib["Tr"])
    velo_dir = os.path.join(a.dataset_root, "sequences", a.sequence, "velodyne")
    entries = sorted(manifest["sequences"].items(),
                     key=lambda kv: kv[1]["extra"].get("source_start", 0))

    payload = {
        "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size, "origin": list(G.origin)},
        "sequence": a.sequence, "history": a.history, "frames": [],
    }

    for gframe in a.frames:
        chunk = chunk_start = None
        for name, meta in entries:
            s = int(meta["extra"].get("source_start", 0))
            if s <= gframe < s + int(meta["num_frames"]):
                chunk, chunk_start = name, s
                break
        if chunk is None:
            print(f"frame {gframe}: not cached, skipped")
            continue
        target, valid = spec.target(gframe)
        if target is None:
            print(f"frame {gframe}: no GT voxels, skipped")
            continue
        gt = (target != G.empty_class) & valid

        # -- LingBot causal prediction -------------------------------------- #
        seq = load_cached(os.path.join(a.cache_dir, f"{chunk}.npz"),
                          manifest["sequences"][chunk])
        t = gframe - chunk_start
        S = umeyama_sim3(seq.pred_pose_c2w[:, :3, 3], seq.gt_pose_c2w[:, :3, 3])
        corr = Sim3(S.s, np.eye(3), np.zeros(3))
        pts_cam = points_in_anchor_camera(seq, t, a.history, corr.s, cfg)
        pred_lb = occupancy_from_points(apply_transform(spec.cam_to_velo, pts_cam), G)

        # -- GT-LiDAR ceiling, identical history ---------------------------- #
        chunks = []
        for f in range(max(0, gframe - a.history + 1), gframe + 1):
            p = np.fromfile(os.path.join(velo_dir, f"{f:06d}.bin"),
                            dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
            if f != gframe:
                T = np.linalg.inv(P_velo[gframe]) @ P_velo[f]
                p = p @ T[:3, :3].T + T[:3, 3]
            chunks.append(p)
        pred_ld = occupancy_from_points(np.concatenate(chunks, 0), G)

        # The labels themselves, as a reference row: what the benchmark actually asks
        # for. Plotted against themselves this is 100 % agreement by definition, which
        # is exactly why it is shown separately from the two reconstructions.
        gt_idx = np.argwhere(gt)
        if len(gt_idx) > 3 * a.max_voxels_per_class:
            gt_idx = gt_idx[rng.choice(len(gt_idx), 3 * a.max_voxels_per_class, replace=False)]
        entry = {
            "frame": gframe,
            "gt_total": int(gt.sum()),
            "gt_xyz_b64": base64.b64encode(
                np.ascontiguousarray(gt_idx.astype(np.int16)).tobytes()).decode(),
            "models": {},
        }
        for name, pred in (("lingbot", pred_lb), ("lidar", pred_ld)):
            sc = binary_occupancy_scores(pred, target, G, valid=valid)
            entry["models"][name] = {
                "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                "n_pred": int(pred[valid].sum()), "n_gt": int(gt.sum()),
                **classify(pred, gt, valid, a.max_voxels_per_class, rng),
            }
            print(f"frame {gframe} {name:8s} IoU {sc['iou']:.4f} P {sc['precision']:.3f} "
                  f"R {sc['recall']:.3f}", flush=True)
        entry["scale"] = float(corr.s)
        payload["frames"].append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(payload, open(a.output, "w"))
    print(f"wrote {a.output} ({os.path.getsize(a.output)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
