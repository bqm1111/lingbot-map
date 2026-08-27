#!/usr/bin/env python
"""Phase-6 visual diagnosis: RGB, both depth maps, LiDAR, error, and occupancy overlays.

Frames are chosen **deterministically** by index across the sequence -- never by
picking the ones that happen to look good.
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

from prompted_lingbot.conventions import Sim3
from prompted_lingbot.external_depth import causal_translation_scale
from prompted_lingbot.occ_datasets import (
    SemanticKittiOccSpec, apply_transform, sparse_depth_from_velodyne,
)
from prompted_lingbot.occ_eval import (
    OccPointConfig, external_points_in_anchor_camera, points_in_anchor_camera,
)
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
from prompted_lingbot.occupancy import binary_occupancy_scores, occupancy_from_points
from prompted_lingbot.runner import load_cached


def overlay(ax, gt, pred, valid, title, axis=2):
    g, p, v = gt.any(axis=axis), pred.any(axis=axis), valid.any(axis=axis)
    img = np.zeros(g.shape + (3,))
    img[v & ~g & ~p] = (0.97, 0.97, 0.97)
    img[g & p] = (0.13, 0.55, 0.20)
    img[~g & p] = (0.85, 0.33, 0.10)
    img[g & ~p] = (0.20, 0.40, 0.80)
    img[~v] = (0.80, 0.80, 0.80)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="data/kitti/dataset")
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--lingbot-cache", required=True)
    ap.add_argument("--external-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 1500, 3000])
    ap.add_argument("--history", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    import cv2
    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence)
    lb_man = json.load(open(os.path.join(args.lingbot_cache, "manifest.json")))
    cfg = OccPointConfig(conf_threshold=1.5, max_depth=60.0, intrinsics="dataset")

    entries = sorted(lb_man["sequences"].items(),
                     key=lambda kv: kv[1]["extra"].get("source_start", 0))
    for gframe in args.frames:
        hit = None
        for name, meta in entries:
            st = int(meta["extra"].get("source_start", 0))
            if st <= gframe < st + int(meta["num_frames"]):
                hit = (name, st); break
        if hit is None:
            continue
        name, start = hit
        ext_path = os.path.join(args.external_cache, f"{name}.npz")
        if not os.path.isfile(ext_path):
            continue
        seq = load_cached(os.path.join(args.lingbot_cache, f"{name}.npz"),
                          lb_man["sequences"][name])
        ext = np.load(ext_path)["external_depth"]
        t = gframe - start
        target, valid = spec.target(gframe)
        if target is None:
            continue
        gt = (target != G.empty_class) & valid
        Ht, Wt = seq.pred_depth.shape[1:]
        lid, lidv = sparse_depth_from_velodyne(
            args.dataset_root, args.sequence, gframe, spec.calib,
            tuple(lb_man["sequences"][name]["source_image_hw"]), (Ht, Wt))

        est = causal_translation_scale(ext[t], seq.pred_depth[t].astype(np.float32),
                                       seq.pred_depth_conf[t].astype(np.float32))
        s_tr = float(np.exp(est[0])) if est else 1.0

        lb_pts = points_in_anchor_camera(seq, t, args.history, s_tr, cfg)
        lb_vol = occupancy_from_points(apply_transform(spec.cam_to_velo, lb_pts), G)
        hy_pts = external_points_in_anchor_camera(seq, ext, t, args.history, s_tr, cfg,
                                                  gate_on_lingbot_conf=True)
        hy_vol = occupancy_from_points(apply_transform(spec.cam_to_velo, hy_pts), G)
        or_pts = external_points_in_anchor_camera(seq, ext, t, args.history, 1.0, cfg,
                                                  poses_c2w=seq.gt_pose_c2w,
                                                  gate_on_lingbot_conf=True)
        or_vol = occupancy_from_points(apply_transform(spec.cam_to_velo, or_pts), G)
        sg_vol = occupancy_from_points(
            apply_transform(spec.cam_to_velo,
                            external_points_in_anchor_camera(seq, ext, t, 1, s_tr, cfg,
                                                             gate_on_lingbot_conf=True)), G)

        fig = plt.figure(figsize=(19, 9))
        img = cv2.cvtColor(cv2.imread(seq_path := os.path.join(
            args.dataset_root, "sequences", args.sequence, "image_2", f"{gframe:06d}.png")),
            cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(3, 4, 1); ax.imshow(img); ax.set_title("RGB", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        lbd = seq.pred_depth[t].astype(np.float32) * s_tr
        for i, (d, ttl) in enumerate([(lbd, f"LingbotMap depth x{s_tr:.1f}"),
                                      (np.asarray(ext[t], np.float32), "external metric depth"),
                                      (np.where(lidv, lid, np.nan), "projected LiDAR")]):
            ax = fig.add_subplot(3, 4, 2 + i)
            im = ax.imshow(d, cmap="turbo", vmin=0, vmax=50)
            ax.set_title(ttl, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        m = lidv & (lid > 0.5)
        for i, (d, ttl) in enumerate([(lbd, "LingbotMap |err| vs LiDAR"),
                                      (np.asarray(ext[t], np.float32), "external |err| vs LiDAR")]):
            e = np.where(m, np.abs(d - lid), np.nan)
            ax = fig.add_subplot(3, 4, 5 + i)
            ax.imshow(e, cmap="magma", vmin=0, vmax=8)
            ar = float(np.nanmean(np.abs(d[m] - lid[m]) / lid[m])) if m.any() else float("nan")
            ax.set_title(f"{ttl}  AbsRel {ar:.3f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        panels = [(lb_vol, "A LingbotMap depth+poses"), (sg_vol, "B external, 1 frame"),
                  (hy_vol, "C hybrid: ext depth + LB poses"),
                  (or_vol, "D ext depth + DATASET poses (oracle)")]
        for i, (vol, ttl) in enumerate(panels):
            sc = binary_occupancy_scores(vol, target, G, valid=valid)
            ax = fig.add_subplot(3, 4, 8 + i) if i < 4 else None
            overlay(ax, gt, vol, valid, f"{ttl}\nIoU {sc['iou']:.4f} P {sc['precision']:.2f} "
                                        f"R {sc['recall']:.2f}")
        fig.legend(handles=[Patch(color=(0.13, 0.55, 0.20), label="TP"),
                            Patch(color=(0.85, 0.33, 0.10), label="FP"),
                            Patch(color=(0.20, 0.40, 0.80), label="FN"),
                            Patch(color=(0.80, 0.80, 0.80), label="not evaluated")],
                   loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"seq {args.sequence} frame {gframe} — history {args.history} "
                     f"(BEV, collapsed over z)", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        out = os.path.join(args.output_dir, f"ds_{args.sequence}_{gframe:06d}_h{args.history}.png")
        fig.savefig(out, dpi=115); plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
