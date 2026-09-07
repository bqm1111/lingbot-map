#!/usr/bin/env python
"""Gate 4 — bird's-eye visualisation of representative Occ3D-nuScenes val scenes."""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config
from gates.voxel_gate.controls import control
from occ3d_zeroshot.grid import NATIVE, canonical_to_native, native_binary_target
from occ3d_zeroshot.pipeline import (
    canonical_features, clip_scale, fuse_to_ego, load_corrector, load_depth_head,
    region_from, run_corrector,
)

TP, FP, FN = "#2e7d32", "#c62828", "#b0bec5"


def bev(pred, gt, keep):
    p, t = (pred & keep).cpu().numpy(), (gt & keep).cpu().numpy()
    tp, fp, fn = (p & t).any(2), (p & ~t).any(2), (~p & t).any(2)
    img = np.ones((*tp.shape, 3), np.float32)
    for m, c in ((fn, FN), (fp, FP), (tp, TP)):
        img[m] = np.array(matplotlib.colors.to_rgb(c))
    return np.transpose(img, (1, 0, 2))[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config); fv = cfg.frozen_values
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    ps = json.load(open(os.path.join(art, "eval", "per_scene_iou.json")))
    order = sorted(ps["T6"], key=ps["T6"].get)
    picks = [(order[-1], "best scene"), (order[len(order) // 2], "median scene"),
             (order[0], "worst scene")]

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)

    head, hck = load_depth_head(os.path.join(REPO_ROOT, cfg.frozen.depth_head), dev)
    full, ckf = load_corrector(os.path.join(REPO_ROOT, cfg.frozen.full_s0), dev)
    occm, cko = load_corrector(os.path.join(REPO_ROOT, cfg.frozen.occ_only_s0), dev)

    cols = ["T1  C3 frozen input", "T2  dilate_r2", "T4  A_occ_only",
            "T6  full V3 seed 0", "REF  in-band oracle"]
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 13.0))
    for i, (scene, label) in enumerate(picks):
        r = by_scene[scene][len(by_scene[scene]) // 2]
        d = np.load(os.path.join(cfg.data.cache_root, r["clip_id"] + ".npz"))
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64); pose = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)
        _, s, _, _ = clip_scale(head, hck, dep, conf, float(fv.s0),
                                float(fv.confidence_threshold), float(fv.min_depth_m),
                                float(fv.max_depth_m), dev)
        pe, fr, cf, dp = fuse_to_ego(dep, conf, K, pose, Tce, s,
                                     float(fv.confidence_threshold),
                                     float(fv.min_depth_m), float(fv.max_depth_m))
        feat5, _ = canonical_features(pe, fr, cf, dp, dev)
        occ = feat5[0] > 0
        R = region_from(occ, 3)
        lab = np.load(os.path.join(cfg.data.occ3d_root, r["anchor_gt_path"]))
        gt_np, keep_np = native_binary_target(lab, True, False,
                                              int(cfg.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev); keep = torch.from_numpy(keep_np).to(dev)
        gtc = gt.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        P = [occ, control(occ, R, **dict(fv.v1_control)),
             run_corrector(occm, feat5, R, cko["norm"], float(cko["threshold"]), True),
             run_corrector(full, feat5, R, ckf["norm"], float(ckf["threshold"]), False),
             gtc & R]
        for j, (pc, name) in enumerate(zip(P, cols)):
            pn = canonical_to_native(pc)
            ax = axes[i, j]
            ax.imshow(bev(pn, gt, keep), interpolation="nearest", aspect="equal")
            p, t = (pn & keep), (gt & keep)
            tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
            iou = tp / max(tp + fp + fn, 1)
            ax.set_title(f"{name}\nIoU {iou:.3f}   {int(p.sum()):,} occ",
                         fontsize=10.5, pad=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd8dc")
        axes[i, 0].set_ylabel(f"{label}\n{scene}\nT6 scene IoU {ps['T6'][scene]:.3f}",
                              fontsize=11, labelpad=12)

    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 4 — frozen SemanticKITTI stack evaluated zero-shot on "
                         "Occ3D-nuScenes val (CAM_FRONT)", ha="center", fontsize=15.5,
             weight="bold")
    fig.text(0.5, 0.958, "bird's-eye view of the official 200x200x16 grid at 0.4 m "
                         "(80 m x 80 m, ego at centre, front camera looks right). Grey = "
                         "ground truth the method missed; only camera-visible voxels are scored.",
             ha="center", fontsize=10.5, color="#546e7a")
    fig.subplots_adjust(top=0.905, bottom=0.055, left=0.055, right=0.99,
                        hspace=0.19, wspace=0.06)
    out = os.path.join(art, "eval", "scenes_zeroshot.png")
    fig.savefig(out, dpi=125)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}  scenes={[s for s, _ in picks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
