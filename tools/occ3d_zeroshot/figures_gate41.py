#!/usr/bin/env python
"""Gate 4.1 — representative-scene visualisation of the 2x2 geometry factorization.

Fixed, documented selection rule, applied to the **deployable** configuration only so the
choice of scenes cannot be influenced by any oracle result:

    rank the 150 official val scenes by G00|v3 scene IoU, then take the
    95th, 50th and 5th percentile scenes (indices 142, 75, 7 of the sorted list).

    python tools/occ3d_zeroshot/figures_gate41.py
"""
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
from occ3d_zeroshot.factorization import fuse, gt_camera_to_world, relative_transforms
from occ3d_zeroshot.grid import canonical_to_native, native_binary_target
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames
from occ3d_zeroshot.pipeline import (
    canonical_features, clip_scale, load_corrector, load_depth_head, region_from,
    run_corrector,
)

TP, FP, FN = "#2e7d32", "#c62828", "#b0bec5"
PCTL = (0.95, 0.50, 0.05)
RULE_ROW = "G00|v3"


def bev(pred, gt, keep):
    p, t = (pred & keep).cpu().numpy(), (gt & keep).cpu().numpy()
    tp, fp, fn = (p & t).any(2), (p & ~t).any(2), (~p & t).any(2)
    img = np.ones((*tp.shape, 3), np.float32)
    for m, c in ((fn, FN), (fp, FP), (tp, TP)):
        img[m] = np.array(matplotlib.colors.to_rgb(c))
    return np.transpose(img, (1, 0, 2))[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/gate41_factorization.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    dev = torch.device(g4.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    ps = json.load(open(os.path.join(art, "factorization_per_scene_iou.json")))
    ranked = sorted(ps[RULE_ROW], key=ps[RULE_ROW].get)
    picks = []
    for p, name in zip(PCTL, ("p95 (strong)", "p50 (median)", "p05 (weak)")):
        i = int(round(p * (len(ranked) - 1)))
        picks.append((ranked[i], f"{name}\nrank {i+1}/150"))

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)
    geo = {}
    import csv as _csv
    for row in _csv.DictReader(open(os.path.join(art, "geometry_per_clip.csv"))):
        geo[row["clip_id"]] = row

    ann = load_annotations(g4.data.occ3d_root)
    head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
    v3, ckv = load_corrector(os.path.join(REPO_ROOT, g4.frozen.full_s0), dev)

    COLS = [("G00", "raw", "G00  const s0 + pred poses"),
            ("G10", "raw", "G10 * const s0 + GT poses"),
            ("G01", "raw", "G01 * LiDAR scale + pred poses"),
            ("G11", "raw", "G11 * LiDAR scale + GT poses"),
            ("G01", "dilate_r2", "G01 * + dilate_r2 (0.4 m)"),
            ("G01", "v3", "G01 * + frozen V3")]
    fig, axes = plt.subplots(3, 6, figsize=(23.0, 13.2))
    for i, (scene, label) in enumerate(picks):
        clips = sorted(by_scene[scene], key=lambda r: r["clip_id"])
        r = clips[len(clips) // 2]
        d = np.load(os.path.join(g4.data.cache_root, r["clip_id"] + ".npz"))
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)
        anchor = dep.shape[0] - 1
        frames = [f for f in scene_frames(ann, scene, g4.data.camera, g4.data.nuscenes_root)
                  if f.token in set(r["sample_tokens"])]
        frames = sorted(frames, key=lambda f: r["sample_tokens"].index(f.token))
        c2w = gt_camera_to_world(frames)
        S = {"const": float(fv.s0), "oracle": float(geo[r["clip_id"]]["s_oracle"])}
        lab = np.load(os.path.join(g4.data.occ3d_root, r["anchor_gt_path"]))
        gt_np, keep_np = native_binary_target(lab, True, False,
                                              int(g4.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev); keep = torch.from_numpy(keep_np).to(dev)

        cache = {}
        for cell, cor, name in COLS:
            stag, pmode = {"G00": ("const", "pred"), "G01": ("oracle", "pred"),
                           "G10": ("const", "gt"), "G11": ("oracle", "gt")}[cell]
            if cell not in cache:
                s = S[stag]
                rel = relative_transforms(pose, c2w, anchor, pmode,
                                          s if pmode == "pred" else float("nan"))
                pe, fr, cf, dp = fuse(dep, conf, K, rel, Tce, s,
                                      float(fv.confidence_threshold),
                                      float(fv.min_depth_m), float(fv.max_depth_m))
                feat5, _ = canonical_features(pe, fr, cf, dp, dev)
                cache[cell] = (feat5, feat5[0] > 0, region_from(feat5[0] > 0, 3))
            feat5, occ, R = cache[cell]
            pc = (occ if cor == "raw" else
                  control(occ, R, **dict(fv.v1_control)) if cor == "dilate_r2" else
                  run_corrector(v3, feat5, R, ckv["norm"], float(ckv["threshold"]), False))
            pn = canonical_to_native(pc)
            j = [c for c in COLS].index((cell, cor, name))
            ax = axes[i, j]
            ax.imshow(bev(pn, gt, keep), interpolation="nearest", aspect="equal")
            p, t = pn & keep, gt & keep
            tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
            ax.set_title(f"{name}\nIoU {tp/max(tp+fp+fn,1):.3f}   {int(p.sum()):,} occ",
                         fontsize=10.0, pad=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd8dc")
        axes[i, 0].set_ylabel(f"{label}\n{scene}", fontsize=10.5, labelpad=12)

    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 4.1 — geometry factorization on Occ3D-nuScenes val "
                         "(scale x poses, frozen stack)", ha="center", fontsize=15.5,
             weight="bold")
    fig.text(0.5, 0.957, "* = non-deployable diagnostic oracle.  Bird's-eye view of the "
                         "official 200x200x16 grid at 0.4 m; ego at centre, camera looks "
                         "right.  Scenes chosen by p95/p50/p05 of the DEPLOYABLE G00|v3 "
                         "scene IoU, before any oracle row was inspected.",
             ha="center", fontsize=10.0, color="#546e7a")
    fig.subplots_adjust(top=0.900, bottom=0.055, left=0.050, right=0.99,
                        hspace=0.19, wspace=0.06)
    out = os.path.join(art, "scenes_factorization.png")
    fig.savefig(out, dpi=125)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}  scenes={[s for s, _ in picks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
