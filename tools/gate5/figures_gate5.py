#!/usr/bin/env python
"""Gate 5 — scale-distribution / correlation plots and representative BEV comparisons.

BEV scene selection rule, fixed and applied to the DEPLOYABLE row only (M0-D), so no
oracle or MoGe result can influence the choice: rank the 150 Occ3D val scenes by M0-D
scene IoU and take the p95, p50 and p05 scenes.
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config
from gates.voxel_gate.controls import control

TP, FP, FN = "#2e7d32", "#c62828", "#b0bec5"
RULE_ROW = "M0-D"


def scale_plots(cfg, art):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))
    for row, ds in enumerate(("occ3d", "kitti")):
        d = json.load(open(os.path.join(art, f"gate5_scale_diagnostics_{ds}.json")))
        rows = list(csv.DictReader(open(os.path.join(art,
                                                     f"gate5_depth_frames_{ds}.csv"))))
        per = {}
        for r in rows:
            per.setdefault(r["clip_id"], r)
        so = np.array([float(v["s_oracle"]) for v in per.values()])
        sm = np.array([float(v["s_moge"]) for v in per.values()])
        sc = np.array([float(v["s_c3"]) for v in per.values()])
        s0 = np.array([float(v["s_c0"]) for v in per.values()])

        ax = axes[row, 0]
        bins = np.linspace(min(so.min(), sm.min()) * 0.9,
                           max(np.quantile(so, .99), np.quantile(sm, .99)) * 1.1, 60)
        ax.hist(so, bins=bins, alpha=.55, label="LiDAR oracle", color="#455a64")
        ax.hist(sm, bins=bins, alpha=.55, label="MoGe-2 (M2)", color="#1565c0")
        ax.hist(sc, bins=bins, alpha=.35, label="learned C3 (M1)", color="#ef6c00")
        ax.axvline(s0[0], color="#c62828", ls="--", lw=2, label="constant C0 (M0)")
        ax.set_xlabel("clip scale"); ax.set_ylabel("clips")
        ax.set_title(f"{ds}: scale distribution", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 1]
        ax.scatter(so, sm, s=6, alpha=.35, color="#1565c0", label="MoGe-2")
        ax.scatter(so, sc, s=6, alpha=.20, color="#ef6c00", label="learned C3")
        lim = [min(so.min(), sm.min()) * .9, max(so.max(), sm.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="identity")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("LiDAR-oracle scale"); ax.set_ylabel("estimated scale")
        ax.set_title(f"{ds}: MoGe r={d['moge']['pearson_log_vs_log_oracle']:.3f}  "
                     f"C3 r={d['c3']['pearson_log_vs_log_oracle']:.3f}", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 2]
        err = {"C0": np.abs(s0 / so - 1), "C3": np.abs(sc / so - 1),
               "MoGe-2": np.abs(sm / so - 1)}
        ax.boxplot(list(err.values()), labels=list(err), showfliers=False)
        for i, (k, v) in enumerate(err.items(), 1):
            ax.text(i, np.median(v), f" {np.median(v):.3f}", va="center", fontsize=9)
        ax.set_ylabel("|relative scale error|")
        ax.set_title(f"{ds}: scale error vs LiDAR oracle", fontsize=11)
    fig.text(0.5, 0.975, "Gate 5 — MoGe-2 metric gauge: clip-scale distributions, "
                         "correlation and error", ha="center", fontsize=14.5, weight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.05, right=0.99, hspace=0.28,
                        wspace=0.22)
    p = os.path.join(art, "gate5_scale_plots.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {os.path.relpath(p, REPO_ROOT)}")


def bev(pred, gt, keep):
    p, t = (pred & keep).cpu().numpy(), (gt & keep).cpu().numpy()
    tp, fp, fn = (p & t).any(2), (p & ~t).any(2), (~p & t).any(2)
    img = np.ones((*tp.shape, 3), np.float32)
    for m, c in ((fn, FN), (fp, FP), (tp, TP)):
        img[m] = np.array(matplotlib.colors.to_rgb(c))
    return np.transpose(img, (1, 0, 2))[::-1]


def bev_figure(cfg, art):
    from occ3d_zeroshot.factorization import fuse, relative_transforms
    from occ3d_zeroshot.grid import canonical_to_native, native_binary_target
    from occ3d_zeroshot.pipeline import (canonical_features, clip_scale,
                                         load_depth_head, region_from)
    g4 = load_config(cfg.experiment.gate4_config)
    g41 = load_config(cfg.experiment.gate41_config)
    fv = g4.frozen_values
    dev = torch.device(cfg.moge.device if torch.cuda.is_available() else "cpu")
    per_unit = json.load(open(os.path.join(art, "gate5_per_unit_iou_occ3d.json")))
    ranked = sorted(per_unit[RULE_ROW], key=per_unit[RULE_ROW].get)
    picks = [(ranked[int(round(p * (len(ranked) - 1)))], n) for p, n in
             ((0.95, "p95 (strong)"), (0.50, "p50 (median)"), (0.05, "p05 (weak)"))]
    ms = {r["clip_id"]: float(r["s_moge"]) for r in
          csv.DictReader(open(os.path.join(art, "moge_scale_occ3d.csv"))) if int(r["ok"])}
    geo = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(
        REPO_ROOT, g41.experiment.output_dir, "geometry_per_clip.csv")))}
    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, cfg.datasets.occ3d.manifest))]
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)
    head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
    v1 = dict(fv.v1_control)

    COLS = [("M0", "raw", "M0-R  constant s0"), ("M0", "dilate_r2", "M0-D  + dilate_r2"),
            ("M2", "raw", "M2-R  MoGe-2 scale"), ("M2", "dilate_r2", "M2-D  + dilate_r2"),
            ("OR", "dilate_r2", "OR-D * LiDAR oracle")]
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 13.0))
    for i, (scene, label) in enumerate(picks):
        clips = sorted(by_scene[scene], key=lambda r: r["clip_id"])
        r = clips[len(clips) // 2]
        d = np.load(os.path.join(cfg.datasets.occ3d.lingbot_cache, r["clip_id"] + ".npz"))
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64); pose = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)
        anchor = dep.shape[0] - 1
        S = {"M0": float(fv.s0), "M2": ms[r["clip_id"]],
             "OR": float(geo[r["clip_id"]]["s_oracle"])}
        lab = np.load(os.path.join(g4.data.occ3d_root, r["anchor_gt_path"]))
        gt_np, keep_np = native_binary_target(lab, True, False,
                                              int(g4.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev); keep = torch.from_numpy(keep_np).to(dev)
        cache = {}
        for j, (cond, cor, name) in enumerate(COLS):
            if cond not in cache:
                s = S[cond]
                rel = relative_transforms(pose, pose, anchor, "pred", s)
                pe, fr, cf, dp = fuse(dep, conf, K, rel, Tce, s,
                                      float(fv.confidence_threshold),
                                      float(fv.min_depth_m), float(fv.max_depth_m))
                feat5, _ = canonical_features(pe, fr, cf, dp, dev)
                occ = feat5[0] > 0
                cache[cond] = (occ, region_from(occ, 3))
            occ, R = cache[cond]
            pc = occ if cor == "raw" else control(occ, R, **v1)
            pn = canonical_to_native(pc)
            ax = axes[i, j]
            ax.imshow(bev(pn, gt, keep), interpolation="nearest", aspect="equal")
            p, t = pn & keep, gt & keep
            tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
            ax.set_title(f"{name}\nIoU {tp/max(tp+fp+fn,1):.3f}   s={S[cond]:.1f}   "
                         f"{int(p.sum()):,} occ", fontsize=10.0, pad=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd8dc")
        axes[i, 0].set_ylabel(f"{label}\n{scene}", fontsize=10.5, labelpad=12)
    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 5 — frozen MoGe-2 metric gauge on Occ3D-nuScenes val",
             ha="center", fontsize=15.5, weight="bold")
    fig.text(0.5, 0.957, "* = non-deployable diagnostic oracle.  MoGe-2 supplies ONE "
                         "scalar per clip; depth shape, intrinsics and poses stay "
                         "LingBot's.  Scenes chosen by p95/p50/p05 of the deployable "
                         "M0-D scene IoU.", ha="center", fontsize=10.0, color="#546e7a")
    fig.subplots_adjust(top=0.900, bottom=0.055, left=0.052, right=0.99, hspace=0.19,
                        wspace=0.06)
    p = os.path.join(art, "gate5_scenes_bev.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    print(f"wrote {os.path.relpath(p, REPO_ROOT)}  scenes={[s for s,_ in picks]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5/moge_metric_gauge.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    scale_plots(cfg, art)
    bev_figure(cfg, art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
