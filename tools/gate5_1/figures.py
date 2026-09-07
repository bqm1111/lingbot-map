#!/usr/bin/env python
"""Gate 5.1 — scale/FOV diagnostic plots and representative BEV comparisons.

BEV scene selection reuses the PRE-EXISTING Gate-5 baseline rule unchanged: rank the 150
Occ3D val scenes by the deployable M0-D scene IoU and take p95 / p50 / p05. The rule does
not consult any G51 result.
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
VARIANTS = ["G51-A", "G51-B", "G51-C", "G51-D"]
COL = {"G51-A": "#455a64", "G51-B": "#1565c0", "G51-C": "#ef6c00", "G51-D": "#2e7d32"}
RULE_ROW = "M0-D"


def scale_plots(cfg, art):
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.8))
    for row, ds in enumerate(("occ3d", "kitti")):
        d = json.load(open(os.path.join(art, f"diagnostics_{ds}.json")))
        rows = list(csv.DictReader(open(os.path.join(art, f"depth_frames_{ds}.csv"))))
        per = {}
        for r in rows:
            per.setdefault(r["clip_id"], r)
        so = np.array([float(v["s_oracle"]) for v in per.values()])

        ax = axes[row, 0]
        lo = min(so.min(), min(np.quantile([float(v[f"s_{x}"]) for v in per.values()], .01)
                               for x in VARIANTS))
        hi = max(np.quantile(so, .99),
                 max(np.quantile([float(v[f"s_{x}"]) for v in per.values()], .99)
                     for x in VARIANTS))
        bins = np.linspace(lo * 0.9, hi * 1.1, 60)
        ax.hist(so, bins=bins, alpha=.45, label="LiDAR oracle", color="#9e9e9e")
        for v in VARIANTS:
            s = np.array([float(x[f"s_{v}"]) for x in per.values()])
            ax.hist(s, bins=bins, histtype="step", lw=1.8, label=v, color=COL[v])
        ax.set_xlabel("clip scale"); ax.set_ylabel("clips")
        ax.set_title(f"{ds}: scale distribution", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 1]
        for v in VARIANTS:
            s = np.array([float(x[f"s_{v}"]) for x in per.values()])
            ax.scatter(so, s, s=5, alpha=.30, color=COL[v],
                       label=f"{v}  r={d[v]['pearson_log_vs_log_oracle']:.3f}")
        lim = [so.min() * .9, np.quantile(so, .995) * 1.1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("LiDAR-oracle scale"); ax.set_ylabel("estimated scale")
        ax.set_title(f"{ds}: estimate vs oracle", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row, 2]
        err = {v: np.abs(np.array([float(x[f"s_{v}"]) for x in per.values()]) / so - 1)
               for v in VARIANTS}
        bp = ax.boxplot(list(err.values()), tick_labels=list(err), showfliers=False)
        ax.axhline(0.05, color="#c62828", ls="--", lw=1.4,
                   label="5 % decision threshold")
        for i, (k, v) in enumerate(err.items(), 1):
            ax.text(i, np.median(v), f" {np.median(v):.3f}", va="center", fontsize=9)
        ax.set_ylabel("|relative scale error|")
        ax.set_title(f"{ds}: scale error vs oracle", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.975, "Gate 5.1 — calibrated MoGe-2 gauge: scale distribution, "
                         "oracle correlation and error", ha="center", fontsize=14.5,
             weight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.05, right=0.99, hspace=0.28,
                        wspace=0.22)
    p = os.path.join(art, "gate5_1_scale_plots.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {os.path.relpath(p, REPO_ROOT)}")


def fov_plot(cfg, art):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, ds in zip(axes, ("occ3d", "kitti")):
        vals, labels, cols = [], [], []
        for v in VARIANTS:
            j = json.load(open(os.path.join(art, f"scales_{v}_{ds}.json")))
            vals.append(j["fov_moge_out_deg"]); labels.append(f"{v}\nout"); cols.append(COL[v])
        j = json.load(open(os.path.join(art, f"scales_G51-A_{ds}.json")))
        cal_full = j["crop_and_calibration"]["calibrated_fov_x_deg"]
        jd = json.load(open(os.path.join(art, f"scales_G51-D_{ds}.json")))
        cal_crop = jd["crop_and_calibration"]["calibrated_fov_x_deg"]
        ax.bar(range(len(vals)), vals, color=cols, alpha=.8)
        ax.axhline(cal_full, color="#c62828", ls="--", lw=1.6,
                   label=f"calibrated (full) {cal_full:.2f}°")
        if abs(cal_crop - cal_full) > 1e-6:
            ax.axhline(cal_crop, color="#6a1b9a", ls=":", lw=1.6,
                       label=f"calibrated (crop) {cal_crop:.2f}°")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_ylabel("MoGe output FOV_x (deg)")
        ax.set_title(f"{ds}: MoGe FOV vs calibration", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Gate 5.1 — inferred versus calibrated horizontal field of view",
                 fontsize=13.5, weight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.22)
    p = os.path.join(art, "gate5_1_fov.png")
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
    from occ3d_zeroshot.pipeline import canonical_features, region_from
    g5 = load_config(cfg.experiment.gate5_config)
    g4 = load_config(cfg.experiment.gate4_config)
    g41 = load_config(cfg.experiment.gate41_config)
    fv = g4.frozen_values
    dev = torch.device(g5.moge.device if torch.cuda.is_available() else "cpu")
    # pre-existing Gate-5 baseline rule, unchanged
    pu5 = json.load(open(os.path.join(REPO_ROOT, g5.experiment.output_dir,
                                      "gate5_per_unit_iou_occ3d.json")))
    ranked = sorted(pu5[RULE_ROW], key=pu5[RULE_ROW].get)
    picks = [(ranked[int(round(p * (len(ranked) - 1)))], n) for p, n in
             ((0.95, "p95 (strong)"), (0.50, "p50 (median)"), (0.05, "p05 (weak)"))]
    load = lambda v: {r["clip_id"]: float(r["s_moge"]) for r in
                      csv.DictReader(open(os.path.join(art, f"scales_{v}_occ3d.csv")))
                      if int(r["ok"])}
    sA, sD = load("G51-A"), load("G51-D")
    geo = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(
        REPO_ROOT, g41.experiment.output_dir, "geometry_per_clip.csv")))}
    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, g5.datasets.occ3d.manifest))]
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)
    v1 = dict(fv.v1_control)
    COLS = [("M0", "raw"), ("M0", "dilate_r2"), ("G51-A", "dilate_r2"),
            ("G51-D", "dilate_r2"), ("OR", "dilate_r2")]
    TITLE = {("M0", "raw"): "M0-R  constant s0", ("M0", "dilate_r2"): "M0-D  + dilate_r2",
             ("G51-A", "dilate_r2"): "G51-A-D  Gate-5 gauge",
             ("G51-D", "dilate_r2"): "G51-D-D  calibrated gauge",
             ("OR", "dilate_r2"): "OR-D *  LiDAR oracle"}
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 13.0))
    for i, (scene, label) in enumerate(picks):
        clips = sorted(by_scene[scene], key=lambda r: r["clip_id"])
        r = clips[len(clips) // 2]
        d = np.load(os.path.join(g5.datasets.occ3d.lingbot_cache, r["clip_id"] + ".npz"))
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64); pose = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)
        anchor = dep.shape[0] - 1
        S = {"M0": float(fv.s0), "G51-A": sA[r["clip_id"]], "G51-D": sD[r["clip_id"]],
             "OR": float(geo[r["clip_id"]]["s_oracle"])}
        lab = np.load(os.path.join(g4.data.occ3d_root, r["anchor_gt_path"]))
        gt_np, keep_np = native_binary_target(lab, True, False,
                                              int(g4.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev); keep = torch.from_numpy(keep_np).to(dev)
        cache = {}
        for j, (cond, cor) in enumerate(COLS):
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
            ax.set_title(f"{TITLE[(cond, cor)]}\nIoU {tp/max(tp+fp+fn,1):.3f}   "
                         f"s={S[cond]:.1f}   {int(p.sum()):,} occ", fontsize=10.0, pad=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd8dc")
        axes[i, 0].set_ylabel(f"{label}\n{scene}", fontsize=10.5, labelpad=12)
    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 5.1 — calibrated MoGe-2 gauge on Occ3D-nuScenes val",
             ha="center", fontsize=15.5, weight="bold")
    fig.text(0.5, 0.957, "* = non-deployable diagnostic oracle.  Occ3D's aspect ratio is "
                         "1.76:1, so the aspect-safe crop is the identity here and G51-D "
                         "differs from G51-A only by the calibrated FOV.  Scenes chosen by "
                         "the pre-existing M0-D rule.", ha="center", fontsize=10.0,
             color="#546e7a")
    fig.subplots_adjust(top=0.900, bottom=0.055, left=0.052, right=0.99, hspace=0.19,
                        wspace=0.06)
    p = os.path.join(art, "gate5_1_scenes_bev.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    print(f"wrote {os.path.relpath(p, REPO_ROOT)}  scenes={[s for s,_ in picks]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_1/calibrated_gauge.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    scale_plots(cfg, art); fov_plot(cfg, art); bev_figure(cfg, art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
