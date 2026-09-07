#!/usr/bin/env python
"""Gate 5.2 step 8 — scale/FOV diagnostic plots and representative BEV panels.

The BEV clips are chosen by a **baseline-only** rule declared in advance: the 10th, 50th
and 90th percentile of ``K360-C0-D`` IoU. The selection therefore never sees A, B or the
oracle, so the panels cannot be cherry-picked in the gauge's favour.

    python tools/gate5_2/figures.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from prompted_lingbot.occ_datasets import apply_transform
from gates.scale_gate.config import REPO_ROOT, load_config
from gates.scale_gate.kitti import read_manifest
from sscbench_kitti360.adapter import (SSCBENCH_KITTI360_GRID as G, binary_target,
                                       load_target)
from gates.voxel_gate.c3 import as4x4, c3_points

COLOURS = {"C0": "#888888", "C3": "#1f77b4", "A": "#ff7f0e", "B": "#2ca02c",
           "OR": "#d62728"}
LABELS = {"C0": "C0 (constant $s_0$)", "C3": "C3 (learned clip head)",
          "A": "A (MoGe, inferred FOV)", "B": "B (MoGe, calibrated FOV)",
          "OR": "LiDAR oracle (diagnostic)"}


def scale_plots(art, rows, out):
    S = {c: np.array([float(r[f"s_{c}"]) for r in rows]) for c in ("C0", "C3", "A", "B")}
    S["OR"] = np.array([float(r["s_oracle"]) for r in rows])
    ok = np.all([np.isfinite(v) & (v > 0) for v in S.values()], axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(8, 45, 80)
    for c in ("C0", "C3", "A", "B", "OR"):
        ax[0].hist(S[c][ok], bins=bins, histtype="step", lw=1.8, color=COLOURS[c],
                   label=LABELS[c])
    ax[0].set_xlabel("clip scale"); ax[0].set_ylabel("clips")
    ax[0].set_title("Scale distributions, SSCBench-KITTI-360 val")
    ax[0].legend(fontsize=8)

    for c in ("C3", "A", "B"):
        ax[1].scatter(S["OR"][ok], S[c][ok], s=3, alpha=.25, color=COLOURS[c],
                      label=LABELS[c])
    lim = [S["OR"][ok].min() * .9, S["OR"][ok].max() * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1, label="identity")
    ax[1].axhline(27.3665, color=COLOURS["C0"], ls=":", lw=1.5, label="$s_0$ = 27.3665")
    ax[1].set_xlim(lim); ax[1].set_xlabel("LiDAR oracle scale")
    ax[1].set_ylabel("estimated scale"); ax[1].legend(fontsize=8)
    ax[1].set_title("Estimate vs oracle")

    for c in ("C0", "C3", "A", "B"):
        ax[2].hist(np.log(S[c][ok] / S["OR"][ok]), bins=np.linspace(-.8, .8, 90),
                   histtype="step", lw=1.8, color=COLOURS[c], label=LABELS[c])
    ax[2].axvline(0, color="k", lw=1)
    ax[2].set_xlabel("log(scale / oracle)"); ax[2].set_ylabel("clips")
    ax[2].set_title("Log-scale error")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fov_plot(art, out):
    a = list(csv.DictReader(open(os.path.join(art, "scales_A.csv"))))
    b = list(csv.DictReader(open(os.path.join(art, "scales_B.csv"))))
    cal = float(a[0]["fov_calibrated_deg"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist([float(r["fov_moge_out_deg"]) for r in a], bins=60, color=COLOURS["A"],
               alpha=.75, label="A: MoGe-inferred")
    ax[0].axvline(cal, color="k", ls="--", lw=2, label=f"calibrated {cal:.2f}$^\\circ$")
    ax[0].set_xlabel("horizontal FOV (deg)"); ax[0].set_ylabel("clips")
    ax[0].set_title("KITTI-360: MoGe-inferred vs calibrated FOV"); ax[0].legend(fontsize=9)
    sa = np.array([float(r["s_moge"]) for r in a])
    sb = np.array([float(r["s_moge"]) for r in b])
    ax[1].scatter(sa, sb, s=3, alpha=.25, color="#444")
    lim = [min(sa.min(), sb.min()) * .95, max(sa.max(), sb.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("A scale (inferred FOV)"); ax[1].set_ylabel("B scale (calibrated FOV)")
    ax[1].set_title(f"Supplying the calibrated FOV moves the scale by "
                    f"{100*np.median(sb/sa-1):+.1f}% (median)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def bev_panels(cfg, art, out):
    per = list(csv.DictReader(open(os.path.join(art, "gate5_2_per_clip.csv"))))
    base = {r["clip_id"]: float(r["iou"]) for r in per
            if r["condition"] == "C0" and r["corrector"] == "dilate_r2"}
    ids = sorted(base, key=lambda c: base[c])
    picks = [ids[int(q * (len(ids) - 1))] for q in (0.10, 0.50, 0.90)]
    recs = {r["clip_id"]: r for r in read_manifest(
        os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))}
    S = {}
    for r in csv.DictReader(open(os.path.join(art, "scales_C0C3.csv"))):
        S.setdefault(r["clip_id"], {}).update({"C0": float(r["s_c0"]),
                                               "C3": float(r["s_c3"])})
    for v in ("A", "B"):
        for r in csv.DictReader(open(os.path.join(art, f"scales_{v}.csv"))):
            S.setdefault(r["clip_id"], {})[v] = float(r["s_moge"])
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)

    conds = ["GT", "C0", "C3", "A", "B"]
    fig, axes = plt.subplots(len(picks), len(conds), figsize=(4 * len(conds),
                                                              3.4 * len(picks)))
    for i, cid in enumerate(picks):
        rec = recs[cid]
        d = np.load(os.path.join(cfg.lingbot.cache_root, cid + ".npz"), allow_pickle=False)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose = as4x4(d["pred_pose_c2w"])
        c2v = d["rect_cam_to_velo"].astype(np.float64)
        t, _ = load_target(cfg.dataset.root, int(rec["anchor"]), G)
        gt, keep = binary_target(t, G)
        for j, c in enumerate(conds):
            ax = axes[i, j]
            if c == "GT":
                vol = gt
            else:
                pts, *_ = c3_points(dep, conf, K, pose, S[cid][c], conf_thr, dmin, dmax)
                pg = apply_transform(c2v, pts)
                idx = np.floor((pg - np.asarray(G.origin)) / G.voxel_size).astype(np.int64)
                m = np.all((idx >= 0) & (idx < np.asarray(G.dims)), axis=1)
                vol = np.zeros(G.dims, bool)
                vol[idx[m, 0], idx[m, 1], idx[m, 2]] = True
            ax.imshow(vol.any(2).T, origin="lower", cmap="Greys",
                      extent=[G.origin[0], G.upper[0], G.origin[1], G.upper[1]])
            ax.set_title(f"{'ground truth' if c == 'GT' else LABELS[c]}"
                         + ("" if c == "GT" else f"  s={S[cid][c]:.1f}"), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{cid.split('_')[-1]}\nC0-D IoU {base[cid]:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Bird's-eye occupancy, clips at the 10/50/90th percentile of the "
                 "K360-C0-D baseline IoU (selection uses the baseline only)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=120); plt.close(fig)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    made = []
    p = os.path.join(art, "oracle_scales.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        scale_plots(art, rows, os.path.join(art, "gate5_2_scales.png"))
        made.append("gate5_2_scales.png")
    fov_plot(art, os.path.join(art, "gate5_2_fov.png"))
    made.append("gate5_2_fov.png")
    if os.path.exists(os.path.join(art, "gate5_2_per_clip.csv")):
        picks = bev_panels(cfg, art, os.path.join(art, "gate5_2_bev.png"))
        made.append("gate5_2_bev.png")
        json.dump({"rule": "10/50/90th percentile of K360-C0-D IoU (baseline only)",
                   "clips": picks},
                  open(os.path.join(art, "bev_selection.json"), "w"), indent=2)
    print("wrote: " + ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
