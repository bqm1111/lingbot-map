"""Why our SC IoU beats OccAny's while our SSC mIoU is lower.

Reads only frozen numbers already in the repository (``artifacts/final_zero_shot/tables/per_class.csv`` and the OccAny
comparison table); no target dataset, prediction or label is opened. SC IoU is class-agnostic occupancy; SSC mIoU is the
UNWEIGHTED mean over classes, so a system that is strong on occupancy and on a few frequent classes, and near zero on many
rare ones, lands exactly where we are.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PER_CLASS = os.path.join(REPO, "artifacts", "final_zero_shot", "tables", "per_class.csv")
OUT_DIR = os.path.join(REPO, "artifacts", "metric_explainer")
#: OccAny published aggregates (Level B, paper Tab. 4) and released-geometry Track P IoU (Level A, same evaluator)
OCCANY = {"semantickitti": {"sc_iou_track_p": 31.28, "ssc_miou_official": 7.28}, "occ3d": {"sc_iou_track_p": 30.05, "ssc_miou_official": 6.66}}
OURS = {"semantickitti": {"sc_iou_track_p": 32.65, "ssc_miou_official": 4.43}, "occ3d": {"sc_iou_track_p": 36.87, "ssc_miou_official": 3.80}}
LABEL = {"semantickitti": "SemanticKITTI 08 (19 classes)", "occ3d": "Occ3D-nuScenes val (17 classes)"}


def per_class(dataset: str) -> List[Dict]:
    rows = [r for r in csv.DictReader(open(PER_CLASS)) if r["dataset"] == dataset]
    return sorted(rows, key=lambda r: -float(r["B3_iou_ssc_P"]))


def figure() -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2.1, 1]})
    for j, ds in enumerate(("semantickitti", "occ3d")):
        rows = per_class(ds)
        iou = np.array([100 * float(r["B3_iou_ssc_P"]) for r in rows])
        share = np.array([100 * float(r["gt_share_central"]) for r in rows])
        names = [r["class"] for r in rows]
        miou, weighted = iou.mean(), float((iou * share).sum() / share.sum())

        ax = axes[j][0]
        ax.bar(np.arange(len(iou)), iou, color=["#2ca02c" if s >= 5 else "#9ecae1" for s in share])
        ax.axhline(miou, color="#d62728", lw=1.6, label=f"our mIoU (unweighted mean) {miou:.2f}")
        ax.axhline(weighted, color="#1f77b4", lw=1.6, ls="--", label=f"our GT-frequency-weighted IoU {weighted:.2f}")
        ax.axhline(OCCANY[ds]["ssc_miou_official"], color="#ff7f0e", lw=1.6, ls=":",
                   label=f"OccAny published mIoU {OCCANY[ds]['ssc_miou_official']:.2f} (Level B, different setting)")
        for x, (v, s) in enumerate(zip(iou, share)):
            if s >= 5:
                ax.text(x, v + 0.6, f"{s:.0f}% of GT", ha="center", fontsize=7, color="#2ca02c")
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("class IoU % (Track P, SSC)"); ax.set_title(f"{LABEL[ds]}: per-class IoU, B3 — green = class holds >= 5 % of GT voxels", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

        ax = axes[j][1]
        groups = ["SC IoU\n(occupancy only)", "SSC mIoU\n(mean over classes)"]
        ours = [OURS[ds]["sc_iou_track_p"], OURS[ds]["ssc_miou_official"]]
        theirs = [OCCANY[ds]["sc_iou_track_p"], OCCANY[ds]["ssc_miou_official"]]
        x = np.arange(2)
        ax.bar(x - 0.2, ours, 0.4, color="#2ca02c", label="ours (1 camera, causal)")
        ax.bar(x + 0.2, theirs, 0.4, color="#ff7f0e", label="OccAny")
        for xi, (a, b) in enumerate(zip(ours, theirs)):
            ax.text(xi - 0.2, a + 0.4, f"{a:.2f}", ha="center", fontsize=8); ax.text(xi + 0.2, b + 0.4, f"{b:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9); ax.set_ylabel("%")
        ax.set_title("geometry (Level A, same evaluator) vs semantics (Level B, published)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Class-agnostic occupancy vs unweighted class mean: the two metrics measure different things", fontsize=12)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "iou_vs_miou.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


if __name__ == "__main__":
    print(figure())
