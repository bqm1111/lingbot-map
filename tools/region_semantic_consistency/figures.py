#!/usr/bin/env python
"""The seven plots of the region-semantic-consistency study.

    python tools/region_semantic_consistency/figures.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.region_semantic_consistency import common as RC                                # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

UNITS = ["voxel"] + list(RC.SCALES)
XLAB = ["voxel\n0.2 m", "small\n0.6 m", "medium\n1.0 m", "large\n1.8 m", "xlarge\n3.0 m"]
COL = {"voxel": "#555555", "small": "#2c6fbb", "medium": "#3a9d5d", "large": "#d98c3a", "xlarge": "#7a5ea8"}
SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}; SHORT["train_pooled"] = "train"


def res(an, d):
    return an["train_pooled"]["results"] if d == "train_pooled" else an["drives"][d]["results"]


def main() -> int:
    an = json.load(open(os.path.join(RC.ART, "analysis.json")))
    fig_dir = os.path.join(RC.ART, "figures"); os.makedirs(fig_dir, exist_ok=True)
    val = SRC.VAL_DRIVE; x = np.arange(len(UNITS))
    # 1. voxel vs region top-1 (0006 and train pooled), with R1/R2 references
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, d in enumerate((val, "train_pooled")):
        R = res(an, d)
        ax[k].bar(x, [R[u]["pairs"]["AB"]["top1"] for u in UNITS], color=[COL[u] for u in UNITS], label="A vs B (true)")
        ax[k].plot(x, [R[u]["controls"]["R1"]["top1"] for u in UNITS], "k_", ms=22, mew=2, label="R1 within-anchor shuffle")
        ax[k].plot(x, [R[u]["controls"]["R2"]["top1"] for u in UNITS], "r_", ms=22, mew=2, label="R2 local shuffle")
        ax[k].plot(x, [R[u]["R0_class_prior_top1"] for u in UNITS], "b.", ms=8, label="R0 pooled class prior")
        ax[k].plot(x, [R[u]["three_way"]["consistent"] for u in UNITS], "g^", ms=7, label="three-way A=B=C")
        ax[k].set_xticks(x); ax[k].set_xticklabels(XLAB); ax[k].set_ylim(0, 1); ax[k].set_ylabel("top-1 cross-view repeatability")
        ax[k].set_title(SHORT[d]); ax[k].grid(alpha=0.3, axis="y"); ax[k].legend(fontsize=7, loc="upper right")
    fig.suptitle("voxel vs region representation: independent cross-view repeatability"); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "1_voxel_vs_region_repeatability.png"), dpi=130); plt.close(fig)
    # 2. region size vs repeatability (by occupied voxels), 0006
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, d in enumerate((val, "train_pooled")):
        R = res(an, d)
        for u in RC.SCALES:
            b = R[u]["by_occupied_voxels"]; ax[k].plot(list(b), [v["top1"] for v in b.values()], "-o", color=COL[u], label=f"{u} (block {RC.SCALES[u]} vox)")
        ax[k].axhline(R["voxel"]["pairs"]["AB"]["top1"], color=COL["voxel"], ls="--", label="voxel")
        ax[k].set_xlabel("occupied voxels in region"); ax[k].set_ylabel("A vs B top-1"); ax[k].set_title(SHORT[d]); ax[k].grid(alpha=0.3); ax[k].legend(fontsize=7)
    fig.suptitle("region size vs repeatability"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "2_region_size_vs_repeatability.png"), dpi=130); plt.close(fig)
    # 3. region scale vs true-minus-shuffled margin; and frames per region
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in list(SRC.TRAIN_DRIVES) + [val]:
        R = res(an, d)
        ax[0].plot(x, [R[u]["pairs"]["AB"]["top1"] - R[u]["controls"]["R1"]["top1"] for u in UNITS], "-o", label=f"{SHORT[d]} vs R1")
        ax[0].plot(x, [R[u]["pairs"]["AB"]["top1"] - R[u]["controls"]["R2"]["top1"] for u in UNITS], ":s", ms=4, color=ax[0].lines[-1].get_color(), label=f"{SHORT[d]} vs R2")
        ax[1].plot(x, [R[u]["size"]["mean_frames_AB"] for u in UNITS], "-o", label=SHORT[d])
    ax[0].set_xticks(x); ax[0].set_xticklabels(XLAB); ax[0].set_ylabel("true - shuffled top-1"); ax[0].axhline(0, color="k", lw=0.8); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("unit-specific identity beyond shuffled references")
    ax[1].set_xticks(x); ax[1].set_xticklabels(XLAB); ax[1].set_ylabel("mean contributing frames in A+B per unit"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title("views actually pooled by aggregation")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "3_scale_vs_margin_and_frames.png"), dpi=130); plt.close(fig)
    # 4. scale vs entropy / dominant share; 5. scale vs coverage
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for d in list(SRC.TRAIN_DRIVES) + [val]:
        R = res(an, d)
        ax[0].plot(x, [R[u]["entropy_nats"] for u in UNITS], "-o", label=SHORT[d])
        ax[1].plot(x, [R[u]["dominant_share"] for u in UNITS], "-o", label=SHORT[d])
        ax[2].plot(x, [100 * R[u]["coverage_of_eligible"]["AB"] for u in UNITS], "-o", label=f"{SHORT[d]} A&B")
        ax[2].plot(x, [100 * R[u]["coverage_of_eligible"]["any_evidence"] for u in UNITS], ":s", ms=4, color=ax[2].lines[-1].get_color(), label=f"{SHORT[d]} any")
    for a_, lab in zip(ax, ("class entropy of A argmax (nats)", "dominant-class share", "coverage of eligible voxels (%)")):
        a_.set_xticks(x); a_.set_xticklabels(XLAB); a_.set_ylabel(lab); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8); ax[2].legend(fontsize=6, ncol=2); ax[2].set_yscale("log")
    fig.suptitle("region scale vs semantic diversity (4) and coverage (5)"); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "4_5_scale_vs_entropy_and_coverage.png"), dpi=130); plt.close(fig)
    # 6. per-class voxel vs region
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    for k, d in enumerate((val, "train_pooled")):
        R = res(an, d); names = sorted(R["voxel"]["per_class"], key=lambda n: -R["voxel"]["per_class"][n]["n"])
        xx = np.arange(len(names)); w = 0.16
        for j, u in enumerate(UNITS):
            ax[k].bar(xx + (j - 2) * w, [R[u]["per_class"].get(n, {"top1": 0})["top1"] for n in names], w, color=COL[u], label=u)
        ax[k].set_xticks(xx); ax[k].set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax[k].set_ylabel("A vs B top-1"); ax[k].set_title(f"{SHORT[d]} (classes with >= 200 A&B units at voxel level)")
        ax[k].grid(alpha=0.3, axis="y"); ax[k].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "6_per_class_voxel_vs_region.png"), dpi=130); plt.close(fig)
    # 7. per-drive
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ds = list(SRC.TRAIN_DRIVES) + [val]; xx = np.arange(len(ds)); w = 0.16
    for j, u in enumerate(UNITS):
        ax.bar(xx + (j - 2) * w, [res(an, d)[u]["pairs"]["AB"]["top1"] for d in ds], w, color=COL[u], label=u)
        ax.scatter(xx + (j - 2) * w, [res(an, d)[u]["controls"]["R1"]["top1"] for d in ds], marker="_", color="k", s=100, zorder=3)
    ax.scatter([], [], marker="_", color="k", s=100, label="R1 shuffled")
    ax.set_xticks(xx); ax.set_xticklabels([SHORT[d] for d in ds]); ax.set_ylabel("A vs B top-1"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, ncol=3)
    ax.set_title("per-drive voxel vs region repeatability"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "7_per_drive_voxel_vs_region.png"), dpi=130); plt.close(fig)
    print(f"figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
