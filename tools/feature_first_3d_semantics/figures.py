#!/usr/bin/env python
"""The seven plots of the feature-first study, from analysis_<population>.json.

    python tools/feature_first_3d_semantics/figures.py --population eligible
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}
COL = {"F": "#2c6fbb", "P_coarse": "#d98c3a", "P_ref": "#7a5ea8"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--population", default="eligible"); a = ap.parse_args()
    an = json.load(open(os.path.join(FF.ART, f"analysis_{a.population}.json")))
    fig_dir = os.path.join(FF.ART, "figures"); os.makedirs(fig_dir, exist_ok=True); tag = a.population
    val = SRC.VAL_DRIVE; drives = list(SRC.TRAIN_DRIVES) + [val]
    ab = {d: an["drives"][d]["pairs"]["AB"] for d in drives}
    # 1. same-surface vs shuffled cosine distributions (0006), feature and probability
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (ctl, title) in enumerate((("C2_local", "local shuffle (same anchor, z band, 4 m cell)"), ("C3_same_class_other_location", "same class, other voxel"))):
        h = ab[val]["hist"][ctl]; bins = np.asarray(h["bins"]); c = 0.5 * (bins[1:] + bins[:-1])
        for key, col, ls in (("F", COL["F"], "-"), ("P_coarse", COL["P_coarse"], "-")):
            t_ = np.asarray(h[f"{key}_true"], float); c_ = np.asarray(h[f"{key}_control"], float)
            ax[k].plot(c, t_ / t_.sum(), ls, color=col, label=f"{key} same surface"); ax[k].plot(c, c_ / c_.sum(), "--", color=col, label=f"{key} {ctl.split('_')[0]}")
        ax[k].set_xlabel("cosine similarity (A vs B)"); ax[k].set_ylabel("fraction"); ax[k].set_title(f"drive 0006 -- {title}"); ax[k].legend(fontsize=7); ax[k].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"1_cosine_distributions_{tag}.png"), dpi=130); plt.close(fig)
    # 2 + 3. same-surface vs local-shuffle similarity, and margins / AUC per drive
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); x = np.arange(len(drives)); w = 0.25
    for j, key in enumerate(("P_ref", "P_coarse", "F")):
        ax[0].bar(x + (j - 1) * w, [ab[d][key]["cosine"] for d in drives], w, color=COL[key], label=f"{key} same surface")
        ax[0].scatter(x + (j - 1) * w, [ab[d]["controls"]["C2_local"][key]["control_mean"] for d in drives], marker="_", color="k", s=120, zorder=3)
        ax[1].bar(x + (j - 1) * w, [ab[d]["controls"]["C2_local"][key]["margin"] for d in drives], w, color=COL[key], label=f"{key} true - C2")
        ax[2].bar(x + (j - 1) * w, [ab[d]["controls"]["C2_local"][key]["auc_true_gt_control"] for d in drives], w, color=COL[key], label=f"{key}")
    for a_ in ax: a_.set_xticks(x); a_.set_xticklabels([SHORT[d] for d in drives]); a_.grid(alpha=0.3, axis="y"); a_.legend(fontsize=7)
    ax[0].set_ylabel("cosine (bars: same surface; ticks: C2 local shuffle)"); ax[1].set_ylabel("cosine margin over C2"); ax[2].set_ylabel("AUC: same-surface > local-shuffle"); ax[2].axhline(0.5, color="k", lw=0.8)
    fig.suptitle("probability-vector vs dense-feature separation from the local-shuffle control"); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"2_3_margins_{tag}.png"), dpi=130); plt.close(fig)
    # 4. classify-then-fuse vs feature-fuse-then-classify, held-out agreement (all rotations)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    rots = list(an["drives"][val]["fusion"])
    for k, d in enumerate((val, SRC.TRAIN_DRIVES[2])):
        fu = an["drives"][d]["fusion"]; xx = np.arange(len(rots))
        ax[k].bar(xx - 0.3, [fu[r]["P_ref_fuse_vs_C_ref"]["top1"] for r in rots], 0.2, color=COL["P_ref"], label="P_ref fuse vs held-out (refined)")
        ax[k].bar(xx - 0.1, [fu[r]["P_coarse_fuse_vs_C_coarse"]["top1"] for r in rots], 0.2, color=COL["P_coarse"], label="classify-then-fuse vs held-out view")
        ax[k].bar(xx + 0.1, [fu[r]["F_fuse_classify_vs_C_coarse"]["top1"] for r in rots], 0.2, color=COL["F"], label="feature-fuse-then-classify vs held-out view")
        ax[k].bar(xx + 0.3, [fu[r]["single_view_A_vs_C"]["P_coarse_top1"] for r in rots], 0.2, color="#bbbbbb", label="single view vs held-out")
        ax[k].set_xticks(xx); ax[k].set_xticklabels([f"{r}\nn={fu[r]['n']}" for r in rots]); ax[k].set_ylabel("held-out top-1 agreement"); ax[k].set_title(SHORT[d]); ax[k].grid(alpha=0.3, axis="y"); ax[k].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"4_fusion_order_{tag}.png"), dpi=130); plt.close(fig)
    # 5. number of fused frames vs held-out consistency
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in drives:
        bf = an["drives"][d]["fusion"]["A+B->C"]["by_frames_fused"]; ks = list(bf)
        ax[0].plot(ks, [bf[k]["F_top1"] for k in ks], "-o", color=None, label=f"{SHORT[d]} F"); ax[0].plot(ks, [bf[k]["P_coarse_top1"] for k in ks], "--s", color=ax[0].lines[-1].get_color(), label=f"{SHORT[d]} P_coarse")
        ax[1].plot(ks, [bf[k]["feature_cosine"] for k in ks], "-o", label=f"{SHORT[d]} cos(f_AB, f_C)"); ax[1].plot(ks, [bf[k]["P_coarse_cosine"] for k in ks], "--s", color=ax[1].lines[-1].get_color(), label=f"{SHORT[d]} cos(p_AB, p_C)")
    ax[0].set_xlabel("frames fused in A+B"); ax[0].set_ylabel("held-out C top-1"); ax[1].set_xlabel("frames fused in A+B"); ax[1].set_ylabel("held-out cosine")
    for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"5_views_{tag}.png"), dpi=130); plt.close(fig)
    # 6. per drive (pairs) already in fig 2/3; per-drive fusion
    fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(drives))
    for j, (key, lab) in enumerate((("P_ref_fuse_vs_C_ref", "P_ref"), ("P_coarse_fuse_vs_C_coarse", "classify-then-fuse"), ("F_fuse_classify_vs_C_coarse", "feature-fuse-then-classify"))):
        ax.bar(x + (j - 1) * 0.25, [an["drives"][d]["fusion"]["A+B->C"][key]["top1"] for d in drives], 0.25, color=list(COL.values())[[2, 1, 0][j]], label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{SHORT[d]}\nn={an['drives'][d]['fusion']['A+B->C']['n']}" for d in drives]); ax.set_ylabel("A+B -> C top-1"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title("per-drive held-out agreement"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"6_per_drive_fusion_{tag}.png"), dpi=130); plt.close(fig)
    # 7. per class
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    pc = an["drives"][val]["per_class_AB"]; names = sorted(pc, key=lambda n: -pc[n]["n"]); xx = np.arange(len(names))
    ax[0].bar(xx - 0.2, [pc[n]["P_coarse_auc_C2"] or 0 for n in names], 0.4, color=COL["P_coarse"], label="P_coarse AUC vs C2")
    ax[0].bar(xx + 0.2, [pc[n]["F_auc_C2"] or 0 for n in names], 0.4, color=COL["F"], label="F AUC vs C2")
    ax[0].axhline(0.5, color="k", lw=0.8); ax[0].set_xticks(xx); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax[0].set_ylabel("AUC same-surface > local shuffle"); ax[0].legend(fontsize=7); ax[0].set_title("drive 0006, per class (A/B pairs)")
    fc = an["drives"][val]["fusion"]["A+B->C"]["per_class"]; names2 = sorted(fc, key=lambda n: -fc[n]["n"]); xx2 = np.arange(len(names2))
    ax[1].bar(xx2 - 0.2, [fc[n]["P_coarse_top1"] for n in names2], 0.4, color=COL["P_coarse"], label="classify-then-fuse")
    ax[1].bar(xx2 + 0.2, [fc[n]["F_top1"] for n in names2], 0.4, color=COL["F"], label="feature-fuse-then-classify")
    ax[1].set_xticks(xx2); ax[1].set_xticklabels([f"{n}\n{fc[n]['n']}" for n in names2], rotation=45, ha="right", fontsize=7); ax[1].set_ylabel("A+B -> C top-1"); ax[1].legend(fontsize=7); ax[1].set_title("drive 0006, per class (held-out C)")
    for a_ in ax: a_.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"7_per_class_{tag}.png"), dpi=130); plt.close(fig)
    print(f"figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
