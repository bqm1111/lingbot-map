#!/usr/bin/env python
"""Plots for the direct-correspondence audit.

    python tools/direct_correspondence_audit/figures.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}


def main() -> int:
    an = json.load(open(os.path.join(DC.ART, "analysis.json"))); mc = json.load(open(os.path.join(DC.ART, "mapper_comparison.json")))
    fig_dir = os.path.join(DC.ART, "figures"); os.makedirs(fig_dir, exist_ok=True); val = SRC.VAL_DRIVE; r = an["drives"][val]
    # 1. distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); h = r["hist"]; b = np.asarray(h["bins"]); c = 0.5 * (b[1:] + b[:-1])
    for k, col, lab in (("feature_same", "#2c6fbb", "same physical point (A vs B)"), ("feature_C2", "#d98c3a", "local control: nearby point 0.3-1.5 m"), ("feature_C0", "#9a9a9a", "random point in drive")):
        v = np.asarray(h[k], float); ax[0].plot(c, v / v.sum(), color=col, label=lab)
    ax[0].set_xlabel("feature cosine"); ax[0].set_ylabel("fraction"); ax[0].set_title("drive 0006, strict visibility"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    pb = np.asarray(h["prob_bins"]); pc = 0.5 * (pb[1:] + pb[:-1])
    for k, col, lab in (("prob_same", "#2c6fbb", "same point"), ("prob_C2", "#d98c3a", "local control")):
        v = np.asarray(h[k], float); ax[1].plot(pc, v / v.sum(), color=col, label=lab)
    ax[1].set_xlabel("class-probability cosine"); ax[1].set_title("per-view coarse probabilities"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "1_distributions.png"), dpi=130); plt.close(fig)
    # 2. same vs controls, and AUC, per drive; with previous mapper-based reference
    drives = [d for d in list(SRC.TRAIN_DRIVES) + [val] if d in an["drives"]]; x = np.arange(len(drives))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, (ctl, col) in enumerate((("C0", "#9a9a9a"), ("C1", "#7a5ea8"), ("C3", "#3a9d5d"), ("C2", "#d98c3a"))):
        ax[0].bar(x + (j - 1.5) * 0.2, [an["drives"][d]["strict"]["controls"][ctl]["feature"]["control_cos"] for d in drives], 0.2, color=col, label=f"{ctl} control")
        ax[1].bar(x + (j - 1.5) * 0.2, [an["drives"][d]["strict"]["controls"][ctl]["feature"]["auc"] for d in drives], 0.2, color=col, label=f"feature AUC vs {ctl}")
    ax[0].plot(x, [an["drives"][d]["strict"]["same"]["feature_cos"] for d in drives], "k_", ms=25, mew=2, label="same point")
    ax[0].set_ylim(0.8, 1.0); ax[0].set_ylabel("feature cosine"); ax[0].set_title("same-point vs control similarity")
    ax[1].axhline(0.5, color="k", lw=0.8); ax[1].axhline(an["previous_mapper_based"]["drive_0006_mapper_voxel_pairs"]["feature_auc_local"], color="#c0392b", ls="--", label="previous mapper-based local AUC 0.52")
    ax[1].set_ylabel("AUC(same > control)"); ax[1].set_title("separability")
    ax[2].bar(x - 0.2, [an["drives"][d]["strict"]["same"]["top1"] for d in drives], 0.4, color="#2c6fbb", label="direct same point")
    ax[2].bar(x + 0.2, [an["drives"][d]["strict"]["controls"]["C2"]["top1"]["control"] for d in drives], 0.4, color="#d98c3a", label="local control point")
    ax[2].axhline(an["previous_mapper_based"]["drive_0006_mapper_voxel_pairs"]["top1_same"], color="#c0392b", ls="--", label="previous mapper same-voxel 0.568 (0006)")
    ax[2].set_ylabel("per-view top-1 class agreement"); ax[2].set_title("class agreement")
    for a_ in ax: a_.set_xticks(x); a_.set_xticklabels([SHORT[d] for d in drives]); a_.grid(alpha=0.3, axis="y"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "2_controls_and_auc.png"), dpi=130); plt.close(fig)
    # 3. viewpoint dependence (0006)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, (key, ttl) in enumerate((("by_baseline", "camera baseline"), ("by_angle", "viewing-angle change"), ("by_depth_B", "depth in B"))):
        names = [n for n, v in r[key].items() if "same" in v]
        ax[k].plot(names, [r[key][n]["same"]["feature_cos"] for n in names], "-o", color="#2c6fbb", label="feature cos same point")
        ax[k].plot(names, [r[key][n]["controls"]["C2"]["feature"]["control_cos"] for n in names], "--o", color="#d98c3a", label="feature cos local control")
        ax[k].plot(names, [r[key][n]["same"]["top1"] for n in names], "-s", color="#3a9d5d", label="top-1 agreement same point")
        ax[k].plot(names, [r[key][n]["controls"]["C2"]["top1"]["control"] for n in names], "--s", color="#7a5ea8", label="top-1 local control")
        ax2 = ax[k].twinx(); ax2.plot(names, [r[key][n]["controls"]["C2"]["feature"]["auc"] for n in names], ":^", color="k", label="AUC same > local"); ax2.set_ylim(0.45, 0.7); ax2.set_ylabel("AUC")
        ax[k].set_title(ttl); ax[k].grid(alpha=0.3); ax[k].legend(fontsize=7, loc="lower left"); ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("drive 0006: consistency of verified same-point observations vs viewpoint change"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "3_viewpoint.png"), dpi=130); plt.close(fig)
    # 4. perturbation vs cross-view
    fig, ax = plt.subplots(figsize=(7, 4)); p = r["perturbation"]
    ks = ["2 px", "5 px", "1 token (11 px)", "cross-view same point", "local control (0.3-1.5 m)"]
    vals = [p["one_minus_cos_2px"], p["one_minus_cos_5px"], p["one_minus_cos_1tok"], p["one_minus_cos_cross_view_same_point"], p["one_minus_cos_local_control"]]
    ax.bar(ks, vals, color=["#9a9a9a"] * 3 + ["#2c6fbb", "#d98c3a"]); ax.set_ylabel("1 - cosine"); ax.set_title("feature change: pixel perturbation vs verified view change (0006)"); ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "4_perturbation.png"), dpi=130); plt.close(fig)
    # 5. mapper vs direct by stream position
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); bs = mc["by_stream_position"]; names = list(bs)
    ax[0].plot(names, [bs[n]["direct_top1_on_both"] for n in names], "-o", color="#2c6fbb", label="direct same-point class agreement")
    ax[0].plot(names, [bs[n]["mapper_top1_on_both"] for n in names], "-s", color="#c0392b", label="mapper same-voxel class agreement (same pairs)")
    ax[0].plot(names, [bs[n]["mapper_has_B_obs"] for n in names], ":^", color="k", label="fraction of pairs where mapper has any B observation in voxel(X)")
    ax[0].set_xlabel("stream position of A (frames)"); ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7); ax[0].set_title("class agreement, |offset| <= 3 stream frames")
    ax[1].plot(names, [bs[n]["pose_reproj_err_px_median"] for n in names], "-o", color="#c0392b", label="LingBot relative-pose reprojection error (px, median)")
    ax[1].plot(names, [bs[n]["mapper_pix_offset_px_median"] for n in names], "-s", color="#d98c3a", label="mapper pixel offset from true projection (px, median)")
    ax2 = ax[1].twinx(); ax2.plot(names, [bs[n]["endpoint_3d_err_m_median"] for n in names], ":^", color="k", label="LingBot endpoint 3D error (m, median)"); ax2.set_ylabel("m")
    ax[1].set_xlabel("stream position of A (frames)"); ax[1].set_ylabel("pixels"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="lower right"); ax[1].set_title("mapper geometry vs ground truth")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "5_mapper_vs_direct.png"), dpi=130); plt.close(fig)
    # 6. per class (0006)
    pc_ = r["per_class"]; names = sorted(pc_, key=lambda n: -pc_[n]["n"]); xx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(xx - 0.2, [pc_[n]["top1"] for n in names], 0.4, color="#2c6fbb", label="same point top-1 agreement"); ax.bar(xx + 0.2, [pc_[n]["C2_top1_control"] for n in names], 0.4, color="#d98c3a", label="local control")
    ax2 = ax.twinx(); ax2.plot(xx, [pc_[n]["C2_feature_auc"] for n in names], "k^", label="feature AUC vs local"); ax2.set_ylim(0.45, 0.7); ax2.axhline(0.5, color="k", lw=0.6)
    ax.set_xticks(xx); ax.set_xticklabels([f"{n}\n{pc_[n]['n']:,}" for n in names], rotation=45, ha="right", fontsize=7); ax.legend(fontsize=7, loc="upper right"); ax2.legend(fontsize=7, loc="upper center"); ax.grid(alpha=0.3, axis="y"); ax.set_title("drive 0006 per class (class = A's per-view top-1)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "6_per_class.png"), dpi=130); plt.close(fig)
    print(f"figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
