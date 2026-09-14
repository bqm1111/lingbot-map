#!/usr/bin/env python
"""The six plots of the target-consistency study, from analysis.json.

    python tools/future_semantic_completion/target_consistency/figures.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

FAM_COLOR = {"any": "#9a9a9a", "F0": "#5a5a5a", "F1": "#2c6fbb", "F2": "#d98c3a", "F3": "#3a9d5d", "F4": "#7a5ea8"}
SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}; SHORT["train_pooled"] = "train"


def main() -> int:
    an = json.load(open(os.path.join(TC.ART, "analysis.json")))
    fig_dir = os.path.join(TC.ART, "figures"); os.makedirs(fig_dir, exist_ok=True)
    F = an["filters"]; val = SRC.VAL_DRIVE
    for g in ("interleaved", "contiguous"):
        R = an["results"][g]
        # 1 + 2: coverage vs repeatability (0006 and train pooled)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for k, (metric, lab) in enumerate((("top1_repeatability", "independent top-1 repeatability (A+B target vs C)"),
                                            ("cosine", "cosine(A+B target, C)"))):
            for d, mk in ((val, "o"), ("train_pooled", "s")):
                for fname, r in R.items():
                    rr = r[d]
                    if rr["n_validated"] < 20:
                        continue
                    ax[k].scatter(100 * rr["coverage_of_eligible"], rr[metric], marker=mk, s=42,
                                  color=FAM_COLOR[F[fname]["family"]], edgecolor="k", linewidth=0.4, alpha=0.9)
                    if F[fname]["family"] in ("any", "F0", "F1") and d == val:
                        ax[k].annotate(fname.split("_")[0] if fname != "any_AB" else "any A|B", (100 * rr["coverage_of_eligible"], rr[metric]),
                                       fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax[k].set_xscale("log"); ax[k].set_xlabel("retained voxels, % of eligible (log)"); ax[k].set_ylabel(lab)
            ref = an["previous_reference"]
            ax[k].axhline(ref["causal_vs_future_top1"] if k == 0 else ref["causal_vs_future_cosine"], color="#c0392b", ls="--", lw=1,
                          label="previous causal-vs-future " + ("0.456" if k == 0 else "0.51"))
            ax[k].axhline(0.60 if k == 0 else 0.6, color="k", ls=":", lw=1, label="0.60 minimum of interest" if k == 0 else None)
            ax[k].grid(alpha=0.3); ax[k].legend(fontsize=8, loc="lower left")
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLOR.items()]
        handles += [plt.Line2D([], [], marker="o", ls="", color="w", markeredgecolor="k", label="drive 0006"),
                    plt.Line2D([], [], marker="s", ls="", color="w", markeredgecolor="k", label="train pooled")]
        ax[1].legend(handles=handles, fontsize=7, loc="lower right")
        fig.suptitle(f"{g} grouping: coverage vs independent repeatability (every filter)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"coverage_vs_repeatability_{g}.png"), dpi=130); plt.close(fig)
        # 3: filter strength vs class entropy / top-1 share / n classes  (F2 by tau, F3 by gamma), 0006
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for fam, key, xs in (("F2", "tau", TC.TAUS[1:]), ("F3", "gamma", TC.GAMMAS[1:])):
            names = ["F1_agree"] + [f"{fam}_{'tau' if key == 'tau' else 'gam'}{x:.2f}" for x in xs]
            xx = [0.0] + list(xs)
            for j, (m, lab) in enumerate((("entropy_nats", "class entropy (nats)"), ("top1_class_share", "dominant-class share"),
                                          ("n_classes_support_ge_500", "classes with >= 500 retained voxels"))):
                ax[j].plot(xx, [R[n][val][m] for n in names], "-o", color=FAM_COLOR[fam], label=f"{fam} ({key})")
                ax[j].set_xlabel("threshold (0 = F1 agreement only)"); ax[j].set_ylabel(lab); ax[j].grid(alpha=0.3)
        for j, m in enumerate(("entropy_nats", "top1_class_share", "n_classes_support_ge_500")):
            ax[j].axhline(R["any_AB"][val][m], color=FAM_COLOR["any"], ls="--", label="any A|B (unfiltered)")
        ax[0].legend(fontsize=8); fig.suptitle(f"{g}: filter strength vs semantic diversity, drive 0006")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"filter_strength_vs_diversity_{g}.png"), dpi=130); plt.close(fig)
        # 4: per-class repeatability (any_AB vs F1) on 0006 and train pooled
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        for k, d in enumerate((val, "train_pooled")):
            pcs = {f: R[f][d]["per_class_repeatability"] for f in ("any_AB", "F1_agree")}
            names = sorted(pcs["any_AB"], key=lambda n: -pcs["any_AB"][n]["n"])
            xx = np.arange(len(names)); w = 0.38
            for j, f in enumerate(("any_AB", "F1_agree")):
                ax[k].bar(xx + (j - 0.5) * w, [pcs[f].get(n, {"top1": 0})["top1"] for n in names], w,
                          color=FAM_COLOR[F[f]["family"]], label=f)
                for i, n in enumerate(names):
                    if n in pcs[f]:
                        ax[k].text(i + (j - 0.5) * w, pcs[f][n]["top1"] + 0.01, str(pcs[f][n]["n"]), ha="center", fontsize=6)
            ax[k].set_xticks(xx); ax[k].set_xticklabels(names, rotation=50, ha="right", fontsize=8)
            ax[k].set_ylabel("top-1 repeatability vs C"); ax[k].set_title(f"{SHORT[d]} (classes with >= 500 validated voxels; n above bars)")
            ax[k].legend(fontsize=8); ax[k].grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"per_class_repeatability_{g}.png"), dpi=130); plt.close(fig)
        # 5: per-drive repeatability with shuffled and class-prior references
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ds = list(SRC.TRAIN_DRIVES) + [val]
        fs = ["any_AB", "F0_existing_rule", "F1_agree", "F2_tau0.60", "F4_tau0.60_gam0.90"]
        xx = np.arange(len(ds)); w = 0.16
        for j, f in enumerate(fs):
            ys = [R[f][d]["top1_repeatability"] for d in ds]; sh = [R[f][d]["shuffled_top1"] for d in ds]
            ax.bar(xx + (j - 2) * w, ys, w, color=FAM_COLOR[F[f]["family"]], label=f, alpha=0.5 if f.startswith("F4") else 0.9)
            ax.scatter(xx + (j - 2) * w, sh, marker="_", color="k", s=120, zorder=3)
        ax.scatter([], [], marker="_", color="k", s=120, label="shuffled within-anchor reference")
        ax.set_xticks(xx); ax.set_xticklabels([SHORT[d] for d in ds]); ax.set_ylabel("top-1 repeatability vs C")
        ax.axhline(0.456, color="#c0392b", ls="--", lw=1, label="previous 0.456"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"{g}: per-drive independent repeatability (bars) and its within-anchor shuffled reference (ticks)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"per_drive_repeatability_{g}.png"), dpi=130); plt.close(fig)
        # 6: confidence calibration (0006, any_AB and F1)
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in ("any_AB", "F1_agree", "F0_existing_rule"):
            bc = R[f][val]["by_confidence"]
            ax.plot(list(bc), [v["top1"] for v in bc.values()], "-o", color=FAM_COLOR[F[f]["family"]], label=f)
        ax.set_xlabel("max(p_target) bin"); ax.set_ylabel("top-1 repeatability vs C"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"{g}: candidate confidence vs independent repeatability, drive 0006")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"confidence_calibration_{g}.png"), dpi=130); plt.close(fig)
    # support structure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for d in list(SRC.TRAIN_DRIVES) + [val]:
        h = np.asarray(an["support"][d]["frames_per_voxel_hist"], float); h = h / max(h.sum(), 1)
        ax[0].plot(range(1, 7), h[1:7], "-o", label=SHORT[d])
        po = np.asarray(an["support"][d]["pairs_by_offset"], float)[1:]; ax[1].plot(range(1, 21), po / po.sum(), "-o", ms=3, label=SHORT[d])
    ax[0].set_xlabel("future frames contributing teacher evidence to a voxel"); ax[0].set_ylabel("fraction of covered voxels"); ax[0].set_yscale("log")
    ax[1].set_xlabel("future frame offset"); ax[1].set_ylabel("fraction of (voxel, frame) evidence pairs")
    for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=8)
    fig.suptitle("evidence structure: most covered voxels are seen by exactly one future frame")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "evidence_structure.png"), dpi=130); plt.close(fig)
    print(f"figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
