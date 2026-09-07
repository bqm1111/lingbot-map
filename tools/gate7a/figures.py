#!/usr/bin/env python
"""Gate 7A figures. Every curve is read from ``artifacts/gate7a/summary_*.json``.

Six compact panels, one column per benchmark, shared axes wherever the quantity is
comparable (fractions and accuracies are; absolute IoU across three different grids and
masks is not, and those panels say so).

    python tools/gate7a/figures.py
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                   # noqa: E402
from gates.gate7a import config as C                                           # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7a")
ORDER = ["semantickitti", "occ3d", "kitti360"]
LABEL = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes",
         "kitti360": "SSCBench-KITTI-360"}
COL = {"oracle": "#1b6ca8", "morph": "#c1440e", "base": "#444444",
       "B-R": "#8a8a8a", "B-D": "#1b6ca8"}


def load(ds):
    p = os.path.join(ART, f"summary_{ds}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _grid(n, title, ylab, sharey=True, h=3.1):
    fig, ax = plt.subplots(1, n, figsize=(4.1 * n, h), sharey=sharey)
    ax = np.atleast_1d(ax)
    fig.suptitle(title, fontsize=11, y=0.99)
    ax[0].set_ylabel(ylab)
    return fig, ax


def _finish(fig, path, ncol=4):
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def fig_reachable(summaries):
    """Reachable miss fraction versus radius, overall and by range band."""
    fig, ax = _grid(len(summaries), "Fraction of coverage misses within a metric radius "
                                    "of the frozen support (B-D primary)",
                    "fraction of B-D misses reachable", h=3.4)
    for i, (ds, s) in enumerate(summaries):
        md = s["miss_distance"]["B-D"]
        R = s["radii_m"]
        ax[i].plot(R, [md["all"]["miss_fraction_within_radius"][f"{r:g}"] for r in R],
                   "o-", color=COL["B-D"], lw=2.2, label="all ranges (B-D)")
        ax[i].plot(R, [s["miss_distance"]["B-R"]["all"]["miss_fraction_within_radius"][f"{r:g}"]
                       for r in R], "s--", color=COL["B-R"], lw=1.4, label="all ranges (B-R)")
        for j, (lo, hi) in enumerate(C.RANGE_BANDS):
            bn = f"{int(lo)}-{int(hi)}m"
            if bn not in md:
                continue
            ax[i].plot(R, [md[bn]["miss_fraction_within_radius"][f"{r:g}"] for r in R],
                       lw=1.0, alpha=0.75,
                       color=plt.cm.viridis(j / max(len(C.RANGE_BANDS) - 1, 1)),
                       label=f"{bn} (B-D)" if i == 0 else None)
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("radius (m)")
        ax[i].set_ylim(0, 1)
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_reachable_miss_fraction.png"), ncol=4)


def _curve(ax, s, key, cond="B-D"):
    R = s["radii_m"]
    base = s["base"][cond][key]
    for constr, mark in (("oracle", "o-"), ("morph", "s-")):
        ax.plot(R, [row[key] for row in s["envelopes"][cond][constr]["by_radius"]],
                mark, color=COL[constr], lw=2.0,
                label=("oracle-local ceiling (uses the target)" if constr == "oracle"
                       else "ordinary metric dilation (deployable)"))
    ax.axhline(base, color=COL["base"], ls=":", lw=1.4, label=f"{cond} base")
    ax.set_xlabel("radius (m)")
    ax.grid(alpha=0.3)


def fig_iou(summaries):
    fig, ax = _grid(len(summaries), "Binary occupancy IoU versus completion radius "
                                    "(B-D base; absolute values are not comparable "
                                    "across benchmarks)",
                    "pooled binary IoU", sharey=False)
    for i, (ds, s) in enumerate(summaries):
        _curve(ax[i], s, "binary_iou")
        ax[i].set_title(LABEL[ds], fontsize=10)
    _finish(fig, os.path.join(ART, "fig_binary_iou_vs_radius.png"), ncol=3)


def fig_miou(summaries):
    fig, ax = _grid(len(summaries), "Full SSC mIoU versus completion radius "
                                    "(B-D base; absolute values are not comparable "
                                    "across benchmarks)",
                    "pooled SSC mIoU", sharey=False)
    for i, (ds, s) in enumerate(summaries):
        _curve(ax[i], s, "ssc_miou")
        ax[i].set_title(LABEL[ds], fontsize=10)
    _finish(fig, os.path.join(ART, "fig_ssc_miou_vs_radius.png"), ncol=3)


def fig_pr(summaries):
    fig, ax = _grid(len(summaries), "Precision-recall of the two constructions "
                                    "(B-D base, radii 0 -> 4 m)", "pooled precision",
                    sharey=True, h=3.4)
    for i, (ds, s) in enumerate(summaries):
        for constr, mark in (("oracle", "o-"), ("morph", "s-")):
            rows = s["envelopes"]["B-D"][constr]["by_radius"]
            ax[i].plot([r["binary_recall"] for r in rows],
                       [r["binary_precision"] for r in rows], mark, color=COL[constr],
                       lw=2.0, label=("oracle-local ceiling" if constr == "oracle"
                                      else "ordinary metric dilation"))
            for r in rows:
                if r["radius_m"] in (0.4, 4.0):
                    ax[i].annotate(f"{r['radius_m']:g} m",
                                   (r["binary_recall"], r["binary_precision"]),
                                   fontsize=7, xytext=(3, 3),
                                   textcoords="offset points")
        b = s["base"]["B-D"]
        ax[i].plot([b["binary_recall"]], [b["binary_precision"]], "*", ms=13,
                   color=COL["base"], label="B-D base")
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("pooled recall")
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_precision_recall.png"), ncol=3)


def fig_transport(summaries):
    fig, ax = _grid(len(summaries), "Semantic accuracy of a propagated label versus how "
                                    "far it was propagated (added true positives, B-D)",
                    "accuracy on added true positives", sharey=True, h=3.4)
    for i, (ds, s) in enumerate(summaries):
        rows = s["semantic_transport"]["B-D"]["by_interval"]
        x = [0.5 * (r["interval_m"][0] + r["interval_m"][1]) for r in rows]
        ax[i].plot(x, [r["top1_accuracy"] for r in rows], "o-", color=COL["oracle"],
                   lw=2.0, label="top-1 accuracy")
        ax[i].plot(x, [r["balanced_recall"] for r in rows], "s--", color=COL["morph"],
                   lw=1.8, label="balanced recall")
        ax[i].plot(x, [r["mean_max_probability"] for r in rows], "^:", color="#5a5a5a",
                   lw=1.4, label="mean max probability")
        base = s["base"]["B-D"]
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("propagation distance (m, interval midpoint)")
        ax[i].set_ylim(0, 1)
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_semantic_transport.png"), ncol=3)


def fig_frustum(summaries):
    fig, ax = _grid(len(summaries), "Where the B-D coverage misses are: inside at least "
                                    "one of the five input frusta, or outside all of them",
                    "fraction of B-D coverage misses", sharey=True, h=3.6)
    cls = ("near_surface", "behind_surface", "in_front_of_surface",
           "no_valid_predicted_depth", "outside_all_frusta")
    nice = ("in-frustum, near predicted surface", "in-frustum, behind predicted surface",
            "in-frustum, in front of predicted surface",
            "in-frustum, no valid predicted depth", "outside all five frusta")
    colours = ("#2e7d32", "#c1440e", "#f2a900", "#7b6ca8", "#9e9e9e")
    bands = [f"{int(lo)}-{int(hi)}m" for lo, hi in C.RANGE_BANDS]
    for i, (ds, s) in enumerate(summaries):
        fr = s["frustum"]["B-D"]
        xs = np.arange(len(bands))
        bottom = np.zeros(len(bands))
        for k, (c, nm, col) in enumerate(zip(cls, nice, colours)):
            vals = np.array([fr[b]["residual_class_fraction"][c] if b in fr else 0.0
                             for b in bands])
            ax[i].bar(xs, vals, bottom=bottom, color=col, width=0.75,
                      label=nm if i == 0 else None)
            bottom += vals
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(bands, fontsize=8)
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("horizontal range band")
        ax[i].set_ylim(0, 1)
    _finish(fig, os.path.join(ART, "fig_frustum_by_range.png"), ncol=3)


def fig_cdf(summaries):
    fig, ax = _grid(len(summaries), "Cumulative distribution of the distance from a "
                                    "coverage miss to the nearest frozen occupied voxel",
                    "cumulative fraction of misses", sharey=True, h=3.3)
    for i, (ds, s) in enumerate(summaries):
        for cond in ("B-R", "B-D"):
            md = s["miss_distance"][cond]["all"]
            q = md["quantiles_m"]
            xs = [0.0] + [q[k] for k in ("p25", "p50", "p75", "p90", "p95", "p99")]
            ys = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            ax[i].plot(xs, ys, "o-", color=COL[cond], lw=2.0, label=f"{cond}")
        for r in C.RADII_M[1:]:
            ax[i].axvline(r, color="#bbbbbb", lw=0.7, ls=":")
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("distance to nearest frozen occupied voxel (m)")
        ax[i].set_xscale("log")
        ax[i].set_ylim(0, 1)
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_miss_distance_cdf.png"), ncol=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    summaries = [(ds, load(ds)) for ds in ORDER]
    summaries = [(ds, s) for ds, s in summaries if s is not None]
    if not summaries:
        raise SystemExit("no Gate-7A summaries; run tools/gate7a/aggregate.py first")
    print(f"figures from {len(summaries)} benchmark(s):")
    fig_reachable(summaries)
    fig_cdf(summaries)
    fig_iou(summaries)
    fig_miou(summaries)
    fig_pr(summaries)
    fig_transport(summaries)
    fig_frustum(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
