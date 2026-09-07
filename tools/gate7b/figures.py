#!/usr/bin/env python
"""Gate 7B figures, all read from ``artifacts/gate7b/*.json``."""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
from gates.gate7b import config as C                                                   # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")
ORDER = ["semantickitti", "occ3d", "kitti360"]
LABEL = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes",
         "kitti360": "SSCBench-KITTI-360"}
COL = {"S1": "#1b6ca8", "S2": "#c1440e", "S3": "#2e7d32", "S4": "#7b3fa0",
       "S0": "#444444", "B-R": "#999999",
       "G-A": "#1b6ca8", "G-B": "#c1440e", "G-C": "#f2a900"}


def S():
    p = os.path.join(ART, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _grid(n, title, ylab, sharey=False, h=3.2):
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, h), sharey=sharey)
    ax = np.atleast_1d(ax)
    fig.suptitle(title, fontsize=11, y=0.99)
    ax[0].set_ylabel(ylab)
    return fig, ax


def _finish(fig, path, ncol=4):
    h, l = fig.axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12 if h else 0.03, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def have(s):
    return [(ds, s["datasets"][ds]) for ds in ORDER
            if ds in s["datasets"] and s["datasets"][ds]["configs"]]


def fig_horizon(s):
    hs = [1, 5, 20, 50, "all"]
    for metric, ylab, name in (("binary_recall", "pooled binary recall",
                                "fig_horizon_recall.png"),
                               ("ssc_miou", "pooled SSC mIoU",
                                "fig_horizon_miou.png")):
        rows = have(s)
        fig, ax = _grid(len(rows), f"{ylab} versus causal history length "
                                   "(S1, fixed anchor gauge)", ylab)
        for i, (ds, d) in enumerate(rows):
            xs, ys = [], []
            for j, h in enumerate(hs):
                cfg = d["configs"].get(f"S1_G-A_h{h}")
                if cfg:
                    xs.append(j)
                    ys.append(cfg[metric])
            ax[i].plot(xs, ys, "o-", color=COL["S1"], lw=2.2, label="S1 streaming")
            for ref, st in (("B-R", "--"), ("B-D", ":")):
                v = d["reference"][ref][metric if metric != "binary_recall"
                                        else "binary_recall"]
                ax[i].axhline(v, ls=st, lw=1.4, color=COL["S0"],
                              label=f"five-frame {ref}")
            ax[i].set_xticks(range(len(hs)))
            ax[i].set_xticklabels([str(h) for h in hs])
            ax[i].set_xlabel("causal history (frames)")
            ax[i].set_title(LABEL[ds], fontsize=10)
            ax[i].grid(alpha=0.3)
        _finish(fig, os.path.join(ART, name), ncol=3)


def fig_scale(s):
    rows = []
    for ds in ORDER:
        for p in sorted(glob.glob(os.path.join(ART, f"eval_{ds}_S1_G-*_hall.json"))):
            rows.append((ds, json.load(open(p))))
    if not rows:
        return
    dss = [ds for ds in ORDER if any(r[0] == ds for r in rows)]
    fig, ax = _grid(len(dss), "Metric gauge versus time, by policy", "scale s(t)")
    for i, ds in enumerate(dss):
        for ds2, e in rows:
            if ds2 != ds:
                continue
            pol = e["scale_policy"]
            seg = next(iter(e["scale_diagnostics"].values()))
            y = np.asarray(seg.get("series", []), float)
            if y.size == 0:
                continue
            ax[i].plot(np.arange(len(y)), y, lw=1.5, color=COL[pol], label=pol,
                       alpha=0.9 if pol != "G-C" else 0.55)
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("stream frame")
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_scale_vs_time.png"), ncol=3)


def fig_jitter(s):
    pts = []
    for ds in ORDER:
        th = os.path.join(ART, f"thickness_{ds}.json")
        if not os.path.exists(th):
            continue
        t = json.load(open(th))
        for p in sorted(glob.glob(os.path.join(ART, f"eval_{ds}_S1_G-*_hall.json"))):
            e = json.load(open(p))
            pol = e["scale_policy"]
            seg = next(iter(e["scale_diagnostics"].values()))
            if pol in t["policies"]:
                pts.append((ds, pol, seg.get("adjacent_log_mad", 0.0),
                            t["policies"][pol]["thickness_mean"],
                            t["policies"][pol]["duplicate_rate_mean"]))
    if not pts:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    fig.suptitle("Gauge jitter against map thickness and duplicate surfaces", fontsize=11)
    mark = {"semantickitti": "o", "occ3d": "s", "kitti360": "^"}
    for k, (lbl, yi) in enumerate((("mean map thickness (voxels per column)", 3),
                                   ("duplicate-surface rate", 4))):
        for ds, pol, j, th, du in pts:
            ax[k].scatter(j, (th if yi == 3 else du), s=64, color=COL[pol],
                          marker=mark[ds],
                          label=f"{pol} · {LABEL[ds]}" if k == 0 else None)
        ax[k].set_xlabel("adjacent-frame log-scale MAD")
        ax[k].set_ylabel(lbl)
        ax[k].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.20, 1, 0.94))
    fig.savefig(os.path.join(ART, "fig_scale_jitter_thickness.png"), dpi=160)
    plt.close(fig)
    print("  fig_scale_jitter_thickness.png")


def fig_pr(s):
    rows = have(s)
    fig, ax = _grid(len(rows), "Precision-recall of the depth-evidence variants "
                               "(all-past causal history, fixed anchor gauge)",
                    "pooled precision", sharey=True, h=3.4)
    # S3 and S4 land almost on top of each other -- that near-identity IS a finding, so
    # S3 is drawn large and filled and S4 as a smaller ring on top, rather than one
    # hiding the other
    style = {"S1": dict(s=110, marker="o"), "S2": dict(s=110, marker="s"),
             "S3": dict(s=190, marker="o", alpha=0.55),
             "S4": dict(s=70, marker="o", edgecolors="black", linewidths=1.1)}
    xs_all, ys_all = [], []
    for i, (ds, d) in enumerate(rows):
        for var in ("S1", "S2", "S3", "S4"):
            tag = f"{var}_G-A_hall" + ("_c0.5" if var == "S2" else "")
            c = d["configs"].get(tag)
            if not c:
                continue
            ax[i].scatter([c["binary_recall"]], [c["binary_precision"]],
                          color=COL[var], label=var, zorder=3, **style[var])
            xs_all.append(c["binary_recall"])
            ys_all.append(c["binary_precision"])
        for ref, m in (("B-R", "X"), ("B-D", "*")):
            r = d["reference"][ref]
            ax[i].scatter([r["binary_recall"]], [r["binary_precision"]], s=150,
                          marker=m, color=COL["S0"], label=f"five-frame {ref}", zorder=3)
            xs_all.append(r["binary_recall"])
            ys_all.append(r["binary_precision"])
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].set_xlabel("pooled recall")
        ax[i].grid(alpha=0.3)
    for a_ in ax:
        a_.set_xlim(0, max(xs_all) * 1.15)
        a_.set_ylim(0, max(ys_all) * 1.2)
    _finish(fig, os.path.join(ART, "fig_precision_recall.png"), ncol=6)


def fig_recovery(s):
    rows = []
    for ds in ORDER:
        p = os.path.join(ART, f"recoverability_{ds}.json")
        if os.path.exists(p):
            rows.append((ds, json.load(open(p))))
    if not rows:
        return
    fig, ax = _grid(len(rows), "Temporal recovery of B-D coverage misses: how long until "
                               "streamed geometry appears?", "cumulative fraction of "
                               "misses", sharey=True, h=3.6)
    keys = ["within_1_frames", "within_5_frames", "within_10_frames",
            "within_20_frames", "within_50_frames", "any_later_frame"]
    xt = ["1", "5", "10", "20", "50", "any"]
    # the curves are all small -- that IS the finding -- so the axis is scaled to the
    # data and the magnitude is stated in words on the panel rather than implied by a
    # large expanse of empty plot
    hi = max(max(d["recovery_latency_fraction"][k] for k in keys)
             + d["fractions"]["recovered_from_causal_past"] for _ds, d in rows)
    for i, (ds, d) in enumerate(rows):
        y = [d["recovery_latency_fraction"][k] for k in keys]
        past = d["fractions"]["recovered_from_causal_past"]
        ax[i].plot(range(len(y)), y, "o-", color=COL["S3"], lw=2.2,
                   label="recovered from a later viewpoint")
        ax[i].axhline(past, ls="--", lw=1.4, color=COL["S1"],
                      label="already in the causal past")
        unreachable = (d["fractions"]["in_frustum_but_never_reconstructed"]
                       + d["fractions"]["outside_all_frusta"])
        ax[i].annotate(f"{unreachable:.0%} never reconstructed\nfrom any viewpoint",
                       xy=(0.03, 0.93), xycoords="axes fraction", fontsize=9,
                       va="top", color="#8a2f2f", fontweight="bold")
        ax[i].set_xticks(range(len(xt)))
        ax[i].set_xticklabels(xt)
        ax[i].set_xlabel("additional frames")
        ax[i].set_ylim(0, hi * 1.55)
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].grid(alpha=0.3)
    _finish(fig, os.path.join(ART, "fig_recovery_latency.png"), ncol=2)


def fig_sources(s):
    rows = have(s)
    fig, ax = _grid(len(rows), "Where the occupied voxels come from "
                               "(all-past causal history)",
                    "fraction of occupied voxels", sharey=True, h=3.4)
    for i, (ds, d) in enumerate(rows):
        tags = [("S1", "S1_G-A_hall"), ("S2", "S2_G-A_hall_c0.5"),
                ("S3", "S3_G-A_hall"), ("S4", "S4_G-A_hall")]
        xs, lb, mg, beyond = [], [], [], []
        for j, (nm, tag) in enumerate(tags):
            c = d["configs"].get(tag)
            if not c:
                continue
            tot = max(c["occ_source"]["lingbot"] + c["occ_source"]["moge_only"], 1)
            xs.append(nm)
            lb.append(c["occ_source"]["lingbot"] / tot)
            mg.append(c["occ_source"]["moge_only"] / tot)
            beyond.append(c["occ_source"]["beyond_five_frames"] / tot)
        idx = np.arange(len(xs))
        ax[i].bar(idx, lb, color=COL["S1"], label="LingBot evidence")
        ax[i].bar(idx, mg, bottom=lb, color=COL["S3"], label="MoGe-only evidence")
        ax[i].plot(idx, beyond, "k^--", lw=1.4, ms=6,
                   label="first seen beyond the five-frame window")
        ax[i].set_xticks(idx)
        ax[i].set_xticklabels(xs)
        ax[i].set_ylim(0, 1.05)
        ax[i].set_title(LABEL[ds], fontsize=10)
        ax[i].grid(alpha=0.3, axis="y")
    _finish(fig, os.path.join(ART, "fig_evidence_sources.png"), ncol=3)


def fig_cost(s):
    rows = have(s)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    fig.suptitle("Map-update cost and memory against causal window length", fontsize=11)
    mark = {"semantickitti": "o", "occ3d": "s", "kitti360": "^"}
    for ds, d in rows:
        for tag, cfg in sorted(d["configs"].items()):
            if not tag.startswith("S1_G-A_h"):
                continue
            ax[0].scatter(cfg["median_window"], cfg["median_seconds_per_anchor"] * 1000,
                          s=60, marker=mark[ds], color=COL["S1"],
                          label=LABEL[ds] if tag.endswith("hall") else None)
            ax[1].scatter(cfg["median_window"], cfg["map_bytes_mean"] / 2 ** 20, s=60,
                          marker=mark[ds], color=COL["S3"],
                          label=LABEL[ds] if tag.endswith("hall") else None)
    ax[0].set_xlabel("median causal window (frames)")
    ax[0].set_ylabel("map-update latency per timestamp (ms)")
    ax[1].set_xlabel("median causal window (frames)")
    ax[1].set_ylabel("evidence volume (MiB)")
    for k in (0, 1):
        ax[k].grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(os.path.join(ART, "fig_cost_vs_window.png"), dpi=160)
    plt.close(fig)
    print("  fig_cost_vs_window.png")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    s = S()
    if s is None:
        raise SystemExit("run tools/gate7b/aggregate.py first")
    print("figures:")
    fig_horizon(s); fig_scale(s); fig_jitter(s); fig_pr(s)
    fig_recovery(s); fig_sources(s); fig_cost(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
