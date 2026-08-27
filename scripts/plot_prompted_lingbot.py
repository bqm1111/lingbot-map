#!/usr/bin/env python
"""Plots for the prompted-LingbotMap study.

    python scripts/plot_prompted_lingbot.py \
        --rows outputs/prompted_lingbot/evaluation/baselines_rows.json \
        --output-dir outputs/prompted_lingbot/evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORDER = ["raw", "first_depth_scale", "running_depth_scale", "causal_pose_sim3",
         "sliding_pose_sim3", "depth_scale_pose_se3", "learned_corrector",
         "offline_oracle_sim3", "per_frame_oracle_scale"]
NON_CAUSAL = {"offline_oracle_sim3", "per_frame_oracle_scale"}
GAPS = [("0", 0), ("1_4", 2.5), ("5_9", 7), ("10_29", 20), ("30_99", 65), ("100plus", 150)]


def _safe_log(ax, axis="y"):
    """Only switch to a log axis when there is positive data to show.

    KITTI has no ground-truth depth, so its depth panels are legitimately empty
    and matplotlib refuses to log-scale them."""
    lim = ax.get_ylim() if axis == "y" else ax.get_xlim()
    data = [c for c in ax.get_children()]
    positive = False
    for line in ax.get_lines():
        y = line.get_ydata() if axis == "y" else line.get_xdata()
        arr = np.asarray(y, float)
        if arr.size and np.isfinite(arr).any() and np.nanmax(arr) > 0:
            positive = True
            break
    if not positive:
        for patch in ax.patches:
            h = patch.get_height() if axis == "y" else patch.get_width()
            if np.isfinite(h) and h > 0:
                positive = True
                break
    if positive:
        ax.set_yscale("log") if axis == "y" else ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "no data\n(this dataset has no ground-truth depth)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#888")


def _style(a):
    return dict(linestyle="--" if a in NON_CAUSAL else "-",
                marker="s" if a in NON_CAUSAL else "o",
                alpha=0.65 if a in NON_CAUSAL else 1.0)


def _mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.mean(v)) if v else np.nan


def group(rows):
    g = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        g[(r["config"], r["anchor"])].append(r)
    return g


def anchors_present(g):
    have = {a for (_, a) in g}
    return [a for a in ORDER if a in have] + sorted(have - set(ORDER))


# --------------------------------------------------------------------------- #
def plot_error_vs_gap(g, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (metric, label) in zip(axes, [("ate_gap_", "position RMSE (m)"),
                                          ("depthscale_gap_", "|log depth-scale error|")]):
        for a in anchors_present(g):
            xs, ys = [], []
            for name, x in GAPS:
                rows = [r for (c, an), rs in g.items() if an == a and "k30" in c and "clean" in c
                        for r in rs]
                if not rows:
                    continue
                v = _mean(rows, f"{metric}{name}")
                if np.isfinite(v):
                    xs.append(x); ys.append(v)
            if xs:
                ax.plot(xs, ys, label=a, **_style(a))
        ax.set_xlabel("frames since last prompt")
        ax.set_ylabel(label)
        _safe_log(ax)
        ax.grid(alpha=0.3)
    axes[0].set_title("Metric error vs frames since the last prompt (k=30, clean)")
    axes[1].set_title("Depth-scale error vs frames since the last prompt")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_1_error_vs_frames_since_prompt.png"), dpi=140)
    plt.close(fig)


def plot_ate_vs_distance(g, out, tag):
    bins = [(k, re.match(r"ate_m_at_(\d+)_(\d+)m", k)) for k in
            sorted({k for rs in g.values() for r in rs for k in r if k.startswith("ate_m_at_")})]
    bins = [(k, (int(m.group(1)) + int(m.group(2))) / 2) for k, m in bins if m]
    bins.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a in anchors_present(g):
        rows = [r for (c, an), rs in g.items() if an == a and c == "both_k30_clean" for r in rs]
        if not rows:
            continue
        xs = [x for _, x in bins]
        ys = [_mean(rows, k) for k, _ in bins]
        ok = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
        if ok:
            ax.plot([o[0] for o in ok], [o[1] for o in ok], label=a, **_style(a))
    ax.set_xlabel("distance travelled (m)")
    ax.set_ylabel("position RMSE (m)")
    _safe_log(ax)
    ax.set_title("ATE vs trajectory length (both prompts, k=30, clean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_2_ate_vs_trajectory_length.png"), dpi=140)
    plt.close(fig)


def plot_vs_interval(g, out, tag):
    ks = [1, 5, 10, 30, 100]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, mode in zip(axes, ["depth", "pose", "both"]):
        for a in anchors_present(g):
            xs, ys = [], []
            for k in ks:
                rows = [r for (c, an), rs in g.items()
                        if an == a and c == f"{mode}_k{k}_clean" for r in rs]
                v = _mean(rows, "ate_rmse_m") if rows else np.nan
                if np.isfinite(v):
                    xs.append(k); ys.append(v)
            if xs:
                ax.plot(xs, ys, label=a, **_style(a))
        ax.set_xscale("log")
        _safe_log(ax)
        ax.set_xlabel("prompt interval K (frames)")
        ax.set_title(f"{mode} prompts")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ATE RMSE (m), no alignment")
    axes[2].legend(fontsize=7)
    fig.suptitle("Error vs prompt interval (clean prompts)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_3_error_vs_prompt_interval.png"), dpi=140)
    plt.close(fig)


def plot_vs_noise(g, out, tag):
    levels = ["clean", "moderate", "heavy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (metric, label) in zip(axes, [("ate_rmse_m", "ATE RMSE (m)"),
                                          ("abs_rel", "depth AbsRel")]):
        width = 0.8 / max(1, len(anchors_present(g)))
        for i, a in enumerate(anchors_present(g)):
            ys = []
            for lv in levels:
                rows = [r for (c, an), rs in g.items()
                        if an == a and c == f"both_k30_{lv}" for r in rs]
                ys.append(_mean(rows, metric) if rows else np.nan)
            ax.bar(np.arange(len(levels)) + i * width, ys, width, label=a,
                   alpha=0.6 if a in NON_CAUSAL else 0.9)
        ax.set_xticks(np.arange(len(levels)) + 0.4)
        ax.set_xticklabels(levels)
        ax.set_ylabel(label)
        _safe_log(ax)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_title("Accuracy vs prompt noise (both prompts, k=30)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_4_accuracy_vs_prompt_noise.png"), dpi=140)
    plt.close(fig)


def plot_runtime(g, out, tag, rows_all):
    lingbot = _mean([r for rs in g.values() for r in rs], "lingbot_ms_per_frame")
    names, vals = [], []
    for a in anchors_present(g):
        rows = [r for (c, an), rs in g.items() if an == a for r in rs]
        names.append(a); vals.append(_mean(rows, "update_ms_per_frame"))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(names, vals, color=["#999" if n in NON_CAUSAL else "#3a7" for n in names])
    if np.isfinite(lingbot):
        ax.axhline(lingbot, color="crimson", linestyle="--",
                   label=f"frozen LingbotMap inference: {lingbot:.0f} ms/frame")
        ax.legend(fontsize=9)
    _safe_log(ax)
    ax.set_ylabel("ms per frame")
    ax.set_title("Correction latency vs frozen LingbotMap inference")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_5_runtime_vs_method.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, nargs="+")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="fig")
    args = ap.parse_args()
    rows = []
    for p in args.rows:
        rows.extend(json.load(open(p)))
    os.makedirs(args.output_dir, exist_ok=True)
    g = group(rows)
    plot_error_vs_gap(g, args.output_dir, args.tag)
    plot_ate_vs_distance(g, args.output_dir, args.tag)
    plot_vs_interval(g, args.output_dir, args.tag)
    plot_vs_noise(g, args.output_dir, args.tag)
    plot_runtime(g, args.output_dir, args.tag, rows)
    print("wrote 5 figures to", args.output_dir)


if __name__ == "__main__":
    main()
