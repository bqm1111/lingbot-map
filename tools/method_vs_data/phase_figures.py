#!/usr/bin/env python
"""Figures for this phase: accuracy versus latency, and regional performance.

Every number plotted is read from a machine-readable artifact written by the tool that measured
it -- never retyped. The accuracy-versus-latency figure refuses to plot a point whose accuracy
and latency come from different configurations: each point names one configuration and both of
its numbers are looked up under that name.

    PYTHONPATH=$PWD python tools/method_vs_data/phase_figures.py --which all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib                                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                        # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MVD = os.path.join(ROOT, "artifacts", "method_vs_data")
RUNTIME = os.path.join(MVD, "runtime")
OBS = os.path.join(MVD, "observed_hidden")
FIG = os.path.join(MVD, "figures", "phase_runtime_hidden")

#: (label, accuracy source, arm key in that source, latency json, window, marker, colour)
POINTS = [
    ("Ours, 5 causal frames", "one_frame_semantickitti_{post}.json", "five_frame_unet",
     "ours_semantickitti_5f_e2e.json", "causal", "o", "#1b6ca8"),
    ("Ours, 1 causal frame", "one_frame_semantickitti_{post}.json", "one_frame_unet",
     "ours_semantickitti_1f_e2e.json", "causal", "s", "#4da3d4"),
    ("Ours, map only (no learned head)", "one_frame_semantickitti_{post}.json",
     "five_frame_map", "DERIVED_ours_map_only", "causal", "v", "#9ecae1"),
    ("OccAny, 1 causal frame", "one_frame_semantickitti_{post}.json", "occany_1f",
     "occany_K_1FRAME_GEOM_e2e.json", "causal", "D", "#b3521b"),
    ("OccAny, 5 forward frames (t..t+40)", "one_frame_semantickitti_{post}.json",
     "occany_5f_fwd", "occany_K_5FRAMES_GEOM_e2e.json", "forward", "^", "#d98042"),
]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def latency(spec: str):
    """(mean_ms, source) for one configuration, or None when it was not measured."""
    if spec.startswith("DERIVED_"):
        st = _load(os.path.join(RUNTIME, "ours_semantickitti_5f_stages.json"))
        e2e = _load(os.path.join(RUNTIME, "ours_semantickitti_5f_e2e.json"))
        if not st or not e2e or not st.get("stages_mean_ms"):
            return None
        unet = st["stages_mean_ms"].get("unet", 0.0)
        return (e2e["latency_per_completed_occupancy_grid"]["mean_ms"] - unet,
                "derived: measured 5-frame end-to-end minus the measured CompletionUNet stage")
    d = _load(os.path.join(RUNTIME, spec))
    if not d:
        return None
    return d["latency_per_completed_occupancy_grid"]["mean_ms"], spec


def accuracy_vs_latency(out: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=False)
    recorded = {}
    for ax, post in zip(axes, ("raw", "pooled")):
        acc = _load(os.path.join(MVD, "results",
                                 f"one_frame_semantickitti_{post}.json"))
        for label, accsrc, arm, latsrc, window, mk, col in POINTS:
            lat = latency(latsrc)
            if acc is None or lat is None or arm not in acc["iou"]["all"]:
                continue
            y = acc["iou"]["all"][arm]["iou"]
            ax.scatter(lat[0], y, s=150, marker=mk, color=col, zorder=3,
                       edgecolor="white", linewidth=1.2,
                       label=label + ("" if window == "causal" else "  [non-causal]"))
            ax.annotate(f"{y:.2f}", (lat[0], y), textcoords="offset points",
                        xytext=(9, -11), fontsize=9, color="#333")
            recorded.setdefault(post, []).append(
                {"configuration": label, "sc_iou": y, "latency_mean_ms": lat[0],
                 "accuracy_source": os.path.join("results", accsrc.format(post=post)),
                 "arm_key": arm, "latency_source": lat[1], "temporal_window": window})
        ax.set_xscale("log")
        ax.set_xticks([150, 200, 300, 400, 600, 900, 1400, 2000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(120, 2600)
        ax.set_xlabel("latency per completed occupancy grid (ms, log scale)")
        ax.set_ylabel("SC IoU (%)")
        ax.set_title(f"SemanticKITTI 08, 801 canonical anchors — {post}")
        ax.grid(alpha=0.25, which="both")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9.5, ncol=3, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Accuracy against measured end-to-end inference latency\n"
                 "same anchors, same scorer; each point's accuracy and latency come from the "
                 "same configuration. [non-causal] reads 4.0 s of future frames.", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    return recorded


def regional(dataset: str, out: str) -> dict:
    d = _load(os.path.join(OBS, f"regional_{dataset}.json"))
    if d is None:
        print("no regional json for", dataset)
        return {}
    regions = d["partition"] + ["never_observed"]
    arms = d["arms"]
    nice = {"k360": "ours (KITTI-360)", "waymo": "ours (Waymo)", "once": "ours (ONCE, exploratory)",
            "map_only": "map only", "occany_1f": "OccAny 1f causal",
            "occany_5f_fwd": "OccAny 5f forward [non-causal]"}
    cols = {"k360": "#1b6ca8", "waymo": "#4da3d4", "once": "#9ecae1", "map_only": "#bdbdbd",
            "occany_1f": "#b3521b", "occany_5f_fwd": "#d98042"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5))
    for ax, post in zip(axes, ("raw", "pooled")):
        w = 0.8 / len(arms)
        x = np.arange(len(regions))
        for i, a in enumerate(arms):
            v = [d["iou"][post][r][a]["iou"] for r in regions]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, label=nice.get(a, a), color=cols.get(a))
            for xx, vv in zip(x + i * w - 0.4 + w / 2, v):
                ax.text(xx, vv + 0.4, f"{vv:.1f}", ha="center", fontsize=7, rotation=90)
        lab = [r.replace("never_observed_", "never obs.\n").replace("_", " ") +
               f"\n{d['composition'][r]['share_of_gt_occupied']*100:.0f}% of GT occupied"
               for r in regions]
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=9)
        ax.set_ylabel("SC IoU (%)")
        ax.set_title(f"{post} — {d['n_samples']} anchors")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    axes[0].legend(fontsize=9, ncol=3, loc="upper left")
    name = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}[dataset]
    fig.suptitle(f"Where the accuracy is: strict visibility regions, {name}\n"
                 "regions are mutually exclusive and partition the valid volume; visibility is "
                 "judged from the five permitted CAUSAL frames, which do not bound OccAny's "
                 "forward arm", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    return {r: {a: d["iou"][p][r][a]["iou"] for a in arms} for p in ("raw",) for r in regions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all",
                    choices=["all", "latency", "regional"])
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rec = {}
    if a.which in ("all", "latency"):
        rec["accuracy_vs_latency"] = accuracy_vs_latency(
            os.path.join(FIG, "accuracy_vs_latency.png"))
    if a.which in ("all", "regional"):
        for ds in ("semantickitti", "occ3d"):
            rec[f"regional_{ds}"] = regional(ds, os.path.join(FIG, f"regional_{ds}.png"))
    json.dump(rec, open(os.path.join(FIG, "figure_data.json"), "w"), indent=1, default=float)
    print("->", os.path.join(FIG, "figure_data.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
