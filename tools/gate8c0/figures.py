#!/usr/bin/env python
"""Gate 8C-0 figures: alignment, oracle history, memorisation curves, channel profiles."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402

P = {"train": "#1f77b4", "source_validation": "#ff7f0e", "heldout": "#2ca02c"}


def load(n):
    p = os.path.join(ART, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig_alignment():
    s2, s2b = load("stage2_alignment"), load("stage2b_target_consistency")
    if not (s2 and s2b):
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sh = s2["aggregate_shift_scan"]
    ks = list(sh); x = np.arange(len(ks))
    ax[0].bar(x, [sh[k]["iou"] for k in ks], color=["#d62728" if k == "(0, 0, 0)" else "#8c8c8c" for k in ks])
    ax[0].set_xticks(x); ax[0].set_xticklabels(ks, rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("IoU vs official target"); ax[0].set_title("KITTI-360: ±1 voxel shift scan\n(red = no shift, the correct one)")
    ax[0].grid(alpha=.3, axis="y")
    k3 = s2b["kitti360"]; sk = s2b["semantickitti"]
    rows = [("ours\nvs SSCBench's own .bin\n(their voxel input)", k3["ours_vs_official_bin"]),
            ("SSCBench .bin\nvs SSCBench label\n(both official, us nowhere)", k3["official_bin_vs_official_label"]),
            ("SemanticKITTI sweep\nvs its label\n(reference implementation)", sk["sweep_vs_official_label"])]
    w = 0.26
    for i, sh_ in enumerate(("(0, 0, 0)", "(0, 0, 1)", "(0, 0, -1)")):
        ax[1].bar(np.arange(3) + (i - 1) * w, [r[1][sh_]["precision"] for r in rows], w,
                  label=f"shift {sh_}")
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels([r[0] for r in rows], fontsize=6, rotation=12, ha="center")
    ax[1].set_ylabel("precision"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("where the +1 z offset lives")
    dz = s2b["kitti360_continuous_dz_scan"]
    xs = sorted(float(k) for k in dz)
    ax[2].plot(xs, [dz[str(k)]["precision"] for k in xs], "-o", ms=3)
    ax[2].axvline(0, color="k", ls=":"); ax[2].axvline(0.2, color="r", ls="--", lw=1, label="one voxel")
    ax[2].set_xlabel("z offset applied to our LiDAR (m)"); ax[2].set_ylabel("precision vs target")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3); ax[2].set_title("continuous z scan (KITTI-360)")
    fig.tight_layout(); p = os.path.join(ART, "fig_alignment.png"); fig.savefig(p, dpi=130)
    plt.close(fig); print("wrote", p)


def fig_oracle():
    s3 = load("stage3_temporal"); s4 = load("stage4_factorization")
    if not s3:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    kinds = ["1", "5", "20", "all_past"]
    for part, v in s3["aggregate"].items():
        ax[0].plot(range(len(kinds)), [v[k]["full_grid"]["recall"] for k in kinds], "-o",
                   color=P[part], label=part)
        ax[1].plot(range(len(kinds)), [v[k]["full_grid"]["precision"] for k in kinds], "-o",
                   color=P[part], label=part)
        ax[0].axhline(v["future"]["full_grid"]["recall"], color=P[part], ls=":", lw=1)
    for i, (t, yl) in enumerate((("oracle GT-LiDAR target coverage", "recall of official occupied"),
                                 ("oracle GT-LiDAR precision", "precision vs official occupied"))):
        ax[i].set_xticks(range(len(kinds))); ax[i].set_xticklabels(kinds)
        ax[i].set_xlabel("causal history (stream frames)"); ax[i].set_ylabel(yl)
        ax[i].legend(fontsize=7); ax[i].grid(alpha=.3); ax[i].set_title(t)
    ax[0].text(.02, .95, "dotted = non-causal t+1..t+20 diagnostic", transform=ax[0].transAxes,
               fontsize=6.5, va="top")
    if s4:
        cells = ["gt_depth_gt_pose", "gt_depth_lb_pose", "lb_depth_gt_pose", "lb_depth_lb_pose"]
        x = np.arange(len(cells))
        for i, part in enumerate(s4["aggregate"]):
            ax[2].bar(x + (i - 1) * 0.26, [s4["aggregate"][part][c]["20"]["iou"] for c in cells],
                      0.25, color=P[part], label=part)
        ax[2].set_xticks(x); ax[2].set_xticklabels([c.replace("_", "\n") for c in cells], fontsize=6.5)
        ax[2].set_ylabel("SC IoU (20-frame history)"); ax[2].legend(fontsize=7)
        ax[2].grid(alpha=.3, axis="y"); ax[2].set_title("depth × pose factorization")
    fig.tight_layout(); p = os.path.join(ART, "fig_oracle.png"); fig.savefig(p, dpi=130)
    plt.close(fig); print("wrote", p)


def fig_memorize():
    s6 = load("stage6_memorize")
    if not s6:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for name, run in s6["runs"].items():
        st = [r["step"] for r in run["log"]]
        ax[0].plot(st, [r["loss"] for r in run["log"]], label=f"batch {name} total")
        ax[0].plot(st, [r["focal"] for r in run["log"]], ls="--", label=f"batch {name} focal")
        ax[0].plot(st, [r["sem_kl"] for r in run["log"]], ls=":", label=f"batch {name} KL")
        ax[1].plot(st, [r["edit"]["ap"] for r in run["log"]], "-o", ms=2.5, label=f"batch {name} edit AP")
        ax[1].plot(st, [r["edit"]["best_iou"] for r in run["log"]], ls="--", label=f"batch {name} best IoU")
        ax[2].plot(st, [r["grad_norm"] for r in run["log"]], label=f"batch {name}")
    ax[0].set_yscale("log"); ax[0].set_title("fixed-batch losses")
    ax[1].axhline(1.0, color="k", ls=":", lw=1); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("editable-region occupancy memorisation")
    ax[2].set_yscale("log"); ax[2].set_title("gradient norm")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=.3); a.legend(fontsize=6.5)
    fig.tight_layout(); p = os.path.join(ART, "fig_memorize.png"); fig.savefig(p, dpi=130)
    plt.close(fig); print("wrote", p)


def fig_channels():
    s7 = load("stage7_channel_stats")
    if not s7:
        return
    parts = list(s7["partitions"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    n = 7
    x = np.arange(n)
    for i, p_ in enumerate(parts):
        v = s7["partitions"][p_]
        ax[0].bar(x + (i - 1) * 0.26, v["channel_mean"][:n], 0.25, yerr=v["channel_sd"][:n],
                  color=P[p_], label=p_, error_kw={"lw": .7})
        ax[1].plot(v["occupancy_z_profile"], np.arange(len(v["occupancy_z_profile"])) * 0.2 - 2.0,
                   "-o", ms=2.5, color=P[p_], label=p_)
        ax[2].plot(np.linspace(0, 51.2, len(v["occupancy_range_profile"])),
                   v["occupancy_range_profile"], "-o", ms=2.5, color=P[p_], label=p_)
    ax[0].set_xticks(x); ax[0].set_xticklabels(s7["channel_names"][:n], rotation=35, ha="right", fontsize=6.5)
    ax[0].set_title("map input channels (mean ± sd)"); ax[0].set_ylabel("normalised value")
    ax[1].set_xlabel("fraction of occupied voxels"); ax[1].set_ylabel("height z (m)")
    ax[1].set_title("causal-map occupancy by height")
    ax[2].set_xlabel("range x (m)"); ax[2].set_ylabel("fraction of occupied voxels")
    ax[2].set_title("causal-map occupancy by range")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); p = os.path.join(ART, "fig_channels.png"); fig.savefig(p, dpi=130)
    plt.close(fig); print("wrote", p)


def main() -> int:
    fig_alignment(); fig_oracle(); fig_memorize(); fig_channels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
