#!/usr/bin/env python
"""Diagnostic plots: training/validation KL curves, the cosine distribution of every
condition on the key set, per-class recall, and accuracy against distance to the nearest
map voxel with teacher evidence.

    python tools/future_semantic_completion/figures.py --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402

COLORS = {"B0": "#9a9a9a", "B1": "#d98c3a", "B2": "#2c6fbb", "B2ref_existing_head": "#7a5ea8"}


def curves(seed, fig_dir):
    tr = json.load(open(os.path.join(ST.ART, f"train_{ST.run_tag(seed)}.json")))
    steps = [m["step"] for m in tr["log"] if "val" not in m]
    kl = [m["loss"] for m in tr["log"] if "val" not in m]
    vs = [m["val"] for m in tr["log"] if "val" in m]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    k = 25
    sm = np.convolve(kl, np.ones(k) / k, mode="valid")
    ax[0].plot(steps[k - 1:], sm, color=COLORS["B2"], label="train KL (25-step mean)")
    ax[0].plot([v["step"] for v in vs], [v["loss"] for v in vs], "o-", color="#c0392b", label="val KL (key set)")
    ax[0].axhline(vs[-1]["b0_loss"], color=COLORS["B0"], ls="--", label="B0 KL on val")
    ax[0].axhline(vs[-1]["existing_head_loss"], color=COLORS["B2ref_existing_head"], ls=":", label="existing joint head KL on val")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("KL(teacher || pred)"); ax[0].legend(fontsize=8); ax[0].set_title(f"seed {seed}: semantic loss")
    ax[1].plot([v["step"] for v in vs], [v["cos"] for v in vs], "o-", color=COLORS["B2"], label="val cosine to target")
    ax[1].plot([v["step"] for v in vs], [v["cos_b0"] for v in vs], "s--", color=COLORS["B0"], label="val cosine to B0 (collapse)")
    ax[1].set_xlabel("step"); ax[1].legend(fontsize=8); ax[1].set_title("feature similarity")
    ax[2].plot([v["step"] for v in vs], [v["top1"] for v in vs], "o-", color=COLORS["B2"], label="val top-1 agreement")
    ax[2].set_xlabel("step"); ax[2].legend(fontsize=8); ax[2].set_title("naming")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"curves_seed{seed}.png"), dpi=130); plt.close(fig)


def source(seed, fig_dir):
    src = json.load(open(os.path.join(ST.ART, "source", f"source_{ST.run_tag(seed)}.json")))
    key = src["populations"]["key"]
    conds = list(key)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.asarray(key["B2"]["cosine_hist_bins"]); c = 0.5 * (bins[1:] + bins[:-1])
    for k in conds:
        h = np.asarray(key[k]["cosine_hist"], float); h /= max(h.sum(), 1)
        ax[0].plot(c, h, "-o", ms=3, color=COLORS[k], label=f"{k} (mean {key[k]['cosine_mean']:.3f})")
    ax[0].set_xlabel("cosine(prediction, future teacher)"); ax[0].set_ylabel("fraction of key voxels")
    ax[0].set_title("drive 0006, unobserved -> later occupied"); ax[0].legend(fontsize=8)
    names = [n for n, v in key["B2"]["per_class"].items() if v["support"] > 0]
    names.sort(key=lambda n: -key["B2"]["per_class"][n]["support"])
    xx = np.arange(len(names)); w = 0.2
    for j, k in enumerate(conds):
        ax[1].bar(xx + (j - 1.5) * w, [key[k]["per_class"][n]["recall"] or 0 for n in names], w, color=COLORS[k], label=k)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("recall of teacher argmax"); ax[1].set_title("per-class recall (support-ordered)"); ax[1].legend(fontsize=8)
    d = src["distance_to_evidence"]
    for k in conds:
        ax[2].plot(d["names"], d["by_condition"][k]["top1"], "-o", color=COLORS[k], label=k)
    ax2 = ax[2].twinx(); ax2.bar(d["names"], np.asarray(d["key_hist"]) / sum(d["key_hist"]), alpha=0.15, color="k")
    ax2.set_ylabel("fraction of key voxels")
    ax[2].set_xlabel("distance to nearest map voxel with teacher evidence (voxels)"); ax[2].set_ylabel("top-1 agreement")
    ax[2].set_title("propagation reach"); ax[2].legend(fontsize=8)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"source_seed{seed}.png"), dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    fig_dir = os.path.join(ST.ART, "figures"); os.makedirs(fig_dir, exist_ok=True)
    for s in a.seed:
        curves(s, fig_dir)
        if os.path.exists(os.path.join(ST.ART, "source", f"source_{ST.run_tag(s)}.json")):
            source(s, fig_dir)
    print(f"figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
