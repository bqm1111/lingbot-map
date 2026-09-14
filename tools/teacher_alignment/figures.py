#!/usr/bin/env python
"""The gradient-budget / gradient-direction figure and the R-S result chart.

    python tools/teacher_alignment/figures.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402

COLOR = {"R": "#1f6fb4", "S": "#c1462f"}


def smooth(x, k=101):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, "valid")
    return np.concatenate([np.full(k - 1, np.nan), c])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ST.ART, "figures"); os.makedirs(out, exist_ok=True)
    rho = json.load(open(ST.rho_path()))["rho"]
    logs = {}
    for c in ST.CONDITIONS:
        for s in ST.SEEDS:
            p = os.path.join(ST.ART, f"train_{ST.run_tag(c, s)}.json")
            if os.path.exists(p):
                logs[(c, s)] = [x for x in json.load(open(p))["log"] if "val" not in x]
    if not logs:
        raise SystemExit("no training records")

    fig, ax = plt.subplots(2, 3, figsize=(14, 6.4), constrained_layout=True)
    for (c, s), rows in sorted(logs.items()):
        st = [r["step"] for r in rows]; lw = 1.2 if s == 0 else 0.7; al = 1.0 if s == 0 else 0.5
        kw = dict(color=COLOR[c], lw=lw, alpha=al, label=f"{c} seed{s}" if s == 0 else None)
        ax[0, 0].plot(st, smooth([r["occ_grad_norm"] for r in rows]), **kw)
        ax[0, 1].plot(st, smooth([r["aux_grad_norm"] for r in rows]), **kw)
        ax[0, 2].plot(st, smooth([r["raw_ratio"] for r in rows]), **kw)
        ax[1, 0].plot(st, [r["achieved_ratio"] for r in rows], **kw)
        ax[1, 1].plot(st, smooth([r["norm_factor"] for r in rows]), **kw)
        ax[1, 2].plot(st, smooth([r["grad_cosine"] for r in rows]), **kw)
    ax[1, 0].axhline(rho, color="k", ls="--", lw=0.8, label=f"rho = {rho:.3f}")
    ax[0, 2].axhline(rho, color="k", ls="--", lw=0.8)
    ax[1, 2].axhline(0, color="k", lw=0.5)
    titles = [("||g_occ|| on shared params (raw)", "log"), ("||g_aux|| = ||d(0.5 KL)|| (raw)", "log"),
              ("raw ratio ||g_aux||/||g_occ||", "log"), ("achieved ratio after matching", "linear"),
              ("normalisation factor c", "log"), ("cos(g_occ, g_aux) on shared params", "linear")]
    for k, (t, sc) in zip(ax.ravel(), titles):
        k.set_title(t, fontsize=9); k.set_yscale(sc); k.grid(alpha=.25); k.tick_params(labelsize=7)
        k.set_xlabel("step", fontsize=8)
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_ylim(0, rho * 1.5)
    fig.suptitle("Gradient budget at the shared trunk -- R (aligned) vs S (shuffled), "
                 "three seeds each (thin = seeds 1, 2)", fontsize=10)
    fig.savefig(os.path.join(out, "gradient_budget.png"), dpi=140); plt.close(fig)

    cp = os.path.join(ST.ART, "comparison.json")
    if os.path.exists(cp):
        cmp_ = json.load(open(cp))
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        panels = [("drive 0006 later-measured", "source", "later_iou", None),
                  ("SemanticKITTI past5", "transfer", "iou", "semantickitti"),
                  ("Occ3D past5", "transfer", "iou", "occ3d")]
        for k, (t, sec, key, ds) in enumerate(panels):
            for i, c in enumerate(ST.CONDITIONS + ("A", "B")):
                tab = cmp_[sec] if c in ST.CONDITIONS else cmp_[sec + "_reference_AB"]
                v = [r[key] for n, r in tab.items()
                     if n.startswith(c + "_") and (ds is None or n.endswith(ds))]
                if not v:
                    continue
                col = COLOR.get(c, "#9a9a9a")
                ax[k].bar(i, np.median(v), color=col, width=.6, alpha=1.0 if c in ST.CONDITIONS else .45)
                ax[k].plot([i] * len(v), v, "o", color="k", ms=4, mfc="none")
            ax[k].set_xticks(range(4)); ax[k].set_xticklabels(["R", "S", "A (ref)", "B (ref)"], fontsize=8)
            ax[k].set_title(f"SC IoU -- {t}", fontsize=9); ax[k].grid(alpha=.25, axis="y")
            ax[k].tick_params(labelsize=8)
        fig.suptitle("bars = median of three seeds, circles = seeds; A/B are the feature study's "
                     "nominal-weight runs, shown for reference only", fontsize=9)
        fig.savefig(os.path.join(out, "result.png"), dpi=140); plt.close(fig)
    print("figures ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
