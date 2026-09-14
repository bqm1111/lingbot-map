#!/usr/bin/env python
"""Training curves and the comparison chart.

    python tools/feature_study/figures.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402

COLOR = {"A": "#7a7a7a", "B": "#1f6fb4", "C": "#c1462f"}


def smooth(x, k=51):
    if len(x) < k:
        return np.asarray(x, float)
    c = np.convolve(np.nan_to_num(np.asarray(x, float), nan=np.nan), np.ones(k) / k, "valid")
    return np.concatenate([np.full(k - 1, np.nan), c])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(ST.ART, "figures")
    os.makedirs(out, exist_ok=True)
    logs = {}
    for c in ST.CONDITIONS:
        p = os.path.join(ST.ART, f"train_{ST.run_tag(c, a.seed)}.json")
        if os.path.exists(p):
            logs[c] = [x for x in json.load(open(p))["log"] if "val" not in x]
    if not logs:
        raise SystemExit("no training records yet")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for c, rows in logs.items():
        st = [r["step"] for r in rows]
        ax[0].plot(st, smooth([r["occ"] for r in rows]), color=COLOR[c], label=c, lw=1.3)
        ax[1].plot(st, smooth([r["iou"] for r in rows]), color=COLOR[c], label=c, lw=1.3)
        if c != "A":
            ax[2].plot(st, smooth([r["aux"] for r in rows]), color=COLOR[c], lw=1.3,
                       label=("B: KL to class probabilities" if c == "B"
                              else "C: cosine distance to features"))
    ax[0].set_title("occupancy loss (focal + Dice)", fontsize=9)
    ax[1].set_title("training-crop SC IoU at logit 0", fontsize=9)
    ax[2].set_title("auxiliary loss\n(different units -- not comparable in level)", fontsize=9)
    for k in ax:
        k.set_xlabel("step", fontsize=8); k.legend(fontsize=7); k.grid(alpha=.25)
        k.tick_params(labelsize=7)
    fig.suptitle(f"feature-supervision study, seed {a.seed}", fontsize=10)
    fig.savefig(os.path.join(out, f"training_seed{a.seed}.png"), dpi=140)
    plt.close(fig)

    cp = os.path.join(ST.ART, "comparison.json")
    if os.path.exists(cp):
        cmp_ = json.load(open(cp))
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
        for k, (title, key) in enumerate((("map-unobserved / later-measured", "later_iou"),
                                          ("full grid", "full_iou"))):
            for i, c in enumerate(ST.CONDITIONS):
                v = [r[key] for n, r in cmp_["source"].items() if n.startswith(c + "_")]
                if not v:
                    continue
                ax[k].bar(i, np.median(v), color=COLOR[c], width=.6)
                ax[k].plot([i] * len(v), v, "o", color="k", ms=4, mfc="none")
            ax[k].set_xticks(range(len(ST.CONDITIONS))); ax[k].set_xticklabels(ST.CONDITIONS)
            ax[k].set_title(f"SC IoU -- {title}", fontsize=9)
            ax[k].grid(alpha=.25, axis="y"); ax[k].tick_params(labelsize=8)
        fig.suptitle("KITTI-360 drive 0006  -  bars are the median over three seeds, "
                     "circles the seeds", fontsize=9)
        fig.savefig(os.path.join(out, "source_comparison.png"), dpi=140)
        plt.close(fig)
    print(f"figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
