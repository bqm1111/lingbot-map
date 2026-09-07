#!/usr/bin/env python
"""Gate 8B figures: PR / IoU-vs-threshold / reliability per target for both settings, and
the three-fold AP-over-prevalence summary.

    python tools/gate8b/figures.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402

G8A = os.path.join(REPO_ROOT, "artifacts", "gate8a")
TARGETS = ("semantickitti", "occ3d", "kitti360")


def blk(art, ds, tag, name, region):
    p = os.path.join(art, f"scores_{ds}_{tag}_{name}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k[len(region) + 1:]: z[k] for k in z.files if k.startswith(region + "_")}


def panel(ax, curves, kind):
    for label, b in curves:
        sw = SC.sweep(b["pos"], b["neg"])
        if kind == "pr":
            ax.plot(sw["recall"], sw["precision"], lw=1.4, label=f"{label} (AP {SC.average_precision(sw):.3f})")
        elif kind == "iou":
            ax.plot(sw["tau"], sw["iou"], lw=1.4, label=label)
        else:
            r = SC.reliability(b["cal_n"], b["cal_p"], b["cal_y"])
            ax.plot(r["confidence"], r["accuracy"], "o-", ms=3, lw=1.2,
                    label=f"{label} (ECE {SC.ece(b['cal_n'], b['cal_p'], b['cal_y']):.3f})")
    if kind == "pr":
        sw = SC.sweep(curves[0][1]["pos"], curves[0][1]["neg"])
        ax.axhline(sw["n_pos"] / sw["n_tot"], color="k", ls=":", lw=1, label="prevalence")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1)
    elif kind == "iou":
        ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_xlim(-6, 6)
        ax.set_xlabel("threshold on final log-odds"); ax.set_ylabel("SC IoU")
    else:
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("mean predicted P(occupied)"); ax.set_ylabel("empirical frequency")
    ax.grid(alpha=.3); ax.legend(fontsize=6)


def main() -> int:
    for t in TARGETS:
        rows = []
        for setting, art, tag in (("all-past streaming", G8A if t == "kitti360" else ART,
                                   "locked" if t == "kitti360" else f"stream_{t}"),
                                  ("matched five-frame", ART, f"clips_{t}")):
            c = [(f"{setting}: mapper", blk(art, t, tag, "mapper", "full")),
                 (f"{setting}: completion", blk(art, t, tag, "selected", "full"))]
            if all(b is not None for _, b in c):
                rows.append((setting, c))
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.6 * len(rows)), squeeze=False)
        for i, (setting, c) in enumerate(rows):
            for j, kind in enumerate(("pr", "iou", "cal")):
                panel(axes[i, j], c, kind); axes[i, j].set_title(f"{kind} · full grid · {setting}")
        fig.suptitle(f"Gate 8B · {t} (held-out target)"); fig.tight_layout()
        p = os.path.join(ART, f"fig_{t}.png"); fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)
    rp = os.path.join(ART, "gate8b_results.json")
    if os.path.exists(rp):
        r = json.load(open(rp)); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        x = np.arange(len(TARGETS))
        for k, setting in enumerate(("stream", "clips")):
            ap = [r["settings"][t][setting]["threshold_free"]["completion_full"]["ap_over_prevalence"]
                  if r["settings"][t][setting] else np.nan for t in TARGETS]
            au = [r["settings"][t][setting]["threshold_free"]["completion_full"]["auroc"]
                  if r["settings"][t][setting] else np.nan for t in TARGETS]
            ax[0].bar(x + 0.35 * k - 0.18, ap, 0.33, label=setting)
            ax[1].bar(x + 0.35 * k - 0.18, au, 0.33, label=setting)
        ax[0].axhline(1.0, color="k", ls=":"); ax[0].set_ylabel("AP / prevalence"); ax[0].set_title("ranking lift on the held-out target")
        ax[1].axhline(0.5, color="k", ls=":"); ax[1].set_ylabel("AUROC"); ax[1].set_title("AUROC on the held-out target")
        for a in ax:
            a.set_xticks(x); a.set_xticklabels(TARGETS); a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(ART, "fig_folds.png"); fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
