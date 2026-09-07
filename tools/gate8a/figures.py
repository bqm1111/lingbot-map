#!/usr/bin/env python
"""Gate 8A figures: PR and calibration for both source datasets and for KITTI-360.

    python tools/gate8a/figures.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")
ORDER = ["mapper", "cellA_best", "cellA_last", "cellB_best", "cellB_last",
         "cellC_best", "cellC_last", "cellD_best", "cellD_last", "selected"]


def blocks(source, tag, region):
    out = {}
    for p in sorted(glob.glob(os.path.join(ART, f"scores_{source}_{tag}_*.npz"))):
        name = os.path.basename(p)[len(f"scores_{source}_{tag}_"):-4]
        z = np.load(p, allow_pickle=False)
        out[name] = {k[len(region) + 1:]: z[k] for k in z.files if k.startswith(region + "_")}
    return out


def _key(n):
    return (ORDER.index(n) if n in ORDER else 99, n)


def panel(ax, bl, kind, region):
    for name in sorted(bl, key=_key):
        b = bl[name]
        sw = SC.sweep(b["pos"], b["neg"])
        if kind == "pr":
            ax.plot(sw["recall"], sw["precision"], lw=1.4,
                    label=f"{name} (AP {SC.average_precision(sw):.3f})")
        elif kind == "iou":
            ax.plot(sw["tau"], sw["iou"], lw=1.4, label=name)
        else:
            r = SC.reliability(b["cal_n"], b["cal_p"], b["cal_y"])
            ax.plot(r["confidence"], r["accuracy"], "o-", ms=3, lw=1.2,
                    label=f"{name} (ECE {SC.ece(b['cal_n'], b['cal_p'], b['cal_y']):.3f})")
    if kind == "pr":
        prev = float(SC.sweep(list(bl.values())[0]["pos"],
                              list(bl.values())[0]["neg"])["n_pos"] /
                     SC.sweep(list(bl.values())[0]["pos"], list(bl.values())[0]["neg"])["n_tot"])
        ax.axhline(prev, color="k", ls=":", lw=1, label=f"prevalence {prev:.3f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1)
    elif kind == "iou":
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("threshold on final log-odds"); ax.set_ylabel("binary IoU")
        ax.set_xlim(-6, 6)
    else:
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("mean predicted P(occupied)"); ax.set_ylabel("empirical frequency")
    ax.set_title(f"{kind} · {region}")
    ax.grid(alpha=.3); ax.legend(fontsize=6)


def figure(source, tag, out):
    bl = {r: blocks(source, tag, r) for r in ("full", "edit")}
    if not bl["full"]:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, r in enumerate(("full", "edit")):
        for j, kind in enumerate(("pr", "iou", "cal")):
            panel(axes[i, j], bl[r], kind, r)
    fig.suptitle(f"Gate 8A · {source} · {tag}")
    fig.tight_layout()
    p = os.path.join(ART, out)
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)
    return p


def main() -> int:
    for src, tag, out in (("semantickitti", "source", "fig_pr_semantickitti.png"),
                          ("occ3d", "source", "fig_pr_occ3d.png"),
                          ("kitti360", "locked", "fig_pr_kitti360.png")):
        figure(src, tag, out)
    res = os.path.join(ART, "gate8a_results.json")
    if os.path.exists(res):
        r = json.load(open(res))
        rows = r.get("ablation", [])
        if rows:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            names = [x["cell"] for x in rows]
            x = np.arange(len(names))
            for k, (lbl, key) in enumerate((("SemanticKITTI", "semantickitti"),
                                            ("Occ3D", "occ3d"))):
                ax[0].bar(x + 0.35 * k - 0.18, [y["ap"][key] for y in rows], 0.33, label=lbl)
                ax[0].plot(x + 0.35 * k - 0.18, [y["prevalence"][key] for y in rows], "k_",
                           ms=14, label="prevalence" if k == 0 else None)
            ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, ha="right")
            ax[0].set_ylabel("average precision"); ax[0].legend(fontsize=7)
            ax[0].set_title("source-validation AP vs prevalence"); ax[0].grid(alpha=.3, axis="y")
            ax[1].bar(x - 0.18, [y["full_iou_at_tau"]["macro"] for y in rows], 0.33,
                      label="macro full IoU @ own tau")
            ax[1].bar(x + 0.18, [y["full_iou_at_zero"]["macro"] for y in rows], 0.33,
                      label="macro full IoU @ 0")
            ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20, ha="right")
            ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
            ax[1].set_title("calibration gain from the source-selected threshold")
            fig.tight_layout(); p = os.path.join(ART, "fig_ablation.png")
            fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
