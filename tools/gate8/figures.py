#!/usr/bin/env python
"""Gate 8 figures from the artifacts."""
from __future__ import annotations
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
ART = os.path.join(REPO_ROOT, "artifacts", "gate8")
LABEL = {"semantickitti": "SemanticKITTI 08 (source val)", "occ3d": "Occ3D-nuScenes (source val)",
         "kitti360": "KITTI-360 (HELD OUT)"}
COL = {"frozen_5frame": "#666666", "mapper": "#1b6ca8", "mapper_plus_completion": "#c1440e"}


def fig_train():
    p = os.path.join(ART, "train_completion.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p)); log = d["log"]
    tr = [x for x in log if "step" in x and "val" not in x]; va = [x["val"] for x in log if "val" in x]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot([x["step"] for x in tr], [x["loss"] for x in tr], lw=0.8, color=COL["mapper"], label="train")
    if va: ax[0].plot([x["step"] for x in va], [x["loss"] for x in va], "o-", color=COL["mapper_plus_completion"], label="val")
    ax[0].set_ylabel("loss"); ax[0].legend(frameon=False)
    ax[1].plot([x["step"] for x in tr], [x["iou"] for x in tr], lw=0.8, color=COL["mapper_plus_completion"], label="completed (train crops)")
    ax[1].plot([x["step"] for x in tr], [x["base_iou"] for x in tr], lw=0.8, color=COL["mapper"], label="frozen map (same crops)")
    if va:
        ax[1].plot([x["step"] for x in va], [x["iou"] for x in va], "o", color=COL["mapper_plus_completion"], label="completed (val)")
        ax[1].plot([x["step"] for x in va], [x["base_iou"] for x in va], "s", color=COL["mapper"], label="frozen map (val)")
    ax[1].set_ylabel("crop binary IoU"); ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot([x["step"] for x in tr], [x["sem_kl"] for x in tr], lw=0.8, color="#2e7d32", label="teacher KL (train)")
    if va: ax[2].plot([x["step"] for x in va], [x["sem_kl"] for x in va], "o-", color="#2e7d32", label="val")
    ax[2].set_ylabel("semantic KL"); ax[2].legend(frameon=False)
    for a in ax: a.set_xlabel("step"); a.grid(alpha=0.3)
    fig.suptitle("Completion training (sources: SemanticKITTI train seqs + Occ3D train scenes)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(ART, "fig_training.png"), dpi=150); plt.close(fig)
    print("  fig_training.png")


def fig_results():
    p = os.path.join(ART, "gate8_results.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    for var in ("raw", "dil"):
        fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
        for i, metric in enumerate(("binary_iou", "ssc_miou", "tp_accuracy")):
            xs, labels = [], []
            for j, ds in enumerate(("semantickitti", "occ3d", "kitti360")):
                m = r["datasets"].get(ds, {}).get("methods", {})
                for k, meth in enumerate(("frozen_5frame", "mapper", "mapper_plus_completion")):
                    v = m.get(f"{meth}_{var}")
                    if v is None:
                        continue
                    ax[i].bar(j * 4 + k, v[metric], color=COL[meth], width=0.9,
                              label=meth.replace("_", " ") if j == 0 else None)
            ax[i].set_xticks([1, 5, 9]); ax[i].set_xticklabels([LABEL[d].split(" (")[0] for d in ("semantickitti", "occ3d", "kitti360")], fontsize=8)
            ax[i].set_title({"binary_iou": "pooled binary IoU", "ssc_miou": "SSC mIoU", "tp_accuracy": "semantic acc. on TP"}[metric], fontsize=10)
            ax[i].grid(alpha=0.3, axis="y")
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle(f"Gate 8 — {'raw' if var == 'raw' else 'fixed 0.4 m dilation'} (KITTI-360 held out, never tuned)", fontsize=10)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95)); fig.savefig(os.path.join(ART, f"fig_results_{var}.png"), dpi=150); plt.close(fig)
        print(f"  fig_results_{var}.png")


def fig_memory():
    ps = [os.path.join(ART, f"runtime_{d}.json") for d in ("kitti360",)]
    r = [json.load(open(p)) for p in ps if os.path.exists(p)]
    if not r:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    for d in r:
        c = d["components"]
        names = [k for k, v in c.items() if v]
        ax.barh(names, [c[k]["median_ms"] for k in names], color="#1b6ca8")
    ax.set_xlabel("median latency per frame (ms)"); ax.set_xscale("log"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "fig_runtime.png"), dpi=150); plt.close(fig)
    print("  fig_runtime.png")


if __name__ == "__main__":
    print("figures:"); fig_train(); fig_results(); fig_memory()
