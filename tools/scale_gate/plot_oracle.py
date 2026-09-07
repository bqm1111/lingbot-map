#!/usr/bin/env python
"""Diagnostic plots for the oracle scale analysis."""
from __future__ import annotations

import argparse, csv, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, load_config

C = {"raw": "#8a8f98", "median": "#d9541a", "oracle": "#218c33", "learned": "#3366cc"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    fig_dir = os.path.join(out, "figures"); os.makedirs(fig_dir, exist_ok=True)

    tv = list(csv.DictReader(open(os.path.join(out, "scale_targets_val.csv"))))
    tt = list(csv.DictReader(open(os.path.join(out, "scale_targets_train.csv"))))
    per = list(csv.DictReader(open(os.path.join(out, "oracle_metrics_per_clip.csv"))))
    fl = lambda rs, k: np.array([float(r[k]) for r in rs if r[k] not in ("", "nan")])
    n_val, n_tr = len(tv), len(tt)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1. log(s*) distribution
    ax[0, 0].hist(fl(tt, "log_s_joint"), bins=45, color=C["median"], alpha=.65,
                  label=f"train (n={n_tr})", density=True)
    ax[0, 0].hist(fl(tv, "log_s_joint"), bins=45, color=C["oracle"], alpha=.6,
                  label=f"val seq 08 (n={n_val})", density=True)
    med = np.median(fl(tt, "log_s_joint"))
    ax[0, 0].axvline(med, color="k", ls="--", lw=1.2,
                     label=f"train median s={np.exp(med):.1f}")
    ax[0, 0].set_xlabel("log(s*)  [joint oracle]"); ax[0, 0].set_ylabel("density")
    ax[0, 0].set_title("Oracle scale varies clip to clip"); ax[0, 0].legend(fontsize=8, frameon=False)

    # 2. per-frame dispersion within a clip
    d = fl(tv, "per_frame_log_s_std")
    ax[0, 1].hist(d, bins=45, color=C["oracle"])
    ax[0, 1].axvline(np.median(d), color="k", ls="--", lw=1.2,
                     label=f"median {np.median(d):.4f}")
    ax[0, 1].set_xlabel("std of per-frame log s within a clip")
    ax[0, 1].set_ylabel(f"clips (n={len(d)})")
    ax[0, 1].set_title("One scalar per clip is well posed"); ax[0, 1].legend(fontsize=8, frameon=False)

    # 3. depth vs pose scale
    sd, sp = fl(tv, "s_depth"), fl(tv, "s_pose")
    n = min(len(sd), len(sp))
    ax[0, 2].scatter(sd[:n], sp[:n], s=11, alpha=.55, color=C["oracle"], linewidths=0)
    lim = [min(sd[:n].min(), sp[:n].min()) * .95, max(sd[:n].max(), sp[:n].max()) * 1.05]
    ax[0, 2].plot(lim, lim, "k--", lw=1, label="y = x")
    ae = fl(tv, "agreement_error_log")
    ax[0, 2].set_xlabel("s* from depth (LiDAR)"); ax[0, 2].set_ylabel("s* from pose (GT translations)")
    ax[0, 2].set_title(f"Depth and pose agree — median |log ratio| {np.median(ae):.3f}")
    ax[0, 2].legend(fontsize=8, frameon=False)

    # 4. agreement error distribution
    ax[1, 0].hist(ae, bins=45, color=C["median"])
    ax[1, 0].axvline(cfg.scale.max_agreement_error_log, color="k", ls="--", lw=1.2,
                     label=f"reliability threshold {cfg.scale.max_agreement_error_log}")
    ax[1, 0].set_xlabel("|log(s_depth / s_pose)|"); ax[1, 0].set_ylabel(f"clips (n={len(ae)})")
    ax[1, 0].set_title("Scale is physically coherent"); ax[1, 0].legend(fontsize=8, frameon=False)

    # 5. scale error vs downstream IoU
    for meth, col, lab in (("global_median_train", C["median"], "global median (deployable)"),
                           ("oracle_joint", C["oracle"], "oracle joint")):
        rs = [r for r in per if r["method"] == meth]
        x = np.array([float(r["abs_log_scale_err"]) for r in rs])
        y = np.array([float(r["iou"]) for r in rs])
        ax[1, 1].scatter(x, y, s=11, alpha=.55, color=col, linewidths=0, label=lab)
    ax[1, 1].set_xlabel("|log(s_est / s*)|"); ax[1, 1].set_ylabel("occupancy IoU")
    ax[1, 1].set_title("Downstream IoU vs scale error"); ax[1, 1].legend(fontsize=8, frameon=False)

    # 6. method comparison
    order = ["raw_canonical", "global_median_train", "oracle_pose", "oracle_joint", "oracle_depth"]
    means, errs, labels, cols = [], [], [], []
    for m in order:
        v = np.array([float(r["iou"]) for r in per if r["method"] == m])
        if not v.size:
            continue
        means.append(v.mean()); errs.append(v.std() / np.sqrt(v.size))
        labels.append(m.replace("_", "\n"))
        cols.append(C["raw"] if m == "raw_canonical"
                    else C["median"] if m.startswith("global") else C["oracle"])
    ax[1, 2].bar(range(len(means)), means, yerr=errs, color=cols, capsize=3)
    for i, v in enumerate(means):
        ax[1, 2].text(i, v + max(means) * .03, f"{v:.4f}", ha="center", fontsize=8.5)
    ax[1, 2].set_xticks(range(len(labels))); ax[1, 2].set_xticklabels(labels, fontsize=7.5)
    ax[1, 2].set_ylabel("occupancy IoU (mean ± SEM)")
    ax[1, 2].set_title("Scale is essential; per-clip oracle adds little")
    for axis in ax.ravel():
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)
        axis.tick_params(labelsize=8)

    fig.suptitle(f"Gate 0 oracle scale diagnosis — SemanticKITTI, {cfg.lingbot.clip_length}-frame "
                 f"clips at stride {cfg.lingbot.frame_stride}, config {cfg.hash}", fontsize=13)
    p = os.path.join(fig_dir, "oracle_scale_diagnosis.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(p, REPO_ROOT))

    # per-sequence comparison (val is one sequence, so this is per-clip-decile on seq 08)
    fig2, ax2 = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ids = sorted({r["clip_id"] for r in per})
    idx = {c: i for i, c in enumerate(ids)}
    for m, col in (("global_median_train", C["median"]), ("oracle_joint", C["oracle"])):
        rs = sorted([r for r in per if r["method"] == m], key=lambda r: idx[r["clip_id"]])
        ax2.plot([idx[r["clip_id"]] for r in rs], [float(r["iou"]) for r in rs],
                 lw=1.1, color=col, label=m.replace("_", " "))
    ax2.set_xlabel("clip index along sequence 08 (chronological)")
    ax2.set_ylabel("occupancy IoU")
    ax2.set_title("Per-clip IoU along the validation sequence")
    ax2.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    p2 = os.path.join(fig_dir, "per_clip_iou_val.png")
    fig2.savefig(p2, dpi=150, facecolor="white"); plt.close(fig2)
    print("wrote", os.path.relpath(p2, REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
