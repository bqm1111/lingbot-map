#!/usr/bin/env python
"""Gate 3.1 — bird's-eye visualisation of representative sequence-08 clips (clean protocol).

    python tools/voxel_gate_validation/figures.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config
from gates.voxel_gate.controls import control
from gates.voxel_gate.models import VoxelCorrector3D, apply_region
from gates.voxel_gate_validation.data import sample, scores, select_clips

TP, FP, FN = "#2e7d32", "#c62828", "#b0bec5"


def bev(pred, gt, keep):
    p, t = (pred & keep).cpu().numpy(), (gt & keep).cpu().numpy()
    tp, fp, fn = (p & t).any(2), (p & ~t).any(2), (~p & t).any(2)
    img = np.ones((*tp.shape, 3), np.float32)
    for m, c in ((fn, FN), (fp, FP), (tp, TP)):
        img[m] = np.array(matplotlib.colors.to_rgb(c))
    return np.transpose(img, (1, 0, 2))[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate_validation/clean_infill.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    radius = int(cfg.region.radius)
    ref = int(cfg.experiment.seed)

    sel = json.load(open(os.path.join(art, "control_selection.json")))
    kw = lambda c: {k: c[k] for k in ("kind", "radius", "min_neighbors") if k in c}
    v1_kw, v2_kw = kw(sel["v1_clean"]), kw(sel["v2_clean"])

    rows = list(csv.DictReader(open(os.path.join(art, "eval", "per_clip.csv"))))
    iou = {}
    for r in rows:
        iou.setdefault(r["config"], {})[r["clip_id"]] = float(r["iou"])
    key = f"V3_s{ref}"
    d = {c: iou[key][c] - iou["C3"][c] for c in iou["C3"]}
    order = sorted(d, key=d.get)
    worst = order[0]
    picks = [(order[-1], "most improved"), (order[len(order) // 2], "median clip"),
             (worst, "most degraded" if d[worst] < 0 else "least improved")]

    def load(run):
        ck = torch.load(os.path.join(art, "runs", run, "best.pt"), map_location="cpu",
                        weights_only=False)
        m = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
        m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
        return m, ck
    full, ck_full = load(f"full_s{ref}")
    occm, ck_occ = load("occ_only_s0")
    c0m, ck_c0 = load("c0_corrector_s0")

    cols = ["C3  frozen input", f"V1_clean  {sel['v1_clean']['name']}",
            f"V2_clean  {sel['v2_clean']['name']}", f"V3  learned (seed {ref})",
            "A_occ_only", "VC  visible set"]
    fig, axes = plt.subplots(3, 6, figsize=(23.0, 13.2))
    for i, (cid, label) in enumerate(picks):
        s3 = sample(cfg, "c3", cid, radius, ck_full["norm"], dev)
        so = sample(cfg, "c3", cid, radius, ck_occ["norm"], dev, occ_only=True)
        occ, R, keep, gt, vc = (s3["occupied"], s3["R_infer"], s3["keep"], s3["gt"], s3["vc"])
        with torch.no_grad():
            pf = torch.sigmoid(full(s3["x"][None])[0, 0])
            po = torch.sigmoid(occm(so["x"][None])[0, 0])
        tau = float(ck_full["threshold"])
        P = [occ, control(occ, R, **v1_kw), control(occ, R, **v2_kw),
             apply_region(pf >= tau, occ, R), apply_region(po >= tau, occ, R), vc]
        for j, (pred, name) in enumerate(zip(P, cols)):
            ax = axes[i, j]
            ax.imshow(bev(pred, gt, keep), interpolation="nearest", aspect="equal")
            sc = scores(pred, gt, keep)
            ax.set_title(f"{name}\nIoU {sc['iou']:.3f}   P {sc['precision']:.2f}   "
                         f"R {sc['recall']:.2f}   {sc['n_pred_occupied']:,} occ",
                         fontsize=10.5, pad=7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd8dc")
        axes[i, 0].set_ylabel(f"{label}\n{cid}\nΔIoU {d[cid]:+.3f}", fontsize=11, labelpad=12)

    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 3.1 — clean bounded local occupancy correction and infill, "
                         "SemanticKITTI sequence 08", ha="center", fontsize=15.5,
             weight="bold")
    fig.text(0.5, 0.958, "bird's-eye view: 256x256x32 grid at 0.2 m collapsed along z "
                         "(51.2 m x 51.2 m, sensor at the left edge). Inference region is a "
                         "pure radius-3 dilation of C3 occupancy — no valid mask.",
             ha="center", fontsize=10.5, color="#546e7a")
    fig.subplots_adjust(top=0.905, bottom=0.055, left=0.048, right=0.99,
                        hspace=0.19, wspace=0.06)
    out = os.path.join(art, "eval", "clips_clean.png")
    fig.savefig(out, dpi=125)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}  ({[c for c, _ in picks]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
