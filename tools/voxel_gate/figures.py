#!/usr/bin/env python
"""Gate 3 — bird's-eye visualisation of representative sequence-08 clips.

Picks the most improved, a median (essentially unchanged) and the most degraded clip by
``IoU(V3) - IoU(V0)``, and draws each configuration's occupancy against the LiDAR target
in the frozen grid frame. Voxels are collapsed along z; a column is coloured by the
strongest evidence it contains (true positive > false positive > false negative).

    python tools/voxel_gate/figures.py --run voxel_cnn3d
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
from gates.voxel_gate.controls import control, enumerate_controls
from gates.voxel_gate.data import sample, scores, select_clips
from gates.voxel_gate.models import VoxelCorrector3D, apply_region

TP, FP, FN = "#2e7d32", "#c62828", "#b0bec5"


def bev(pred: torch.Tensor, gt: torch.Tensor, keep: torch.Tensor) -> np.ndarray:
    """Collapse z into an RGB bird's-eye image: TP green, FP red, FN grey."""
    p, t = (pred & keep).cpu().numpy(), (gt & keep).cpu().numpy()
    tp, fp, fn = (p & t).any(2), (p & ~t).any(2), (~p & t).any(2)
    img = np.ones((*tp.shape, 3), np.float32)
    for m, c in ((fn, FN), (fp, FP), (tp, TP)):                # last write wins
        img[m] = np.array(matplotlib.colors.to_rgb(c))
    return np.transpose(img, (1, 0, 2))[::-1]                  # x right, y up


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate/visible_correction.yaml")
    ap.add_argument("--run", default="voxel_cnn3d")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    sel = json.load(open(os.path.join(art, "region_selection.json")))
    summ = json.load(open(os.path.join(art, "eval", f"summary_{a.run}.json")))
    radius = int(sel["selected_radius"])
    cands = enumerate_controls(cfg)
    v1_kw = {k: v for k, v in summ["v1_control"].items() if k != "name"}
    v2_kw = {k: v for k, v in summ["v2_control"].items()
             if k not in ("name", "mean_occupied")}

    rows = list(csv.DictReader(open(os.path.join(art, "eval", f"per_clip_{a.run}.csv"))))
    iou = {}
    for r in rows:
        iou.setdefault(r["config"], {})[r["clip_id"]] = float(r["iou"])
    d = {c: iou["V3"][c] - iou["V0"][c] for c in iou["V0"]}
    order = sorted(d, key=d.get)
    worst = order[0]
    picks = [(order[-1], "most improved"),
             (order[len(order) // 2], "median clip"),
             (worst, "most degraded" if d[worst] < 0 else "least improved")]

    ck = torch.load(os.path.join(art, "runs", a.run, "best.pt"), map_location="cpu",
                    weights_only=False)
    model = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    tau = float(ck["threshold"])

    cols = ["V0  frozen C3", f"V1  {summ['v1_control']['name']}",
            f"V2  {summ['v2_control']['name']}", "V3  learned", "VC  visible ceiling"]
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 13.2))
    for i, (cid, label) in enumerate(picks):
        s = sample(cfg, cid, radius, ck["norm"], dev)
        occ, R, keep, gt, vc = (s["occupied"], s["region"], s["keep"], s["gt"], s["vc"])
        with torch.no_grad():
            p = torch.sigmoid(model(s["x"][None])[0, 0])
        P = [occ, control(occ, R, **v1_kw), control(occ, R, **v2_kw),
             apply_region(p >= tau, occ, R), vc]
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
        axes[i, 0].set_ylabel(f"{label}\n{cid}\nΔIoU {d[cid]:+.3f}",
                              fontsize=11, labelpad=12)

    fig.legend(handles=[Patch(facecolor=TP, label="true positive"),
                        Patch(facecolor=FP, label="false positive"),
                        Patch(facecolor=FN, label="missed (false negative)")],
               loc="lower center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.978, "Gate 3 — learned visible-voxel correction on SemanticKITTI "
                         "sequence 08", ha="center", fontsize=15.5, weight="bold")
    fig.text(0.5, 0.958, "bird's-eye view: the 256x256x32 grid at 0.2 m collapsed along z "
                         "(51.2 m x 51.2 m, sensor at the left edge). "
                         "V3 is the only configuration that can also delete voxels.",
             ha="center", fontsize=10.5, color="#546e7a")
    fig.subplots_adjust(top=0.905, bottom=0.055, left=0.055, right=0.99,
                        hspace=0.19, wspace=0.06)
    out = os.path.join(art, "eval", f"clips_{a.run}.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}  ({[c for c, _ in picks]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
