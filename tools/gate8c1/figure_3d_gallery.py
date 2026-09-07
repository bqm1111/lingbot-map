#!/usr/bin/env python
"""Gate 8C-1: 3D gallery of the completion on a single benchmark, several input frames.

This is the counterpart to ``figure_3d_compare.py``. That script needs the released OccAny
checkpoint's saved voxels, which exist only for Occ3D-nuScenes; here nothing external is
needed, so it runs on **SemanticKITTI** (and on Occ3D too, if a single-method view is
wanted). What it shows instead of a rival method is the *contribution of the trained
module*: the causal map is drawn next to the completion built on top of it.

Per input frame, six panels at a fixed viewpoint plus one from the side:

    camera at t | ground truth | causal map (input) | recovered | + invented | side view

The side view earns its place because the SemanticKITTI failure *was* a height failure -- a
slab of predicted occupancy in mid-air, invisible from above. That slab is fixed (see
``artifacts/gate8c1/ceiling_fix.md``), and the elevation is kept precisely so the fix stays
checkable: a figure of this system that omits an elevation cannot show whether it holds.

    python tools/gate8c1/figure_3d_gallery.py --dataset semantickitti --n 6 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.patches import Patch                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import occ_to_eval_grid, to_eval_grid                        # noqa: E402
from gates.gate8c1 import render3d as R3                                               # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402
from tools.gate8c1.figure_3d_compare import (C_FP, C_GT, C_MISS, C_TP,           # noqa: E402
                                             GHOST, SOLID)

C_MAP = (0.22, 0.50, 0.72)                    # the mapper's own occupancy, before completion
NICE = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}

#: an oblique view for structure, and a near-level view for height -- the second is the one
#: that exposes predicted occupancy floating above the scene
VIEWS = {"semantickitti": {"oblique": dict(eye_offset=(-34.0, -26.0, 30.0), fov=46.0),
                           "side": dict(eye_offset=(-6.0, -62.0, 7.0), fov=32.0)},
         "occ3d": {"oblique": dict(eye_offset=(-26.0, -20.0, 26.0), fov=48.0),
                   "side": dict(eye_offset=(-4.0, -54.0, 6.0), fov=32.0)}}


def mesh(mask, colour, grid, shrink=SOLID):
    idx = np.argwhere(mask)
    if not len(idx):
        return None
    return R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)),
                        float(grid.voxel_size), grid.origin, shrink=shrink)


def anchors(ds, n):
    """Deterministic spread over the benchmark, never two frames from the same clip."""
    segs = S.segments(ds, REPO_ROOT)
    cand, seen = [], set()
    for seg in segs:
        for i in sorted(seg.anchors):
            if i + min(PAST5_OFFSETS) < 0:
                continue
            key = (seg.name, str(seg.frames[i].gt_ref.get("clip_id", i)))
            if key in seen:
                continue
            seen.add(key)
            cand.append((seg, i))
    if not cand:
        return []
    return [cand[int(round(q * (len(cand) - 1)))] for q in np.linspace(0.05, 0.95, n)]


def predict(ds, seg, i, seed, dev, net, tau, into, C, MAP, EVAL, edims):
    """The frozen causal inference, identical to eval_target.py. Returns the eval grids."""
    feed = CachedFeed(seg, dev)
    with np.load(S.scale_path(ds, seg.name)) as z:
        scale = float(np.exp(np.median(z["log_s"][:5])))
    f = seg.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None:
        return None
    m, used = window_map(feed, seg, i, PAST5_OFFSETS, scale, into, C, dev)
    P = feed.pose[i].copy()
    P[:3, 3] *= scale
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    base = q["logodds"]
    fin, _ = net.raw(q, MAP)
    s_fin = to_eval_grid(fin, base, ds)[0]
    occ_native = q["occupied"] & (q["sem_w"] > 0)
    mp = occ_to_eval_grid(occ_native, ds).cpu().numpy().reshape(edims)
    comp = (s_fin >= tau).cpu().numpy().reshape(edims)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims)
    val = keep.reshape(edims)
    del m
    torch.cuda.empty_cache()
    return {"map": mp & val, "pred": comp & val, "gt": gt, "valid": val,
            "img": f.path, "clip": str(f.gt_ref.get("clip_id", i)),
            "seg": seg.name, "i": i, "used": used}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="semantickitti",
                    choices=["semantickitti", "occ3d"])
    ap.add_argument("--n", type=int, default=6, help="input frames to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = a.dataset
    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    C = len(vocab.load(ds))
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)

    picks = anchors(ds, a.n)
    print(f"{len(picks)} anchors: "
          f"{', '.join(f'{s.name}#{i}' for s, i in picks)}", flush=True)

    rows = []
    nz = edims[2]
    fp_h, gt_h = np.zeros(nz), np.zeros(nz)
    for seg, i in picks:
        r = predict(ds, seg, i, a.seed, dev, net, tau, into, C, MAP, EVAL, edims)
        if r is not None:
            rows.append(r)
            fp_h += np.bincount(np.argwhere(r["pred"] & ~r["gt"] & r["valid"])[:, 2],
                                minlength=nz)
            gt_h += np.bincount(np.argwhere(r["gt"] & r["valid"])[:, 2], minlength=nz)
            print(f"  ran {seg.name}#{i}", flush=True)
    if not rows:
        print("no anchors produced a target")
        return 1

    # the slab: how much of the hallucination sits above everything real?
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    zc = (np.arange(nz) + 0.5) * vs + z0
    hi = zc >= 2.0
    slab_pct = 100 * fp_h[hi].sum() / max(fp_h.sum(), 1)
    gt_hi_pct = 100 * gt_h[hi].sum() / max(gt_h.sum(), 1)
    top = int(np.argmax(fp_h))
    print(f"  {slab_pct:.0f}% of false positives sit above 2 m, where only "
          f"{gt_hi_pct:.0f}% of the ground truth lives; worst layer z={zc[top]:+.1f} m",
          flush=True)
    with open(os.path.join(ART, f"slab_profile_{ds}.json"), "w") as f:
        json.dump({"n_frames": len(rows), "z_centres_m": zc.tolist(),
                   "false_positive_voxels_by_height": fp_h.tolist(),
                   "gt_occupied_by_height": gt_h.tolist(),
                   "fp_above_2m_pct": float(slab_pct),
                   "gt_above_2m_pct": float(gt_hi_pct),
                   "worst_layer_m": float(zc[top]),
                   "note": ("a mid-air slab: the completion asserts occupancy in a "
                            "horizontal sheet near the top of the grid where almost no "
                            "real geometry exists")}, f, indent=2)

    V = VIEWS[ds]
    fig, axes = plt.subplots(len(rows), 6, figsize=(35, 4.9 * len(rows)), squeeze=False)
    for r, rec in enumerate(rows):
        gt, val, pr, mp = rec["gt"], rec["valid"], rec["pred"], rec["map"]
        tp = int((pr & gt & val).sum())
        fp = int((pr & ~gt & val).sum())
        fn = int((~pr & gt & val).sum())
        iou = tp / max(tp + fp + fn, 1)
        ax = axes[r]
        try:
            from PIL import Image
            with Image.open(rec["img"]) as im:
                ax[0].imshow(np.asarray(im))
        except Exception:
            ax[0].text(.5, .5, "image unavailable", ha="center")
        ax[0].set_axis_off()
        ax[0].set_title(f"1 · camera at t\n{rec['clip'][:38]}", fontsize=10.5)

        ghost = mesh(~pr & gt & val, C_MISS, EVAL, GHOST)
        scenes = [
            ([mesh(gt & val, C_GT, EVAL)],
             "2 · ground truth",
             "oblique"),
            ([mesh(gt & val, C_MISS, EVAL, GHOST), mesh(mp, C_MAP, EVAL)],
             "3 · causal map — the module's input", "oblique"),
            ([ghost, mesh(pr & gt & val, C_TP, EVAL)],
             "4 · completion — what it recovered", "oblique"),
            ([ghost, mesh(pr & gt & val, C_TP, EVAL), mesh(pr & ~gt & val, C_FP, EVAL)],
             "5 · + what it invented", "oblique"),
            ([ghost, mesh(pr & gt & val, C_TP, EVAL), mesh(pr & ~gt & val, C_FP, EVAL)],
             "6 · the same, seen from ground level", "side"),
        ]
        for c, (ms, ttl, view) in enumerate(scenes):
            axes[r][c + 1].imshow(R3.render(ms, size=(980, 700), **V[view]))
            axes[r][c + 1].set_axis_off()
            axes[r][c + 1].set_title(ttl, fontsize=10.5)
        print(f"  rendered row {r + 1}/{len(rows)}", flush=True)

    handles = [Patch(color=C_GT, label="ground truth — the shape to be reproduced"),
               Patch(color=C_MAP, label="observed by the mapper (input, before completion)"),
               Patch(color=C_TP, label="recovered correctly"),
               Patch(color=C_MISS, label="left out by the completion"),
               Patch(color=C_FP, label="predicted where the ground truth says free")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Visualization on {NICE[ds]}", fontsize=17)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.subplots_adjust(hspace=0.14)
    out = a.out or os.path.join(ART, f"fig_3d_gallery_{ds}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
