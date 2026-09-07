#!/usr/bin/env python
"""Gate 8C-1 DIAGNOSTIC: what does MoGe-2's metric scale actually buy?

LingBot-Map reconstructs up to one unknown global factor. ``ScaleState`` fixes that factor
once, from the first five frames, as the median of ``log(D_moge) - log(D_lingbot)``, and the
same scalar multiplies depth and pose translation. This script asks what happens without it,
under three conditions:

  ``raw``       s = 1 -- LingBot's canonical units used directly as metres. This is
                "LingBot-Map alone", and it is not a small perturbation: the canonical unit
                is ~25x smaller than a metre, so the whole scene collapses toward the origin.
  ``constant``  s = one global constant for every scene (the median over the split). The
                steelman for dropping MoGe: if the canonical scale were a fixed property of
                the checkpoint, a hard-coded number would do and MoGe would be redundant.
  ``moge``      the deployed per-scene estimate.

Both the mapper's own occupancy and the completion on top of it are scored, because the
scale enters the geometry, not the network -- and the network was trained under ``moge``, so
its numbers under the other two are out of distribution and are labelled as such.

    python tools/gate8c1/scale_ablation.py --dataset occ3d --n 30 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import occ_to_eval_grid, to_eval_grid                        # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402
from tools.gate8c1.figure_3d_gallery import anchors                              # noqa: E402


def moge_scale(ds, name):
    with np.load(S.scale_path(ds, name)) as z:
        return float(np.exp(np.median(z["log_s"][:5])))


def run_one(ds, seg, i, scale, dev, net, tau, into, C, MAP, EVAL, edims):
    """One anchor at a chosen scale. Everything else is the frozen causal path."""
    feed = CachedFeed(seg, dev)
    f = seg.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None:
        return None
    m, _ = window_map(feed, seg, i, PAST5_OFFSETS, scale, into, C, dev)
    P = feed.pose[i].copy()
    P[:3, 3] *= scale
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    fin, _ = net.raw(q, MAP)
    s_fin = to_eval_grid(fin, q["logodds"], ds)[0]
    occ_native = q["occupied"] & (q["sem_w"] > 0)
    mp = occ_to_eval_grid(occ_native, ds).cpu().numpy().reshape(edims)
    comp = (s_fin >= tau).cpu().numpy().reshape(edims)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims)
    val = keep.reshape(edims)
    del m
    torch.cuda.empty_cache()
    return {"map": mp & val, "completion": comp & val, "gt": gt, "valid": val}


def counts(pred, gt, valid):
    p, g = pred & valid, gt & valid
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="occ3d", choices=["occ3d", "semantickitti"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
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

    alls = []
    for seg in S.segments(ds, REPO_ROOT):
        if os.path.exists(S.scale_path(ds, seg.name)):
            alls.append(moge_scale(ds, seg.name))
    alls = np.array(alls)
    CONST = float(np.median(alls))
    print(f"MoGe scale over {len(alls)} segments: min {alls.min():.2f}  median {CONST:.2f}  "
          f"max {alls.max():.2f}  (spread {alls.max()/alls.min():.1f}x)", flush=True)

    picks = anchors(ds, a.n)
    MODES = {"raw (s=1, LingBot alone)": None,
             f"constant (s={CONST:.2f} for every scene)": CONST,
             "moge (per scene, deployed)": "moge"}
    agg = {k: {"map": [0, 0, 0], "completion": [0, 0, 0]} for k in MODES}
    n = 0
    for seg, i in picks:
        s_moge = moge_scale(ds, seg.name)
        ok = True
        got = {}
        for k, v in MODES.items():
            s = 1.0 if v is None else (s_moge if v == "moge" else float(v))
            r = run_one(ds, seg, i, s, dev, net, tau, into, C, MAP, EVAL, edims)
            if r is None:
                ok = False
                break
            got[k] = r
        if not ok:
            continue
        n += 1
        for k, r in got.items():
            for w in ("map", "completion"):
                for j, x in enumerate(counts(r[w], r["gt"], r["valid"])):
                    agg[k][w][j] += x
        if n % 10 == 0:
            print(f"  {n} anchors", flush=True)

    out = {"dataset": ds, "n_anchors": n, "seed": a.seed, "threshold": tau,
           "moge_scale_stats": {"n_segments": int(len(alls)), "min": float(alls.min()),
                                "median": CONST, "max": float(alls.max()),
                                "spread": float(alls.max() / alls.min())},
           "NOTE": ("the completion network was trained under per-scene MoGe scale, so its "
                    "rows under 'raw' and 'constant' are out of distribution; the 'map' "
                    "rows are the clean measure of what the scale does to the geometry"),
           "rows": {}}
    print(f"\n{'condition':40s} {'map IoU':>8} {'map P':>7} {'map R':>7} | "
          f"{'compl IoU':>9} {'compl P':>8} {'compl R':>8}   (n={n})")
    for k in MODES:
        row = {}
        for w in ("map", "completion"):
            tp, fp, fn = agg[k][w]
            row[w] = {"iou": tp / max(tp + fp + fn, 1),
                      "precision": tp / max(tp + fp, 1),
                      "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
        out["rows"][k] = row
        print(f"{k:40s} {100*row['map']['iou']:8.2f} {100*row['map']['precision']:7.2f} "
              f"{100*row['map']['recall']:7.2f} | {100*row['completion']['iou']:9.2f} "
              f"{100*row['completion']['precision']:8.2f} "
              f"{100*row['completion']['recall']:8.2f}")
    with open(os.path.join(ART, f"scale_ablation_{ds}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote artifacts/gate8c1/scale_ablation_{ds}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
