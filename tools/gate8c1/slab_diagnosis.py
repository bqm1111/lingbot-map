#!/usr/bin/env python
"""Gate 8C-1 DIAGNOSTIC: what does the mid-air slab cost on SemanticKITTI?

**This is not a result and must never be quoted as one.** Choosing a height cut by looking
at SemanticKITTI is exactly the target-domain tuning the gate forbids; the frozen number
stays whatever the frozen threshold produces. What this measures is how much of the failure
is attributable to one artefact -- which decides whether the next gate chases a single bug
or redesigns the module.

    python tools/gate8c1/slab_diagnosis.py --n 24 --device cuda:2
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, vocab                                            # noqa: E402
from gates.gate8 import vocab as V8                                                    # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from tools.gate8c1.eval_target import MANIFEST                                   # noqa: E402
from tools.gate8c1.figure_3d_gallery import anchors, predict                     # noqa: E402


def iou(pred, gt, valid):
    p, g = pred & valid, gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="semantickitti")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    ds = a.dataset
    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    C = len(vocab.load(ds))
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    zc = (np.arange(edims[2]) + 0.5) * vs + z0
    CUTS = [None, 4.0, 3.5, 3.0, 2.5, 2.0]
    agg = {str(c): [0, 0, 0] for c in CUTS}
    n = 0
    for seg, i in anchors(ds, a.n):
        r = predict(ds, seg, i, a.seed, dev, net, tau, into, C, MAP, EVAL, edims)
        if r is None:
            continue
        n += 1
        for c in CUTS:
            p = r["pred"] if c is None else (r["pred"] & (zc < c)[None, None, :])
            for j, x in enumerate(iou(p, r["gt"], r["valid"])):
                agg[str(c)][j] += x
        if n % 8 == 0:
            print(f"  {n} frames", flush=True)
    out = {"n_frames": n, "seed": a.seed, "threshold": tau,
           "WARNING": ("post-hoc diagnosis only -- the cut height was chosen by looking at "
                       "the target dataset, so no row but 'None' is a legitimate result"),
           "rows": {}}
    print(f"\n{'cut':>10}  {'SC IoU':>7} {'prec':>7} {'recall':>7}   (n={n})")
    for c in CUTS:
        tp, fp, fn = agg[str(c)]
        m = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
             "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
        out["rows"]["frozen" if c is None else f"drop above {c} m"] = m
        lab = "none (frozen)" if c is None else f"> {c} m"
        print(f"{lab:>10}  {100*m['iou']:7.2f} {100*m['precision']:7.2f} "
              f"{100*m['recall']:7.2f}")
    with open(os.path.join(ART, f"slab_diagnosis_{ds}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
