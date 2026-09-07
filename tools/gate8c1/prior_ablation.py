#!/usr/bin/env python
"""Gate 8C-1 DIAGNOSTIC: how much of the score is a learned prior, not the observations?

The scale ablation turned this up: with s=1 the map scores 0.00 IoU on Occ3D, yet the
completion built on that useless map still scores 24.4. That is only explicable if the
network, handed an input that says "nothing has been observed", emits a generic scene it
learned during training -- and the residual lock permits exactly that, since the gate
``|base_logodds| < 2`` is open everywhere when nothing is observed.

This measures it cleanly. ``empty`` hands the network a map in which every cell is
unobserved: no log-odds, no counts, no semantics. Whatever it predicts then is pure prior,
with zero information from the camera. ``moge`` is the deployed system.

    python tools/gate8c1/prior_ablation.py --dataset occ3d --n 30 --device cuda:2
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import occ_to_eval_grid, to_eval_grid                        # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402
from tools.gate8c1.figure_3d_gallery import anchors                              # noqa: E402
from tools.gate8c1.scale_ablation import counts, moge_scale                      # noqa: E402


def blank(q):
    """Every cell unobserved: the network is told the camera saw nothing, anywhere."""
    z = {}
    for k, v in q.items():
        if k in ("first_time", "last_time"):
            z[k] = torch.full_like(v, -1)
        elif v.dtype == torch.bool:
            z[k] = torch.zeros_like(v)
        else:
            z[k] = torch.zeros_like(v)
    return z


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
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    C = len(vocab.load(ds))
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)

    agg = {k: [0, 0, 0] for k in ("empty (pure prior)", "moge (deployed)", "map only")}
    n = 0
    for seg, i in anchors(ds, a.n):
        feed = CachedFeed(seg, dev)
        f = seg.frames[i]
        target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
        if target is None:
            continue
        s = moge_scale(ds, seg.name)
        m, _ = window_map(feed, seg, i, PAST5_OFFSETS, s, into, C, dev)
        P = feed.pose[i].copy(); P[:3, 3] *= s
        Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
        q = m.query(MAP, Tgw)
        q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        gt = ((target != EVAL.empty_class) & keep).reshape(edims)
        val = keep.reshape(edims)
        occ_native = q["occupied"] & (q["sem_w"] > 0)
        preds = {"map only": occ_to_eval_grid(occ_native, ds).cpu().numpy().reshape(edims)}
        for name, qq in (("moge (deployed)", q), ("empty (pure prior)", blank(q))):
            fin, _ = net.raw(qq, MAP)
            s_fin = to_eval_grid(fin, qq["logodds"], ds)[0]
            preds[name] = (s_fin >= tau).cpu().numpy().reshape(edims)
        for k, p in preds.items():
            for j, x in enumerate(counts(p & val, gt, val)):
                agg[k][j] += x
        del m; torch.cuda.empty_cache()
        n += 1
        if n % 10 == 0:
            print(f"  {n} anchors", flush=True)

    out = {"dataset": ds, "n_anchors": n, "threshold": tau, "rows": {}}
    print(f"\n{'condition':22s} {'IoU':>7} {'prec':>7} {'recall':>7}   (n={n})")
    for k, (tp, fp, fn) in agg.items():
        r = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
             "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
        out["rows"][k] = r
        print(f"{k:22s} {100*r['iou']:7.2f} {100*r['precision']:7.2f} {100*r['recall']:7.2f}")
    d = out["rows"]
    out["information_gain_iou"] = (d["moge (deployed)"]["iou"]
                                   - d["empty (pure prior)"]["iou"])
    print(f"\nobservations are worth {100*out['information_gain_iou']:+.2f} IoU over a "
          f"blind prior")
    with open(os.path.join(ART, f"prior_ablation_{ds}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
