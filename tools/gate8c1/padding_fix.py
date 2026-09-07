#!/usr/bin/env python
"""Gate 8C-1: does correcting the convolution boundary remove the ceiling slab?

The completion input carries ``observed`` and ``1 - observed`` in channels 2 and 3, so those
two always sum to 1 in real data. ``nn.Conv3d(padding=1)`` pads with zeros, which at the
array boundary manufactures ``observed = 0`` and ``unobserved = 0`` together -- a state the
network never saw. With only 32 voxels of depth, the response to that impossible input
reaches a large share of the grid's ceiling.

This measures the fix: pad the volume in z with the CORRECT encoding of "outside the grid"
(unobserved), run, crop back. No retraining, no threshold change, nothing tuned on the
target dataset -- the padded margin is a statement about the grid, not about the scene.

    python tools/gate8c1/padding_fix.py --dataset semantickitti --n 24 --device cuda:2
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
from gates.gate8a.regions import to_eval_grid                                          # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402
from tools.gate8c1.figure_3d_gallery import anchors                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="semantickitti",
                    choices=["semantickitti", "occ3d"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pads", type=int, nargs="+", default=[0, 4, 8, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    ds = a.dataset
    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    zc = (np.arange(edims[2]) + .5) * vs + z0

    agg = {p: [0, 0, 0] for p in a.pads}
    prof = {p: np.zeros(edims[2]) for p in a.pads}
    gt_prof = np.zeros(edims[2]); nz = np.zeros(edims[2]); n = 0
    for seg, i in anchors(ds, a.n):
        f = seg.frames[i]
        target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
        if target is None:
            continue
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(ds, seg.name)) as z:
            sc = float(np.exp(np.median(z["log_s"][:5])))
        m, _ = window_map(feed, seg, i, PAST5_OFFSETS, sc, into, C, dev)
        P = feed.pose[i].copy(); P[:3, 3] *= sc
        Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
        q = m.query(MAP, Tgw)
        q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        gt = ((target != EVAL.empty_class) & keep).reshape(edims)
        val = keep.reshape(edims)
        gt_prof += (gt & val).sum(axis=(0, 1)); nz += val.sum(axis=(0, 1))
        for p in a.pads:
            fin, _ = net.raw(q, MAP, pad_z=p)
            pr = (to_eval_grid(fin, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
            pv, g = pr & val, gt & val
            agg[p][0] += int((pv & g).sum()); agg[p][1] += int((pv & ~g).sum())
            agg[p][2] += int((~pv & g).sum())
            prof[p] += pv.sum(axis=(0, 1))
        del m; torch.cuda.empty_cache(); n += 1
        if n % 8 == 0:
            print(f"  {n} anchors", flush=True)

    out = {"dataset": ds, "n_anchors": n, "threshold": tau, "pads": {}}
    print(f"\n{'pad_z':>6} {'SC IoU':>8} {'precision':>10} {'recall':>8}   (n={n})")
    for p in a.pads:
        tp, fp, fn = agg[p]
        r = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
             "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
        out["pads"][str(p)] = r
        print(f"{p:6d} {100*r['iou']:8.2f} {100*r['precision']:10.2f} "
              f"{100*r['recall']:8.2f}")
    print(f"\npredicted-occupied share by height (%), ground truth for reference")
    hdr = "".join(f"{('pad ' + str(p)):>9}" for p in a.pads)
    print(f"{'z (m)':>7} {'GT':>7}{hdr}")
    for k in range(edims[2]):
        row = "".join(f"{100*prof[p][k]/max(nz[k],1):9.2f}" for p in a.pads)
        print(f"{zc[k]:+7.1f} {100*gt_prof[k]/max(nz[k],1):7.2f}{row}")
    out["profile"] = {"z_centres_m": zc.tolist(),
                      "gt_occupied_pct": (100 * gt_prof / np.maximum(nz, 1)).tolist(),
                      "pred_occupied_pct": {str(p): (100 * prof[p] / np.maximum(nz, 1)).tolist()
                                            for p in a.pads}}
    with open(os.path.join(ART, f"padding_fix_{ds}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote artifacts/gate8c1/padding_fix_{ds}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
