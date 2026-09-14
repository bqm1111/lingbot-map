#!/usr/bin/env python
"""A fast dial you can turn: run the pipeline on a few anchors and watch numbers move.

The point is the feedback loop, not the score. On ~20 anchors a run takes well under a
minute, so you can change one thing, rerun, and see what it did. Every knob below
corresponds to a specific line in a specific file -- the flag exists so you can feel what
that line controls before you edit it.

    python tools/skyfix/probe.py --limit 20                  # the model as shipped
    python tools/skyfix/probe.py --limit 20 --no-net         # what the map alone gives you
    python tools/skyfix/probe.py --limit 20 --lock 0.0       # freeze the network out
    python tools/skyfix/probe.py --limit 20 --frames 1       # starve the map of history
    python tools/skyfix/probe.py --dataset occ3d --limit 20  # the other benchmark

Nothing here writes anything. Read-only.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from core.run._common import REPO_ROOT, default_device                                # noqa: E402
from core.datasets import grids as G6G, vocab, sources as S, union_vocab as V8        # noqa: E402
from core.evaluation import official_targets as G6T                                   # noqa: E402
from core.mapping.mapper import IncrementalMapper                                     # noqa: E402
from core.mapping.feed import CachedFeed                                              # noqa: E402
from core.evaluation.regions import to_eval_grid, occ_to_eval_grid                    # noqa: E402
import core.model.net as NET                                                          # noqa: E402


def counts(pred, gt):
    return (int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()))


def row(name, tp, fp, fn):
    iou = tp / max(tp + fp + fn, 1)
    return (f"  {name:14s} {iou*100:6.2f}  {tp/max(tp+fp,1)*100:6.2f}  "
            f"{tp/max(tp+fn,1)*100:6.2f}   {(tp+fp)/max(tp+fn,1):5.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="semantickitti", choices=["semantickitti", "occ3d"])
    ap.add_argument("--limit", type=int, default=20, help="anchors to score (speed dial)")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--manifest", default=os.environ.get(
        "G8C1_MANIFEST", os.path.join(ROOT, "artifacts/gate8c1/frozen_manifest_sky.json")))
    ap.add_argument("--tau", type=float, default=None,
                    help="occupancy threshold; default from the manifest")
    ap.add_argument("--frames", type=int, default=5,
                    help="past frames the map integrates (core/mapping/mapper.py)")
    ap.add_argument("--lock", type=float, default=None,
                    help="LOCK_LOGODDS: the net may only edit voxels with |base| below this "
                         "(core/model/net.py::apply_residual). 0 freezes the net out")
    ap.add_argument("--pad-z", type=int, default=0,
                    help="pad the grid in z with 'unobserved' before the convolutions")
    ap.add_argument("--no-net", action="store_true", help="score the raw map, no completion")
    ap.add_argument("--min-frame", type=int, default=0,
                    help="skip anchors before this stream index; use when comparing "
                         "different --frames values so every setting scores the same anchors")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    DS = a.dataset
    fz = json.load(open(a.manifest))["seeds"][str(a.seed)]
    tau = a.tau if a.tau is not None else float(fz["occupancy_threshold"])
    if a.lock is not None:
        NET.LOCK_LOGODDS = a.lock                 # apply_residual reads it at call time
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    v = vocab.load(DS); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[DS], G6G.EVAL_GRID[DS]
    into = V8.into_matrix(DS)
    net = NET.load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    offs = list(range(-(a.frames - 1), 1))
    # first eval-grid z index whose voxel CENTRE sits above +2.0 m
    zc = EVAL.origin[2] + (np.arange(int(EVAL.dims[2])) + 0.5) * EVAL.voxel_size
    z2 = int(np.searchsorted(zc, 2.0, side="right"))

    print(f"probe  {DS}  manifest={os.path.basename(a.manifest)} seed={a.seed} "
          f"tau={tau:g} frames={a.frames} lock={NET.LOCK_LOGODDS:g} pad_z={a.pad_z}"
          f"{' NO-NET' if a.no_net else ''}", flush=True)

    agg = {"map": [0, 0, 0], "completion": [0, 0, 0]}
    edited = tot_valid = hi_pred = hi_valid = hi_gt = n = 0
    t0 = time.time()
    for seg in S.segments(DS, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(DS, seg.name)) as zz:
            scale = float(np.exp(np.median(zz["log_s"][:5])))
        for i in sorted(seg.anchors):
            if i + min(offs) < 0 or i < a.min_frame:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(DS, f.gt_ref, REPO_ROOT)
            if target is None:
                continue
            m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
            m.scale_state.force(scale)
            for o in offs:
                if 0 <= i + o < len(seg):
                    m.step(feed.frame(i + o))
            P = feed.pose[i].copy(); P[:3, 3] *= scale
            q = m.query(MAP, P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64)))
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            base = q["logodds"]
            fin = base if a.no_net else net.raw(q, MAP, pad_z=a.pad_z)[0]

            valid = torch.from_numpy(keep.reshape(-1)).to(dev)
            gt = torch.from_numpy(((target != EVAL.empty_class) & keep).reshape(-1)).to(dev)
            # the map's own occupancy decision -- the `mapper_native` baseline, not a
            # threshold on the log-odds (unobserved voxels sit at 0, which is above tau)
            preds = {"map": occ_to_eval_grid(q["occupied"] & (q["sem_w"] > 0), DS) & valid,
                     "completion": (to_eval_grid(fin, base, DS)[0] >= tau) & valid}
            for name, p_ in preds.items():
                for j, x in enumerate(counts(p_, gt)):
                    agg[name][j] += x
            # reduce the edit mask to the eval grid too, so the percentage means the
            # same thing on Occ3D (400x400x32 predicted -> 200x200x16 scored) as on
            # SemanticKITTI, where the two grids coincide
            ed = occ_to_eval_grid((fin - base).abs() > 1e-6, DS)
            edited += int((ed & valid).sum())
            tot_valid += int(valid.sum())
            d = tuple(int(x) for x in EVAL.dims)
            hi_pred += int(preds["completion"].reshape(d)[:, :, z2:].sum())
            hi_valid += int(valid.reshape(d)[:, :, z2:].sum())
            hi_gt += int(gt.reshape(d)[:, :, z2:].sum())
            n += 1
            if n >= a.limit:
                break
        if n >= a.limit:
            break

    print(f"\n{n} anchors in {time.time()-t0:.0f}s\n")
    print("                    IoU       P       R   pred/GT")
    for k, nm in (("map", "map only"), ("completion", "completion")):
        print(row(nm, *agg[k]))
    gtp = agg["completion"][0] + agg["completion"][2]        # TP + FN = GT positives
    print(f"\n  network edited {edited/max(tot_valid,1)*100:.1f}% of scored voxels")
    print(f"  above +2.0 m (eval z>={z2}):")
    print(f"     phi  = {hi_pred/max(hi_valid,1)*100:6.2f}%  of scored high voxels predicted occupied")
    print(f"     gt   = {hi_gt/max(hi_valid,1)*100:6.2f}%  of them are actually occupied")
    print(f"     beta = {(hi_valid-hi_gt)/max(gtp,1):6.2f}   scored-empty-high : all-GT-occupied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
