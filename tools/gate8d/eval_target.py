#!/usr/bin/env python
"""Gate 8D Phase 7: the one final target evaluation.

Refuses to start without ``artifacts/gate8d/frozen_manifest.json`` and accepts **no**
checkpoint or threshold override -- both come from the manifest and nowhere else. This is
the only Gate 8D entry point permitted to open a target benchmark.

    python tools/gate8d/eval_target.py --dataset semantickitti --protocol past5 --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8a.regions import occ_to_eval_grid, to_eval_grid                        # noqa: E402
from gates.gate8b import pooling as PL                                                 # noqa: E402
from gates.gate8d.net import load_checkpoint                                           # noqa: E402
from tools.gate8a.evaluate import dilate_with_semantics                          # noqa: E402
from tools.gate8c1.eval_target import (OCCANY_FWD_OFFSETS, PAST5_OFFSETS,        # noqa: E402
                                       window_map)

MANIFEST = os.path.join(ART, "frozen_manifest.json")
PROTOCOLS = {"past5": PAST5_OFFSETS, "occany_fwd": OCCANY_FWD_OFFSETS, "stream": None}


def require_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit("Gate 8D target evaluation refused: the frozen manifest does not "
                         "exist. Run tools/gate8d/freeze_manifest.py first.")
    return json.load(open(MANIFEST))


def counts(pred, gt, valid):
    p, g = pred & valid, gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "pred_over_gt": (tp + fp) / max(tp + fn, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--protocol", default="past5", choices=list(PROTOCOLS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    #: deliberately no --checkpoint and no --threshold: the manifest is the only source
    man = require_manifest()
    tau = float(man["occupancy_threshold"])
    ds = a.dataset
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    C = len(vocab.load(ds))
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(ds)
    comp = load_checkpoint(os.path.join(REPO_ROOT, man["checkpoint"]), dev)
    offs = PROTOCOLS[a.protocol]
    rows, agg = [], {}
    METHODS = ("completion", "completion_pooled", "mapper_native", "mapper_dilate")
    for m in METHODS:
        agg[m] = [0, 0, 0]
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    zc = (np.arange(edims[2]) + .5) * vs + z0
    prof = {m: np.zeros(edims[2]) for m in METHODS}
    gt_prof = np.zeros(edims[2]); nz = np.zeros(edims[2])
    t0 = time.time()
    for seg in S.segments(ds, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(ds, seg.name)) as z:
            scale = float(np.exp(np.median(z["log_s"][:5])))
        for i in sorted(seg.anchors):
            if a.limit and len(rows) >= a.limit:
                break
            use = offs if offs is not None else list(range(-i, 1))
            if i + min(use) < 0 or i + max(use) >= len(seg):
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
            if target is None:
                continue
            m_, _ = window_map(feed, seg, i, use, scale, into, C, dev)
            P_ = feed.pose[i].copy(); P_[:3, 3] *= scale
            Tgw = P_ @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
            q = m_.query(MAP, Tgw)
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            fin, _ = comp.raw(q, MAP)
            comp_pred = (to_eval_grid(fin, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
            occ_native = q["occupied"] & (q["sem_w"] > 0)
            mp = occ_to_eval_grid(occ_native, ds).cpu().numpy().reshape(edims)
            flat_n = occ_native.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
            probs_map = q["sem"] / q["sem_w"].clamp_min(1e-9).unsqueeze(1)
            dil_flat, _ = dilate_with_semantics(flat_n, probs_map[occ_native],
                                                tuple(MAP.dims), dev)
            dil = torch.zeros_like(occ_native)
            if len(dil_flat):
                dil[torch.as_tensor(dil_flat, device=dev)] = True
            mdil = occ_to_eval_grid(dil, ds).cpu().numpy().reshape(edims)
            gt = ((target != EVAL.empty_class) & keep).reshape(edims)
            val = keep.reshape(edims)
            preds = {"completion": comp_pred,
                     "completion_pooled": PL.geometry_dilation(
                         torch.from_numpy(comp_pred)).numpy(),
                     "mapper_native": mp, "mapper_dilate": mdil}
            row = {"seg": seg.name, "i": int(i)}
            for k, p in preds.items():
                c = counts(p, gt, val)
                for j, x in enumerate((c["tp"], c["fp"], c["fn"])):
                    agg[k][j] += x
                row[k] = c
                prof[k] += (p & val).sum(axis=(0, 1))
            gt_prof += (gt & val).sum(axis=(0, 1)); nz += val.sum(axis=(0, 1))
            rows.append(row)
            del m_; torch.cuda.empty_cache()
            if len(rows) % 25 == 0:
                print(f"  {len(rows)} scored ({(time.time()-t0)/len(rows):.2f}s each)",
                      flush=True)
    out = {"dataset": ds, "protocol": a.protocol,
           "causal": a.protocol != "occany_fwd",
           "n_anchors": len(rows), "threshold": tau,
           "manifest_checkpoint": man["checkpoint"],
           "manifest_sha256": man["checkpoint_sha256"],
           "pooled": {}, "z_centres_m": zc.tolist(),
           "gt_occupied_pct_by_z": (100 * gt_prof / np.maximum(nz, 1)).tolist(),
           "pred_occupied_pct_by_z": {k: (100 * v / np.maximum(nz, 1)).tolist()
                                      for k, v in prof.items()},
           "per_sample": rows[:200], "seconds": time.time() - t0}
    for k, (tp, fp, fn) in agg.items():
        out["pooled"][k] = {"tp": tp, "fp": fp, "fn": fn,
                            "iou": tp / max(tp + fp + fn, 1),
                            "precision": tp / max(tp + fp, 1),
                            "recall": tp / max(tp + fn, 1),
                            "pred_over_gt": (tp + fp) / max(tp + fn, 1)}
    hi = zc >= 2.0
    for k in METHODS:
        p = np.asarray(out["pred_occupied_pct_by_z"][k])
        out["pooled"][k]["high_altitude_pred_pct"] = float(p[hi].mean())
    write_json(os.path.join(ART, f"eval_{ds}_{a.protocol}.json"), out)
    print(f"\n{ds} / {a.protocol}  ({len(rows)} anchors, tau {tau:+.4f})")
    print(f"{'method':22s} {'IoU':>7} {'prec':>7} {'rec':>7} {'pred/GT':>8} {'>2m %':>7}")
    for k, m in out["pooled"].items():
        print(f"{k:22s} {100*m['iou']:7.2f} {100*m['precision']:7.2f} "
              f"{100*m['recall']:7.2f} {m['pred_over_gt']:8.2f} "
              f"{m['high_altitude_pred_pct']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
