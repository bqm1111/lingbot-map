#!/usr/bin/env python
"""Phase 4 — does the Waymo geometry change semantic occupancy, and how?

Both arms are scored here by ONE function, so the two columns cannot differ by scorer. The
semantic pathway is frozen and identical in both: Trident-H evidence, the B0 nearest-evidence
rule, the same text readout. Only the CompletionUNet that produced ``occupied`` differs.

Populations, each stated with its denominator:

  full                 every valid voxel (label != 255). SC IoU and Track-O SSC mIoU.
  observed             valid AND ``pred.observed`` -- the causal hard map resolved it.
  completed            valid AND NOT ``pred.observed`` -- the network had to invent it.
  correct_occupied     valid AND predicted occupied AND GT occupied, PER ARM. Semantic
                       accuracy here answers "when the geometry is right, is the label
                       right?", but the voxel SET differs between arms, so a difference
                       mixes semantics with occupancy.
  shared_correct       the INTERSECTION of the two arms' ``correct_occupied`` sets. The
                       voxel set is identical by construction, so a difference here is
                       semantics alone. This is the only cell that isolates it.

No pooling is applied: the frozen protocol's own convention. Pooling changes the level of
every row, not the sign of the difference.

Uncertainty uses the preregistered statistics (``practical.protocol.Statistics``): a paired
block bootstrap, 2,000 resamples, seed 0, blocks being contiguous runs of 25 evaluated
anchors on SemanticKITTI and one block per scene on Occ3D. The resampled statistic is the
pooled metric of each arm on the resampled anchors, and the reported quantity is Waymo minus
KITTI-360. Pairing is by anchor: both arms are always resampled together, which is what makes
the interval an interval on the DIFFERENCE rather than on two independent levels.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/score_semantic.py --dataset semantickitti
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison import scoring as SC                                       # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm.cli.predict_semantic import PREDICTION_ROOT as WAYMO_PRED          # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.metrics import confusion, occany_ssc_miou           # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS            # noqa: E402

K360_PRED = ZC.PREDICTION_ROOT
POPS = ("full", "observed", "completed")
IGNORE = 255


def _path(root: str, dataset: str, sid: str) -> str:
    return os.path.join(root, dataset, f"{sid}.npz")


def _blocks(dataset: str, segs: list) -> list:
    """Preregistered resampling blocks: one per Occ3D scene, 25 contiguous anchors on SK."""
    if dataset == "occ3d":
        by: Dict[str, list] = {}
        for i, sg in enumerate(segs):
            by.setdefault(sg, []).append(i)
        return [v for _, v in sorted(by.items())]
    n = STATISTICS.semantickitti_block_samples
    return [list(range(a, min(a + n, len(segs)))) for a in range(0, len(segs), n)]


def score(dataset: str, method: str = "B0", limit: int | None = None) -> dict:
    spec = DATASET_SPECS[dataset]
    nc = spec.n_classes
    samples = ZS.evaluated_samples(dataset)
    acc: Dict[str, np.ndarray] = defaultdict(lambda: 0)
    per: Dict[str, list] = defaultdict(list)
    segs: list = []
    n = 0
    for s in samples[:limit]:
        pk, pw = _path(K360_PRED, dataset, s.sample_id), _path(WAYMO_PRED, dataset, s.sample_id)
        if not (os.path.exists(pk) and os.path.exists(pw)):
            continue
        a_k, a_w = CanonicalPrediction.load(pk), CanonicalPrediction.load(pw)
        label = SC.load_occany(dataset, s.occany_pkl, a_k.dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)
        arms = {"k360": a_k, "waymo": a_w}
        tp_sets = {}
        for arm, p in arms.items():
            occ = p.occupied
            tp_sets[arm] = valid & occ & gt_occ
            dense = np.where(occ, p.dense_labels(method, spec.empty_class), spec.empty_class)
            for pop, m in (("full", valid), ("observed", valid & p.observed),
                           ("completed", valid & ~p.observed)):
                acc[f"occ__{arm}__{pop}"] = acc[f"occ__{arm}__{pop}"] + np.array(
                    [int((occ & gt_occ & m).sum()), int((occ & ~gt_occ & m).sum()),
                     int((~occ & gt_occ & m).sum())], np.int64)
                c_ = confusion(label[m], dense[m], nc)
                acc[f"conf__{arm}__{pop}"] = acc[f"conf__{arm}__{pop}"] + c_
                per[f"conf__{arm}__{pop}"].append(c_)
                per[f"occ__{arm}__{pop}"].append(acc[f"occ__{arm}__{pop}"] * 0 + np.array(
                    [int((occ & gt_occ & m).sum()), int((occ & ~gt_occ & m).sum()),
                     int((~occ & gt_occ & m).sum())], np.int64))
            t = tp_sets[arm]
            acc[f"corr__{arm}"] = acc[f"corr__{arm}"] + np.array(
                [int((dense[t] == label[t]).sum()), int(t.sum())], np.int64)
        shared = tp_sets["k360"] & tp_sets["waymo"]
        for arm, p in arms.items():
            dense = np.where(p.occupied, p.dense_labels(method, spec.empty_class),
                             spec.empty_class)
            v_ = np.array([int((dense[shared] == label[shared]).sum()), int(shared.sum())],
                          np.int64)
            acc[f"shared__{arm}"] = acc[f"shared__{arm}"] + v_
            per[f"shared__{arm}"].append(v_)
        segs.append(s.segment)
        n += 1

    def iou(c):
        tp, fp, fn = (float(x) for x in c)
        return 100 * tp / max(tp + fp + fn, 1.0)

    out = {"dataset": dataset, "method": method, "n_samples": n,
           "k360_predictions": K360_PRED, "waymo_predictions": WAYMO_PRED,
           "populations": {}, "per_class": {}, "accuracy": {}}
    for pop in POPS:
        rec = {}
        for arm in ("k360", "waymo"):
            c = acc[f"conf__{arm}__{pop}"]
            m = occany_ssc_miou(c, nc, spec.empty_class, spec.occany_other_class,
                                spec.occany_ignore_other)
            rec[arm] = {"sc_iou": iou(acc[f"occ__{arm}__{pop}"]),
                        "ssc_miou": 100 * m["miou"],
                        "n_gt_voxels": int(c.sum())}
            if pop == "full":
                out["per_class"][arm] = {spec.class_names[int(i)]: 100 * v
                                         for i, v in m["per_class_iou"].items()
                                         if int(i) < nc}
        rec["delta_sc_iou"] = rec["waymo"]["sc_iou"] - rec["k360"]["sc_iou"]
        rec["delta_ssc_miou"] = rec["waymo"]["ssc_miou"] - rec["k360"]["ssc_miou"]
        out["populations"][pop] = rec
    for key, tag in (("corr", "correct_occupied_per_arm"), ("shared", "shared_correct_occupied")):
        rec = {}
        for arm in ("k360", "waymo"):
            hit, tot = (int(x) for x in acc[f"{key}__{arm}"])
            rec[arm] = {"accuracy": 100 * hit / max(tot, 1), "n_voxels": tot}
        rec["delta"] = rec["waymo"]["accuracy"] - rec["k360"]["accuracy"]
        rec["same_voxel_set"] = (key == "shared")
        out["accuracy"][tag] = rec
    out["per_class_delta"] = {k: out["per_class"]["waymo"][k] - v
                              for k, v in out["per_class"]["k360"].items()}

    # ---- paired block bootstrap on the deltas
    blocks = _blocks(dataset, segs)
    stack = {k: np.stack(v) for k, v in per.items()}
    rng = np.random.default_rng(STATISTICS.seed)
    picks = []
    for _ in range(STATISTICS.resamples):
        j = rng.integers(0, len(blocks), len(blocks))
        picks.append(np.concatenate([blocks[x] for x in j]))

    def ci(f):
        d = np.array([f(p) for p in picks], np.float64)
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        return {"lo": lo, "hi": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

    for pop in POPS:
        def d_sc(p, pop=pop):
            return (iou(stack[f"occ__waymo__{pop}"][p].sum(0))
                    - iou(stack[f"occ__k360__{pop}"][p].sum(0)))

        def d_mi(p, pop=pop):
            g = lambda arm: 100 * occany_ssc_miou(
                stack[f"conf__{arm}__{pop}"][p].sum(0), nc, spec.empty_class,
                spec.occany_other_class, spec.occany_ignore_other)["miou"]
            return g("waymo") - g("k360")

        out["populations"][pop]["ci_delta_sc_iou"] = ci(d_sc)
        out["populations"][pop]["ci_delta_ssc_miou"] = ci(d_mi)

    def d_sh(p):
        g = lambda arm: (lambda t: 100 * t[0] / max(t[1], 1))(stack[f"shared__{arm}"][p].sum(0))
        return g("waymo") - g("k360")

    out["accuracy"]["shared_correct_occupied"]["ci_delta"] = ci(d_sh)
    out["statistics"] = {"n_blocks": len(blocks), "resamples": STATISTICS.resamples,
                         "bootstrap_seed": STATISTICS.seed,
                         "block_rule": ("one per scene" if dataset == "occ3d"
                                        else f"{STATISTICS.semantickitti_block_samples} "
                                             "contiguous anchors"),
                         "paired": "both arms resampled on the same anchors"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    ap.add_argument("--method", default="B0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = {"version": A.VERSION, "results": {}}
    for ds in a.datasets.split(","):
        r = score(ds, a.method, a.limit)
        res["results"][ds] = r
        print(f"\n=== {ds}  ({r['n_samples']} samples, method {a.method})")
        print(f"  {'population':14s} {'SC IoU k360':>12s} {'waymo':>8s} {'delta':>8s} "
              f"{'mIoU k360':>10s} {'waymo':>8s} {'delta':>8s}")
        for pop, v in r["populations"].items():
            c1, c2 = v["ci_delta_sc_iou"], v["ci_delta_ssc_miou"]
            print(f"  {pop:14s} {v['k360']['sc_iou']:12.2f} {v['waymo']['sc_iou']:8.2f} "
                  f"{v['delta_sc_iou']:+8.2f} {v['k360']['ssc_miou']:10.2f} "
                  f"{v['waymo']['ssc_miou']:8.2f} {v['delta_ssc_miou']:+8.2f}")
            print(f"  {'':14s} {'':12s} {'':8s} [{c1['lo']:+.2f},{c1['hi']:+.2f}]"
                  f"{'*' if c1['significant_at_95'] else ' '}"
                  f"{'':10s} {'':8s} [{c2['lo']:+.2f},{c2['hi']:+.2f}]"
                  f"{'*' if c2['significant_at_95'] else ' '}")
        print(f"  blocks {r['statistics']['n_blocks']} "
              f"({r['statistics']['block_rule']}), {r['statistics']['resamples']} resamples, "
              f"seed {r['statistics']['bootstrap_seed']}; * = 95% CI excludes 0")
        for tag, v in r["accuracy"].items():
            c = v.get("ci_delta")
            ci_s = (f" CI [{c['lo']:+.2f},{c['hi']:+.2f}]"
                    f"{'*' if c['significant_at_95'] else ''}") if c else ""
            print(f"  {tag:28s} k360 {v['k360']['accuracy']:6.2f}%  "
                  f"waymo {v['waymo']['accuracy']:6.2f}%  delta {v['delta']:+6.2f}{ci_s}  "
                  f"n={v['waymo']['n_voxels']:,}  same_voxels={v['same_voxel_set']}")
        worst = sorted(r["per_class_delta"].items(), key=lambda kv: kv[1])[:5]
        best = sorted(r["per_class_delta"].items(), key=lambda kv: -kv[1])[:5]
        print(f"  largest per-class drops : {[(k, round(v, 2)) for k, v in worst]}")
        print(f"  largest per-class gains : {[(k, round(v, 2)) for k, v in best]}")
    d = os.path.join(A.ARTIFACT_DIR, "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"semantic_impact_{a.method}.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
