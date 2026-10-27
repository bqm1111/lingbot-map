#!/usr/bin/env python
"""Phase 5 — where the semantic completion error actually comes from.

Every GT-occupied valid voxel is placed in exactly one bucket, so the buckets sum to the
whole population and nothing is double counted:

  occ_miss        GT occupied, predicted empty. No label can be right: an occupancy failure.
  ev_right        predicted occupied, has semantic evidence, B0 label correct.
  ev_wrong        predicted occupied, has semantic evidence, B0 label wrong. B0's label at an
                  evidence voxel is that voxel's own teacher readout (its nearest evidence is
                  itself, distance 0), so this bucket IS the 2D teacher's error rate where the
                  teacher spoke directly.
  noev_right      predicted occupied, no evidence, B0 label correct -- extrapolation worked.
  noev_wrong      predicted occupied, no evidence, B0 label wrong -- extrapolation failed.

Separately, ``occ_false`` counts predicted-occupied voxels that are GT-free; they carry no GT
label, so they sit outside the partition but still cost IoU as FP.

The wrong buckets are then split by whether the predicted class shares OccAny's official
superclass with the GT class. An intra-superclass error (road vs sidewalk, car vs truck) is a
candidate for genuine semantic ambiguity; an inter-superclass error (building vs vegetation)
is not.

Causal attribution comes from oracles, not from bucket sizes. Each oracle substitutes GT
labels into one bucket only, leaves the predicted occupancy mask untouched, and the full
confusion is recomputed, so the reported gain is what perfecting that one stage would buy:

  oracle_teacher        GT labels on correctly-occupied evidence voxels (ev_right + ev_wrong)
  oracle_extrapolation  GT labels on correctly-occupied non-evidence voxels (noev_*)
  oracle_semantics      both of the above: perfect semantics on the occupancy we got right.
                        Its gap to 100 is the share of the metric occupancy alone owns.

Every oracle is confined to GT-occupied voxels on purpose. Substituting the GT label on a
false positive would set it to "free" and delete the false positive, so such an oracle would
be scoring occupancy repair as a semantic gain. False positives are reported separately and
never repaired here.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/bottleneck.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison import scoring as SC                                       # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm.cli.predict_semantic import PREDICTION_ROOT as WAYMO_PRED          # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.metrics import confusion, occany_ssc_miou           # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import (DATASET_SPECS, EVIDENCE_DISTANCE_EDGES_M,
                                                     EVIDENCE_DISTANCE_NAMES)           # noqa: E402

ROOTS = {"k360": ZC.PREDICTION_ROOT, "waymo": WAYMO_PRED}
BUCKETS = ("occ_miss", "ev_right", "ev_wrong", "noev_right", "noev_wrong")
IGNORE = 255


def diagnose(dataset: str, arm: str = "k360", method: str = "B0", limit: int | None = None) -> dict:
    spec = DATASET_SPECS[dataset]
    nc = spec.n_classes
    sup = np.asarray(spec.superclass_of, np.int64)
    root = ROOTS[arm]
    acc = defaultdict(lambda: 0)
    conf = {k: np.zeros((nc, nc), np.int64) for k in ("base", "oracle_teacher",
                                                      "oracle_extrapolation", "oracle_semantics")}
    n = 0
    for s in ZS.evaluated_samples(dataset)[:limit]:
        p = os.path.join(root, dataset, f"{s.sample_id}.npz")
        if not os.path.exists(p):
            continue
        a = CanonicalPrediction.load(p)
        label = SC.load_occany(dataset, s.occany_pkl, a.dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)
        occ = a.occupied
        pred = np.where(occ, a.dense_labels(method, spec.empty_class), spec.empty_class)
        ev = a.evidence
        right = pred == label

        tp = gt_occ & occ
        b = {"occ_miss": gt_occ & ~occ,
             "ev_right": gt_occ & occ & ev & right,
             "ev_wrong": gt_occ & occ & ev & ~right,
             "noev_right": gt_occ & occ & ~ev & right,
             "noev_wrong": gt_occ & occ & ~ev & ~right}
        assert int(sum(int(m.sum()) for m in b.values())) == int(gt_occ.sum()), s.sample_id
        for k, m in b.items():
            acc[k] = acc[k] + int(m.sum())
        acc["occ_false"] = acc["occ_false"] + int((occ & valid & ~gt_occ).sum())
        acc["valid"] = acc["valid"] + int(valid.sum())

        for k in ("ev_wrong", "noev_wrong"):
            m = b[k]
            same = sup[label[m]] == sup[pred[m]]
            acc[f"{k}__intra_superclass"] = acc[f"{k}__intra_superclass"] + int(same.sum())
            # how often the wrong prediction was "free" -- i.e. no identity reached the voxel
            acc[f"{k}__predicted_empty"] = acc[f"{k}__predicted_empty"] + \
                int((pred[m] == spec.empty_class).sum())

        # Error rate against distance to the nearest evidence voxel, on correctly occupied
        # voxels. Reported twice: over all of them, and over the non-evidence ones alone.
        # An evidence voxel sits at distance 0 and so lands in the 0-1 m bin, where it would
        # otherwise mix the teacher's own accuracy into the extrapolation curve.
        dist = np.full(a.n_voxels, np.inf, np.float32)
        dist[a.query_index] = a.evidence_distance
        db_all = np.digitize(dist[tp], np.asarray(EVIDENCE_DISTANCE_EDGES_M))
        r = right[tp]
        no_ev = ~ev[tp]
        for i in range(len(EVIDENCE_DISTANCE_NAMES)):
            sel = db_all == i
            acc[f"dist{i}__n"] = acc[f"dist{i}__n"] + int(sel.sum())
            acc[f"dist{i}__right"] = acc[f"dist{i}__right"] + int((sel & r).sum())
            s_ = sel & no_ev
            acc[f"distnoev{i}__n"] = acc[f"distnoev{i}__n"] + int(s_.sum())
            acc[f"distnoev{i}__right"] = acc[f"distnoev{i}__right"] + int((s_ & r).sum())

        conf["base"] += confusion(label[valid], pred[valid], nc)
        for name, m in (("oracle_teacher", tp & ev),
                        ("oracle_extrapolation", tp & ~ev),
                        ("oracle_semantics", tp)):
            o = np.where(m, label, pred)                 # GT labels only inside the bucket
            conf[name] += confusion(label[valid], o[valid], nc)
        n += 1

    miou = {k: 100 * occany_ssc_miou(c, nc, spec.empty_class, spec.occany_other_class,
                                     spec.occany_ignore_other)["miou"] for k, c in conf.items()}
    tot = sum(int(acc[k]) for k in BUCKETS)
    out = {"dataset": dataset, "arm": arm, "method": method, "n_samples": n,
           "predictions": root, "geometry_seed": ZC.GEOMETRY_SEED,
           "gt_occupied_voxels": tot, "valid_voxels": int(acc["valid"]),
           "false_positive_voxels": int(acc["occ_false"]),
           "buckets": {k: {"voxels": int(acc[k]), "share": int(acc[k]) / max(tot, 1)}
                       for k in BUCKETS},
           "error_character": {}, "distance": {}, "distance_no_evidence": {}, "miou": miou,
           "oracle_gain": {k: miou[k] - miou["base"] for k in miou if k != "base"}}
    for k in ("ev_wrong", "noev_wrong"):
        nk = int(acc[k])
        out["error_character"][k] = {
            "voxels": nk,
            "intra_superclass_share": int(acc[f"{k}__intra_superclass"]) / max(nk, 1),
            "predicted_free_share": int(acc[f"{k}__predicted_empty"]) / max(nk, 1)}
    for i, name in enumerate(EVIDENCE_DISTANCE_NAMES):
        for key, tag in (("dist", "distance"), ("distnoev", "distance_no_evidence")):
            nn = int(acc[f"{key}{i}__n"])
            out[tag][name] = {"voxels": nn,
                              "accuracy": 100 * int(acc[f"{key}{i}__right"]) / max(nn, 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    ap.add_argument("--arms", default="k360,waymo")
    ap.add_argument("--method", default="B0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = {"version": A.VERSION, "results": {}}
    for ds in a.datasets.split(","):
        for arm in a.arms.split(","):
            r = diagnose(ds, arm, a.method, a.limit)
            res["results"][f"{ds}/{arm}"] = r
            print(f"\n=== {ds} / {arm}  ({r['n_samples']} samples, {r['gt_occupied_voxels']:,} "
                  f"GT-occupied voxels, {r['false_positive_voxels']:,} false positives)")
            for k, v in r["buckets"].items():
                print(f"  {k:12s} {v['voxels']:>12,}  {v['share']*100:5.1f}% of GT occupied")
            for k, v in r["error_character"].items():
                print(f"  {k:12s} intra-superclass {v['intra_superclass_share']*100:5.1f}%, "
                      f"predicted free {v['predicted_free_share']*100:5.1f}%")
            print("  accuracy on correctly occupied voxels, by distance to nearest evidence")
            print(f"    {'bin':8s} {'all n':>12s} {'acc':>7s} {'no-ev n':>12s} {'acc':>7s}")
            for k, v in r["distance"].items():
                w = r["distance_no_evidence"][k]
                print(f"    {k:8s} {v['voxels']:>12,} {v['accuracy']:6.2f}% "
                      f"{w['voxels']:>12,} {w['accuracy']:6.2f}%")
            print(f"  mIoU base {r['miou']['base']:.2f}")
            for k, v in r["oracle_gain"].items():
                print(f"    {k:22s} {r['miou'][k]:6.2f}  (+{v:.2f})")
    d = os.path.join(A.ARTIFACT_DIR, "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"bottleneck_{a.method}.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
