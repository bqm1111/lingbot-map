#!/usr/bin/env python
"""Assemble the three-arm + OccAny comparison from stored evaluation records.

Reads only what already exists on disk and keeps the two protocols apart, because they are not
interchangeable:

* **frozen evaluator** (163 SemanticKITTI / 1,182 Occ3D anchors, its own masks, 3 seeds) --
  the protocol the KITTI-360 (26.17 / 31.22) and Waymo (23.14 / 22.86) rows were measured on,
  and therefore the only one on which the ONCE arm can be compared with them;
* **canonical prediction protocol** (801 / 4,219 anchors, OccAny's own labels, placement and
  scorer, seed 0) -- the only one on which OccAny can be compared with anything.

A row from one may never be put in a column with a row from the other, and raw is never mixed
with pooled. This script therefore prints two tables and labels both.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/compare.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from tools.method_vs_data.once_arm import config as A                            # noqa: E402

#: frozen-evaluator rows, from artifacts/waymo_arm/waymo_vs_kitti360_report.md §2-§4
FROZEN_REF = {
    "semantickitti": {"kitti360": {"full": 26.17, "observed": 33.23, "completed": 24.74},
                      "waymo": {"full": 23.14, "observed": 28.26, "completed": 21.46}},
    "occ3d": {"kitti360": {"full": 31.22, "observed": 39.03, "completed": 27.50},
              "waymo": {"full": 22.86, "observed": 35.05, "completed": 15.86}},
}
#: canonical-protocol OccAny rows, verified against artifacts/occany_comparison/analysis.json
OCCANY = {
    "semantickitti": {"1frame_causal": {"raw": 20.32, "pooled": 24.06},
                      "5frame_forward": {"raw": 25.21, "pooled": 25.93}},
    "occ3d": {"5frame_forward": {"raw": 20.66, "pooled": 23.55}},
}
REGION = {"full": "full", "observed": "H_evidence", "completed": "no_H_evidence"}


def read(ds: str) -> dict | None:
    f = os.path.join(A.ARTIFACT_DIR, "zero_shot_eval", f"zero_shot_{ds}.json")
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    out = {"n_anchors": d["n_anchors"], "map_only": 100 * d["map_only"]["full"]["iou"],
           "checkpoint_sha256": d["checkpoint_sha256"], "regions": {}}
    for name, key in REGION.items():
        g = lambda m: [100 * d["per_seed"][s][key][m] for s in d["per_seed"]]     # noqa: E731
        i, p, r = g("iou"), g("precision"), g("recall")
        out["regions"][name] = {
            "iou_mean": float(np.mean(i)), "iou_sd": float(np.std(i, ddof=1)),
            "iou_per_seed": [float(x) for x in i],
            "precision": float(np.mean(p)), "recall": float(np.mean(r))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(A.ARTIFACT_DIR, "logs", "compare.json"))
    a = ap.parse_args()
    res = {"provenance": A.provenance(), "protocols": {
        "frozen_evaluator": "163 SK / 1,182 Occ3D anchors, own masks, 3 seeds, RAW",
        "canonical": "801 SK / 4,219 Occ3D anchors, OccAny labels+scorer, seed 0"},
        "frozen_reference": FROZEN_REF, "occany_canonical": OCCANY, "once": {}}

    for ds in ("semantickitti", "occ3d"):
        o = read(ds)
        if o is None:
            print(f"[compare] {ds}: no ONCE record yet")
            continue
        res["once"][ds] = o
        ref = FROZEN_REF[ds]
        print(f"\n=== {ds} -- FROZEN EVALUATOR, {o['n_anchors']} anchors, RAW, 3 seeds ===")
        print(f"{'arm':12s} {'full':>14s} {'observed':>10s} {'completed':>10s} "
              f"{'prec':>7s} {'recall':>7s}")
        print(f"{'KITTI-360':12s} {ref['kitti360']['full']:14.2f} "
              f"{ref['kitti360']['observed']:10.2f} {ref['kitti360']['completed']:10.2f} "
              f"{'--':>7s} {'--':>7s}")
        print(f"{'Waymo':12s} {ref['waymo']['full']:14.2f} "
              f"{ref['waymo']['observed']:10.2f} {ref['waymo']['completed']:10.2f} "
              f"{'--':>7s} {'--':>7s}")
        g = o["regions"]
        print(f"{'ONCE':12s} {g['full']['iou_mean']:8.2f}+-{g['full']['iou_sd']:.2f} "
              f"{g['observed']['iou_mean']:10.2f} {g['completed']['iou_mean']:10.2f} "
              f"{g['full']['precision']:7.2f} {g['full']['recall']:7.2f}")
        print(f"  ONCE per-seed full: "
              f"{' / '.join(f'{x:.2f}' for x in g['full']['iou_per_seed'])}")
        print(f"  map-only (must be identical across arms): {o['map_only']:.4f}")
        print(f"  ONCE - KITTI-360 = {g['full']['iou_mean'] - ref['kitti360']['full']:+.2f}   "
              f"ONCE - Waymo = {g['full']['iou_mean'] - ref['waymo']['full']:+.2f}")
        print(f"\n  OccAny, CANONICAL protocol (different anchors -- NOT a column of the "
              f"table above):")
        for k, v in OCCANY[ds].items():
            print(f"    {k:16s} raw {v['raw']:6.2f}   pooled {v['pooled']:6.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\n[compare] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
