#!/usr/bin/env python
"""Phase 3 — Waymo arm against the KITTI-360 arm, on identical anchors and scoring code.

Both arms are evaluated by the SAME frozen harness; only the CompletionUNet checkpoint
differs. This reads the two result sets and applies the preregistered statistics.

Populations and denominators, stated once so no table is ambiguous:

  full            every voxel inside the dataset's official evaluation mask for that anchor.
  H_evidence      the subset where the causal five-frame hard map resolved something
                  (``S | F`` reduced to the evaluation grid) -- the OBSERVED region.
  no_H_evidence   the complement inside the mask -- the region the network must COMPLETE.
                  This is not the same as "outside the camera frustum": a voxel can be
                  inside the frustum and still unresolved, and a voxel outside it can be
                  resolved from an earlier frame of the clip. The frustum split is a
                  different measurement and is not produced here.

Statistics: paired block bootstrap, 2,000 resamples, seed 0. Blocks are contiguous runs of
25 evaluated anchors on SemanticKITTI and one block per scene on Occ3D, per
``zero_shot_eval/practical/protocol.Statistics``. The statistic resampled is the seed-mean
pooled IoU of each arm, and the reported quantity is Waymo minus KITTI-360.

Per-anchor counts exist only for the ``full`` region, so ``full`` carries a bootstrap CI
while the two sub-regions carry the across-seed mean and spread only. That asymmetry is
reported rather than papered over.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/compare.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as BR                                  # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.zero_shot_eval.practical.protocol import Statistics                           # noqa: E402

ST = Statistics()
E0_DIR = os.path.join(BR.ARTIFACT_DIR, "zero_shot_eval")
E1_DIR = os.path.join(A.ARTIFACT_DIR, "zero_shot_eval")
REGIONS = ("full", "H_evidence", "no_H_evidence")
#: the prediction fixed in artifacts/waymo_arm/predeclaration.md, before any of this ran
PREDICTION = {"semantickitti": "decrease", "occ3d": "increase"}


def _load(d: str, dataset: str):
    with open(os.path.join(d, f"zero_shot_{dataset}.json")) as fh:
        summary = json.load(fh)
    with open(os.path.join(d, f"per_anchor_{dataset}.json")) as fh:
        rows = json.load(fh)
    return summary, {r["clip_id"]: r for r in rows}


def _iou(tp, fp, fn):
    return tp / max(tp + fp + fn, 1.0)


def _pooled(rows: List[dict], ids: List[str], seed: int):
    tp = sum(rows[i][f"seed{seed}"]["tp"] for i in ids)
    fp = sum(rows[i][f"seed{seed}"]["fp"] for i in ids)
    fn = sum(rows[i][f"seed{seed}"]["fn"] for i in ids)
    return _iou(tp, fp, fn), tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)


def _blocks(dataset: str, ids: List[str], rows: Dict[str, dict]) -> List[List[str]]:
    if dataset == "occ3d":
        by: Dict[str, List[str]] = {}
        for i in ids:
            by.setdefault(rows[i]["segment"], []).append(i)
        return [v for _, v in sorted(by.items())]
    n = ST.semantickitti_block_samples
    return [ids[a:a + n] for a in range(0, len(ids), n)]


def compare(dataset: str) -> dict:
    s0, r0 = _load(E0_DIR, dataset)
    s1, r1 = _load(E1_DIR, dataset)
    ids = sorted(set(r0) & set(r1))
    assert ids, f"{dataset}: no shared anchors"

    seeds = [int(s) for s in s0["seeds"]]
    per_seed = {"k360": {}, "waymo": {}}
    for arm, rows in (("k360", r0), ("waymo", r1)):
        for s in seeds:
            i, p, rc = _pooled(rows, ids, s)
            per_seed[arm][s] = {"iou": 100 * i, "precision": 100 * p, "recall": 100 * rc}

    def seed_mean(arm, key):
        return float(np.mean([per_seed[arm][s][key] for s in seeds]))

    def seed_sd(arm, key):
        return float(np.std([per_seed[arm][s][key] for s in seeds], ddof=1))

    blocks = _blocks(dataset, ids, r0)
    rng = np.random.default_rng(ST.seed)
    point = seed_mean("waymo", "iou") - seed_mean("k360", "iou")
    draws = np.empty(ST.resamples)
    for b in range(ST.resamples):
        pick = [x for j in rng.integers(0, len(blocks), len(blocks)) for x in blocks[j]]
        a = float(np.mean([100 * _pooled(r1, pick, s)[0] for s in seeds]))
        c = float(np.mean([100 * _pooled(r0, pick, s)[0] for s in seeds]))
        draws[b] = a - c
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))

    regions = {}
    for reg in REGIONS:
        rec = {}
        for arm, summ in (("k360", s0), ("waymo", s1)):
            v = [100 * summ["per_seed"][str(s)][reg]["iou"] for s in seeds]
            rec[arm] = {"seed_mean": float(np.mean(v)), "seed_sd": float(np.std(v, ddof=1)),
                        "per_seed": v}
        rec["delta_seed_mean"] = rec["waymo"]["seed_mean"] - rec["k360"]["seed_mean"]
        rec["ci"] = None if reg != "full" else {"lo": lo, "hi": hi}
        regions[reg] = rec

    direction = "increase" if point > 0 else ("decrease" if point < 0 else "none")
    significant = bool(lo > 0 or hi < 0)
    return {"dataset": dataset, "n_anchors_shared": len(ids), "n_blocks": len(blocks),
            "block_rule": ("one per scene" if dataset == "occ3d"
                           else f"{ST.semantickitti_block_samples} contiguous anchors"),
            "resamples": ST.resamples, "bootstrap_seed": ST.seed,
            "per_seed": {a: {str(s): v for s, v in d.items()} for a, d in per_seed.items()},
            "seed_mean": {a: {k: seed_mean(a, k) for k in ("iou", "precision", "recall")}
                          for a in ("k360", "waymo")},
            "seed_sd": {a: {k: seed_sd(a, k) for k in ("iou", "precision", "recall")}
                        for a in ("k360", "waymo")},
            "delta_iou": point, "ci_low": lo, "ci_high": hi,
            "significant_at_95": significant,
            "observed_direction": direction,
            "predicted_direction": PREDICTION[dataset],
            "prediction_supported": bool(direction == PREDICTION[dataset] and significant),
            "prediction_falsified": bool(direction != PREDICTION[dataset] and significant),
            "regions": regions,
            # the raw hard map scored the same way, i.e. the input the two arms share.
            # It must be IDENTICAL across arms: only the network downstream of it changed.
            "map_only_iou": {"k360": 100 * s0["map_only"]["full"]["iou"],
                             "waymo": 100 * s1["map_only"]["full"]["iou"]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    a = ap.parse_args()
    out = {"version": A.VERSION, "comparisons": {}}
    for ds in a.datasets.split(","):
        out["comparisons"][ds] = compare(ds)
    d = os.path.join(A.ARTIFACT_DIR, "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "waymo_vs_kitti360.json")
    json.dump(out, open(path, "w"), indent=1, default=float)

    for ds, c in out["comparisons"].items():
        print(f"\n=== {ds}  ({c['n_anchors_shared']} shared anchors, {c['n_blocks']} blocks)")
        print(f"{'arm':8s} {'SC IoU':>16s} {'precision':>12s} {'recall':>12s}")
        for arm in ("k360", "waymo"):
            m, s = c["seed_mean"][arm], c["seed_sd"][arm]
            print(f"{arm:8s} {m['iou']:9.2f} +-{s['iou']:4.2f} {m['precision']:12.2f} "
                  f"{m['recall']:12.2f}")
        print(f"  delta SC IoU  {c['delta_iou']:+.2f}  CI [{c['ci_low']:+.2f}, "
              f"{c['ci_high']:+.2f}]  significant={c['significant_at_95']}")
        print(f"  predicted {c['predicted_direction']}, observed {c['observed_direction']}"
              f"  -> supported={c['prediction_supported']} "
              f"falsified={c['prediction_falsified']}")
        print(f"  {'region':16s} {'k360':>10s} {'waymo':>10s} {'delta':>9s}")
        for reg, r in c["regions"].items():
            print(f"  {reg:16s} {r['k360']['seed_mean']:10.2f} "
                  f"{r['waymo']['seed_mean']:10.2f} {r['delta_seed_mean']:+9.2f}")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
