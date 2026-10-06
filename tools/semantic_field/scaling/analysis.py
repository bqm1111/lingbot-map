"""Source analysis of the scaling study.

Per pool: the FROZEN ``tools/semantic_field/evaluation/analysis.analyse`` (regions, per-class IoU, distance bins, masked diagnostic,
paired block bootstrap, S1-S6) on that pool's evaluation shards, with its output directory redirected to
``artifacts/semantic_field_scaling/analysis/<pool>`` so the original B4 records are never overwritten. For the expanded pool this
IS the promotion test: candidate = expanded B4 (mean of seeds), comparator = expanded B3, B0 frozen.

Cross-pool contrasts (same frozen bootstrap ``_boot``: 20-anchor blocks, 2,000 resamples, rng seed 0, seed-pooled confusions):
    source_pool_effect      expanded B4 (40,056 updates) - current B4 (40,056 updates)
    optimisation_effect     current B4 (40,056 updates)  - original B4 (6,000 updates, artifacts/semantic_field/eval)
    method_same_pool        expanded B4 - expanded B3     (reported by the frozen analysis as B4_minus_B3)
    absolute                expanded B4 - B0              (reported by the frozen analysis as B4_minus_B0)
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from tools.delayed_semantic import analysis as PA
from tools.semantic_field.config.schema import SEEDS, Paths
from tools.semantic_field.evaluation import analysis as SFA
from tools.semantic_field.scaling.config import ARTIFACT_DIR
from tools.semantic_field.scaling.evaluate import pool_paths

CONTRAST_REGIONS = ("completed_never_observed", "ssc_full", "primary_candidate_supported", "primary_candidate_unsupported",
                    "completed_geo_unobserved", "never_observed", "completed_later_revealed")


def frozen_analysis(pool: str) -> Dict:
    out = os.path.join(ARTIFACT_DIR, "analysis", pool); os.makedirs(out, exist_ok=True)
    original = SFA.ARTIFACT_DIR
    SFA.ARTIFACT_DIR = out
    try:
        return SFA.analyse(pool_paths(pool))
    finally:
        SFA.ARTIFACT_DIR = original


def _pooled_b4(z: Dict, region: str) -> np.ndarray:
    return sum(z[f"conf__B4_s{s}__{region}"] for s in SEEDS)


def cross_pool_contrasts() -> Dict:
    exp, cur = SFA.load(pool_paths("expanded")), SFA.load(pool_paths("current"))
    orig = SFA.load(Paths())                                                      # the original 6,000-update B4 evaluation
    for name, z in (("current", cur), ("original", orig)):
        if not np.array_equal(z["ids"], exp["ids"]):
            raise RuntimeError(f"{name} evaluation anchors differ from the expanded evaluation")
    for name, z in (("current", cur), ("original", orig)):
        if not np.array_equal(z["occ"], exp["occ"]):
            raise RuntimeError(f"{name} occupancy counts differ from the expanded evaluation")
    blk = PA.blocks(len(exp["ids"])); miou = lambda cm: 100 * PA.fast_miou(cm)
    out = {"source_pool_effect": {}, "optimisation_effect": {}, "occupancy_identical_across_conditions": True}
    for r in CONTRAST_REGIONS:
        out["source_pool_effect"][r] = SFA._boot(miou, _pooled_b4(exp, r), _pooled_b4(cur, r), blk)
        out["optimisation_effect"][r] = SFA._boot(miou, _pooled_b4(cur, r), _pooled_b4(orig, r), blk)
    return out


def run() -> Dict:
    rec = {"current": frozen_analysis("current"), "expanded": frozen_analysis("expanded"), "cross_pool": cross_pool_contrasts()}
    json.dump(rec["cross_pool"], open(os.path.join(ARTIFACT_DIR, "analysis", "cross_pool_contrasts.json"), "w"), indent=1, default=float)
    return rec


__all__ = ["CONTRAST_REGIONS", "frozen_analysis", "cross_pool_contrasts", "run"]
