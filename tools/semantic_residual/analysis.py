"""B5 source decision: pool the evaluation shards, score B0 against B5, apply the pre-registered rule.

The rule was written into ``config.DECISION_RULE`` before training. This module reports every quantity it names --
and the effect sizes and uncertainty even where a threshold is missed.

Uncertainty is a paired block bootstrap over 20-anchor temporal blocks (the pilot's convention): blocks are resampled,
confusions are summed within the resample, and mIoU is recomputed from the summed confusion. Per-block mIoUs are never
averaged, and neighbouring voxels are never treated as independent.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.delayed_semantic import config as C
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS
from tools.semantic_residual import config as B5
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_residual.evaluate import DECISION_REGION

SEEDS = (0, 1, 2)


def miou(conf: np.ndarray) -> float:
    """mIoU from a summed 19x19 confusion, using the REPO'S frozen definition.

    ``semantic_metrics.summarize`` averages IoU over classes 1..18 with GT support -- the empty class 0 is excluded.
    Reimplementing this is how an earlier version of this file silently reported SSC mIoU 6.42 where the frozen
    protocol says 1.55: including class 0, which is large and easy, inflates any region that contains empty voxels."""
    r = SM.summarize(conf.astype(np.int64), semantic_only=True)["miou"]
    return float("nan") if r is None else float(100 * r)


def accuracy(conf: np.ndarray) -> float:
    r = SM.summarize(conf.astype(np.int64), semantic_only=True)["overall_accuracy"]
    return float(100 * r)


def load_shards() -> Dict[str, np.ndarray]:
    """All shards of ONE sharding. A leftover file from a different run is an error, not something to concatenate.

    An earlier version globbed every ``eval_shard*of*.npz``, so a 4-anchor smoke test written as ``shard0of1``
    was merged with the real ``*of4`` shards and four anchors were counted twice."""
    files = sorted(glob.glob(os.path.join(B5.DATA_ROOT, "eval", "eval_shard*of*.npz")))
    if not files:
        raise FileNotFoundError("no evaluation shards; run tools.semantic_residual.evaluate first")
    groups: Dict[str, List[str]] = {}
    for f in files:
        groups.setdefault(os.path.basename(f).split("of")[1][:-4], []).append(f)
    if len(groups) > 1:
        raise RuntimeError(f"evaluation shards from different runs in {os.path.dirname(files[0])}: "
                           f"{ {k: len(v) for k, v in groups.items()} }. Remove the stale ones and re-run.")
    n_expected = int(next(iter(groups)))
    if len(files) != n_expected:
        raise RuntimeError(f"expected {n_expected} shards, found {len(files)}")
    parts = [np.load(f, allow_pickle=True) for f in files]
    keys = parts[0].files
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if len(np.unique(out["ids"])) != len(out["ids"]):
        raise RuntimeError("duplicate anchors across shards")
    order = np.argsort(out["ids"], kind="stable")                    # anchors back in temporal order for blocking
    return {k: v[order] for k, v in out.items()}


def blocks(n: int, size: int) -> List[np.ndarray]:
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int, block: int) -> Tuple[float, float, float]:
    """(mean delta, ci_lo, ci_hi) of mIoU(b) - mIoU(a), resampling 20-anchor blocks of per-anchor confusions."""
    bl = blocks(len(a), block)
    rng = np.random.default_rng(seed)
    point = miou(b.sum(0)) - miou(a.sum(0))
    d = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(bl), len(bl))
        idx = np.concatenate([bl[j] for j in pick])
        d[i] = miou(b[idx].sum(0)) - miou(a[idx].sum(0))
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run() -> Dict:
    d = load_shards()
    rule = B5.DECISION_RULE; bs = rule["bootstrap"]
    methods = ["B0"] + [f"B5_s{s}" for s in SEEDS]
    regions = list(SEMANTIC_REGIONS) + list(SSC_REGIONS)

    table: Dict[str, Dict[str, float]] = {}
    for r in regions:
        table[r] = {m: miou(d[f"conf__{m}__{r}"].sum(0)) for m in methods}
        table[r]["B5_mean"] = float(np.mean([table[r][f"B5_s{s}"] for s in SEEDS]))
        table[r]["delta_mean"] = table[r]["B5_mean"] - table[r]["B0"]

    # the decision region, per seed, with paired CIs
    per_seed, cis = {}, {}
    for s in SEEDS:
        a = d[f"conf__B0__{DECISION_REGION}"]; b = d[f"conf__B5_s{s}__{DECISION_REGION}"]
        per_seed[f"seed{s}"] = miou(b.sum(0)) - miou(a.sum(0))
        cis[f"seed{s}"] = paired_bootstrap(a, b, bs["resamples"], bs["seed"], bs["block_anchors"])

    # pooled over seeds: sum the three seeds' confusions is wrong (they are the same voxels three times),
    # so the pooled statement is the mean of the per-seed deltas, with the CI of the median seed reported alongside
    mean_delta = float(np.mean(list(per_seed.values())))
    mid_seed = sorted(SEEDS, key=lambda s: per_seed[f"seed{s}"])[1]
    point, lo, hi = cis[f"seed{mid_seed}"]

    teacher = {m: {"top1_agreement": float(d[f"teacher__{m}"][:, 0].sum() / max(d[f"teacher__{m}"][:, 1].sum(), 1))}
               for m in methods}

    checks = {
        "mean_miou_gain_pp": mean_delta,
        "mean_miou_gain_pp_min": rule["pass_requires"]["mean_miou_gain_pp_min"],
        "gain_meets_threshold": mean_delta >= rule["pass_requires"]["mean_miou_gain_pp_min"],
        "all_three_seeds_positive": all(v > 0 for v in per_seed.values()),
        "paired_ci95": [point, lo, hi], "ci_lower_above_zero": lo > 0.0,
        "candidate_supported_change_pp": table["completed_geo_observed"]["delta_mean"],
        "candidate_supported_ok": table["completed_geo_observed"]["delta_mean"] >= rule["pass_requires"]["candidate_supported_change_pp_min"],
        "full_source_ssc_change_pp": table["ssc_full"]["delta_mean"],
        "full_source_ssc_ok": table["ssc_full"]["delta_mean"] >= rule["pass_requires"]["full_source_ssc_change_pp_min"],
        "direct_evidence_identical": bool(d["evidence_path_equal"].all()),
        "occupancy_identical": bool(d["occupancy_identical"].all()),
    }
    passed = (checks["gain_meets_threshold"] and checks["all_three_seeds_positive"] and checks["ci_lower_above_zero"]
              and checks["candidate_supported_ok"] and checks["full_source_ssc_ok"]
              and checks["direct_evidence_identical"] and checks["occupancy_identical"])

    out = {"version": B5.VERSION, "n_anchors": int(len(d["ids"])), "decision_region": DECISION_REGION,
           "future_offsets": list(B5.FUTURE_OFFSETS), "methods": methods, "miou_by_region": table,
           "decision_region_delta_per_seed": per_seed, "decision_region_ci_per_seed": cis,
           "teacher_agreement": teacher, "checks": checks,
           "outcome": "SOURCE PASS" if passed else "SOURCE FAIL", "rule": rule}
    os.makedirs(B5.ARTIFACT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(B5.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    r = run()
    t = r["miou_by_region"]
    print(f"B5 source evaluation on 0006 -- {r['n_anchors']} anchors, offsets +{B5.FUTURE_OFFSETS}\n")
    print("%-32s %8s %8s %8s" % ("region (mIoU)", "B0", "B5 mean", "delta"))
    for k, v in t.items():
        print("%-32s %8.2f %8.2f %+8.2f" % (k, v["B0"], v["B5_mean"], v["delta_mean"]))
    print("\ndecision region: %s" % r["decision_region"])
    for s, v in r["decision_region_delta_per_seed"].items():
        lo, hi = r["decision_region_ci_per_seed"][s][1], r["decision_region_ci_per_seed"][s][2]
        print("  %-8s delta %+6.2f pp   95%% CI [%+.2f, %+.2f]" % (s, v, lo, hi))
    print("\nchecks:")
    for k, v in r["checks"].items():
        print("  %-34s %s" % (k, v))
    print("\nOUTCOME:", r["outcome"])
