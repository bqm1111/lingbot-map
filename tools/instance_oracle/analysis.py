"""Gate 17 analysis: pool the shards, build the tables, classify the outcome.

Metric is the repository's frozen one (`semantic_metrics.summarize`, mIoU over classes 1..18 with GT support,
empty class excluded). Uncertainty is the pilot's paired block bootstrap: 20-anchor temporal blocks, confusions
summed within a resample, mIoU recomputed from the sum -- never an average of per-block mIoUs.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.instance_oracle import config as G
from tools.semantic_completion.evaluation import semantic_metrics as SM

BLOCK, RESAMPLES = 20, 2000


def miou(conf: np.ndarray) -> float:
    r = SM.summarize(conf.astype(np.int64), semantic_only=True)["miou"]
    return float("nan") if r is None else float(100 * r)


def per_class(conf: np.ndarray) -> Dict[str, float]:
    d = SM.summarize(conf.astype(np.int64), semantic_only=True)["per_class_iou"]
    return {G.CLASS_NAMES[c - 1]: (None if d[c] is None else 100 * d[c]) for c in d}


def load() -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(G.DATA_ROOT, "eval", "gate17_shard*of*.npz")))
    if not files:
        raise FileNotFoundError("no shards; run tools.instance_oracle.evaluate")
    groups: Dict[str, List[str]] = {}
    for f in files:
        groups.setdefault(os.path.basename(f).split("of")[1][:-4], []).append(f)
    if len(groups) > 1:
        raise RuntimeError(f"shards from different runs: { {k: len(v) for k, v in groups.items()} }")
    parts = [np.load(f, allow_pickle=True) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    if len(np.unique(out["ids"])) != len(out["ids"]):
        raise RuntimeError("duplicate anchors across shards")
    o = np.argsort(out["ids"], kind="stable")
    return {k: v[o] for k, v in out.items()}


def paired_ci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    bl = [np.arange(i, min(i + BLOCK, len(a))) for i in range(0, len(a), BLOCK)]
    rng = np.random.default_rng(G.SEED)
    point = miou(b.sum(0)) - miou(a.sum(0))
    d = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        idx = np.concatenate([bl[j] for j in rng.integers(0, len(bl), len(bl))])
        d[i] = miou(b[idx].sum(0)) - miou(a[idx].sum(0))
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run() -> Dict:
    d = load()
    main = {m: {p: miou(d[f"conf__{m}__{p}"].sum(0)) for p in G.POPULATIONS} for m in G.METHODS}
    cov = float(d["coverage"][:, 0].sum() / max(d["coverage"][:, 1].sum(), 1))
    P2 = "P2_correct_completion"; P1 = "P1_ssc_deployable"
    base = d[f"conf__C0_b0__{P2}"]
    cis = {m: paired_ci(base, d[f"conf__{m}__{P2}"]) for m in G.METHODS if m != "C0_b0"}
    ci_p1 = {m: paired_ci(d[f"conf__C0_b0__{P1}"], d[f"conf__{m}__{P1}"]) for m in ("A_oracle_instance", "C2_gt_class")}

    bins = {}
    for axis, key, names in (("distance", "confdist", G.DISTANCE_NAMES), ("visibility", "confvis", G.VISIBILITY_NAMES)):
        bins[axis] = {names[b]: {m: miou(d[f"{key}__{m}"][:, b].sum(0)) for m in G.METHODS}
                      for b in range(len(names))}
        bins[axis + "_counts"] = {names[b]: int(d[f"confdist__C0_b0"][:, b].sum() if key == "confdist"
                                                else d[f"confvis__C0_b0"][:, b].sum()) for b in range(len(names))}
    split = {s: {m: miou(d[f"conf{s}__{m}"].sum(0)) for m in G.METHODS} for s in ("thing", "stuff")}
    split_counts = {s: int(d[f"conf{s}__C0_b0"].sum()) for s in ("thing", "stuff")}

    gain_A_P2 = main["A_oracle_instance"][P2] - main["C0_b0"][P2]
    gain_A_P1 = main["A_oracle_instance"][P1] - main["C0_b0"][P1]
    gain_shuf = main["C1_shuffled_instance"][P2] - main["C0_b0"][P2]
    ceiling = main["C2_gt_class"][P2] - main["C0_b0"][P2]

    # Outcome per the brief. "Substantial" on P2 is read against the headroom the semantic ceiling actually offers.
    if gain_A_P2 >= 1.0 and cis["A_oracle_instance"][1] > 0 and gain_A_P1 >= 1.0:
        outcome, why = "A", "oracle instance transfer improves both the isolated semantic population and the deployable SSC read"
    elif gain_A_P2 >= 1.0 and cis["A_oracle_instance"][1] > 0:
        outcome, why = "B", "instance semantics work on correctly completed voxels, but the deployable SSC gain is small: geometry completion limits it"
    else:
        outcome, why = "C", "perfect instance grouping does not materially beat nearest-evidence propagation"

    out = {"version": G.VERSION, "n_anchors": int(len(d["ids"])), "coverage_P2": cov,
           "instances_per_anchor": float(d["n_instances"].mean()),
           "instances_with_evidence_per_anchor": float(d["n_inst_with_evidence"].mean()),
           "population_voxels": {p: int(d["n_pop"][:, i].sum()) for i, p in enumerate(G.POPULATIONS)},
           "miou": main, "paired_ci_P2": cis, "paired_ci_P1": ci_p1, "bins": bins,
           "thing_stuff": split, "thing_stuff_voxels": split_counts,
           "per_class_P2": {m: per_class(d[f"conf__{m}__{P2}"].sum(0)) for m in G.METHODS},
           "gains": {"A_minus_B0_P2": gain_A_P2, "A_minus_B0_P1": gain_A_P1,
                     "shuffled_minus_B0_P2": gain_shuf, "gt_ceiling_minus_B0_P2": ceiling},
           "oracle_use": G.ORACLE_USE, "outcome": outcome, "outcome_reason": why}
    os.makedirs(G.ARTIFACT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(G.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    r = run()
    print("Gate 17 -- %d anchors on 0006, instance coverage of P2 %.3f\n" % (r["n_anchors"], r["coverage_P2"]))
    print("%-24s %14s %14s %14s" % ("method", "P1 deployable", "P2 correct", "P3 oracle geo"))
    for m in G.METHODS:
        v = r["miou"][m]
        print("%-24s %14.2f %14.2f %14.2f" % (m, v["P1_ssc_deployable"], v["P2_correct_completion"], v["P3_oracle_geometry"]))
    print("\npaired 95%% CI vs B0 on P2 (correct completion):")
    for m, (p, lo, hi) in r["paired_ci_P2"].items():
        print("  %-24s %+6.2f  [%+.2f, %+.2f]" % (m, p, lo, hi))
    print("\nthing vs stuff on P2 (voxels: thing %d, stuff %d):" % (r["thing_stuff_voxels"]["thing"], r["thing_stuff_voxels"]["stuff"]))
    for s in ("thing", "stuff"):
        print("  %-6s " % s + "  ".join("%s %.2f" % (m.split("_")[0], r["thing_stuff"][s][m]) for m in G.METHODS))
    print("\nby distance to evidence (P2 mIoU):")
    for b in G.DISTANCE_NAMES:
        print("  %-10s " % b + "  ".join("%s %6.2f" % (m.split("_")[0], r["bins"]["distance"][b][m]) for m in G.METHODS))
    print("\nby instance visible fraction (P2 mIoU):")
    for b in G.VISIBILITY_NAMES:
        print("  %-10s " % b + "  ".join("%s %6.2f" % (m.split("_")[0], r["bins"]["visibility"][b][m]) for m in G.METHODS))
    print("\nOUTCOME %s -- %s" % (r["outcome"], r["outcome_reason"]))
