#!/usr/bin/env python
"""Gate 6 analysis — permutation control, paired bootstrap, and the predeclared verdict.

Reads only the per-clip count blocks written by ``tools/gate6/evaluate.py``; no voxel is
re-scored and no target is re-opened, so the control and the intervals are exactly
consistent with the reported metrics.

    python tools/gate6/analyze.py --dataset kitti360
    python tools/gate6/analyze.py --combine
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                      # noqa: E402
from gates.gate6 import permute, stats, vocab                                  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate6")
CONDITIONS = ("B-R", "B-D")
PRIMARY = "B-D"
N_PERM, PERM_SEED = 100, 0
N_BOOT, BOOT_SEED = 10000, 0
BLOCK_SIZE = 20


# ------------------------------- metrics on a summed count block ----------- #
def m_ssc_miou(b):
    tp, fp, fn = b["tp"].astype(np.float64), b["fp"].astype(np.float64), b["fn"].astype(np.float64)
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    return float(np.nanmean(iou[den > 0])) if np.any(den > 0) else 0.0


def m_tp_accuracy(b):
    c = b["conf"].astype(np.float64)
    n = c.sum()
    return float(np.trace(c) / n) if n else 0.0


def m_tp_balanced(b):
    c = b["conf"].astype(np.float64)
    rows = c.sum(axis=1)
    rec = np.where(rows > 0, np.diag(c) / np.maximum(rows, 1), np.nan)
    return float(np.nanmean(rec[rows > 0])) if np.any(rows > 0) else 0.0


def m_coverage_miss(b):
    d = b["decomp"].astype(np.float64)
    return float(d[:, 0].sum() / d.sum()) if d.sum() else 0.0


def m_naming_error(b):
    d = b["decomp"].astype(np.float64)
    return float(d[:, 1].sum() / d.sum()) if d.sum() else 0.0


def m_binary_iou(b):
    tp, fp, fn = b["binary"][0], b["binary"][1], b["binary"][2]
    den = tp + fp + fn
    return float(tp / den) if den else 0.0


METRICS = {"ssc_miou": m_ssc_miou, "tp_accuracy": m_tp_accuracy,
           "tp_balanced_recall": m_tp_balanced, "coverage_miss_fraction": m_coverage_miss,
           "naming_error_fraction": m_naming_error, "binary_iou": m_binary_iou}
CI_METRICS = ("ssc_miou", "tp_accuracy", "coverage_miss_fraction", "naming_error_fraction")
BLOCK_KEYS = ("tp", "fp", "fn", "conf", "decomp", "fp_empty", "binary")


def load_counts(ds, cond):
    z = np.load(os.path.join(ART, f"counts_{ds}_{cond}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def unit_ids(ds, clip_ids, groups):
    """The benchmark's resampling unit for each clip."""
    if ds == "occ3d":
        return [str(g) for g in groups], "scene"
    if ds == "kitti360":
        return [f"block{int(g):03d}" for g in groups], "contiguous block of 20 clips"
    order = np.argsort(clip_ids)                       # chronological by construction
    blk = np.empty(len(clip_ids), np.int64)
    blk[order] = stats.contiguous_blocks(len(clip_ids), BLOCK_SIZE)
    return [f"block{b:03d}" for b in blk], "contiguous block of 20 clips"


def per_unit_blocks(z, units):
    out = {}
    for i, u in enumerate(units):
        d = out.setdefault(u, {k: np.zeros_like(z[k][0], dtype=np.int64) for k in BLOCK_KEYS})
        for k in BLOCK_KEYS:
            d[k] = d[k] + z[k][i].astype(np.int64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(vocab.DATASETS))
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()

    if a.combine:
        return combine()

    ds = a.dataset
    v = vocab.load(ds)
    res = {"dataset": ds, "n_classes": len(v), "class_names": list(v.names),
           "permutation_control": {}, "bootstrap": {}, "point": {}}

    for cond in CONDITIONS:
        z = load_counts(ds, cond)
        tot = {k: z[k].astype(np.int64).sum(axis=0) for k in BLOCK_KEYS}
        res["point"][cond] = {name: fn(tot) for name, fn in METRICS.items()}
        res["permutation_control"][cond] = permute.run_control(
            tot["conf"], tot["decomp"][:, 0], tot["fp_empty"], N_PERM, PERM_SEED)

    zr, zd = load_counts(ds, "B-R"), load_counts(ds, "B-D")
    assert list(zr["clip_id"]) == list(zd["clip_id"]), "conditions must be clip-aligned"
    units, unit_name = unit_ids(ds, list(zr["clip_id"]), list(zr["group"]))
    res["bootstrap_unit"] = unit_name
    res["n_units"] = len(set(units))
    res["blocks_are_not_independent_scenes"] = (ds != "occ3d")
    pu = {"B-R": per_unit_blocks(zr, units), "B-D": per_unit_blocks(zd, units)}
    uids = sorted(set(units))
    for name in CI_METRICS:
        res["bootstrap"][name] = stats.paired_bootstrap(
            uids, pu, METRICS[name], N_BOOT, BOOT_SEED)

    # ---------------- predeclared per-dataset decision ---------------- #
    pc = res["permutation_control"][PRIMARY]
    man = json.load(open(os.path.join(ART, f"prediction_manifest_{ds}.json")))
    res["decision"] = {
        "criterion_1_miou_exceeds_p95": pc["ssc_miou"]["real_exceeds_p95"],
        "criterion_2_tp_balanced_exceeds_p95": pc["tp_balanced_recall"]["real_exceeds_p95"],
        "criterion_3_prediction_target_independent":
            not man["audit"].get("violations") and man["audit"].get("n_opened", 0) > 0,
        "criterion_4_mapping_and_evaluation_tests": None,   # filled by --combine from pytest
        "primary_condition": PRIMARY,
    }
    write_json(os.path.join(ART, f"analysis_{ds}.json"), res)
    p = res["point"][PRIMARY]
    print(f"[{ds} {PRIMARY}] mIoU={p['ssc_miou']:.4f} "
          f"(perm p95={pc['ssc_miou']['permutation_p95']:.4f}, "
          f"pct={pc['ssc_miou']['real_percentile']:.1f}) "
          f"TPbal={p['tp_balanced_recall']:.4f} "
          f"(perm p95={pc['tp_balanced_recall']['permutation_p95']:.4f}) "
          f"miss={p['coverage_miss_fraction']:.4f} name={p['naming_error_fraction']:.4f} "
          f"units={res['n_units']}")
    return 0


def combine() -> int:
    tests_ok = os.path.exists(os.path.join(ART, "tests_passed.flag"))
    out = {"datasets": {}, "n_permutations": N_PERM, "n_boot": N_BOOT}
    passes = []
    for ds in vocab.DATASETS:
        p = os.path.join(ART, f"analysis_{ds}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        r["decision"]["criterion_4_mapping_and_evaluation_tests"] = tests_ok
        ok = all(bool(r["decision"][k]) for k in r["decision"] if k.startswith("criterion_"))
        r["decision"]["dataset_passes"] = ok
        passes.append(ok)
        write_json(p, r)
        out["datasets"][ds] = {"passes": ok, "point": r["point"],
                               "permutation": {c: {m: r["permutation_control"][c][m]
                                                   for m in ("ssc_miou", "tp_balanced_recall")}
                                               for c in CONDITIONS},
                               "decision": r["decision"], "n_units": r["n_units"]}
    n = sum(passes)
    out["transfer_diagnosis"] = ("FROZEN_TRIDENT_SEMANTICS_TRANSFER" if n == 3 else
                                 "FROZEN_TRIDENT_SEMANTICS_PARTIAL" if n >= 1 else
                                 "FROZEN_TRIDENT_SEMANTICS_FAIL")
    cov = sum(1 for ds, d in out["datasets"].items()
              if d["point"][PRIMARY]["coverage_miss_fraction"]
              > d["point"][PRIMARY]["naming_error_fraction"])
    nam = len(out["datasets"]) - cov
    out["bottleneck_diagnosis"] = ("COVERAGE_DOMINATES" if cov >= 2 else
                                   "NAMING_DOMINATES" if nam >= 2 else "MIXED_BOTTLENECK")
    out["n_datasets_passing"] = n
    out["coverage_dominates_on"] = cov
    out["naming_dominates_on"] = nam
    write_json(os.path.join(ART, "diagnosis.json"), out)
    print(json.dumps({k: out[k] for k in ("transfer_diagnosis", "bottleneck_diagnosis",
                                          "n_datasets_passing", "coverage_dominates_on")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
