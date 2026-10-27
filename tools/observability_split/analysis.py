"""Tables and paired block-bootstrap intervals for the observed / unobserved split.

Regions come from ``evaluate.py``; ``unobserved`` is the sum of ``occluded`` and ``out_of_view``.
Metrics are always computed from counts pooled over the resampled blocks, never by averaging
per-frame metrics -- the same convention as OccAny's ``SSCMetrics``. Blocks are 25 consecutive
evaluated frames of sequence 08 (~12.5 s of driving), the unit used by every earlier comparison in
this repository.

    PYTHONPATH=$PWD python -m tools.observability_split.analysis
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.observability_split.evaluate import ARTIFACT_DIR
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, Statistics

REGIONS = ("all", "observed", "occluded", "out_of_view", "unobserved")
BLOCK = Statistics().semantickitti_block_samples
RESAMPLES = Statistics().resamples
SEED = Statistics().seed


def _load(dataset: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(ARTIFACT_DIR, f"counts_{dataset}.npz")) as z:
        return {k: z[k] for k in z.files}


def occ_counts(d, key: str, post: str, region: str) -> np.ndarray:
    """[S, 3] per-sample (tp, fp, fn)."""
    if region == "unobserved":
        return d[f"ps_occ__{key}__{post}__occluded"] + d[f"ps_occ__{key}__{post}__out_of_view"]
    return d[f"ps_occ__{key}__{post}__{region}"]


def cls_counts(d, key: str, post: str, region: str, what: str = "cls") -> np.ndarray:
    """[S, C, 3] per-sample (tp, fp, fn) per class."""
    if region == "unobserved":
        return d[f"ps_{what}__{key}__{post}__occluded"] + d[f"ps_{what}__{key}__{post}__out_of_view"]
    return d[f"ps_{what}__{key}__{post}__{region}"]


def iou(c: np.ndarray) -> float:
    tp, fp, fn = c.sum(0).astype(np.float64)
    return 100 * tp / max(tp + fp + fn, 1.0)


def recall(c: np.ndarray) -> float:
    tp, _, fn = c.sum(0).astype(np.float64)
    return 100 * tp / max(tp + fn, 1.0)


def precision(c: np.ndarray) -> float:
    tp, fp, _ = c.sum(0).astype(np.float64)
    return 100 * tp / max(tp + fp, 1.0)


def miou(c: np.ndarray, valid: List[int]) -> float:
    s = c.sum(0).astype(np.float64)[valid]
    return 100 * float(np.mean(s[:, 0] / (s.sum(1) + 1e-5)))


def _blocks(n: int) -> List[np.ndarray]:
    return [np.arange(a, min(a + BLOCK, n)) for a in range(0, n, BLOCK)]


def bootstrap(a: np.ndarray, b: np.ndarray, stat, n: int) -> Tuple[float, float, float]:
    """Paired block bootstrap of ``stat(a) - stat(b)``; both arrays are [S, ...] counts."""
    rng = np.random.default_rng(SEED)
    blk = _blocks(n)
    point = stat(a) - stat(b)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = np.concatenate([blk[j] for j in rng.integers(0, len(blk), len(blk))])
        draws[i] = stat(a[pick]) - stat(b[pick])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def report(dataset: str = "semantickitti") -> Dict:
    d = _load(dataset)
    spec = DATASET_SPECS[dataset]
    n = len(d["samples"])
    sem_valid = [c for c in range(1, spec.n_classes)]                       # 1..19, OccAny's mIoU classes
    sup_valid = [c for c in range(len(spec.superclass_names)) if c != spec.superclass_empty]
    out: Dict = {"dataset": dataset, "n_samples": n, "block_frames": BLOCK, "resamples": RESAMPLES}

    out["composition"] = {}
    for r in REGIONS:
        if r == "unobserved":
            v, g = (d["n__occluded"] + d["n__out_of_view"]).tolist()
        else:
            v, g = d[f"n__{r}"].tolist()
        tot = int(d["n__all"][1])
        out["composition"][r] = {"valid_voxels": int(v), "gt_occupied": int(g),
                                 "share_of_gt_surface": 100 * g / max(tot, 1),
                                 "share_of_valid": 100 * v / max(int(d["n__all"][0]), 1)}

    occ_runs = ["ours", "occany_1f_geom", "occany_5f_geom", "occany_5f_sem", "occany_1f_sem"]
    out["occupancy"] = {}
    for key in occ_runs:
        for post in ("raw", "pooled"):
            for r in REGIONS:
                c = occ_counts(d, key, post, r)
                out["occupancy"].setdefault(key, {}).setdefault(post, {})[r] = {
                    "iou": iou(c), "precision": precision(c), "recall": recall(c)}

    def b3_mean(what: str, post: str, r: str) -> np.ndarray:
        return sum(cls_counts(d, f"ours_B3_s{s}", post, r, what) for s in (0, 1, 2)) / 3.0

    out["semantics"] = {}
    for key, get in [("ours_B0", lambda w, p, r: cls_counts(d, "ours_B0", p, r, w)),
                     ("ours_B3", b3_mean),
                     ("occany_1f_sem", lambda w, p, r: cls_counts(d, "occany_1f_sem", p, r, w)),
                     ("occany_5f_sem", lambda w, p, r: cls_counts(d, "occany_5f_sem", p, r, w))]:
        for post in ("raw", "pooled"):
            for r in REGIONS:
                out["semantics"].setdefault(key, {}).setdefault(post, {})[r] = {
                    "ssc_miou": miou(get("cls", post, r), sem_valid),
                    "superclass_miou": miou(get("sup", post, r), sup_valid),
                    "surface_recall": recall(occ_counts(d, "ours" if key.startswith("ours") else key.replace("_sem", "_sem"), post, r))}

    #: headline paired deltas: ours minus each OccAny configuration, per region
    out["deltas"] = {}
    for post in ("raw", "pooled"):
        for opp in ("occany_1f_geom", "occany_5f_geom"):
            for r in REGIONS:
                p, lo, hi = bootstrap(occ_counts(d, "ours", post, r), occ_counts(d, opp, post, r), iou, n)
                out["deltas"].setdefault("sc_iou", {}).setdefault(post, {}).setdefault(opp, {})[r] = {
                    "delta": p, "ci_low": lo, "ci_high": hi}
    for opp in ("occany_1f_sem", "occany_5f_sem"):
        for r in REGIONS:
            a = b3_mean("cls", "pooled", r); b = cls_counts(d, opp, "pooled", r, "cls")
            p, lo, hi = bootstrap(a, b, lambda x: miou(x, sem_valid), n)
            out["deltas"].setdefault("ssc_miou", {}).setdefault("pooled", {}).setdefault(opp, {})[r] = {
                "delta": p, "ci_low": lo, "ci_high": hi}
            a = b3_mean("sup", "pooled", r); b = cls_counts(d, opp, "pooled", r, "sup")
            p, lo, hi = bootstrap(a, b, lambda x: miou(x, sup_valid), n)
            out["deltas"].setdefault("superclass_miou", {}).setdefault("pooled", {}).setdefault(opp, {})[r] = {
                "delta": p, "ci_low": lo, "ci_high": hi}

    json.dump(out, open(os.path.join(ARTIFACT_DIR, f"analysis_{dataset}.json"), "w"), indent=1, default=float)
    return out


def _fmt(out: Dict) -> str:
    L: List[str] = []
    cols = ["all", "observed", "occluded", "out_of_view", "unobserved"]
    L.append(f"n = {out['n_samples']} frames, sequence 08\n")
    L.append("REGION COMPOSITION")
    L.append(f"  {'region':13s} {'valid voxels':>16s} {'% of valid':>11s} {'GT surfaces':>14s} {'% of GT':>9s}")
    for r in cols:
        c = out["composition"][r]
        L.append(f"  {r:13s} {c['valid_voxels']:>16,} {c['share_of_valid']:>10.1f}% {c['gt_occupied']:>14,} {c['share_of_gt_surface']:>8.1f}%")
    for post in ("raw", "pooled"):
        L.append(f"\nSC IoU  [{post}]")
        L.append(f"  {'run':16s} " + " ".join(f"{c:>12s}" for c in cols))
        for k, v in out["occupancy"].items():
            L.append(f"  {k:16s} " + " ".join(f"{v[post][c]['iou']:12.2f}" for c in cols))
        L.append(f"  {'-- recall':16s}")
        for k, v in out["occupancy"].items():
            L.append(f"  {k:16s} " + " ".join(f"{v[post][c]['recall']:12.2f}" for c in cols))
    L.append("\nSSC mIoU  [semantic pooling]")
    L.append(f"  {'run':16s} " + " ".join(f"{c:>12s}" for c in cols))
    for k, v in out["semantics"].items():
        L.append(f"  {k:16s} " + " ".join(f"{v['pooled'][c]['ssc_miou']:12.2f}" for c in cols))
    L.append("\nsuperclass mIoU  [semantic pooling]")
    for k, v in out["semantics"].items():
        L.append(f"  {k:16s} " + " ".join(f"{v['pooled'][c]['superclass_miou']:12.2f}" for c in cols))
    for metric, by_post in out["deltas"].items():
        for post, by_opp in by_post.items():
            for opp, by_r in by_opp.items():
                L.append(f"\nDELTA ours - {opp}  [{metric}, {post}]")
                for r in cols:
                    z = by_r[r]
                    star = "  *" if z["ci_low"] > 0 or z["ci_high"] < 0 else ""
                    L.append(f"  {r:13s} {z['delta']:+8.2f}  [{z['ci_low']:+7.2f}, {z['ci_high']:+7.2f}]{star}")
    return "\n".join(L)


if __name__ == "__main__":
    o = report()
    print(_fmt(o))
