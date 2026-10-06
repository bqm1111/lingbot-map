"""Pooled metrics per subset and variant, seed statistics, and paired block-bootstrap deltas against OccAny.

Every IoU is pooled over samples (global TP/FP/FN), as OccAny's evaluator reports it. Ours is a
three-seed model: each seed is pooled separately and the headline is the seed mean (median alongside).
The paired bootstrap resamples whole blocks -- Occ3D scenes, SemanticKITTI runs of 25 consecutive
target frames -- applies the same draw to both methods, and recomputes ``mean_seed IoU(ours) - IoU(OccAny)``.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from tools.baseline_reconstruction.metrics.occupancy import seed_statistics
from tools.occany_comparison import config as C


def pooled_rates(counts: np.ndarray) -> Dict[str, float]:
    """``counts[..., 3]`` summed over every leading axis -> IoU / precision / recall / pred-over-GT."""
    tp, fp, fn = (int(x) for x in counts.reshape(-1, 3).sum(0))
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "pred_over_gt": (tp + fp) / max(tp + fn, 1)}


def _iou(t: np.ndarray) -> np.ndarray:
    return t[..., 0] / np.maximum(t.sum(-1), 1)


def bootstrap_delta(ours: np.ndarray, ref: np.ndarray, blocks: np.ndarray, n_boot: int = C.BOOTSTRAP_N,
                    seed: int = C.BOOTSTRAP_SEED) -> Dict[str, float]:
    """``ours [n, seeds, 3]``, ``ref [n, 3]``, ``blocks [n]`` -> point, 95 % CI and P(delta <= 0) of mean-seed IoU delta."""
    ids, inv = np.unique(blocks, return_inverse=True)
    nb = len(ids)
    bo = np.zeros((nb,) + ours.shape[1:], np.float64); br = np.zeros((nb, 3), np.float64)
    np.add.at(bo, inv, ours); np.add.at(br, inv, ref)
    point = float(_iou(bo.sum(0)).mean() - _iou(br.sum(0)))
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(n_boot)]).astype(np.float64)
    d = _iou(np.einsum("bk,ksc->bsc", w, bo)).mean(1) - _iou(w @ br)
    return {"point": point, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_le_0": float((d <= 0).mean()), "n_blocks": int(nb), "n_boot": int(n_boot)}


def load_dataset(dataset: str) -> Dict[str, np.ndarray]:
    files = sorted(f for f in glob.glob(os.path.join(C.EVAL_DIR, f"{dataset}_shard*of*.npz")) if "_limit" not in f)
    assert files, f"no evaluation shards for {dataset}"
    parts = [dict(np.load(f)) for f in files]
    assert len({int(os.path.basename(f).split("of")[1].split(".")[0]) for f in files}) == 1, "mixed shard counts"
    keys = [str(k) for k in parts[0]["keys"]]
    per_sample = [k for k, v in parts[0].items() if k != "keys" and np.ndim(v) >= 1 and len(v) == len(parts[0]["sample_id"])]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in per_sample}
    order = np.argsort(merged["sample_id"])
    merged = {k: v[order] for k, v in merged.items()}
    merged["keys"] = np.asarray(keys)
    assert len(merged["sample_id"]) == C.OCCANY[dataset].expected_samples, f"{dataset}: {len(merged['sample_id'])} samples scored"
    assert len(set(merged["sample_id"].tolist())) == len(merged["sample_id"]), "duplicate samples across shards"
    return merged


def subset_mask(d: Dict[str, np.ndarray], subset: str) -> np.ndarray:
    if subset == "occany_all":
        return np.ones(len(d["sample_id"]), bool)
    if subset == "causal_eligible":
        return d["causal_ok"].astype(bool)
    if subset == "official_anchors":
        return d["official_anchor"].astype(bool) & d["causal_ok"].astype(bool) & d["forward_ok"].astype(bool)
    raise KeyError(subset)


def blocks_of(dataset: str, d: Dict[str, np.ndarray]) -> np.ndarray:
    if dataset == "occ3d":
        return d["segment"]
    return d["target_pos"] // C.SK_BLOCK_FRAMES


def analyse_dataset(dataset: str) -> Dict:
    d = load_dataset(dataset)
    keys = [str(k) for k in d["keys"]]; col = {k: i for i, k in enumerate(keys)}
    cnt = d["official"]
    blocks = blocks_of(dataset, d)
    out = {"dataset": dataset, "n_samples": int(len(d["sample_id"])),
           "checks": {"fast_equals_official": bool(np.array_equal(d["fast"], d["official"])),
                      "label_identical": int(d["label_identical"].sum()),
                      "T_occany_minus_ours_maxabs": float(d["T_occany_minus_ours_maxabs"].max()),
                      "T_occany_minus_ours_median": float(np.median(d["T_occany_minus_ours_maxabs"])),
                      "scale_missing_forward": int(np.sum(d["forward_ok"] & ~np.isfinite(d["scale_forward"]))),
                      "scale_missing_causal": int(np.sum(d["causal_ok"] & ~np.isfinite(d["scale_causal"])))},
           "subsets": {}}
    for subset in C.SUBSETS:
        m = subset_mask(d, subset)
        sub = {"n_samples": int(m.sum()), "n_blocks": int(len(np.unique(blocks[m]))), "occany": {}, "ours": {}, "maponly": {}}
        for post in C.POSTPROCESS:
            sub["occany"][post] = pooled_rates(cnt[m, col[f"occany|{post}"]])
        for pname, protocol in C.PROTOCOLS.items():
            if subset == "occany_all" and pname == "causal" and not d["causal_ok"].all():
                continue          # the causal window does not exist for every OccAny sample
            for cname in C.CALIBRATIONS[dataset]:
                for post in C.POSTPROCESS:
                    v = f"{pname}|{cname}|{post}"
                    sc = np.stack([cnt[m, col[f"ours|{v}|seed{s}"]] for s in C.SEEDS], 1)     # [n, seeds, 3]
                    assert (sc >= 0).all(), (dataset, subset, v)
                    per_seed = {f"seed{s}": pooled_rates(sc[:, k]) for k, s in enumerate(C.SEEDS)}
                    row = {"per_seed": per_seed,
                           "stats": {q: seed_statistics([per_seed[f"seed{s}"][q] for s in C.SEEDS]) for q in ("iou", "precision", "recall", "pred_over_gt")},
                           "delta_vs_occany_same_post": bootstrap_delta(sc, cnt[m, col[f"occany|{post}"]], blocks[m])}
                    if post == "raw":     # the comparison as originally quoted: our raw number vs OccAny's pooled one
                        row["delta_vs_occany_pooled"] = bootstrap_delta(sc, cnt[m, col["occany|pooled"]], blocks[m])
                    sub["ours"][v] = row
                    sub["maponly"][v] = pooled_rates(cnt[m, col[f"maponly|{v}"]])
        out["subsets"][subset] = sub
    # the published OccAny number must be reproduced on its full sample set
    out["occany_reproduction"] = {"published_sc_iou": C.OCCANY[dataset].published_sc_iou,
                                  "ours_scoring_pooled": out["subsets"]["occany_all"]["occany"]["pooled"]["iou"],
                                  "ours_scoring_raw": out["subsets"]["occany_all"]["occany"]["raw"]["iou"]}
    # per-sample IoU for the figures (primary variant, seed mean of per-sample IoU)
    pv = f"forward|{C.PRIMARY['calibration'][dataset]}|{C.PRIMARY['post']}"
    ours_s = np.stack([cnt[:, col[f"ours|{pv}|seed{s}"]] for s in C.SEEDS], 1)
    out["per_sample_primary"] = {"sample_id": d["sample_id"].tolist(), "block": [str(b) for b in blocks],
                                 "occany_iou": _iou(cnt[:, col["occany|pooled"]].astype(np.float64)).tolist(),
                                 "ours_iou_seed_mean": _iou(ours_s.astype(np.float64)).mean(1).tolist(),
                                 "n_gt": d["n_gt"].tolist()}
    return out


def analyse_all() -> Dict:
    res = {ds: analyse_dataset(ds) for ds in C.DATASETS}
    res["primary"] = C.PRIMARY
    os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(C.ARTIFACT_DIR, "analysis.json"), "w"), indent=1)
    return res


def load_analysis() -> Dict:
    return json.load(open(os.path.join(C.ARTIFACT_DIR, "analysis.json")))


__all__ = ["pooled_rates", "bootstrap_delta", "load_dataset", "subset_mask", "blocks_of", "analyse_dataset", "analyse_all", "load_analysis"]
