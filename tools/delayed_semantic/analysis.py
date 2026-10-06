"""Pooled metrics, paired block bootstrap, decision criteria K1-K6. Reads the evaluation shards only.

mIoU (semantic regions): mean over classes 1..18 with GT support in the region of TP / (GT + predicted - TP),
confusion pooled over all anchors. SSC regions: same over the 19x19 matrix with empty predictions (class 0
predictions count against no class). Uncertainty: 2,000 paired resamples of contiguous 20-anchor blocks
(0006 anchor order; a trailing partial block joins the last full one); both methods use the same resampled
blocks. One head seed: the interval covers scene sampling only, not training randomness.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from tools.delayed_semantic import config as C, rules as R
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS
from tools.semantic_completion import config as G15C
from tools.semantic_completion.evaluation import semantic_metrics as SM

ORDER = ["B0", "B0g", "B1", "B2", "B2s"]
PAIRS = [("B2", "B1"), ("B2", "B0"), ("B1", "B0"), ("B0g", "B0"), ("B2", "B0g"), ("B2s", "B1"), ("B2s", "B0")]
G15_EVAL = os.path.join(G15C.ARTIFACT_DIR, "source_eval")


def load_eval(paths: C.Paths = C.Paths()) -> Dict[str, np.ndarray]:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(paths.eval, "eval_shard*of*.npz")))]
    ids = np.concatenate([p["ids"] for p in parts]); order = np.argsort(ids)
    assert len(np.unique(ids)) == len(ids), "duplicate anchors across shards"
    return {k: np.concatenate([p[k] for p in parts])[order] for k in parts[0]}


def methods_of(z) -> List[str]:
    return [m for m in ORDER if f"conf__{m}__{PRIMARY}" in z]


def blocks(n: int, size: int = 20) -> np.ndarray:
    return np.minimum(np.arange(n) // size, max(n // size, 1) - 1)


def fast_miou(cm: np.ndarray) -> float:
    tp = np.diag(cm)[1:].astype(np.float64); g = cm[1:].sum(1); p = cm[:, 1:].sum(0); sup = g > 0
    return float(np.mean(tp[sup] / (g[sup] + p[sup] - tp[sup]))) if sup.any() else float("nan")


def bootstrap(conf_a: np.ndarray, conf_b: np.ndarray, blk: np.ndarray, resamples: int = 2000, seed: int = 0) -> Dict:
    nb = int(blk.max()) + 1
    ba = np.stack([conf_a[blk == b].sum(0) for b in range(nb)]); bb = np.stack([conf_b[blk == b].sum(0) for b in range(nb)])
    idx = np.random.default_rng(seed).integers(nb, size=(resamples, nb))
    d = np.array([fast_miou(ba[i].sum(0)) - fast_miou(bb[i].sum(0)) for i in idx])
    return {"a": fast_miou(conf_a.sum(0)), "b": fast_miou(conf_b.sum(0)), "delta": fast_miou(conf_a.sum(0)) - fast_miou(conf_b.sum(0)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "blocks": nb, "block_wins": None}


def gate15_reference(ids: np.ndarray) -> Dict[str, np.ndarray]:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(G15_EVAL, "source_eval_stride1_shard*.npz")))]
    gid = np.concatenate([p["ids"] for p in parts]); order = np.argsort(gid)
    pos = np.searchsorted(gid[order], ids); assert np.array_equal(gid[order][pos], ids)
    take = lambda k: np.concatenate([p[k] for p in parts])[order][pos]
    return {"occ_seed0": take("occ_seed0"), "b0_primary": take("conf__B0_seed0__completed_unobserved"), "b0_ssc_full": take("conf__B0_seed0__ssc_full")}


def criteria(z, pooled, deltas, ref, candidate: str) -> Dict:
    d1, d2 = deltas[f"{candidate}-B1"][PRIMARY], deltas[f"{candidate}-B0"][PRIMARY]
    c = pooled[candidate][PRIMARY]; b = pooled["B0"][PRIMARY]
    sup = [k for k in range(1, 19) if c["per_class_iou"][k] is not None]
    gain = {k: c["per_class_iou"][k] - b["per_class_iou"][k] for k in sup}
    top = max(gain, key=gain.get)
    without = float(np.mean([c["per_class_iou"][k] for k in sup if k != top]) - np.mean([b["per_class_iou"][k] for k in sup if k != top]))
    K = {"K1": {"delta": d1["delta"], "ci95": d1["ci95"], "pass": d1["delta"] > 0 and d1["ci95"][0] > 0},
         "K2": {"delta": d2["delta"], "ci95": d2["ci95"], "pass": d2["delta"] >= 0.02 and d2["ci95"][0] > 0},
         "K3": {"anchors_equal": int(z["evidence_path_equal"].sum()), "anchors": len(z["ids"]), "pass": bool(z["evidence_path_equal"].all())},
         "K4": {"classes_positive": int(sum(v > 0 for v in gain.values())), "largest_gain_class": top, "gain_without_it": without,
                "pass": sum(v > 0 for v in gain.values()) >= 5 and without > 0},
         "K5": {candidate: pooled[candidate]["ssc_full"]["miou"], "B0": pooled["B0"]["ssc_full"]["miou"],
                "pass": pooled[candidate]["ssc_full"]["miou"] >= pooled["B0"]["ssc_full"]["miou"]},
         "K6": {"pass": bool(np.array_equal(z["occ"], ref["occ_seed0"]))}}
    K["all_pass"] = all(v["pass"] for k, v in K.items() if k.startswith("K"))
    return K


def analyse(paths: C.Paths = C.Paths()) -> Dict:
    R.check("decision")
    rel = json.load(open(os.path.join(paths.reliability, "reliability.json")))["gate"]
    z = load_eval(paths); ms = methods_of(z); n = len(z["ids"]); blk = blocks(n)
    regions = SEMANTIC_REGIONS + SSC_REGIONS
    pooled = {m: {r: SM.summarize(z[f"conf__{m}__{r}"].sum(0), r in SEMANTIC_REGIONS) for r in regions} for m in ms}
    deltas = {f"{a}-{b}": {r: bootstrap(z[f"conf__{a}__{r}"], z[f"conf__{b}__{r}"], blk) for r in regions} for a, b in PAIRS if a in ms and b in ms}
    for key, per in deltas.items():                                   # block-level wins on the primary region
        a, b = key.split("-")
        wins = [fast_miou(z[f"conf__{a}__{PRIMARY}"][blk == i].sum(0)) > fast_miou(z[f"conf__{b}__{PRIMARY}"][blk == i].sum(0)) for i in range(int(blk.max()) + 1)]
        per[PRIMARY]["block_wins"] = [int(sum(wins)), len(wins)]
    ref = gate15_reference(z["ids"])
    dist = {m: [SM.summarize(z[f"confdist__{m}"][:, b].sum(0), True) for b in range(len(C.DISTANCE_NAMES))] for m in ms}
    reveal = {m: [SM.summarize(z[f"confreveal__{m}"][:, b].sum(0), True) for b in range(len(C.REVEAL_NAMES))] for m in ms}
    target_agree = {}
    for m in ms:
        cm = z[f"conftarget__{m}"].sum(0)
        target_agree[m] = {"voxels": int(cm[1:].sum()), "agreement": float(np.trace(cm[1:, 1:]) / max(cm[1:].sum(), 1)), "miou": fast_miou(cm)}
    rec = {"n_anchors": n, "methods": ms, "blocks": int(blk.max()) + 1, "reliability_gate": rel, "pooled": pooled, "deltas": deltas,
           "distance": {m: [{k: s[k] for k in ("miou", "overall_accuracy", "n_voxels")} for s in v] for m, v in dist.items()},
           "reveal": {m: [{k: s[k] for k in ("miou", "overall_accuracy", "n_voxels")} for s in v] for m, v in reveal.items()},
           "delayed_target_agreement": target_agree,
           "occupancy": {"TP_FP_FN": z["occ"].sum(0).tolist(), "iou": float(z["occ"].sum(0)[0] / z["occ"].sum(0).sum()),
                         "equal_to_gate15_seed0": bool(np.array_equal(z["occ"], ref["occ_seed0"]))},
           "b0_reproduces_gate15": {"primary_per_anchor_equal": bool(np.array_equal(z[f"conf__B0__{PRIMARY}"], ref["b0_primary"])),
                                    "ssc_full_per_anchor_equal": bool(np.array_equal(z["conf__B0__ssc_full"], ref["b0_ssc_full"]))},
           "geodesic_reached_fraction": float(z["geodesic_reached"][:, 0].sum() / max(z["geodesic_reached"][:, 1].sum(), 1))}
    if not rel["pass"]:
        rec["decision"] = "Stop: delayed targets are not reliable enough"
    else:
        rec["criteria"] = criteria(z, pooled, deltas, ref, "B2")
        if "B2s" in ms:
            rec["criteria_B2s_sensitivity"] = criteria(z, pooled, deltas, ref, "B2s")
        rec["decision"] = ("Proceed to a matched persistent-streaming source study" if rec["criteria"]["all_pass"]
                           else "Revise the semantic mechanism or supervision")
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "analysis.json"), "w"), indent=1)
    return rec


__all__ = ["load_eval", "blocks", "fast_miou", "bootstrap", "criteria", "analyse"]
