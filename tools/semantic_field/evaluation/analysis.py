"""Source analysis and the frozen promotion rule (S1-S6). Reads the source evaluation shards only.

mIoU = mean over classes 1..18 with GT support of TP / (GT + pred - TP), confusions pooled over anchors. Seeds: per seed and mean.
Bootstrap: 20-anchor blocks, 2,000 resamples, seed 0; B4 seed-pooled confusions vs 3 x B0, or vs B3 seed-pooled confusions.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from tools.delayed_semantic import analysis as PA
from tools.delayed_semantic import config as PILOT
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_field import freeze
from tools.semantic_field.config.schema import ARTIFACT_DIR, SEEDS, Paths
from tools.semantic_field.evaluation.evaluate_source import REGIONS

PRIMARY = "completed_never_observed"          # = Gate 15 completed_unobserved
NAMES = ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking", "sidewalk", "other ground", "building",
         "fence", "vegetation", "terrain", "pole", "traffic sign", "other structure", "other object"]
REPORT_REGIONS = {"oracle_observed": "oracle_observed", "completed_observed": "completed_observed", PRIMARY: "completed_unobserved (PRIMARY)",
                  "completed_geo_unobserved": "geometry_unobserved", "never_observed": "no_semantic_evidence", "ssc_full": "SSC full",
                  "primary_candidate_supported": "completed_unobserved, candidate-supported", "primary_candidate_unsupported": "completed_unobserved, candidate-unsupported",
                  "completed_later_revealed": "completed_unobserved, later revealed"}


def load(paths: Paths = Paths()) -> Dict:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(paths.eval, "source_eval_shard*of*.npz")))]
    ids = np.concatenate([p["ids"] for p in parts]); order = np.argsort(ids)
    z = {k: np.concatenate([p[k] for p in parts])[order] for k in parts[0] if k.startswith("conf__") or k in ("occ", "evidence_path_equal", "ids")}
    for k in parts[0]:
        if k.startswith(("masked_", "dist__")):
            z[k] = sum(p[k] for p in parts)
    return z


def entropy(freq: List[float]) -> float:
    f = np.asarray(freq, np.float64); f = f[f > 0]
    return float(-(f * np.log2(f)).sum())


def summaries(z: Dict, region: str, methods: List[str]) -> Dict:
    return {m: SM.summarize(z[f"conf__{m}__{region}"].sum(0), region not in ("ssc_full", "ssc_never_observed")) for m in methods}


def analyse(paths: Paths = Paths()) -> Dict:
    freeze.check("source_predeclaration"); freeze.check("centering"); freeze.check("source_code")
    z = load(paths); blk = PA.blocks(len(z["ids"]))
    b3 = [f"B3_s{s}" for s in SEEDS]; b4 = [f"B4_s{s}" for s in SEEDS]; methods = ["B0"] + b3 + b4
    pooled = lambda ms, r: sum(z[f"conf__{m}__{r}"] for m in ms)
    miou = lambda cm: 100 * PA.fast_miou(cm)
    rec: Dict = {"n_anchors": int(len(z["ids"])), "regions": {}}
    for r in REPORT_REGIONS:
        s = summaries(z, r, methods)
        rec["regions"][r] = {"label": REPORT_REGIONS[r], "voxels": s["B0"]["n_voxels"], "B0": 100 * s["B0"]["miou"],
                             "B3": float(np.mean([100 * s[m]["miou"] for m in b3])), "B3_per_seed": [100 * s[m]["miou"] for m in b3],
                             "B4": float(np.mean([100 * s[m]["miou"] for m in b4])), "B4_per_seed": [100 * s[m]["miou"] for m in b4],
                             "accuracy": {"B0": 100 * s["B0"]["overall_accuracy"], "B3": float(np.mean([100 * s[m]["overall_accuracy"] for m in b3])),
                                          "B4": float(np.mean([100 * s[m]["overall_accuracy"] for m in b4]))}}
        d0 = _boot(miou, pooled(b4, r), 3 * z[f"conf__B0__{r}"], blk); d3 = _boot(miou, pooled(b4, r), pooled(b3, r), blk)
        rec["regions"][r]["B4_minus_B0"] = d0; rec["regions"][r]["B4_minus_B3"] = d3
    prim = summaries(z, PRIMARY, methods)
    per_class = {}
    for c in range(1, 19):
        b0i = prim["B0"]["per_class_iou"][c]
        if b0i is None:
            continue
        per_class[NAMES[c - 1]] = {"B0": 100 * b0i, "B3": float(np.mean([100 * prim[m]["per_class_iou"][c] for m in b3])),
                                   "B4": float(np.mean([100 * prim[m]["per_class_iou"][c] for m in b4])), "gt_share": 100 * prim["B0"]["gt_frequency"][c - 1],
                                   "B4_pred_share": float(np.mean([100 * prim[m]["pred_frequency"][c - 1] for m in b4]))}
    rec["per_class_primary"] = per_class
    rec["distance"] = {PILOT.DISTANCE_NAMES[b]: {m_: 100 * PA.fast_miou(sum(z[f"dist__{m}"][b] for m in ms) if ms != ["B0"] else z["dist__B0"][b])
                                                  for m_, ms in (("B0", ["B0"]), ("B3", b3), ("B4", b4))} | {"voxels": int(z["dist__B0"][b][1:].sum())}
                       for b in range(len(PILOT.DISTANCE_NAMES))}
    rec["masked_diagnostic"] = {}
    for name, ms in (("nearest remaining evidence (B0)", ["B0"]), ("B3", b3), ("B4", b4)):
        tch = sum(z[f"masked_teacher__{m}"] for m in ms); gtc = sum(z[f"masked_gt__{m}"] for m in ms)
        rec["masked_diagnostic"][name] = {"hidden_voxels": int(tch[0] // len(ms)), "teacher_class_agreement": 100 * tch[1] / max(tch[0], 1),
                                          "gt_accuracy": 100 * np.trace(gtc[1:, 1:]) / max(gtc[1:].sum(), 1), "gt_miou": miou(gtc)}
    # ---- promotion rule ----
    reg = rec["regions"][PRIMARY]; ref = PA.gate15_reference(z["ids"])
    gains_seed = [v - reg["B0"] for v in reg["B4_per_seed"]]
    classes_up = sum(1 for v in per_class.values() if v["B4"] > v["B0"])
    dom = float(np.mean([prim[m]["dominant_pred_fraction"] for m in b4])); gt_dom = prim["B0"]["dominant_gt_fraction"]
    k1 = float(np.mean([prim[m]["n_classes_ge_1pct_of_predictions"] for m in b4]))
    ent_pred = float(np.mean([entropy(prim[m]["pred_frequency"]) for m in b4])); ent_gt = entropy(prim["B0"]["gt_frequency"])
    S = {"S1": {"gain": reg["B4"] - reg["B0"], "pass": reg["B4"] - reg["B0"] >= 2.0},
         "S2": {"seed_gains": gains_seed, "ci95": reg["B4_minus_B0"]["ci95"], "pass": all(g > 0 for g in gains_seed) and reg["B4_minus_B0"]["ci95"][0] > 0},
         "S3": {"gain_vs_B3": reg["B4"] - reg["B3"], "ci95": reg["B4_minus_B3"]["ci95"], "pass": reg["B4"] - reg["B3"] >= 1.0 and reg["B4_minus_B3"]["ci95"][0] > 0},
         "S4": {"classes_improved_vs_B0": classes_up, "pass": classes_up >= 8},
         "S5": {"dominant_fraction": dom, "limit": max(0.60, gt_dom + 0.10), "classes_ge_1pct": k1, "entropy_pred": ent_pred, "entropy_gt": ent_gt,
                "pass": dom <= max(0.60, gt_dom + 0.10) and k1 >= 5 and ent_pred >= 0.75 * ent_gt},
         "S6": {"pass": bool(np.array_equal(z["occ"], ref["occ_seed0"]))}}
    decision = "SOURCE PASS" if all(v["pass"] for v in S.values()) else "SOURCE FAIL"
    rec["criteria"] = S; rec["decision"] = decision
    rec["reproduction"] = {"B0_equal_gate15": bool(np.array_equal(z[f"conf__B0__{PRIMARY}"], ref["b0_primary"])), "evidence_path_equal": bool(z["evidence_path_equal"].all())}
    json.dump(rec, open(os.path.join(ARTIFACT_DIR, "source_analysis.json"), "w"), indent=1, default=float)
    json.dump({"decision": decision, "criteria": S, "frozen_rule": "artifacts/semantic_field/source_predeclaration.md"},
              open(os.path.join(ARTIFACT_DIR, "source_promotion.json"), "w"), indent=1, default=float)
    return rec


def _boot(metric, a: np.ndarray, b: np.ndarray, blk: np.ndarray) -> Dict:
    nb = int(blk.max()) + 1
    ba = np.stack([a[blk == i].sum(0) for i in range(nb)]); bb = np.stack([b[blk == i].sum(0) for i in range(nb)])
    idx = np.random.default_rng(0).integers(nb, size=(2000, nb))
    d = np.array([metric(ba[i].sum(0)) - metric(bb[i].sum(0)) for i in idx])
    return {"delta": metric(a.sum(0)) - metric(b.sum(0)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


__all__ = ["PRIMARY", "load", "analyse"]
