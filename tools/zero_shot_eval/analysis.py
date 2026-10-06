"""Final zero-shot analysis: Track P and Track O metrics, B3 - B0 with paired block bootstrap, OccAny geometry, paper decision.

B3 = seeds 0/1/2: metric per seed and seed mean; bootstrap on seed-pooled confusions against 3 x B0 (same anchors per resample).
Track P mIoU = mean IoU over classes with GT support in the population; Track O mIoU = OccAny SSCMetrics definition.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List

import numpy as np

from tools.zero_shot_eval import config as C, freeze
from tools.zero_shot_eval.canonical.metrics import occany_ssc_miou, occupancy_scores, semantic_summary, superclass_confusion
from tools.zero_shot_eval.practical.evaluator import CENTRAL, POPULATIONS, REGIONS
from tools.zero_shot_eval.practical.protocol import (B3_SEED_METHODS, DATASET_SPECS, EVIDENCE_DISTANCE_NAMES, PAPER, REGION, STATISTICS,
                                                    DatasetSpec)
from tools.zero_shot_eval.run_eval import SCORE_DIR

KEY_POPULATIONS = ("ssc", "ssc_no_evidence", "gt_occupied", "observed_evidence", "no_evidence", "completed_no_evidence", "geometrically_unobserved")


def load_scores(dataset: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(SCORE_DIR, f"scores_{dataset}.npz")) as z:
        return {k: z[k] for k in z.files}


def track_p_miou(conf: np.ndarray, classes: List[int]) -> float:
    tp = np.diag(conf).astype(np.float64); gt = conf.sum(1); pr = conf.sum(0)
    sup = [c for c in classes if gt[c] > 0]
    return float(np.mean([tp[c] / (gt[c] + pr[c] - tp[c]) for c in sup])) if sup else float("nan")


def block_bootstrap(metric: Callable[[np.ndarray], float], a: np.ndarray, b: np.ndarray, blocks: np.ndarray) -> Dict:
    nb = int(blocks.max()) + 1
    ba = np.stack([a[blocks == i].sum(0) for i in range(nb)]); bb = np.stack([b[blocks == i].sum(0) for i in range(nb)])
    idx = np.random.default_rng(STATISTICS.seed).integers(nb, size=(STATISTICS.resamples, nb))
    d = np.array([metric(ba[i].sum(0)) - metric(bb[i].sum(0)) for i in idx])
    wins = sum(metric(ba[i]) > metric(bb[i]) for i in range(nb) if bb[i].sum() and ba[i].sum())
    return {"delta": metric(a.sum(0)) - metric(b.sum(0)), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))], "blocks": nb, "block_wins": int(wins)}


def semantic_tables(z: Dict, spec: DatasetSpec) -> Dict:
    classes = list(spec.semantic_classes); out = {}
    for region in REGIONS:
        for pop in KEY_POPULATIONS:
            conf = {m: z[f"ps__conf__{region}__{pop}__{m}"] for m in ("B0",) + B3_SEED_METHODS}
            rec = {"B0": semantic_summary(conf["B0"].sum(0), classes, spec.empty_class)}
            seeds = {m: semantic_summary(conf[m].sum(0), classes, spec.empty_class) for m in B3_SEED_METHODS}
            rec["B3_per_seed"] = {m: seeds[m]["miou"] for m in B3_SEED_METHODS}
            rec["B3"] = {"miou": float(np.mean([seeds[m]["miou"] for m in B3_SEED_METHODS])),
                         "overall_accuracy": float(np.mean([seeds[m]["overall_accuracy"] for m in B3_SEED_METHODS])),
                         "mean_class_accuracy": float(np.mean([seeds[m]["mean_class_accuracy"] for m in B3_SEED_METHODS])),
                         "per_class_iou": {c: (float(np.mean([seeds[m]["per_class_iou"][c] for m in B3_SEED_METHODS])) if seeds[B3_SEED_METHODS[0]]["per_class_iou"][c] is not None else None) for c in classes},
                         "gt_classes_predicted": float(np.mean([seeds[m]["gt_classes_predicted"] for m in B3_SEED_METHODS])),
                         "predicted_frequency": {c: float(np.mean([seeds[m]["predicted_frequency"][c] for m in B3_SEED_METHODS])) for c in classes},
                         "gt_voxels": seeds[B3_SEED_METHODS[0]]["gt_voxels"]}
            pooled3 = sum(conf[m] for m in B3_SEED_METHODS)
            metric = lambda cm: track_p_miou(cm, classes) * 100
            rec["delta_B3_minus_B0"] = block_bootstrap(metric, pooled3, 3 * conf["B0"], z["block"])
            rec["delta_per_seed"] = {m: 100 * (seeds[m]["miou"] - rec["B0"]["miou"]) for m in B3_SEED_METHODS}
            sc = [superclass_confusion(c_.sum(0), list(spec.superclass_of), len(spec.superclass_names)) for c_ in (conf["B0"], pooled3)]
            sup_classes = [i for i in range(len(spec.superclass_names)) if i != spec.superclass_empty]
            rec["superclass_miou"] = {"B0": track_p_miou(sc[0], sup_classes), "B3_seed_pooled": track_p_miou(sc[1], sup_classes)}
            out[f"{region}__{pop}"] = rec
    return out


def bins(z: Dict, spec: DatasetSpec, prefix: str, names) -> Dict:
    classes = list(spec.semantic_classes); out = {}
    for b, name in enumerate(names):
        b0 = z[f"pool__{prefix}__B0"][b]; b3 = [z[f"pool__{prefix}__{m}"][b] for m in B3_SEED_METHODS]
        s0 = semantic_summary(b0, classes, spec.empty_class); s3 = [semantic_summary(c, classes, spec.empty_class) for c in b3]
        out[name] = {"voxels": int(b0.sum()), "B0_miou": s0["miou"], "B3_miou": float(np.mean([s["miou"] for s in s3])) if s0["miou"] is not None else None,
                     "B0_accuracy": s0["overall_accuracy"], "B3_accuracy": float(np.mean([s["overall_accuracy"] for s in s3]))}
    return out


def occupancy_tables(z: Dict) -> Dict:
    out = {}
    for src, prefix in (("ours", "ps__occ"), ("occany_raw", "ps__occany_raw"), ("occany_pooled", "ps__occany_pooled")):
        for region in REGIONS:
            for part in ("all", "observed", "unobserved"):
                out[f"{src}__{region}__{part}"] = occupancy_scores(z[f"{prefix}__{region}__{part}"].sum(0))
    out["ours__official"] = occupancy_scores(z["ps__official_occ"].sum(0))
    for tag in ("raw", "pooled"):
        out[f"occany_{tag}__official"] = occupancy_scores(z[f"ps__occany_{tag}__official"].sum(0))
    out["ours__range"] = {REGION.range_names[b]: occupancy_scores(z["pool__occ_range"][b]) for b in range(len(REGION.range_names))}
    for tag in ("raw", "pooled"):
        out[f"occany_{tag}__range"] = {REGION.range_names[b]: occupancy_scores(z[f"ps__occany_{tag}__range"].sum(0)[b]) for b in range(len(REGION.range_names))}
    iou = lambda c: 100 * c[0] / max(c.sum(), 1)
    for tag in ("raw", "pooled"):
        out[f"delta_ours_minus_occany_{tag}__P"] = block_bootstrap(iou, z["ps__occ__P__all"], z[f"ps__occany_{tag}__P__all"], z["block"])
        out[f"delta_ours_minus_occany_{tag}__official"] = block_bootstrap(iou, z["ps__official_occ"], z[f"ps__occany_{tag}__official"], z["block"])
    return out


def official_tables(z: Dict, spec: DatasetSpec) -> Dict:
    out = {}
    for m in ("B0",) + B3_SEED_METHODS:
        conf = z[f"ps__official_conf__{m}"].sum(0)
        r = occany_ssc_miou(conf, spec.n_classes, spec.empty_class, spec.occany_other_class, spec.occany_ignore_other)
        sc = superclass_confusion(conf, list(spec.superclass_of), len(spec.superclass_names))
        rs = occany_ssc_miou(sc, len(spec.superclass_names), spec.superclass_empty, -1, False)
        out[m] = {"miou": r["miou"], "per_class_iou": r["per_class_iou"], "superclass_miou": rs["miou"]}
    out["B3"] = {"miou": float(np.mean([out[m]["miou"] for m in B3_SEED_METHODS])), "superclass_miou": float(np.mean([out[m]["superclass_miou"] for m in B3_SEED_METHODS]))}
    return out


def consistency(z: Dict) -> Dict:
    """Track O and Track P-full read the same predictions: their full-region SSC confusions must be identical."""
    return {m: bool(np.array_equal(z[f"ps__official_conf__{m}"], z[f"ps__conf__full__ssc__{m}"])) for m in ("B0",) + B3_SEED_METHODS}


def analyse() -> Dict:
    freeze.check("protocol"); freeze.check("evaluators")
    rec: Dict = {"datasets": {}}
    for dataset in C.DATASETS:
        spec = DATASET_SPECS[dataset]; z = load_scores(dataset)
        rec["datasets"][dataset] = {"n_samples": int(len(z["sample_id"])), "blocks": int(z["block"].max()) + 1,
                                    "semantic": semantic_tables(z, spec), "occupancy": occupancy_tables(z), "official": official_tables(z, spec),
                                    "distance": bins(z, spec, "dist", EVIDENCE_DISTANCE_NAMES), "range": bins(z, spec, "range", REGION.range_names),
                                    "track_o_equals_track_p_full": consistency(z)}
    gen, meaningful, competitive = {}, {}, {}
    published = dict(PAPER.occany_published_miou)
    for dataset, d in rec["datasets"].items():
        c = d["semantic"][f"P__{CENTRAL}"]
        gen[dataset] = c["delta_B3_minus_B0"]["delta"] > 0 and c["delta_B3_minus_B0"]["ci95"][0] > 0 and all(v > 0 for v in c["delta_per_seed"].values())
        meaningful[dataset] = c["delta_B3_minus_B0"]["delta"] >= PAPER.meaningful_gain_points
        competitive[dataset] = 100 * d["official"]["B3"]["miou"] >= published[dataset]
    decision = "PAPER C" if not all(gen.values()) else ("PAPER A" if all(meaningful.values()) and all(competitive.values()) else "PAPER B")
    rec["criteria"] = {"generalises": gen, "meaningful_gain": meaningful, "competitive_with_published_occany": competitive, "occany_published_miou": published}
    rec["decision"] = decision
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    return rec


__all__ = ["load_scores", "track_p_miou", "block_bootstrap", "analyse"]
