"""The predeclared promotion rule, frozen before any trained head is evaluated, and its guard.

``write_promotion_rule`` records the thresholds together with the sha256 of every file that decides
what is measured (evidence, B0, regions, metrics, evaluator, config). ``check_promotion_rule`` refuses
an analysis whose rule file or measuring code changed after the freeze.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List

import yaml

from tools.semantic_completion import config as C

RULE_PATH = os.path.join(C.ARTIFACT_DIR, "promotion_rule.json")
MEASURING_CODE: List[str] = [
    "tools/semantic_completion/config.py", "tools/semantic_completion/evidence/trident.py", "tools/semantic_completion/evidence/lift.py",
    "tools/semantic_completion/evidence/build.py", "tools/semantic_completion/frozen_geometry.py", "tools/semantic_completion/dataset.py",
    "tools/semantic_completion/evaluation/baseline_nearest.py", "tools/semantic_completion/evaluation/regions.py",
    "tools/semantic_completion/evaluation/semantic_metrics.py", "tools/semantic_completion/evaluation/evaluate.py",
]

RULE: Dict = {
    "primary_region": "completed_unobserved = GT occupied (labelled) & frozen predicted occupied & no direct causal evidence",
    "statistic": "per seed: mIoU of the confusion pooled over all 590 anchors of 0006; decisions use the mean over seeds 0/1/2",
    "P1_unobserved_gain": {"metric": "mIoU(B1) - mIoU(B0) on completed_unobserved", "mean_over_seeds_at_least": 0.02,
                           "all_seeds_positive": True, "paired_clip_block_bootstrap_ci95_lower_above": 0.0, "blocks": 20, "resamples": 2000},
    "P2_overall_ssc_gain": {"metric": "SSC mIoU(B1) - SSC mIoU(B0) on ssc_full under frozen predicted occupancy", "mean_over_seeds_above": 0.0, "all_seeds_positive": True},
    "P3_breadth_no_collapse": {"min_classes_with_positive_mean_iou_gain": 5, "b1_dominant_pred_fraction_at_most": "max(0.60, GT dominant fraction + 0.10)",
                               "b1_min_classes_with_ge_1pct_of_predictions": 5, "region": "completed_unobserved"},
    "P4_occupancy_unchanged": "per seed, occupancy TP/FP/FN over the 590 anchors (Gate 12 regions 'full') equal the frozen baseline's source_eval counts exactly; geometry state digests unchanged by training",
    "decision_order": [
        "invalid (stop) if P4 fails",
        "D if the teacher readout of the evidence on oracle_observed has overall accuracy < 0.40 or mIoU < 0.15",
        "A if P1, P2, P3 and P4 all pass",
        "C if P1 fails and the collapse clause of P3 fails (dominant fraction above its limit or < 5 classes with >= 1 % of predictions)",
        "B otherwise",
    ],
    "B0": "nearest evidence voxel label (3D Euclidean, no radius); evaluation/baseline_nearest.py",
    "checkpoint_rule": "final step 6000 of each head seed, no selection",
    "disclosure": "The Decision-D thresholds were written after a 10-anchor smoke test of the evaluator with an untrained (20-step) head, "
                  "which showed a teacher readout of ~0.23 mIoU / 0.64 accuracy on observed voxels; the thresholds were the ones considered "
                  "before that run and are not tuned to it. No trained-head result existed when this rule was frozen.",
}


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_promotion_rule() -> Dict:
    if os.path.exists(RULE_PATH):
        raise RuntimeError(f"{RULE_PATH} exists; the rule is frozen once")
    rec = {"frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "protocol": C.PROTOCOL_VERSION, "rule": RULE,
           "measuring_code_sha256": {f: sha256_file(os.path.join(C.REPO_ROOT, f)) for f in MEASURING_CODE},
           "trained_head_results_existing_at_freeze": sorted(os.listdir(C.Paths().source_eval)) if os.path.isdir(C.Paths().source_eval) else []}
    os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(RULE_PATH, "w"), indent=1)
    cfg = {"protocol": C.PROTOCOL_VERSION, "paths": dataclasses.asdict(C.Paths()), "head": dataclasses.asdict(C.HeadConfig()),
           "train": dataclasses.asdict(C.TrainConfig()), "train_drives": list(C.TRAIN_DRIVES), "val_drive": C.VAL_DRIVE, "seeds": list(C.SEEDS),
           "teacher": {"model": "Trident-H (frozen), CLIP ViT-H-14 token feature", "dim": C.TEACHER_DIM, "token_grid_hw": list(C.TOKEN_GRID_HW),
                       "normalisation": "unit L2 per token; lifted voxel feature = normalize(count-weighted mean of unit ray features)",
                       "text": "Trident query_features (CLIP text, ImageNet templates), 18 KITTI-360 classes, unit norm; readout argmax cosine"},
           "evidence": "grid voxels whose world key holds >= 1 accepted causal ray endpoint and that the frozen hard map marks occupied",
           "occupancy": "frozen Current Geometry-Only Baseline, final log-odds > 0", "distance_edges_m": list(C.DISTANCE_EDGES_M)}
    yaml.safe_dump(cfg, open(os.path.join(C.ARTIFACT_DIR, "semantic_config.yaml"), "w"), sort_keys=False, width=120)
    return rec


def check_promotion_rule() -> Dict:
    rec = json.load(open(RULE_PATH))
    changed = [f for f, h in rec["measuring_code_sha256"].items() if sha256_file(os.path.join(C.REPO_ROOT, f)) != h]
    if rec["rule"] != json.loads(json.dumps(RULE)) or changed:
        raise RuntimeError(f"promotion rule or measuring code changed since the freeze: {changed or 'rule text'}")
    return rec


__all__ = ["RULE_PATH", "MEASURING_CODE", "RULE", "write_promotion_rule", "check_promotion_rule"]
