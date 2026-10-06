"""Predeclared rules, frozen in stages. Each stage records the sha256 of predeclaration.md and of the code
that measures it, plus the list of result files that already existed. ``check`` refuses to analyse if any
recorded file changed.

    stage "predeclaration"  before any target statistic: the rule text only
    stage "reliability"     before the full target build: target construction code
    stage "decision"        before trained heads are evaluated: evaluation code
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List

from tools.delayed_semantic import config as C

PREDECLARATION = os.path.join(C.ARTIFACT_DIR, "predeclaration.md")
STAGE_FILES: Dict[str, List[str]] = {
    "predeclaration": [],
    # thresholds are frozen through RELIABILITY_GATE; the analysis code (reliability.py) is not hashed
    "reliability": ["tools/delayed_semantic/config.py", "tools/delayed_semantic/stream.py", "tools/delayed_semantic/targets.py"],
    "addendum_A": ["artifacts/delayed_semantic_pilot/predeclaration_addendum_A.md"],
    "decision": ["artifacts/delayed_semantic_pilot/predeclaration_addendum_A.md",
                 "tools/delayed_semantic/config.py", "tools/delayed_semantic/regions.py", "tools/delayed_semantic/baselines.py",
                 "tools/delayed_semantic/evaluate.py", "tools/delayed_semantic/analysis.py", "tools/semantic_completion/evaluation/semantic_metrics.py"],
}

RELIABILITY_GATE = {
    "G1_registration": {"offsets": [1, 5, 10, 20], "median_max_m": 0.10, "p90_max_m": 0.20},
    "G2_coverage": {"mean_D_per_train_anchor_min": 2000, "classes_min": 8, "voxels_per_class_min": 10000},
    "G3_repeatability": {"train_agreement_min": 0.65, "train_margin_over_shuffle_min": 0.10, "dev_agreement_min": 0.60, "dev_margin_over_shuffle_min": 0.10},
    "G4_reobservation": {"agreement_min": 0.65},
    "tier_B_rule": "tier B enters the delayed loss only if tier B alone meets G3 on the training drives",
    "stop_rule": "any of G1-G4 failing for tier A => Stop: delayed targets are not reliable enough; nothing is trained",
}
DECISION = {
    "primary_region": "completed_never_observed = GT occupied & predicted occupied & no student evidence in H_t",
    "K1": {"metric": "mIoU(B2) - mIoU(B1)", "ci95_lower_above": 0.0},
    "K2": {"metric": "mIoU(B2) - mIoU(B0)", "min_points": 2.0, "ci95_lower_above": 0.0},
    "K3": "predictions on evidence voxels identical across methods",
    "K4": {"min_classes_positive_gain_vs_B0": 5, "gain_without_largest_class_above": 0.0},
    "K5": "ssc_full mIoU(B2) >= ssc_full mIoU(B0)",
    "K6": "occupancy TP/FP/FN identical to the frozen seed-0 counts of Gate 15 source evaluation",
    "bootstrap": {"block_anchors": 20, "resamples": 2000, "seed": 0},
    "outcomes": {"proceed": "K1-K6 all pass", "stop": "reliability gate fails", "revise": "otherwise"},
}


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _path(stage: str) -> str:
    return os.path.join(C.ARTIFACT_DIR, f"frozen_{stage}.json")


def freeze(stage: str) -> Dict:
    p = _path(stage)
    if os.path.exists(p):
        raise RuntimeError(f"{p} exists; stages are frozen once")
    files = [os.path.relpath(PREDECLARATION, C.REPO_ROOT)] + STAGE_FILES[stage]
    existing = {d: sorted(os.listdir(d)) if os.path.isdir(d) else [] for d in (C.Paths().targets_root, C.Paths().eval, C.Paths().checkpoints)}
    rec = {"stage": stage, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "protocol": C.PROTOCOL_VERSION,
           "sha256": {f: _sha(os.path.join(C.REPO_ROOT, f)) for f in files}, "reliability_gate": RELIABILITY_GATE, "decision": DECISION,
           "existing_results": {os.path.basename(d): len(v) for d, v in existing.items()}}
    os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(p, "w"), indent=1)
    return rec


def check(stage: str) -> Dict:
    rec = json.load(open(_path(stage)))
    changed = [f for f, h in rec["sha256"].items() if _sha(os.path.join(C.REPO_ROOT, f)) != h]
    if changed or rec["reliability_gate"] != json.loads(json.dumps(RELIABILITY_GATE)) or rec["decision"] != json.loads(json.dumps(DECISION)):
        raise RuntimeError(f"stage {stage} changed since its freeze: {changed or 'rule dicts'}")
    return rec


__all__ = ["RELIABILITY_GATE", "DECISION", "freeze", "check"]
