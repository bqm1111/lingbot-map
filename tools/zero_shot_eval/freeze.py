"""Hash freezes for the final evaluation. A stage records sha256 of its files and which predictions / scores existed.

    protocol     before any target semantic prediction is scored: predeclaration + protocol constants
    evaluators   before scoring: canonical prediction format, masks, metrics, evaluators, analysis
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from typing import Dict, List

from tools.zero_shot_eval import config as C

STAGE_FILES: Dict[str, List[str]] = {
    "protocol": ["artifacts/final_zero_shot/practical_evaluation_predeclaration.md", "tools/zero_shot_eval/practical/protocol.py"],
    "evaluators": ["artifacts/final_zero_shot/practical_evaluation_predeclaration.md", "tools/zero_shot_eval/practical/protocol.py",
                   "tools/zero_shot_eval/canonical/prediction.py", "tools/zero_shot_eval/canonical/metrics.py", "tools/zero_shot_eval/practical/masks.py",
                   "tools/zero_shot_eval/practical/evaluator.py", "tools/zero_shot_eval/official/evaluator.py", "tools/zero_shot_eval/analysis.py"],
}


def _sha(rel: str) -> str:
    return hashlib.sha256(open(os.path.join(C.REPO_ROOT, rel), "rb").read()).hexdigest()


def _path(stage: str) -> str:
    return os.path.join(C.ARTIFACT_DIR, f"frozen_{stage}.json")


def freeze(stage: str) -> Dict:
    if os.path.exists(_path(stage)):
        raise RuntimeError(f"{_path(stage)} exists; stages are frozen once")
    existing = {d: len(glob.glob(os.path.join(C.PREDICTION_ROOT, d, "*.npz"))) for d in C.DATASETS}
    scores = len(glob.glob(os.path.join(C.ARTIFACT_DIR, "scores", "*")))
    rec = {"stage": stage, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "sha256": {f: _sha(f) for f in STAGE_FILES[stage]}, "existing_predictions": existing, "existing_scores": scores}
    os.makedirs(C.ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(_path(stage), "w"), indent=1)
    return rec


def check(stage: str) -> Dict:
    rec = json.load(open(_path(stage)))
    changed = [f for f, h in rec["sha256"].items() if _sha(f) != h]
    if changed:
        raise RuntimeError(f"stage {stage} changed since its freeze: {changed}")
    return rec


__all__ = ["STAGE_FILES", "freeze", "check"]
