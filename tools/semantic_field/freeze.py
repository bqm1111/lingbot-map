"""Hash freezes for the semantic-field experiment. Each stage records sha256 of its files, an optional payload, and what
checkpoints / evaluation results / target predictions already existed.

    source_predeclaration  before any B4 statistic: rule text + configuration
    centering              after mu is computed from training drives, before training
    source_code            before source evaluation of trained B4 heads: evaluation / analysis code
    final                  after SOURCE PASS only, before any target access: code, checkpoints, mu, prompts, protocols
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from typing import Dict, List, Optional

from tools.semantic_field.config.schema import ARTIFACT_DIR, REPO_ROOT, Paths

PRE = "artifacts/semantic_field/source_predeclaration.md"
STAGE_FILES: Dict[str, List[str]] = {
    "source_predeclaration": [PRE, "tools/semantic_field/config/schema.py"],
    "centering": [PRE, "tools/semantic_field/config/schema.py", "tools/semantic_field/data/feature_mean.py"],
    "source_code": [PRE, "tools/semantic_field/config/schema.py", "tools/semantic_field/model/semantic_field_head.py",
                    "tools/semantic_field/data/semantic_evidence.py", "tools/semantic_field/evaluation/predict.py",
                    "tools/semantic_field/evaluation/evaluate_source.py", "tools/semantic_field/evaluation/analysis.py",
                    "tools/delayed_semantic/regions.py", "tools/delayed_semantic/gate16/candidate_regions.py"],
}


def _sha(rel: str) -> str:
    return hashlib.sha256(open(os.path.join(REPO_ROOT, rel), "rb").read()).hexdigest()


def _path(stage: str) -> str:
    return os.path.join(ARTIFACT_DIR, f"frozen_{stage}.json")


def freeze(stage: str, payload: Optional[Dict] = None, extra_files: Optional[List[str]] = None) -> Dict:
    if os.path.exists(_path(stage)):
        raise RuntimeError(f"{_path(stage)} exists; stages are frozen once")
    p = Paths()
    files = STAGE_FILES.get(stage, []) + list(extra_files or [])
    rec = {"stage": stage, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "sha256": {f: _sha(f) for f in files}, "payload": payload or {},
           "existing": {"checkpoints": len(glob.glob(os.path.join(p.checkpoints, "*.pt"))), "eval": len(glob.glob(os.path.join(p.eval, "*.npz"))),
                        "target_predictions": len(glob.glob(os.path.join(p.predictions, "semantickitti", "*"))) + len(glob.glob(os.path.join(p.predictions, "occ3d", "*")))}}
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(_path(stage), "w"), indent=1)
    return rec


def check(stage: str) -> Dict:
    rec = json.load(open(_path(stage)))
    changed = [f for f, h in rec["sha256"].items() if _sha(f) != h]
    if changed:
        raise RuntimeError(f"semantic-field stage {stage} changed since its freeze: {changed}")
    return rec


def is_frozen(stage: str) -> bool:
    return os.path.exists(_path(stage))


def require_source_pass() -> Dict:
    """Target evaluation guard: a SOURCE PASS promotion record and the final freeze must both exist and verify."""
    promo = os.path.join(ARTIFACT_DIR, "source_promotion.json")
    if not os.path.exists(promo) or json.load(open(promo)).get("decision") != "SOURCE PASS":
        raise RuntimeError("target datasets are locked: no SOURCE PASS promotion record")
    return check("final")


__all__ = ["STAGE_FILES", "freeze", "check", "is_frozen", "require_source_pass"]
