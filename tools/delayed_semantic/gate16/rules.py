"""Gate 16 freezes. Each stage records sha256 of its files (and an optional payload) plus what results already existed.

    predeclaration   before any B3 model or reconstructed ceiling: the rule text
    ceiling          after the candidate ceiling is reconstructed, before any B3 score: the numeric thresholds
    code             before B3 evaluation: evaluation / analysis code
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List, Optional

from tools.delayed_semantic.gate16 import config as G

PRE = "artifacts/gate16_candidate_selection/predeclaration.md"
STAGE_FILES: Dict[str, List[str]] = {
    "predeclaration": [PRE, "tools/delayed_semantic/gate16/config.py"],
    "ceiling": [PRE, "artifacts/gate16_candidate_selection/candidate_ceiling.json", "tools/delayed_semantic/gate16/ceiling.py",
                "tools/delayed_semantic/gate16/candidates.py", "tools/delayed_semantic/gate16/candidate_regions.py"],
    "code": [PRE, "artifacts/gate16_candidate_selection/candidate_ceiling.json", "tools/delayed_semantic/gate16/candidates.py",
             "tools/delayed_semantic/gate16/candidate_targets.py", "tools/delayed_semantic/gate16/candidate_regions.py",
             "tools/delayed_semantic/gate16/candidate_metrics.py", "tools/delayed_semantic/gate16/evaluate.py",
             "tools/delayed_semantic/gate16/analysis.py", "tools/delayed_semantic/regions.py", "tools/delayed_semantic/model.py"],
}


def _sha(rel: str) -> str:
    return hashlib.sha256(open(os.path.join(G.REPO_ROOT, rel), "rb").read()).hexdigest()


def _path(stage: str) -> str:
    return os.path.join(G.ARTIFACT_DIR, f"frozen_{stage}.json")


def freeze(stage: str, payload: Optional[Dict] = None) -> Dict:
    if os.path.exists(_path(stage)):
        raise RuntimeError(f"{_path(stage)} exists; stages are frozen once")
    existing = {os.path.basename(d): len(os.listdir(d)) if os.path.isdir(d) else 0 for d in (G.CHECKPOINTS, G.EVAL)}
    rec = {"stage": stage, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "protocol": G.PROTOCOL_VERSION,
           "sha256": {f: _sha(f) for f in STAGE_FILES[stage]}, "payload": payload or {}, "existing_results": existing}
    os.makedirs(G.ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(_path(stage), "w"), indent=1)
    return rec


def check(stage: str) -> Dict:
    rec = json.load(open(_path(stage)))
    changed = [f for f, h in rec["sha256"].items() if _sha(f) != h]
    if changed:
        raise RuntimeError(f"Gate 16 stage {stage} changed since its freeze: {changed}")
    return rec


__all__ = ["STAGE_FILES", "freeze", "check"]
