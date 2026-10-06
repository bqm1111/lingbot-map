"""Hash freezes of the scaling study (same convention as ``tools/semantic_field/freeze.py``). A stage is frozen once.

    source      before any scaling training step: declaration, manifest, scaling code, frozen configs, storage layout, cache
                identities (sha256 of every cache file the two methods read for every anchor of both pools) and budgets
    evaluation  after training, before source evaluation: evaluator adapter, analysis code, final checkpoints
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from tools.semantic_field.scaling.config import ARTIFACT_DIR, METHODS, POOLS, SEEDS, REPO_ROOT, StudyPaths

DECLARATION = "artifacts/semantic_field_scaling/source_declaration.md"
SOURCE_FILES: List[str] = [DECLARATION, "artifacts/semantic_field_scaling/manifests/source_pool_expanded.json",
                           "artifacts/semantic_field_scaling/prep/storage_layout.json", "artifacts/semantic_field_scaling/prep/reliability_expanded.json",
                           "tools/semantic_field/config/schema.py", "tools/delayed_semantic/config.py", "tools/delayed_semantic/gate16/config.py",
                           "tools/semantic_field/training/trainer.py", "tools/delayed_semantic/train.py", "tools/delayed_semantic/model.py",
                           "tools/semantic_field/model/semantic_field_head.py", "tools/semantic_field/losses/semantic_loss.py",
                           "tools/semantic_field/data/semantic_evidence.py"] + sorted(
    os.path.relpath(p, REPO_ROOT) for p in glob.glob(os.path.join(REPO_ROOT, "tools", "semantic_field", "scaling", "*.py")))
EVALUATION_FILES: List[str] = ["tools/semantic_field/scaling/evaluate.py", "tools/semantic_field/scaling/analysis.py",
                               "tools/semantic_field/evaluation/evaluate_source.py", "tools/semantic_field/evaluation/analysis.py"]


def sha256(path: str) -> str:
    """sha256 of a file; its page cache is released afterwards (hashing ~400 GB must not exhaust free memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def hash_many(paths: List[str], workers: int = 16) -> Dict[str, str]:
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(paths, ex.map(sha256, paths)))


def _stage_path(stage: str) -> str:
    return os.path.join(ARTIFACT_DIR, f"frozen_{stage}.json")


def freeze(stage: str, files: List[str], payload: Dict) -> Dict:
    path = _stage_path(stage)
    if os.path.exists(path):
        raise RuntimeError(f"{path} exists; stages are frozen once")
    rec = {"stage": stage, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "sha256": {f: sha256(os.path.join(REPO_ROOT, f)) for f in files}, "payload": payload,
           "existing_final_checkpoints": sorted(glob.glob(os.path.join(ARTIFACT_DIR, "runs", "*", "*_step*.pt")))}
    json.dump(rec, open(path, "w"), indent=1)
    return rec


def check(stage: str) -> Dict:
    rec = json.load(open(_stage_path(stage)))
    changed = [f for f, h in rec["sha256"].items() if sha256(os.path.join(REPO_ROOT, f)) != h]
    if changed:
        raise RuntimeError(f"scaling stage {stage} changed since its freeze: {changed}")
    return rec


def final_checkpoints(steps: Dict[str, int], paths: StudyPaths = StudyPaths()) -> List[str]:
    return [paths.final_checkpoint(m, p, s, steps[m]) for m in METHODS for p in POOLS for s in SEEDS]


__all__ = ["DECLARATION", "SOURCE_FILES", "EVALUATION_FILES", "sha256", "hash_many", "freeze", "check", "final_checkpoints"]
