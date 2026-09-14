"""The source-side freeze: what is hashed, when it is written, and the guard that reads it.

``write_source_freeze`` runs once, after the KITTI-360 result is confirmed and before any
target dataset is opened. It hashes every file the zero-shot evaluation will execute or
consume (this package, the core modules it imports, the frozen LingBot/MoGe code, the
recipe, the three final checkpoints) and records the timestamp. ``require_source_freeze``
is called by the target-evaluation entry points: it refuses to run if the freeze record
is missing or if any hashed file has changed since -- so a target result can never come
from code that differs from the frozen one without the record saying so.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from typing import Dict, List

from tools.baseline_reconstruction import config as C

FREEZE_PATH = os.path.join(C.ARTIFACT_DIR, "code_freeze.json")

#: repository files whose behaviour the frozen system depends on (relative to REPO_ROOT)
CORE_FILES: List[str] = [
    "core/mapping/mapper.py", "core/mapping/depth.py", "core/mapping/rays.py", "core/mapping/voxmap.py", "core/mapping/scale.py",
    "core/model/net.py", "core/model/losses.py", "core/model/sampler.py", "core/run/_gate8a_train.py", "core/run/_gate8_train.py",
    "core/datasets/sources.py", "core/datasets/streams.py", "core/datasets/frames.py", "core/datasets/grids.py",
    "core/datasets/occ3d_grid.py", "core/datasets/occupancy.py", "core/datasets/kitti360_splits.py", "core/datasets/audit.py",
    "core/evaluation/official_targets.py", "core/evaluation/regions.py", "core/datasets/occ_datasets.py", "core/datasets/nuscenes.py",
    "core/datasets/kitti_odometry.py", "core/supervision/rawtarget.py",
    "gates/gate7b/replay.py", "gates/gate7b/scale.py", "gates/gate7b/depth.py",
    "checkpoints/lingbot-map/204754b/lingbot-map.pt",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: package parts that only *render* results after the fact and never touch a prediction or a metric;
#: they may be edited after the freeze without invalidating the evaluation
NOT_EVALUATION_CODE = ("tools/baseline_reconstruction/reports/", "tools/baseline_reconstruction/historical/",
                       "tools/baseline_reconstruction/cli/audit_history.py", "tools/baseline_reconstruction/cli/report.py",
                       "tools/baseline_reconstruction/README.md")


def package_files(evaluation_only: bool = True) -> List[str]:
    root = os.path.join(C.REPO_ROOT, "tools", "baseline_reconstruction")
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py") or f.endswith(".md"):
                rel = os.path.relpath(os.path.join(d, f), C.REPO_ROOT)
                if evaluation_only and rel.startswith(NOT_EVALUATION_CODE):
                    continue
                out.append(rel)
    return sorted(out)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", C.REPO_ROOT, *args], text=True).strip()
    except subprocess.CalledProcessError:
        return ""


def frozen_file_list(paths: C.CurrentPaths) -> List[str]:
    files = package_files() + CORE_FILES + [os.path.relpath(paths.training_config, C.REPO_ROOT)]
    files += [os.path.relpath(paths.checkpoint_path(s, 6000), C.REPO_ROOT) for s in C.SEEDS]
    return files


def hash_files(files: List[str]) -> Dict[str, str]:
    return {f: sha256_file(os.path.join(C.REPO_ROOT, f)) for f in files}


def write_source_freeze(paths: C.CurrentPaths, source_summary: Dict, reason: str) -> Dict:
    """Write code_freeze.json. Refuses to overwrite an existing freeze."""
    if os.path.exists(FREEZE_PATH):
        raise RuntimeError(f"{FREEZE_PATH} exists; the freeze is written once. Delete it deliberately to re-freeze.")
    files = frozen_file_list(paths)
    rec = {"experiment": C.PROTOCOL_VERSION, "frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "git_head": _git("rev-parse", "HEAD"), "git_status_dirty_files": len(_git("status", "--porcelain").splitlines()),
           "reason": reason, "sha256": hash_files(files),
           "frozen_decisions": {"seeds": list(C.SEEDS), "checkpoint_rule": "final step 6000 of every seed; no selection",
                                "occupancy_threshold": C.OCCUPANCY_THRESHOLD, "seed_statistic": "mean over seeds primary; median secondary",
                                "train_drives": list(C.TRAIN_DRIVES), "val_drive": C.VAL_DRIVE, "clip_offsets": list(C.CLIP_OFFSETS),
                                "target_protocols": {k: v.__dict__ for k, v in C.TARGET_PROTOCOLS.items()}},
           "excluded_from_freeze_render_only": list(NOT_EVALUATION_CODE),
           "source_summary": source_summary,
           "target_datasets_opened_before_freeze": False}
    os.makedirs(os.path.dirname(FREEZE_PATH), exist_ok=True)
    json.dump(rec, open(FREEZE_PATH, "w"), indent=1)
    return rec


class SourceFreezeMissing(RuntimeError):
    pass


class SourceFreezeViolated(RuntimeError):
    pass


def require_source_freeze() -> Dict:
    """Load the freeze and verify every hashed file is unchanged. Raises otherwise."""
    if not os.path.exists(FREEZE_PATH):
        raise SourceFreezeMissing(f"target evaluation locked: {FREEZE_PATH} does not exist (run cli/freeze_source.py after the source result is confirmed)")
    rec = json.load(open(FREEZE_PATH))
    changed = [f for f, h in rec["sha256"].items() if not os.path.exists(os.path.join(C.REPO_ROOT, f)) or sha256_file(os.path.join(C.REPO_ROOT, f)) != h]
    if changed:
        raise SourceFreezeViolated("target evaluation locked: files changed since the source freeze: " + ", ".join(changed))
    return rec


__all__ = ["FREEZE_PATH", "CORE_FILES", "NOT_EVALUATION_CODE", "sha256_file", "package_files", "frozen_file_list", "hash_files",
           "write_source_freeze", "SourceFreezeMissing", "SourceFreezeViolated", "require_source_freeze"]
