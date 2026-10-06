"""Explicit registration of the added training drives 0000 / 0002 / 0004 / 0005 for the unchanged cache tools.

The frozen tools enumerate training drives from module tuples (``kitti360_folds.K360_TRAIN_DRIVES`` for the k360_train stream;
``gate12_suf.g12.TRAIN_DRIVES`` for the clip generator's guard). ``register_added_drives`` appends the four drives to those tuples
in the CURRENT PROCESS only, after checking that every added drive is in the official SSCBench training split and is neither the
development drive 0006 nor the reserved drive 0009. Stream, eligibility, target and evidence rules are not touched.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from core.datasets import kitti360 as K3, kitti360_folds as KF, kitti360_splits as SRC

ADDED_DRIVES: Tuple[str, ...] = tuple(f"2013_05_28_drive_{d}_sync" for d in ("0000", "0002", "0004", "0005"))
RESERVED: Tuple[str, ...] = (SRC.VAL_DRIVE, "2013_05_28_drive_0009_sync")


def check_added_drives(drives: Tuple[str, ...] = ADDED_DRIVES) -> None:
    for d in drives:
        if d not in K3.OFFICIAL_SPLIT["train"] or d in RESERVED:
            raise RuntimeError(f"{d} is not an official training drive or is reserved; refusing to register it")


def register_added_drives(drives: Tuple[str, ...] = ADDED_DRIVES) -> Tuple[str, ...]:
    """Idempotent. Returns the k360_train drive tuple after registration."""
    check_added_drives(drives)
    KF.K360_TRAIN_DRIVES = tuple(KF.K360_TRAIN_DRIVES) + tuple(d for d in drives if d not in KF.K360_TRAIN_DRIVES)
    try:                                                   # the Gate 12 clip-generator guard (tools/gate12_suf is a script dir)
        import g12
        g12.TRAIN_DRIVES = tuple(g12.TRAIN_DRIVES) + tuple(d for d in drives if d not in g12.TRAIN_DRIVES)
    except ImportError:
        pass
    return KF.K360_TRAIN_DRIVES


def kept_stream_indices(manifest_path: str) -> Dict[str, List[int]]:
    """Kept anchors per added drive from the expanded-pool manifest (draft before the freeze, frozen after)."""
    rec = json.load(open(manifest_path))
    kept = {d: [int(t) for t in rec["added_anchors"][d]] for d in rec["added_drives"]}
    check_added_drives(tuple(kept))
    return kept


def default_manifest_path() -> str:
    from tools.semantic_field.scaling.config import StudyPaths
    frozen = os.path.join(StudyPaths().manifests, "source_pool_expanded.json")
    return frozen if os.path.exists(frozen) else os.path.join(StudyPaths().manifests, "source_pool_expanded.DRAFT.json")


__all__ = ["ADDED_DRIVES", "RESERVED", "check_added_drives", "register_added_drives", "kept_stream_indices", "default_manifest_path"]
