"""Source pools and split guards.

``current``  = every cached anchor of the frozen training drives 0003 / 0007 / 0010 (``G15B.anchor_ids``), the original pool.
``expanded`` = ``current`` + the anchors listed in the frozen expanded manifest (``manifests/source_pool_expanded.json``).
Anchor order is (drive order, then t ascending) -- it indexes the samplers, so it is part of the experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from core.datasets import kitti360 as K3
from tools.semantic_completion.evidence import build as G15B
from tools.semantic_field.config.schema import DEV_DRIVE, TRAIN_DRIVES
from tools.semantic_field.scaling.anchor_store import AnchorKey
from tools.semantic_field.scaling.config import POOLS, StudyPaths

#: the only drives that may ever contribute training anchors: the official SSCBench-KITTI-360 training split
ALLOWED_TRAIN_DRIVES: Tuple[str, ...] = tuple(K3.OFFICIAL_SPLIT["train"])
RESERVED_DRIVES: Tuple[str, ...] = (DEV_DRIVE,) + tuple(K3.OFFICIAL_SPLIT["test"])     # 0006 (development), 0009 (reserved)


class SourcePoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourcePool:
    name: str
    anchors: Tuple[AnchorKey, ...]

    def digest(self) -> str:
        return hashlib.sha256(json.dumps([list(a) for a in self.anchors]).encode()).hexdigest()

    def drives(self) -> List[str]:
        return sorted({d for d, _ in self.anchors})


def check_membership(anchors: Tuple[AnchorKey, ...]) -> None:
    """Raise if any anchor comes from outside the training split, from 0006 / 0009, or is duplicated."""
    bad = [a for a in anchors if a[0] not in ALLOWED_TRAIN_DRIVES or a[0] in RESERVED_DRIVES]
    if bad:
        raise SourcePoolError(f"{len(bad)} anchors outside the training split or from a reserved drive, e.g. {bad[:3]}")
    if len(set(anchors)) != len(anchors):
        raise SourcePoolError("duplicate anchors in the pool")


def current_pool() -> SourcePool:
    anchors = tuple((d, int(t)) for d in TRAIN_DRIVES for t in G15B.anchor_ids(d))
    check_membership(anchors)
    return SourcePool("current", anchors)


def expanded_manifest_path(paths: StudyPaths = StudyPaths()) -> str:
    return os.path.join(paths.manifests, "source_pool_expanded.json")


def expanded_pool(paths: StudyPaths = StudyPaths()) -> SourcePool:
    """The frozen expanded pool. Refuses to build one from anything but the frozen manifest."""
    path = expanded_manifest_path(paths)
    if not os.path.exists(path):
        raise SourcePoolError(f"{path} does not exist: the expanded pool is defined only by the frozen manifest")
    rec = json.load(open(path))
    added = tuple((d, int(t)) for d in rec["added_drives"] for t in rec["added_anchors"][d])
    pool = SourcePool("expanded", current_pool().anchors + added)
    check_membership(pool.anchors)
    if pool.digest() != rec["expanded_pool_sha256"] or len(pool.anchors) != rec["n_expanded"]:
        raise SourcePoolError(f"{path}: pool digest / size do not match the frozen record")
    return pool


def pool_by_name(name: str, paths: StudyPaths = StudyPaths()) -> SourcePool:
    if name not in POOLS:
        raise SourcePoolError(f"unknown pool {name!r}; expected one of {POOLS}")
    return current_pool() if name == "current" else expanded_pool(paths)


__all__ = ["ALLOWED_TRAIN_DRIVES", "RESERVED_DRIVES", "SourcePoolError", "SourcePool", "check_membership", "current_pool",
           "expanded_manifest_path", "expanded_pool", "pool_by_name"]
