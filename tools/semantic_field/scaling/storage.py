"""Where the added drives' caches live. The frozen tools read and write fixed paths under /media/SSD1; for an added drive the
per-drive directory at that fixed path is a SYMLINK to a directory on the chosen tier:

    hot  (NVMe; read by every training step): inputs, evidence, provenance, delayed targets, B3 train-cache student / query
    cold (USB MINH1, ~400 MB/s; read only while preparing): local-reset clips, Trident token maps

Trident tokens live in one flat directory per source (``trident_tokens/k360_train/<key>.npz``), so they are extracted straight to the
cold tier and linked file by file afterwards. Existing drives (0003 / 0006 / 0007 / 0010) are never touched. Every link is recorded in
``artifacts/semantic_field_scaling/prep/storage_layout.json``.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple

from tools.delayed_semantic import config as PILOT, train_cache as TC
from tools.feature_first_3d_semantics import common as FF
from tools.semantic_field.scaling.config import ARTIFACT_DIR
from tools.semantic_field.scaling.registration import ADDED_DRIVES

HOT_ROOTS: Tuple[str, ...] = ("/media/welf/SSD2/MINH_DATASETS/lingbot_b4_scaling/hot", "/media/SSD1/MINH_DATASETS/lingbot_b4_scaling/hot")
COLD_ROOT = "/media/minh/MINH1/lingbot_b4_scaling/cold"
MIN_FREE_GB = 20.0
LAYOUT_PATH = os.path.join(ARTIFACT_DIR, "prep", "storage_layout.json")


def _g15():
    return PILOT.Paths().g15


#: kind -> (function giving the fixed parent directory the frozen tools use, tier)
CACHE_KINDS: Dict[str, tuple] = {
    "clips": (lambda: _g15().clips_train_root, "cold"),
    "inputs": (lambda: _g15().inputs_root, "hot"),
    "evidence": (lambda: _g15().evidence_root, "hot"),
    "provenance": (lambda: PILOT.Paths().provenance_root, "hot"),
    "delayed_targets": (lambda: PILOT.Paths().targets_root, "hot"),
    "train_cache_student": (lambda: os.path.join(TC.CACHE_ROOT, "student"), "hot"),
    "train_cache_query": (lambda: os.path.join(TC.CACHE_ROOT, "query"), "hot"),
}


@dataclass(frozen=True)
class Placement:
    kind: str
    drive: str
    link: str        # fixed path the frozen tools use
    target: str      # real directory on the tier


def free_gb(path: str) -> float:
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


def _load_layout() -> Dict:
    return json.load(open(LAYOUT_PATH)) if os.path.exists(LAYOUT_PATH) else {"placements": {}, "token_links": {}}


def place(kind: str, drive: str, hot_root: str | None = None, need_gb: float = 0.0) -> Placement:
    """Create (or confirm) the tier directory and the symlink for one cache kind of one added drive."""
    if drive not in ADDED_DRIVES:
        raise RuntimeError(f"{drive} is not an added drive; existing caches are never relocated")
    parent_fn, tier = CACHE_KINDS[kind]
    link = os.path.join(parent_fn(), drive)
    if tier == "cold":
        root = COLD_ROOT
    else:
        root = hot_root or HOT_ROOTS[0]
        if root not in HOT_ROOTS:
            raise ValueError(f"{root} is not a declared hot root")
    target = os.path.join(root, kind, drive)
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(target):
            raise RuntimeError(f"{link} already points to {os.readlink(link)}, not {target}")
    else:
        if os.path.exists(link):
            raise RuntimeError(f"{link} exists as a real directory; refusing to replace it")
        if free_gb(root) - need_gb < MIN_FREE_GB:
            raise RuntimeError(f"{root}: {free_gb(root):.0f} GB free, {need_gb:.0f} GB needed + {MIN_FREE_GB:.0f} GB floor")
        os.makedirs(target, exist_ok=True); os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)
    layout = _load_layout(); layout["placements"][f"{kind}/{drive}"] = {"link": link, "target": target, "tier": tier}
    os.makedirs(os.path.dirname(LAYOUT_PATH), exist_ok=True); json.dump(layout, open(LAYOUT_PATH, "w"), indent=1)
    return Placement(kind, drive, link, target)


def cold_token_dir() -> str:
    return os.path.join(COLD_ROOT, "trident_tokens")


def link_tokens(drive: str) -> int:
    """After extraction to the cold tier: symlink every token of ``drive`` into the fixed flat token directory. Returns the count."""
    if drive not in ADDED_DRIVES:
        raise RuntimeError(f"{drive} is not an added drive")
    src_dir = os.path.join(cold_token_dir(), "k360_train"); dst_dir = os.path.join(FF.TOKEN_DIR, "k360_train")
    prefix = drive[17:21] + "_"; n = 0
    for f in sorted(os.listdir(src_dir)):
        if not (f.startswith(prefix) and f.endswith(".npz")):
            continue
        dst = os.path.join(dst_dir, f)
        if os.path.lexists(dst):
            if not os.path.islink(dst) or os.path.realpath(dst) != os.path.realpath(os.path.join(src_dir, f)):
                raise RuntimeError(f"{dst} exists and is not our link")
        else:
            os.symlink(os.path.join(src_dir, f), dst)
        n += 1
    layout = _load_layout(); layout["token_links"][drive] = {"count": n, "from": src_dir, "into": dst_dir}
    json.dump(layout, open(LAYOUT_PATH, "w"), indent=1)
    return n


__all__ = ["HOT_ROOTS", "COLD_ROOT", "MIN_FREE_GB", "LAYOUT_PATH", "CACHE_KINDS", "Placement", "free_gb", "place", "cold_token_dir", "link_tokens"]
