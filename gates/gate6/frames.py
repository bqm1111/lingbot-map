"""The unique-RGB-frame index of each benchmark, and where its pixels live.

Trident is run **once per unique rectified frame**, not once per clip: the five-frame
clips overlap heavily (Occ3D slides by one keyframe, KITTI-360 anchors are shared between
consecutive clips), so caching per frame is what makes the gate affordable and also what
guarantees that two clips sharing a frame see identical semantics.

Nothing here opens a target, a LiDAR sweep or an oracle table -- only manifests and image
files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

SEMANTICKITTI_ROOT = "data/kitti/dataset"
OCC3D_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes"
KITTI360_ROOT = "/media/welf/MINH/datasets/kitti360/KITTI-360"
KITTI360_SEQUENCE = "2013_05_28_drive_0006_sync"

MANIFESTS = {
    "semantickitti": "artifacts/scale_gate/manifests/val.jsonl",
    "occ3d": "manifests/occ3d_zeroshot/val.jsonl",
    "kitti360": "manifests/gate5_2/val.jsonl",
}

LINGBOT_CACHE = {
    "semantickitti": "artifacts/scale_gate/cache/lingbot",
    "occ3d": "/media/SSD1/MINH_DATASETS/lingbot_occ3d_zeroshot/cache_lingbot",
    "kitti360": "/media/SSD1/MINH_DATASETS/lingbot_gate5_2/cache_lingbot",
}

# Where the semantic tensors live. '/' is full on this machine; the bulk caches of every
# previous gate already sit on SSD1 for the same reason.
SEMANTIC_CACHE_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics"


def _repo(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def image_root(dataset: str, repo_root: str) -> str:
    if dataset == "semantickitti":
        return os.path.join(repo_root, SEMANTICKITTI_ROOT)
    if dataset == "occ3d":
        return OCC3D_ROOT
    if dataset == "kitti360":
        return os.path.join(KITTI360_ROOT, "data_2d_raw", KITTI360_SEQUENCE)
    raise KeyError(dataset)


def frame_key(dataset: str, rel_path: str) -> str:
    """A stable, filesystem-safe id for one rectified frame."""
    stem = rel_path.replace("\\", "/")
    if dataset == "semantickitti":                       # sequences/08/image_2/000000.png
        parts = stem.split("/")
        return f"{parts[1]}_{os.path.splitext(parts[-1])[0]}"
    if dataset == "kitti360":                            # image_00/data_rect/0000000137.png
        return os.path.splitext(os.path.basename(stem))[0]
    return os.path.splitext(os.path.basename(stem))[0]   # nuScenes filenames are unique


@dataclass
class ClipRecord:
    clip_id: str
    rel_images: Tuple[str, ...]
    keys: Tuple[str, ...]
    raw: dict


def read_manifest(dataset: str, repo_root: str) -> List[ClipRecord]:
    path = _repo(MANIFESTS[dataset], repo_root)
    out: List[ClipRecord] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rel = tuple(d["image_paths"])
            out.append(ClipRecord(clip_id=d["clip_id"], rel_images=rel,
                                  keys=tuple(frame_key(dataset, r) for r in rel), raw=d))
    return out


def unique_frames(dataset: str, repo_root: str) -> List[Tuple[str, str]]:
    """``[(frame_key, absolute image path)]``, sorted by key -- deterministic everywhere."""
    root = image_root(dataset, repo_root)
    seen: Dict[str, str] = {}
    for rec in read_manifest(dataset, repo_root):
        for rel, key in zip(rec.rel_images, rec.keys):
            seen.setdefault(key, os.path.join(root, rel))
    return sorted(seen.items())


def semantic_cache_dir(dataset: str, root: str = SEMANTIC_CACHE_ROOT) -> str:
    return os.path.join(root, dataset)


def semantic_cache_path(dataset: str, key: str, root: str = SEMANTIC_CACHE_ROOT) -> str:
    return os.path.join(semantic_cache_dir(dataset, root), f"{key}.npz")


def lingbot_cache_path(dataset: str, clip_id: str, repo_root: str) -> str:
    return os.path.join(_repo(LINGBOT_CACHE[dataset], repo_root), f"{clip_id}.npz")


def shard(items: Sequence, index: int, count: int) -> List:
    """Deterministic contiguous-stride sharding; the union over shards is the whole list."""
    if count <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % count == index]


__all__ = ["ClipRecord", "read_manifest", "unique_frames", "frame_key", "image_root",
           "semantic_cache_dir", "semantic_cache_path", "lingbot_cache_path", "shard",
           "MANIFESTS", "LINGBOT_CACHE", "SEMANTIC_CACHE_ROOT", "KITTI360_SEQUENCE"]
