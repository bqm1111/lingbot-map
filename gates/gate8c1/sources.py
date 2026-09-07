"""The Gate 8C-1 dataset firewall, drives, anchors and splits.

KITTI-360 is the only dataset any training or selection code may touch. The firewall is
not a convention here: :func:`assert_no_target_access` is called by every training and
selection tool, and ``tests/gate8c1`` fails if a target-dataset name appears in any of
their import closures.

Splits follow the brief and Gate 8B's official partition:

* **training** drives 0003, 0007, 0010 (official SSCBench train drives);
* **source validation** drive 0006 (official SSCBench val drive).

Anchors are every frame of the frozen LingBot stream that has five preceding frames (so
the scale anchor is complete) and a usable future window -- not only the frames SSCBench
happened to label, since Gate 8C-1 does not read those labels at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from gates.gate8 import sources as S
from gates.gate8b.sources import G8B_ROOT, K360_TRAIN_DRIVES, K360_VAL_DRIVE
from gates.gate8c0 import transforms as TF

#: ``G8C1_DATA_ROOT`` redirects targets and samples, so a rebuilt supervision set can be
#: produced without overwriting the one the frozen gate was trained on.
G8C1_ROOT = os.environ.get("G8C1_DATA_ROOT",
                           "/media/SSD1/MINH_DATASETS/lingbot_gate8c1")
TRAIN_DRIVES = tuple(K360_TRAIN_DRIVES)          # 0003, 0007, 0010
VAL_DRIVE = K360_VAL_DRIVE                       # 0006
#: Datasets no training or selection code may read.
TARGET_DATASETS = ("semantickitti", "occ3d", "nuscenes", "sscbench")
#: Substrings that must not appear in any path opened before the manifests are frozen.
FORBIDDEN_PATH_TOKENS = ("data/kitti/dataset", "occ3d_gt", "nuscenes",
                         "lingbot_occ3d_zeroshot", "scale_gate/cache",
                         "preprocess/labels")     # SSCBench completion labels included
#: Privileged future horizon, in stream frames. Frozen from Gate 8.
FUTURE_FRAMES = 20
#: Frames the scale anchor consumes. Frozen.
SCALE_FRAMES = 5

SSCBENCH_ROOT = "/media/SSD1/MINH_DATASETS/sscbench_kitti360"
KITTI360_ROOT = "/media/welf/MINH/datasets/kitti360/KITTI-360"


class TargetAccessViolation(RuntimeError):
    pass


def assert_no_target_access(paths: Sequence[str]) -> None:
    """Raise if any path names a target dataset or an SSCBench completion label."""
    bad = [p for p in paths
           if any(t in str(p).replace("\\", "/") for t in FORBIDDEN_PATH_TOKENS)]
    if bad:
        raise TargetAccessViolation(f"forbidden path(s) before the firewall lifts: {bad[:4]}")


def source_of(drive: str) -> str:
    """Which Gate-8 cache source holds this drive's frozen stream."""
    return "kitti360" if drive == VAL_DRIVE else "k360_train"


def partition_of(drive: str) -> str:
    return "source_validation" if drive == VAL_DRIVE else "train"


@dataclass
class Anchor:
    drive: str
    partition: str
    stream_index: int
    key: str
    native_frame: int
    future_stream_indices: List[int]
    future_natives: List[int]
    n_future: int


def anchors(drive: str, repo_root: str, future_frames: int = FUTURE_FRAMES,
            min_future: int = 5) -> List[Anchor]:
    """Every eligible stream frame of ``drive``, with its causal and future windows.

    Eligible means: at least ``SCALE_FRAMES`` frames precede it (so the frozen five-frame
    scale anchor is complete and the map is warm) and at least ``min_future`` future stream
    frames exist for the privileged target.
    """
    src = source_of(drive)
    seg = [s for s in S.segments(src, repo_root) if s.name == drive]
    if not seg:
        return []
    seg = seg[0]
    geo = TF.DriveGeometry(drive, SSCBENCH_ROOT, KITTI360_ROOT)
    natives = [int(f.order) for f in seg.frames]
    out = []
    n = len(seg.frames)
    for i in range(SCALE_FRAMES - 1, n):
        fut_i = [j for j in range(i + 1, min(i + 1 + future_frames, n))]
        if len(fut_i) < min_future:
            continue
        fut_n = [natives[j] for j in fut_i]
        if not all(f in geo.cam0_to_world for f in [natives[i]] + fut_n):
            continue
        out.append(Anchor(drive=drive, partition=partition_of(drive), stream_index=i,
                          key=seg.frames[i].key, native_frame=natives[i],
                          future_stream_indices=fut_i, future_natives=fut_n,
                          n_future=len(fut_i)))
    return out


def all_anchors(repo_root: str) -> Dict[str, List[Anchor]]:
    return {d: anchors(d, repo_root) for d in list(TRAIN_DRIVES) + [VAL_DRIVE]}


def target_path(drive: str, stream_index: int) -> str:
    return f"{G8C1_ROOT}/targets/{drive}/{stream_index:05d}.npz"


def sample_path(drive: str, stream_index: int) -> str:
    return f"{G8C1_ROOT}/samples/{drive}/{stream_index:05d}.npz"


__all__ = ["G8C1_ROOT", "TRAIN_DRIVES", "VAL_DRIVE", "TARGET_DATASETS", "FUTURE_FRAMES",
           "SCALE_FRAMES", "Anchor", "anchors", "all_anchors", "target_path", "sample_path",
           "source_of", "partition_of", "assert_no_target_access", "TargetAccessViolation",
           "SSCBENCH_ROOT", "KITTI360_ROOT", "FORBIDDEN_PATH_TOKENS"]
