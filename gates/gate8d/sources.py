"""Gate 8D: the dataset boundary, and the runtime audit that proves it held.

KITTI-360 is the only dataset any module in this package may reach. The forbidden tokens
below cover both target benchmarks and the defective SSCBench completion label; opening a
matching path inside a ``FileAudit`` raises. Gate 8C-1's firewall is inherited and extended
with the Gate 8D data root.
"""

from __future__ import annotations

import os
from typing import Sequence

from gates.gate8c1.sources import (FORBIDDEN_PATH_TOKENS as _G8C1_TOKENS, FUTURE_FRAMES,
                             KITTI360_ROOT, SSCBENCH_ROOT, TargetAccessViolation,
                             all_anchors, anchors, assert_no_target_access,
                             partition_of, source_of)
from gates.gate8d import protocol as P

#: Gate 8D writes its own targets, samples and caches; Gate 8C-1's stay byte-for-byte.
G8D_ROOT = os.environ.get("G8D_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate8d")

TRAIN_DRIVES = P.TRAIN_DRIVES
VAL_DRIVE = P.VAL_DRIVE
ALL_DRIVES = TRAIN_DRIVES + (VAL_DRIVE,)

#: every token Gate 8C-1 forbade, plus both target names spelled both ways and the
#: SSCBench completion label Gate 8C-0 showed to be geometrically inconsistent.
#:
#: Gate 8C-1's own *KITTI-360* targets are deliberately NOT forbidden. The boundary this
#: firewall enforces is the dataset boundary -- SemanticKITTI, Occ3D/nuScenes, and the
#: defective `_1_1.npy` label -- not isolation from earlier gates. The Phase 3 audit has to
#: compare the new supervision against those baselines, and they are built from the same
#: KITTI-360 drives by the same rules, so reading them crosses no boundary.
FORBIDDEN_PATH_TOKENS: Sequence[str] = tuple(sorted(set(_G8C1_TOKENS) | {
    "semantickitti", "SemanticKITTI", "nuscenes", "nuScenes", "occ3d", "Occ3D",
    "_1_1.npy"}))


def target_path(drive: str, stream_index: int) -> str:
    return f"{G8D_ROOT}/targets/{drive}/{stream_index:05d}.npz"


def sample_path(drive: str, stream_index: int) -> str:
    return f"{G8D_ROOT}/samples/{drive}/{stream_index:05d}.npz"


def moge_future_path(drive: str, native_frame: int) -> str:
    return f"{G8D_ROOT}/moge_future/{drive}/{native_frame:010d}.npz"


def sky_future_path(drive: str, native_frame: int) -> str:
    return f"{G8D_ROOT}/sky_future/{drive}/{native_frame:010d}.npz"


def image_path(drive: str, native_frame: int) -> str:
    return (f"{KITTI360_ROOT}/data_2d_raw/{drive}/image_00/data_rect/"
            f"{native_frame:010d}.png")


__all__ = ["G8D_ROOT", "TRAIN_DRIVES", "VAL_DRIVE", "ALL_DRIVES", "FORBIDDEN_PATH_TOKENS",
           "target_path", "sample_path", "moge_future_path", "sky_future_path",
           "image_path", "anchors", "all_anchors", "source_of", "partition_of",
           "FUTURE_FRAMES", "KITTI360_ROOT",
           "SSCBENCH_ROOT", "assert_no_target_access", "TargetAccessViolation"]
