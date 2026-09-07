# extracted from gate8b/sources.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""KITTI-360 as a completion-*training* source, and the fold definitions.

The benchmark KITTI-360 (drive 0006, SSCBench validation) is what every prior gate scored.
Gate 8B needs KITTI-360 on the *other* side as well: a training partition that is
disjoint from any evaluation, plus a source-validation partition for checkpoint and
threshold selection in the two new folds. The partition follows the official SSCBench
split exactly:

* ``k360_train``  -- official SSCBench **train** drives 0003, 0007, 0010 (raw images and
  velodyne already on disk; their occupancy targets fetched from the official archive by
  ``tools/gate8b/fetch_k360_labels.py``). Streamed at SSCBench-index stride 5, so every
  stream frame is a labelled anchor -- the same construction as ``sk_train``.
* ``kitti360``    -- drive 0006, the official **val** drive, unchanged from Gates 5.2-8A,
  used as the KITTI-360 source-validation domain.

The two are disjoint by drive. Registering ``k360_train`` through
:func:`gate8.sources.register_source` makes it visible to Gate 8's untouched cache tools
and feed, with all of its caches under a separate Gate 8B root.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from core.datasets import frames as G6F
from core.datasets import sources as S
from core.datasets import kitti360 as K3

G8B_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate8b"
K360_TRAIN_DRIVES = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync",
                     "2013_05_28_drive_0010_sync")
K360_VAL_DRIVE = K3.SEQUENCE                     # 2013_05_28_drive_0006_sync
K360_INDEX_STRIDE = 5                            # SSCBench anchors sit every 5 indices
SOURCE = "k360_train"

#: fold name -> (train sources, source-validation sources, held-out target)
FOLDS: Dict[str, dict] = {
    "kitti360": {"train": ["sk_train", "occ3d_train"], "val": ["semantickitti", "occ3d"],
                 "target": "kitti360", "status": "existing (Gate 8A), reused unmodified"},
    "semantickitti": {"train": ["occ3d_train", SOURCE], "val": ["occ3d", "kitti360"],
                      "target": "semantickitti", "status": "new"},
    "occ3d": {"train": ["sk_train", SOURCE], "val": ["semantickitti", "kitti360"],
              "target": "occ3d", "status": "new"},
}

for _f, _d in FOLDS.items():
    assert _d["target"] not in _d["train"] and _d["target"] not in _d["val"]
    assert not any(S.DATASET_OF.get(x, x) == _d["target"] for x in _d["train"] + _d["val"]
                   if x != SOURCE), _f
assert K360_VAL_DRIVE not in K360_TRAIN_DRIVES
assert all(d in K3.OFFICIAL_SPLIT["train"] for d in K360_TRAIN_DRIVES)


def _k360_train_segments(repo_root: str) -> List[S.Seg]:
    from core.evaluation.official_targets import KITTI360_TARGET_ROOT
    calib = K3.parse_calibration(os.path.join(KITTI360_TARGET_ROOT, "calibration"))
    out = []
    for drive in K360_TRAIN_DRIVES:
        frames = K3.pose_frames(os.path.join(KITTI360_TARGET_ROOT, "data_poses", drive,
                                             "poses.txt"))
        ldir = os.path.join(KITTI360_TARGET_ROOT, "preprocess", "labels", drive)
        have = {int(f.split("_")[0]) for f in os.listdir(ldir) if f.endswith("_1_1.npy")} \
            if os.path.isdir(ldir) else set()
        seg = S.Seg(source=SOURCE, name=drive)
        img_root = os.path.join(G6F.KITTI360_ROOT, "data_2d_raw", drive)
        i = 0
        # the stream is defined by the pose file alone (every 5th SSCBench index that
        # resolves to a native frame); a target, where one exists, only marks an anchor
        for idx in range(0, len(frames) - 1, K360_INDEX_STRIDE):
            native = int(K3.sscbench_to_native(idx, frames))
            path = os.path.join(img_root, "image_00", "data_rect", f"{native:010d}.png")
            if not os.path.isfile(path):
                continue
            gt = ({"anchor": idx, "sequence": drive, "clip_id": f"{drive}_{idx:06d}",
                   "dataset": "sscbench_kitti360"} if idx in have else None)
            seg.frames.append(S.SFrame(
                key=f"{drive[-9:-5]}_{native:010d}", path=path, order=native, index=i,
                gt_ref=gt, T_cam_to_grid=calib.rect_cam_to_velo, K_native=calib.K,
                native_hw=calib.native_hw))
            i += 1
        out.append(seg)
    return out


S.register_source(SOURCE, "kitti360", G8B_ROOT, _k360_train_segments)

__all__ = ["G8B_ROOT", "K360_TRAIN_DRIVES", "K360_VAL_DRIVE", "SOURCE", "FOLDS"]
