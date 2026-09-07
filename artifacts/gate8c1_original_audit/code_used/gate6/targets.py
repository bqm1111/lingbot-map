"""Official semantic targets and evaluation masks. **Evaluator-only.**

Every function here opens a ground-truth file, so nothing in this module may be reached
from the prediction stage -- ``gate6.audit.Gate6Audit`` raises if it is. The rules are the
official ones of each benchmark, taken from the loaders the previous gates already
verified elementwise rather than re-derived here.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from prompted_lingbot.occ_datasets import load_semantickitti_target
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID
from occ3d_zeroshot.grid import NATIVE as OCC3D_NATIVE
from sscbench_kitti360.adapter import SSCBENCH_KITTI360_GRID, load_target as k360_target
from sscbench_kitti360.adapter import SEQUENCE as KITTI360_SEQUENCE_DEFAULT

SEMANTICKITTI_ROOT = "data/kitti/dataset"
KITTI360_TARGET_ROOT = "/media/SSD1/MINH_DATASETS/sscbench_kitti360"
OCC3D_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval"

# Frozen Occ3D evaluation setting, inherited from Gate 4 (configs/occ3d_zeroshot).
OCC3D_APPLY_CAMERA_MASK = True
OCC3D_APPLY_LIDAR_MASK = False
OCC3D_SINGLE_CAMERA_X_CUT = 100


def semantic_target(dataset: str, rec: dict, repo_root: str):
    """``(target, keep)``: official integer labels and the official evaluation mask.

    ``target`` carries the benchmark's own label ids; ``keep`` is ``target != ignore``.
    Occupied is ``target != empty`` inside ``keep``.
    """
    if dataset == "semantickitti":
        root = os.path.join(repo_root, SEMANTICKITTI_ROOT)
        t, _valid = load_semantickitti_target(root, rec["sequence"],
                                              int(rec["frame_ids"][-1]),
                                              SEMANTICKITTI_GRID)
        if t is None:
            return None, None
        return t.astype(np.int32), t != SEMANTICKITTI_GRID.ignore_label

    if dataset == "kitti360":
        # the drive is read from the record when present (every official val record
        # carries the val drive, so this is the identity for Gates 5.2-8A); Gate 8B's
        # training records name an official train drive instead
        t, keep = k360_target(KITTI360_TARGET_ROOT, int(rec["anchor"]),
                              SSCBENCH_KITTI360_GRID,
                              sequence=rec.get("sequence", KITTI360_SEQUENCE_DEFAULT))
        return t.astype(np.int32), keep

    if dataset == "occ3d":
        p = os.path.join(OCC3D_ROOT, rec["anchor_gt_path"])
        if not os.path.isfile(p):
            return None, None
        with np.load(p) as d:
            label = np.asarray(d["semantics"]).astype(np.int32)
            mask_camera = np.asarray(d["mask_camera"]).astype(bool)
            mask_lidar = np.asarray(d["mask_lidar"]).astype(bool)
        if label.shape != tuple(OCC3D_NATIVE.dims):
            raise ValueError(f"{p}: {label.shape} != {OCC3D_NATIVE.dims}")
        if OCC3D_APPLY_CAMERA_MASK:
            label[~mask_camera] = OCC3D_NATIVE.ignore_label
        if OCC3D_APPLY_LIDAR_MASK:
            label[~mask_lidar] = OCC3D_NATIVE.ignore_label
        if OCC3D_SINGLE_CAMERA_X_CUT:
            label[:OCC3D_SINGLE_CAMERA_X_CUT, :, :] = OCC3D_NATIVE.ignore_label
        return label, label != OCC3D_NATIVE.ignore_label

    raise KeyError(dataset)


__all__ = ["semantic_target", "OCC3D_ROOT", "KITTI360_TARGET_ROOT", "SEMANTICKITTI_ROOT"]
