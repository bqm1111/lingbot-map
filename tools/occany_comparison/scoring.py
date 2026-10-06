"""OccAny's saved outputs, the independent label check, OccAny's pooling and OccAny's scorer.

* :func:`load_occany` -- the released model's full-completion occupancy, its stored ground truth
  (``voxel_label``, 255 = ignore) and its camera-to-grid transform.
* :func:`independent_label` -- the official label files read directly with the corrected rule of the
  Gate 8C-1 audit (SemanticKITTI: invalid voxels and raw labels mapped to 0 other than raw 0 are
  ignored; Occ3D: ``mask_camera``, no lidar mask, x-index < 100 ignored). Used only to *check* that
  OccAny's stored label is the official one; both methods are scored against OccAny's stored label.
* :func:`pool` -- OccAny ``apply_majority_pooling(is_geometry_only=True)`` default = 3x3x3 max-pool,
  applied to the *unmasked* prediction (their pool-then-mask order).
* :func:`official_counts` -- ``SSCMetrics.get_score_completion`` from the OccAny checkout;
  :func:`fast_counts` is the GPU tally cross-checked against it on every scored volume.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from core.evaluation import official_targets as OT
from gates.gate8b import pooling as PL
from gates.gate8c1 import occany_eval as OE
from tools.occany_comparison import config as C


@dataclass(frozen=True)
class OccAnyOutput:
    label: np.ndarray            # uint8 [X, Y, Z], 255 = ignore
    pred: np.ndarray             # bool  [X, Y, Z]
    T_cam_to_grid: np.ndarray    # [4, 4]


def load_occany(dataset: str, path: str, dims) -> OccAnyOutput:
    run = C.OCCANY[dataset]
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    raw_label = np.asarray(d["voxel_label"]).reshape(dims)
    assert np.array_equal(raw_label, np.round(raw_label)) and raw_label.min() >= 0 and raw_label.max() <= 255, path
    label = raw_label.astype(np.uint8)
    pred_raw = np.asarray(d[run.pred_key]).reshape(dims)
    return OccAnyOutput(label, pred_raw != run.empty_class, np.asarray(d["T_cam_to_voxel"], np.float64))


def independent_label(dataset: str, gt_ref: dict) -> np.ndarray:
    """Official label with the corrected ignore rule, uint8 [X, Y, Z], 255 = ignore."""
    if dataset == "semantickitti":
        import yaml
        base = os.path.join(C.REPO_ROOT, OT.SEMANTICKITTI_ROOT, "sequences", gt_ref["sequence"], "voxels", f"{gt_ref['frame_ids'][-1]:06d}")
        raw = np.fromfile(base + ".label", dtype="<u2").reshape(256, 256, 32)
        invalid = np.unpackbits(np.fromfile(base + ".invalid", dtype=np.uint8), bitorder="big").reshape(raw.shape).astype(bool)
        lm = yaml.safe_load(open(C.OCCANY_SK_LEARNING_MAP))["learning_map"]
        assert set(np.unique(raw)).issubset(lm), "unmapped raw label"
        lut = np.array([lm.get(i, 0) for i in range(65536)], dtype=np.uint16)
        label = lut[raw].astype(np.int32)
        valid = ~invalid & ((raw == 0) | (label != 0))
        label[~valid] = C.IGNORE
        return label.astype(np.uint8)
    with np.load(os.path.join(OT.OCC3D_ROOT, gt_ref["anchor_gt_path"])) as z:
        label = z["semantics"].astype(np.int32)
        valid = z["mask_camera"].astype(bool) & (label != C.IGNORE)
    valid[:OT.OCC3D_SINGLE_CAMERA_X_CUT] = False
    label[~valid] = C.IGNORE
    return label.astype(np.uint8)


def pool(occ: torch.Tensor) -> torch.Tensor:
    """OccAny geometry pooling (3x3x3 max, stride 1, pad 1) on the unmasked boolean volume."""
    return PL.geometry_dilation(occ)


def official_counts(dataset: str, occ: np.ndarray, label: np.ndarray) -> Tuple[int, int, int]:
    run = C.OCCANY[dataset]
    pred_label = np.where(occ, 1, run.empty_class).astype(np.uint8)
    return OE.official_sc_counts(pred_label, label, n_classes=run.n_classes, empty_class=run.empty_class)


def fast_counts(occ: torch.Tensor, gt: torch.Tensor, keep: torch.Tensor) -> Tuple[int, int, int]:
    p, g = occ & keep, gt & keep
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


__all__ = ["OccAnyOutput", "load_occany", "independent_label", "pool", "official_counts", "fast_counts"]
