"""Gate 3.1 — frozen geometry evidence, rebuilt from raw with auditable provenance.

The Gate-3 defect was that the inference region was intersected with the SemanticKITTI
per-sample ``valid``/``invalid`` mask, which is label-side information. This module is
written so that the same mistake is structurally impossible:

* :func:`geometry_evidence` takes **no target, valid or label argument at all**. Its
  signature is the guarantee, and a test asserts the signature.
* the only inputs are the frozen LingBot cache, the frozen depth head's clip scalar and
  the sequence calibration -- none of which is derived from the evaluation target.

Two geometries are supported, both frozen:

    C0   depth = s0 * D_lingbot,        translation = s0 * t          (Gate-2 deployable)
    C3   depth = s_learned * D_lingbot, translation = s_learned * t   (Gate-2 SCALE_ONLY)

with ``s_learned = s0 * exp(a_clip)`` and ``a_clip`` the clip-median residual of the
frozen ``depth_cnn`` head over the unchanged C0 fusion support. ``r_shape`` is never formed.
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional

import numpy as np

from gates.voxel_gate.c3 import c3_points, clip_scale
from gates.voxel_gate.voxels import sparse_voxel_features

from prompted_lingbot.occ_datasets import apply_transform

GEOMETRIES = ("c0", "c3")


def geometry_evidence(dep: np.ndarray, conf: np.ndarray, K: np.ndarray,
                      pose_pred: np.ndarray, cam_to_velo: np.ndarray, scale: float,
                      conf_thr: float, dmin: float, dmax: float) -> Dict[str, np.ndarray]:
    """Frozen per-voxel evidence for one clip. **No target-side argument exists.**

    ``scale`` multiplies depth *and* the relative pose translations (rotations untouched),
    so the fused cloud is a pure similarity of the canonical reconstruction.
    """
    pts, fr, cf, pd, _ = c3_points(dep, conf, K, pose_pred, float(scale),
                                   conf_thr, dmin, dmax)
    pg = apply_transform(cam_to_velo, pts) if len(pts) else pts
    sp = sparse_voxel_features(pg, fr, cf, pd)
    sp["n_points"] = np.int64(len(pts))
    return sp


# The audit test compares against this list; adding a target-side parameter breaks it.
EVIDENCE_SIGNATURE = tuple(inspect.signature(geometry_evidence).parameters)

FORBIDDEN_SIGNATURE_TOKENS = ("valid", "invalid", "target", "gt", "label", "occupancy",
                              "visible", "oracle", "keep", "lidar")


def clip_scales(model, ck, dep: np.ndarray, conf: np.ndarray, s0: float, conf_thr: float,
                dmin: float, dmax: float, rgb: Optional[np.ndarray], device):
    """``{"c0": s0, "c3": s0 * exp(a_clip)}`` plus ``a_clip`` -- frozen head, scalar only."""
    a_clip, s_learned, support = clip_scale(model, ck, dep, conf, s0, conf_thr,
                                            dmin, dmax, rgb, device)
    return {"c0": float(s0), "c3": float(s_learned)}, a_clip, support
