"""Track P region and population masks, computed from a canonical prediction and the official label volume.

All masks are flat [N] bool on the evaluation grid.

    valid         label != 255
    field_of_view voxel centre in front of the target camera and between the rays through image columns 0 and W
    range_m       horizontal range sqrt(x_cam^2 + z_cam^2) of the voxel centre, metres (camera OpenCV frame)
    region        valid & field_of_view & range_m <= 40 m                         (Track P)
    gt_occupied   valid & label not empty
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.protocol import REGION, DatasetSpec

IGNORE = 255


def voxel_centres_grid(pred: CanonicalPrediction) -> np.ndarray:
    X, Y, Z = pred.dims
    ijk = np.stack(np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"), -1).reshape(-1, 3)
    return (ijk + 0.5) * pred.voxel_size + np.asarray(pred.origin, np.float64)


def camera_coordinates(pred: CanonicalPrediction) -> np.ndarray:
    """[N, 3] voxel centres in the target camera frame (OpenCV), metres."""
    T = np.linalg.inv(np.asarray(pred.T_cam_to_grid, np.float64))
    c = voxel_centres_grid(pred)
    return c @ T[:3, :3].T + T[:3, 3]


def field_of_view(cam: np.ndarray, K: np.ndarray, image_width: int) -> np.ndarray:
    """Horizontal field of view: z > 0 and 0 <= u <= W with u = fx x / z + cx (no vertical limit)."""
    z = cam[:, 2]; ok = z > 0
    u = np.where(ok, K[0, 0] * cam[:, 0] / np.where(ok, z, 1.0) + K[0, 2], -1.0)
    return ok & (u >= 0) & (u <= image_width)


@dataclass
class TrackPMasks:
    valid: np.ndarray
    region: np.ndarray
    range_bin: np.ndarray       # [N] int, 0..3 inside the 40 m range, 4 beyond
    gt_occupied: np.ndarray


def track_p_masks(pred: CanonicalPrediction, label: np.ndarray, spec: DatasetSpec) -> TrackPMasks:
    valid = label != IGNORE
    cam = camera_coordinates(pred)
    rng = np.hypot(cam[:, 0], cam[:, 2])
    fov = field_of_view(cam, np.asarray(pred.K, np.float64), pred.image_width)
    inside = rng <= REGION.max_range_m
    rbin = np.where(inside, np.digitize(rng, np.asarray(REGION.range_edges_m)), len(REGION.range_names))
    return TrackPMasks(valid, valid & fov & inside, rbin, valid & (label != spec.empty_class))


def populations(pred: CanonicalPrediction, m: TrackPMasks, region: np.ndarray) -> Dict[str, np.ndarray]:
    """Semantic populations inside ``region`` (Track P region or the full valid region)."""
    G = region & m.gt_occupied; O = pred.occupied; E = pred.evidence; S = pred.observed
    return {"gt_occupied": G, "observed_evidence": G & E, "no_evidence": G & ~E, "completed_no_evidence": G & O & ~E,
            "geometrically_unobserved": G & O & ~S}


__all__ = ["IGNORE", "camera_coordinates", "field_of_view", "TrackPMasks", "track_p_masks", "populations"]
