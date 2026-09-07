# extracted from voxel_gate/voxels.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Voxel-space feature construction and the inference-computable correction region.

Two hard rules, tested in ``tests/voxel_gate``:

* **every model input is computable at inference time from frozen C3 alone** -- no LiDAR
  occupancy, no LiDAR depth, no ground-truth pose, no oracle scale, no oracle visibility;
* the grid, voxel size, floor-binning voxeliser and evaluation mask are the frozen ones
  from ``prompted_lingbot.occupancy``; nothing here re-derives or re-resolutions them.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.datasets.occupancy import SEMANTICKITTI_GRID as G, voxelize_points

# Channel order is part of the trained checkpoint; never reorder without retraining.
FEATURE_NAMES = ("occupied", "log1p_count", "n_frames", "mean_confidence",
                 "mean_point_depth_m", "in_correction_region")
N_FEATURES = len(FEATURE_NAMES)

# Names of every quantity that must never reach the model. Asserted in the tests.
FORBIDDEN_INPUTS = ("lidar_occupancy", "lidar_depth", "gt_pose", "oracle_scale",
                    "visible_ceiling", "semantic_label", "target", "valid")


def flat_index(idx: np.ndarray) -> np.ndarray:
    return np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), G.dims)


def sparse_voxel_features(points: np.ndarray, frame: np.ndarray, conf: np.ndarray,
                          depth: np.ndarray) -> Dict[str, np.ndarray]:
    """Aggregate fused C3 points into per-voxel evidence.

    Returns flat voxel indices plus, for each, the point count, the number of *distinct*
    contributing frames, the mean LingBot confidence and the mean metric point depth.
    All four are properties of the prediction, available at inference.
    """
    idx, keep = voxelize_points(points, G)
    if idx.size == 0:
        z = np.zeros((0,), np.float32)
        return {"flat": np.zeros((0,), np.int64), "count": z, "n_frames": z,
                "sum_conf": z, "sum_depth": z}
    flat = flat_index(idx)
    fr, cf, dp = frame[keep], conf[keep], depth[keep]
    uniq, inv = np.unique(flat, return_inverse=True)
    n = len(uniq)
    count = np.bincount(inv, minlength=n).astype(np.float32)
    sum_conf = np.bincount(inv, weights=cf.astype(np.float64), minlength=n)
    sum_depth = np.bincount(inv, weights=dp.astype(np.float64), minlength=n)
    nf = np.zeros(n, np.float32)
    for f in np.unique(fr):                      # distinct frames, not point counts
        nf[np.unique(inv[fr == f])] += 1.0
    return {"flat": uniq.astype(np.int64), "count": count, "n_frames": nf,
            "sum_conf": sum_conf.astype(np.float32),
            "sum_depth": sum_depth.astype(np.float32)}


def dense_from_sparse(sp: Dict[str, np.ndarray], device) -> torch.Tensor:
    """Sparse per-voxel evidence -> dense ``[5, X, Y, Z]`` float32 feature volume.

    Channels 0-4 of :data:`FEATURE_NAMES`; the correction-region channel is appended by
    :func:`build_input` once the region is known.
    """
    n = int(np.prod(G.dims))
    vol = torch.zeros(5, n, dtype=torch.float32, device=device)
    if sp["flat"].size:
        f = torch.from_numpy(sp["flat"]).to(device)
        c = torch.from_numpy(sp["count"]).to(device)
        vol[0, f] = 1.0
        vol[1, f] = torch.log1p(c)
        vol[2, f] = torch.from_numpy(sp["n_frames"]).to(device)
        vol[3, f] = torch.from_numpy(sp["sum_conf"]).to(device) / c
        vol[4, f] = torch.from_numpy(sp["sum_depth"]).to(device) / c
    return vol.view(5, *G.dims)


def dilate(vol: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation by a cube of half-width ``radius`` (Chebyshev ball)."""
    if radius <= 0:
        return vol > 0
    x = (vol > 0).float()[None, None]
    k = 2 * radius + 1
    return (F.max_pool3d(x, kernel_size=k, stride=1, padding=radius)[0, 0] > 0)


def correction_region(occupied: torch.Tensor, radius: int) -> torch.Tensor:
    """The band the model is allowed to change: ``dilate(C3 occupancy, radius)``.

    Inference-computable by construction -- it is a morphological operation on the frozen
    C3 occupancy and nothing else. Because dilation is extensive, the band always
    **contains** every C3-occupied voxel, so "preserve C3 outside the region" and "force
    empty outside the region" are the same statement (see the report, §4).
    """
    return dilate(occupied, radius)


def build_input(feat5: torch.Tensor, region: torch.Tensor,
                norm: Dict[str, float]) -> torch.Tensor:
    """Assemble the normalised ``[6, X, Y, Z]`` network input.

    Continuous channels are standardised with statistics computed on **source training
    clips only**; the occupancy and region channels are already 0/1.
    """
    occ = feat5[0:1]
    cnt = (feat5[1:2] - norm["log1p_count_mean"]) / norm["log1p_count_std"] * occ
    nfr = (feat5[2:3] - norm["n_frames_mean"]) / norm["n_frames_std"] * occ
    cnf = (feat5[3:4] - norm["mean_confidence_mean"]) / norm["mean_confidence_std"] * occ
    dpt = (feat5[4:5] - norm["mean_point_depth_mean"]) / norm["mean_point_depth_std"] * occ
    return torch.cat([occ, cnt, nfr, cnf, dpt, region[None].float()], dim=0)


def occupancy_from_flat(flat: np.ndarray) -> np.ndarray:
    """Flat voxel indices -> boolean volume of the frozen grid dimensions."""
    vol = np.zeros(int(np.prod(G.dims)), bool)
    if flat.size:
        vol[flat] = True
    return vol.reshape(G.dims)


def pack(vol: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(vol, bool).reshape(-1))


def unpack(bits: np.ndarray) -> np.ndarray:
    n = int(np.prod(G.dims))
    return np.unpackbits(bits)[:n].astype(bool).reshape(G.dims)


def distance_bins(bins: Tuple[Tuple[float, float], ...]) -> Dict[str, np.ndarray]:
    """Voxel masks split by horizontal range from the anchor sensor origin.

    The grid frame is the velodyne frame of the anchor frame, so the sensor sits at the
    origin and range is read straight off the voxel centres -- no extra information.
    """
    ix, iy, iz = np.meshgrid(*[np.arange(d) for d in G.dims], indexing="ij")
    c = np.stack([ix, iy, iz], -1).astype(np.float64)
    xyz = (c + 0.5) * G.voxel_size + np.asarray(G.origin, float)
    rng = np.linalg.norm(xyz[..., :2], axis=-1)
    return {f"{int(lo)}-{int(hi)}m": (rng >= lo) & (rng < hi) for lo, hi in bins}

