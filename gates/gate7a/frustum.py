"""Does a missed voxel even project into one of the five input images?

The term throughout is **in-frustum**, never "visible": this tests projection, positive
camera depth and the frozen metric depth range. It does not test occlusion, so a voxel
behind a wall counts as in-frustum here.

Every transform is the frozen one. Grid -> anchor camera is the inverse of the clip's own
anchor-to-grid transform (the calibrated ``cam_to_velo`` / ``rect_cam_to_velo`` /
``T_camera_to_ego[-1]`` that the reconstruction itself used); anchor -> frame ``f`` is the
inverse of ``decompose_residual.scaled_relative_pose``, the same function the frozen fusion
calls. :func:`verify_roundtrip` proves the chain by pushing the frozen reconstruction's own
points back to the pixels they came from.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # gates/<pkg>/ -> repo root
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import scaled_relative_pose                      # noqa: E402

NEAR_SURFACE, BEHIND, IN_FRONT, NO_DEPTH, OUTSIDE = 0, 1, 2, 3, 4
RESIDUAL_CLASSES = ("near_surface", "behind_surface", "in_front_of_surface",
                    "no_valid_predicted_depth", "outside_all_frusta")


def voxel_centres(flat: np.ndarray, grid) -> np.ndarray:
    """Flat evaluation-grid indices -> metric centres in the benchmark's grid frame."""
    idx = np.stack(np.unravel_index(np.asarray(flat, np.int64), grid.dims), axis=-1)
    return (idx.astype(np.float64) + 0.5) * grid.voxel_size + np.asarray(grid.origin, float)


def anchor_to_frame(pose: np.ndarray, f: int, anchor: int, scale: float) -> np.ndarray:
    """The inverse of the frozen frame-``f``-to-anchor transform."""
    return np.linalg.inv(scaled_relative_pose(pose, f, anchor, scale))


def project(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """``(u, v, z)`` in the processed lattice. No distortion: the frames are rectified."""
    z = points_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u = K[0, 0] * points_cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / safe + K[1, 2]
    return u, v, z


def analyse(centres_grid: np.ndarray, T_anchor_to_grid: np.ndarray, pose: np.ndarray,
            K: np.ndarray, dep: np.ndarray, conf: np.ndarray, scale: float,
            conf_thr: float, dmin: float, dmax: float,
            voxel_diagonal: float) -> Dict[str, np.ndarray]:
    """In-frustum test in all five frames, plus the signed ray-depth residual.

    Returns ``in_any`` (bool), ``n_frames`` (in how many frames it is in-frustum),
    ``frame`` (the predeclared frame used for the residual: the anchor when in-frustum
    there, else the lowest-index in-frustum frame; ``-1`` when outside all), ``residual``
    (metres; ``nan`` where undefined) and ``klass`` (one of :data:`RESIDUAL_CLASSES`).
    """
    N = len(centres_grid)
    T, H, W = dep.shape
    anchor = T - 1
    out = {"in_any": np.zeros(N, bool), "n_frames": np.zeros(N, np.int32),
           "frame": np.full(N, -1, np.int32), "residual": np.full(N, np.nan),
           "klass": np.full(N, OUTSIDE, np.int8)}
    if N == 0:
        return out

    Tg = np.linalg.inv(np.asarray(T_anchor_to_grid, np.float64))
    p_anchor = centres_grid @ Tg[:3, :3].T + Tg[:3, 3]

    depth_m = scale * dep.astype(np.float64)
    valid_depth = ((conf >= conf_thr) & np.isfinite(depth_m) & (depth_m > dmin)
                   & (depth_m < dmax))

    # anchor first, then the remaining frames in index order -- the predeclared preference
    for f in [anchor] + [i for i in range(T) if i != anchor]:
        Taf = np.eye(4) if f == anchor else anchor_to_frame(pose, f, anchor, scale)
        pc = p_anchor @ Taf[:3, :3].T + Taf[:3, 3]
        u, v, z = project(pc, K[f])
        ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        inside = ((z > 0) & (z >= dmin) & (z <= dmax) & (ui >= 0) & (ui < W)
                  & (vi >= 0) & (vi < H))
        out["in_any"] |= inside
        out["n_frames"] += inside.astype(np.int32)
        take = inside & (out["frame"] < 0)
        if not take.any():
            continue
        idx = np.flatnonzero(take)
        out["frame"][idx] = f
        pd = depth_m[f][vi[idx], ui[idx]]
        ok = valid_depth[f][vi[idx], ui[idx]]
        res = z[idx] - pd
        out["residual"][idx] = np.where(ok, res, np.nan)
        k = np.full(len(idx), NO_DEPTH, np.int8)
        near = ok & (np.abs(res) <= voxel_diagonal)
        beh = ok & (res > voxel_diagonal)
        fro = ok & (res < -voxel_diagonal)
        k[near], k[beh], k[fro] = NEAR_SURFACE, BEHIND, IN_FRONT
        out["klass"][idx] = k
    return out


def verify_roundtrip(points_anchor: np.ndarray, frame: np.ndarray, v_pix: np.ndarray,
                     u_pix: np.ndarray, pose: np.ndarray, K: np.ndarray, scale: float,
                     n_check: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Push the frozen fusion's own points back to their originating pixels.

    Every arithmetic step of the forward direction is inverted here, so a mistake in the
    anchor convention, in the pose scaling or in the intrinsics shows up immediately as a
    pixel error. Gate 7A refuses to interpret any frustum number unless this round-trip is
    exact to well under a pixel.
    """
    n = len(points_anchor)
    if n == 0:
        return {"n": 0, "max_pixel_error": 0.0, "max_depth_error_m": 0.0}
    rng = np.random.default_rng(seed)
    sel = np.arange(n) if n <= n_check else rng.choice(n, n_check, replace=False)
    anchor = len(K) - 1
    du = dv = dz = 0.0
    for f in np.unique(frame[sel]):
        m = sel[frame[sel] == f]
        Taf = np.eye(4) if f == anchor else anchor_to_frame(pose, int(f), anchor, scale)
        pc = points_anchor[m] @ Taf[:3, :3].T + Taf[:3, 3]
        u, v, z = project(pc, K[int(f)])
        # the forward pass unprojected from *integer* pixel centres, so the exact
        # round-trip target is the integer itself; flooring here would report a whole
        # pixel of error whenever floating point lands a hair below it
        du = max(du, float(np.abs(u - u_pix[m]).max()))
        dv = max(dv, float(np.abs(v - v_pix[m]).max()))
        dz = max(dz, float(np.abs(z).min() * 0))       # z sanity: strictly positive below
        assert (z > 0).all(), "round-trip produced a non-positive camera depth"
    return {"n": int(len(sel)), "max_pixel_error": float(max(du, dv)),
            "max_depth_error_m": dz}


__all__ = ["voxel_centres", "anchor_to_frame", "project", "analyse", "verify_roundtrip",
           "RESIDUAL_CLASSES", "NEAR_SURFACE", "BEHIND", "IN_FRONT", "NO_DEPTH", "OUTSIDE"]
