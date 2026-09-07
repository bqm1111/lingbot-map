"""Frozen 3D semantic lifting: teacher pixels -> fused voxel probability vectors.

Three operations, each of which must be exactly the frozen one:

* :func:`points_with_pixels` reproduces ``voxel_gate.c3.c3_points`` **bit-for-bit** while
  additionally returning each point's pixel of origin. The test suite asserts the identity
  against the frozen function rather than trusting the copy;
* :func:`fuse_voxel_probs` averages contributing probability vectors with *uniform*
  weights -- no confidence weighting, no class prior, no frame-count normalisation;
* :func:`propagate_dilation` gives each dilation-only voxel the probability vector of its
  nearest raw-occupied source inside the same fixed neighbourhood, averaging exact-distance
  ties. It reads no target and no target-derived mask.

Nothing here is learned and nothing is tuned. The only free parameters are the frozen
``dilate_r2`` radius and the frozen depth/confidence gates, all inherited from Gate 4.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # gates/<pkg>/ -> repo root
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import scaled_relative_pose                       # noqa: E402


def points_with_pixels(dep: np.ndarray, conf: np.ndarray, K: np.ndarray,
                       pose_pred: np.ndarray, scale: float, conf_thr: float,
                       dmin: float, dmax: float):
    """Frozen five-frame fusion into the anchor camera, carrying pixel provenance.

    Identical in every arithmetic step to ``voxel_gate.c3.c3_points`` /
    ``occ3d_zeroshot.pipeline.fuse_to_ego``: the same scale multiplies depth *and* the
    pose translations, rotations are untouched, the anchor is the last frame, and the
    per-frame mask and iteration order are the same -- so the returned point array is
    element-for-element the frozen one.

    Returns ``(points, frame, conf, depth, v_pix, u_pix, mask)``.
    """
    depth_m = scale * dep.astype(np.float64)
    mask = (conf >= conf_thr) & np.isfinite(depth_m) & (depth_m > dmin) & (depth_m < dmax)
    T, H, W = depth_m.shape
    anchor = T - 1
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts, fr, cf, dp, vp, up = [], [], [], [], [], []
    for f in range(T):
        m = mask[f]
        if not m.any():
            continue
        d = depth_m[f][m]
        p = np.stack([(u[m] - K[f][0, 2]) * d / K[f][0, 0],
                      (v[m] - K[f][1, 2]) * d / K[f][1, 1], d], axis=-1)
        if f != anchor:
            Trel = scaled_relative_pose(pose_pred, f, anchor, scale)
            p = p @ Trel[:3, :3].T + Trel[:3, 3]
        pts.append(p)
        fr.append(np.full(len(p), f, np.int8))
        cf.append(conf[f][m].astype(np.float32))
        dp.append(d.astype(np.float32))
        vp.append(v[m].astype(np.int32))
        up.append(u[m].astype(np.int32))
    if not pts:
        z = np.zeros((0,), np.float32)
        zi = np.zeros((0,), np.int32)
        return (np.zeros((0, 3)), np.zeros((0,), np.int8), z, z, zi, zi, mask)
    return (np.concatenate(pts, 0), np.concatenate(fr), np.concatenate(cf),
            np.concatenate(dp), np.concatenate(vp), np.concatenate(up), mask)


def sample_probs(sem: torch.Tensor, frame: np.ndarray, v: np.ndarray, u: np.ndarray,
                 device) -> torch.Tensor:
    """Gather each point's teacher probability vector.

    ``sem`` is ``[T, C, H, W]`` on the LingBot processed lattice -- the same lattice the
    depth map lives on -- so the lookup is an exact integer index, never a resample.
    """
    f = torch.from_numpy(frame.astype(np.int64)).to(device)
    vv = torch.from_numpy(v.astype(np.int64)).to(device)
    uu = torch.from_numpy(u.astype(np.int64)).to(device)
    return sem[f, :, vv, uu]                                    # [N, C]


def fuse_voxel_probs(flat: np.ndarray, probs: torch.Tensor, n_voxels: int, device):
    """Uniform mean of the contributing probability vectors, per occupied voxel.

    Returns ``(unique_flat_indices, mean_probs [K, C], counts [K])``. The mean is over
    every contributing *point* across every frame, which is the predeclared rule: a voxel
    seen in four frames is not weighted differently from one seen in one, beyond the
    number of points that actually landed in it.
    """
    if flat.size == 0:
        C = int(probs.shape[1]) if probs.ndim == 2 else 0
        return (np.zeros((0,), np.int64), torch.zeros((0, C), device=device),
                torch.zeros((0,), device=device))
    uniq, inv = np.unique(flat, return_inverse=True)
    ii = torch.from_numpy(inv.astype(np.int64)).to(device)
    K, C = len(uniq), int(probs.shape[1])
    acc = torch.zeros(K, C, dtype=torch.float32, device=device)
    acc.index_add_(0, ii, probs.float())
    cnt = torch.zeros(K, dtype=torch.float32, device=device)
    cnt.index_add_(0, ii, torch.ones_like(ii, dtype=torch.float32))
    return uniq.astype(np.int64), acc / cnt.unsqueeze(1), cnt


# --------------------------------------------------------------------------- #
# Dilation-only propagation
# --------------------------------------------------------------------------- #
def _offsets_by_distance(radius: int):
    """The Chebyshev-ball offsets, grouped by exact Euclidean distance, nearest first."""
    r = range(-radius, radius + 1)
    groups: Dict[int, list] = {}
    for dx in r:
        for dy in r:
            for dz in r:
                if dx == dy == dz == 0:
                    continue
                groups.setdefault(dx * dx + dy * dy + dz * dz, []).append((dx, dy, dz))
    return [(d2, groups[d2]) for d2 in sorted(groups)]


def propagate_dilation(dims: Tuple[int, int, int], src_flat: np.ndarray,
                       src_probs: torch.Tensor, dst_flat: np.ndarray, radius: int,
                       device):
    """Probability vectors for the dilation-only voxels.

    For each destination voxel: the raw-occupied sources inside the same fixed
    ``radius`` (Chebyshev) neighbourhood are ranked by Euclidean distance; the nearest
    distance shell that contains any source wins, and every source in that shell is
    averaged. Averaging is the tie rule, so the result does not depend on iteration order
    and is bit-reproducible.

    Dilation is extensive, so every dilation-only voxel has at least one source in range;
    the returned ``assigned`` mask asserts it rather than assuming it.
    """
    X, Y, Z = dims
    C = int(src_probs.shape[1])
    D = len(dst_flat)
    if D == 0:
        return torch.zeros((0, C), device=device), np.zeros((0,), bool)

    lut = torch.full((X * Y * Z,), -1, dtype=torch.int32, device=device)
    sf = torch.from_numpy(src_flat.astype(np.int64)).to(device)
    lut[sf] = torch.arange(len(src_flat), dtype=torch.int32, device=device)

    df = torch.from_numpy(dst_flat.astype(np.int64)).to(device)
    dz_i = df % Z
    dy_i = (df // Z) % Y
    dx_i = df // (Z * Y)

    acc = torch.zeros(D, C, dtype=torch.float32, device=device)
    cnt = torch.zeros(D, dtype=torch.float32, device=device)
    assigned = torch.zeros(D, dtype=torch.bool, device=device)
    sp = src_probs.float()

    for _d2, offs in _offsets_by_distance(radius):
        todo = ~assigned
        if not bool(todo.any()):
            break
        g_acc = torch.zeros(D, C, dtype=torch.float32, device=device)
        g_cnt = torch.zeros(D, dtype=torch.float32, device=device)
        for ox, oy, oz in offs:
            nx, ny, nz = dx_i + ox, dy_i + oy, dz_i + oz
            ok = ((nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y) & (nz >= 0) & (nz < Z) & todo)
            if not bool(ok.any()):
                continue
            nf = (nx.clamp(0, X - 1) * Y + ny.clamp(0, Y - 1)) * Z + nz.clamp(0, Z - 1)
            j = lut[nf]
            hit = ok & (j >= 0)
            if not bool(hit.any()):
                continue
            idx = hit.nonzero(as_tuple=True)[0]
            g_acc.index_add_(0, idx, sp[j[idx].long()])
            g_cnt.index_add_(0, idx, torch.ones(len(idx), device=device))
        newly = todo & (g_cnt > 0)
        if bool(newly.any()):
            acc[newly] = g_acc[newly]
            cnt[newly] = g_cnt[newly]
            assigned |= newly

    out = torch.zeros(D, C, dtype=torch.float32, device=device)
    got = assigned
    out[got] = acc[got] / cnt[got].unsqueeze(1)
    return out, got.cpu().numpy()


def labels_from_probs(probs: torch.Tensor, vocab_labels) -> np.ndarray:
    """``argmax`` over teacher channels, mapped to official benchmark label ids."""
    if probs.numel() == 0:
        return np.zeros((0,), np.uint8)
    ch = probs.argmax(1).cpu().numpy()
    return np.asarray(vocab_labels, np.uint8)[ch]


__all__ = ["points_with_pixels", "sample_probs", "fuse_voxel_probs", "propagate_dilation",
           "labels_from_probs", "_offsets_by_distance"]
