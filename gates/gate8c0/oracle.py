"""Oracle geometry: ground-truth LiDAR through the adapter path, and how well it lands.

Nothing here is a deployable baseline. Stage 2 uses the anchor frame's own velodyne sweep
to ask a single question -- *do measured surfaces land on official occupied voxels?* --
and Stage 3 extends it over causal history. Both use ground-truth poses and ground-truth
LiDAR, so any misalignment they find is a property of the pipeline's conventions, not of a
learned model.

The distance statistics answer the complaint that a sparse sweep cannot recall a dense
completion target: recall is expected to be low, but **precision and surface distance must
be high and small**. A one-voxel systematic offset shows up as a median distance of
0.2 m rather than 0 m, and as a large improvement under an integer shift.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from gates.gate8c0 import transforms as TF


def occupied_from_target(target: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Official occupied mask: a valid voxel whose class is not the empty class."""
    return (target != TF.GRID.empty_class) & keep


def volume_from_points(pts_grid_frame: np.ndarray, grid=TF.GRID) -> np.ndarray:
    """Boolean occupancy volume of points already in the grid frame."""
    idx, _ = TF.voxelize(pts_grid_frame, grid)
    vol = np.zeros(tuple(int(d) for d in grid.dims), bool)
    if len(idx):
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return vol


def counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    p = pred & valid
    g = gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "n_pred": int(p.sum()), "n_gt": int(g.sum()), "n_valid": int(valid.sum()),
            "pred_fraction": float(p.sum() / max(valid.sum(), 1)),
            "gt_fraction": float(g.sum() / max(valid.sum(), 1))}


def distance_stats(pred: np.ndarray, gt_occ: np.ndarray, voxel: float = TF.VOXEL
                   ) -> Dict[str, float]:
    """Distance in metres from every predicted voxel to the nearest GT-occupied voxel."""
    from scipy import ndimage
    if not gt_occ.any() or not pred.any():
        return {"n": int(pred.sum()), "median_m": float("nan"), "p90_m": float("nan"),
                "p95_m": float("nan"), "mean_m": float("nan"), "frac_within_1_voxel": float("nan")}
    d = ndimage.distance_transform_edt(~gt_occ, sampling=(voxel, voxel, voxel))
    v = d[pred]
    return {"n": int(v.size), "median_m": float(np.median(v)),
            "p90_m": float(np.percentile(v, 90)), "p95_m": float(np.percentile(v, 95)),
            "mean_m": float(v.mean()),
            "frac_within_1_voxel": float((v <= voxel * 1.001).mean()),
            "frac_exact": float((v == 0.0).mean())}


def shift_volume(vol: np.ndarray, shift: Sequence[int]) -> np.ndarray:
    """Integer roll with zero fill -- a diagnostic only, never applied to a result."""
    out = np.zeros_like(vol)
    sl_src, sl_dst = [], []
    for n, s in zip(vol.shape, shift):
        if s >= 0:
            sl_src.append(slice(0, n - s)); sl_dst.append(slice(s, n))
        else:
            sl_src.append(slice(-s, n)); sl_dst.append(slice(0, n + s))
    out[tuple(sl_dst)] = vol[tuple(sl_src)]
    return out


SHIFTS = [(0, 0, 0)] + [tuple(s * (a == k) for a in range(3)) for k in range(3) for s in (-1, 1)]


def shift_scan(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
               shifts: Iterable[Sequence[int]] = SHIFTS) -> Dict[str, Dict[str, float]]:
    """IoU/precision under +-1 voxel shifts along each axis. Diagnosis of a mismatch."""
    return {str(tuple(s)): counts(shift_volume(pred, s), gt, valid) for s in shifts}


def frustum_mask(K: np.ndarray, hw: Tuple[int, int], T_grid_to_cam: np.ndarray,
                 grid=TF.GRID, near: float = 1.0, far: float = 60.0) -> np.ndarray:
    """Grid voxels whose centre projects inside the image and lies in [near, far]."""
    dims = tuple(int(d) for d in grid.dims)
    ii, jj, kk = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    idx = np.stack([ii, jj, kk], -1).reshape(-1, 3)
    c = TF.centres(idx, grid)
    pc = TF.apply(T_grid_to_cam, c)
    z = pc[:, 2]
    ok = (z > near) & (z < far)
    u = np.full(len(pc), -1.0); v = np.full(len(pc), -1.0)
    K = np.asarray(K, np.float64)
    u[ok] = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
    H, W = hw
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return ok.reshape(dims)


__all__ = ["occupied_from_target", "volume_from_points", "counts", "distance_stats",
           "shift_volume", "shift_scan", "frustum_mask", "SHIFTS"]
