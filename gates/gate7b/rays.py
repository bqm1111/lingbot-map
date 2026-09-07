"""Casting one frame's accepted depth rays into an evidence volume.

The rule, fixed in the precommit and identical on all three benchmarks:

1. **free** evidence on the voxels the ray passes through, from the near plane up to
   ``band_half_m`` before the surface;
2. **occupied** evidence on the voxels whose centre lies within ``band_half_m`` of the
   surface point along the ray;
3. **nothing at all** beyond the surface. The region behind a predicted surface stays
   *unknown*, never free -- that is what makes "is this voxel occluded or did the depth
   fail?" answerable at all.

Free-space carving uses a fixed 4x4 pixel decimation (one ray in sixteen) because free
space is spatially redundant and full carving costs sixteen times more for a volume that
is already saturated. Occupied evidence uses **every** accepted ray. The decimation is
declared, uniform across benchmarks and never tuned.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

#: Half-width of the occupied band around the surface, in metres.
BAND_HALF_M = 0.2
#: Pixel decimation of the free-space carve, in each image axis.
CARVE_STRIDE = 4
#: Near plane of the carve, in metres.
CARVE_NEAR_M = 1.0


def pixel_rays(K: np.ndarray, hw: Tuple[int, int], device) -> torch.Tensor:
    """Unit-``z`` camera-frame directions ``[H, W, 3]`` for every pixel."""
    H, W = int(hw[0]), int(hw[1])
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float64),
                          torch.arange(W, device=device, dtype=torch.float64),
                          indexing="ij")
    Kt = torch.as_tensor(np.asarray(K, np.float64), device=device)
    x = (u - Kt[0, 2]) / Kt[0, 0]
    y = (v - Kt[1, 2]) / Kt[1, 1]
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)


def to_grid(points_world: torch.Tensor, T_world_to_grid: torch.Tensor, origin, voxel_size,
            dims) -> Tuple[torch.Tensor, torch.Tensor]:
    """World points -> ``(flat index, inside mask)`` on the benchmark grid."""
    p = points_world @ T_world_to_grid[:3, :3].T + T_world_to_grid[:3, 3]
    idx = torch.floor((p - origin) / float(voxel_size)).to(torch.int64)
    ok = torch.ones(idx.shape[:-1], dtype=torch.bool, device=idx.device)
    for a in range(3):
        ok &= (idx[..., a] >= 0) & (idx[..., a] < int(dims[a]))
    ix = idx[..., 0].clamp(0, int(dims[0]) - 1)
    iy = idx[..., 1].clamp(0, int(dims[1]) - 1)
    iz = idx[..., 2].clamp(0, int(dims[2]) - 1)
    flat = (ix * int(dims[1]) + iy) * int(dims[2]) + iz
    return flat, ok


def cast_frame(vol, dirs_cam: torch.Tensor, depth_m: torch.Tensor,
               accepted: torch.Tensor, T_cam_to_world: torch.Tensor,
               T_world_to_grid: torch.Tensor, origin: torch.Tensor, voxel_size: float,
               dims, time_index: int, source: int, weight: float,
               probs: Optional[torch.Tensor] = None,
               sem_weight: Optional[torch.Tensor] = None,
               carve: bool = True, band_half_m: float = BAND_HALF_M,
               carve_stride: int = CARVE_STRIDE, carve_near_m: float = CARVE_NEAR_M,
               max_depth_m: float = 60.0) -> dict:
    """Fuse one frame. Returns per-frame counts for the runtime and volume tables."""
    H, W = depth_m.shape
    R = T_cam_to_world[:3, :3]
    t = T_cam_to_world[:3, 3]

    # ---- occupied band, every accepted ray
    sel = accepted.reshape(-1).nonzero(as_tuple=True)[0]
    stats = {"n_rays": int(sel.numel()), "n_occ_updates": 0, "n_free_updates": 0}
    if sel.numel() == 0:
        return stats
    d = depth_m.reshape(-1)[sel]
    dirs = dirs_cam.reshape(-1, 3)[sel]
    ray_len = dirs.norm(dim=1)                       # |dir| per unit z
    # offsets along the ray in metres, symmetric about the surface
    n_off = max(int(np.floor(band_half_m / voxel_size)), 0)
    offsets = torch.arange(-n_off, n_off + 1, device=d.device, dtype=torch.float64)
    flats, oks = [], []
    for o in offsets:
        dz = d + o * float(voxel_size) / ray_len     # move o metres ALONG the ray
        p = dirs * dz.unsqueeze(1)
        pw = p @ R.T + t
        f, ok = to_grid(pw, T_world_to_grid, origin, voxel_size, dims)
        flats.append(f)
        oks.append(ok & (dz > 0))
    fo = torch.cat(flats)
    ko = torch.cat(oks)
    fo = fo[ko]
    vol.add_occupied(fo, time_index, source=source, weight=weight)
    stats["n_occ_updates"] = int(fo.numel())

    if probs is not None and sem_weight is not None:
        # Semantics go on EVERY voxel of the ray's occupied band, not only the surface
        # voxel. The band is what the occupancy update writes, so labelling only its
        # centre leaves the neighbours with no semantic evidence and forces the readout
        # to invent a class for them. This is the Gate-6 dilation convention -- a band
        # voxel inherits the vector of the ray that created it -- applied at ray level,
        # weighted by geometry reliability and never by the teacher's own confidence.
        sw = sem_weight.reshape(-1)[sel]
        pv = probs.reshape(-1, probs.shape[-1])[sel]
        nrep = len(offsets)
        vol.add_semantics(fo, pv.repeat(nrep, 1)[ko], sw.repeat(nrep)[ko])

    # ---- free space, decimated rays, stopping band_half_m before the surface
    if carve:
        mask2d = torch.zeros_like(accepted)
        mask2d[::carve_stride, ::carve_stride] = True
        csel = (accepted & mask2d).reshape(-1).nonzero(as_tuple=True)[0]
        if csel.numel():
            cd = depth_m.reshape(-1)[csel]
            cdir = dirs_cam.reshape(-1, 3)[csel]
            clen = cdir.norm(dim=1)
            stop = (cd - band_half_m / clen).clamp(min=carve_near_m)
            n_steps = int(np.ceil(float(max_depth_m) / float(voxel_size)))
            step = torch.arange(n_steps, device=cd.device, dtype=torch.float64)
            zz = carve_near_m + step.unsqueeze(0) * float(voxel_size) / clen.unsqueeze(1)
            valid = zz < stop.unsqueeze(1)
            pts = cdir.unsqueeze(1) * zz.unsqueeze(2)
            pw = pts @ R.T + t
            f, ok = to_grid(pw, T_world_to_grid, origin, voxel_size, dims)
            ff = f[ok & valid]
            vol.add_free(ff, weight=weight)
            stats["n_free_updates"] = int(ff.numel())
    vol.clamp()
    return stats


__all__ = ["pixel_rays", "to_grid", "cast_frame", "BAND_HALF_M", "CARVE_STRIDE",
           "CARVE_NEAR_M"]
