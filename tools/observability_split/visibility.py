"""Camera visibility of the ground-truth surface, computed once per sample and shared by every method.

The split answers "could the input camera see this voxel at prediction time?". It is derived from the
*ground truth* occupancy and the *input camera* alone -- no method's output enters it -- so the same mask
is applied to ours and to OccAny.

A ray is cast through every ``pixel_stride``-th pixel of the input image and marched in ``step_m`` steps.
Every voxel a ray enters before (and including) the first GT-occupied voxel on that ray is **observed**:
free space the camera sees through, plus the surface it terminates on. The rest of the grid splits into
**occluded** (inside the image frustum, behind a surface) and **out of view** (outside the frustum).

Observed therefore contains both empty and occupied voxels, which is what makes IoU well defined inside it:
false positives are counted in the region where they occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Regions:
    """Flat [N] bool masks over the evaluation grid; ``observed | occluded | out_of_view`` covers everything."""
    observed: np.ndarray
    occluded: np.ndarray
    out_of_view: np.ndarray


def _ray_directions(K: np.ndarray, W: int, H: int, stride: int, device) -> torch.Tensor:
    u = torch.arange(0, W, stride, dtype=torch.float64, device=device) + 0.5
    v = torch.arange(0, H, stride, dtype=torch.float64, device=device) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    Kinv = torch.from_numpy(np.linalg.inv(np.asarray(K, np.float64))).to(device)
    pix = torch.stack([uu.reshape(-1), vv.reshape(-1), torch.ones(uu.numel(), dtype=torch.float64, device=device)], 1)
    d = pix @ Kinv.T
    return d / d.norm(dim=1, keepdim=True)


def observed_from_camera(gt_occupied: np.ndarray, T_cam_to_grid: np.ndarray, K: np.ndarray, image_size: Tuple[int, int],
                         dims, voxel_size: float, origin, *, pixel_stride: int = 2, step_m: float = 0.1,
                         max_range_m: float = 80.0, device: str = "cuda", chunk_rays: int = 16384) -> np.ndarray:
    """[N] bool: voxels the input camera sees through or terminates on, given the GT occupancy."""
    W, H = image_size
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X, Y, Z = (int(d) for d in dims)
    occ = torch.from_numpy(np.ascontiguousarray(gt_occupied.reshape(-1))).to(dev)
    out = torch.zeros(X * Y * Z, dtype=torch.bool, device=dev)

    T = torch.from_numpy(np.asarray(T_cam_to_grid, np.float64)).to(dev)
    cam = T[:3, 3]
    dirs = _ray_directions(K, W, H, pixel_stride, dev) @ T[:3, :3].T          # [R, 3] in grid frame
    ts = (torch.arange(int(round(max_range_m / step_m)), dtype=torch.float64, device=dev) + 0.5) * step_m
    org = torch.from_numpy(np.asarray(origin, np.float64)).to(dev)
    shape = torch.tensor([X, Y, Z], device=dev)

    for a in range(0, dirs.shape[0], chunk_rays):
        d = dirs[a:a + chunk_rays]
        pts = cam[None, None, :] + d[:, None, :] * ts[None, :, None]           # [r, S, 3]
        idx = torch.floor((pts - org) / voxel_size).long()
        inside = ((idx >= 0) & (idx < shape)).all(-1)
        flat = (idx[..., 0].clamp(0, X - 1) * Y + idx[..., 1].clamp(0, Y - 1)) * Z + idx[..., 2].clamp(0, Z - 1)
        hit = inside & occ[flat]
        before = torch.cumsum(hit.to(torch.int16), 1) - hit.to(torch.int16)    # hits strictly before this step
        seen = inside & (before == 0)
        out[flat[seen]] = True
    return out.cpu().numpy()


def frustum(T_cam_to_grid: np.ndarray, K: np.ndarray, image_size: Tuple[int, int], dims, voxel_size: float,
            origin) -> np.ndarray:
    """[N] bool: voxel centre in front of the camera and inside the image rectangle."""
    W, H = image_size
    X, Y, Z = (int(d) for d in dims)
    ijk = np.stack(np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"), -1).reshape(-1, 3)
    c = (ijk + 0.5) * voxel_size + np.asarray(origin, np.float64)
    Ti = np.linalg.inv(np.asarray(T_cam_to_grid, np.float64))
    cam = c @ Ti[:3, :3].T + Ti[:3, 3]
    z = cam[:, 2]; ok = z > 0
    zs = np.where(ok, z, 1.0)
    u = np.asarray(K)[0, 0] * cam[:, 0] / zs + np.asarray(K)[0, 2]
    v = np.asarray(K)[1, 1] * cam[:, 1] / zs + np.asarray(K)[1, 2]
    return ok & (u >= 0) & (u <= W) & (v >= 0) & (v <= H)


def regions(gt_occupied: np.ndarray, valid: np.ndarray, T_cam_to_grid, K, image_size, dims, voxel_size, origin,
            **kw) -> Regions:
    obs = observed_from_camera(gt_occupied, T_cam_to_grid, K, image_size, dims, voxel_size, origin, **kw)
    fov = frustum(T_cam_to_grid, K, image_size, dims, voxel_size, origin)
    obs = obs & fov                                                            # rays never leave the frustum; belt and braces
    return Regions(valid & obs, valid & ~obs & fov, valid & ~obs & ~fov)


__all__ = ["Regions", "observed_from_camera", "frustum", "regions"]
