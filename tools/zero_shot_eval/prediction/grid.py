"""Grid-parametrised versions of the voxel helpers the frozen pipeline hard-codes for the KITTI-360 grid.

Every formula is copied from the frozen code with the grid constants replaced by a :class:`GridSpec`; with the KITTI-360
grid each function is bit-identical to the original (``tests/zero_shot_eval``).

Conventions: flat index = ravel_multi_index((x, y, z)) in C order; voxel centre = (ijk + 0.5) * voxel_size + origin, metres,
in the grid frame (SemanticKITTI / KITTI-360: velodyne of the target frame; Occ3D: ego of the target keyframe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from core.mapping.mapper import pack

#: the mapper's world-lattice key size (``core/mapping``); Gate 15 keys rays with 0.2 m as well
MAPPER_KEY_VOXEL_M = 0.2


@dataclass(frozen=True)
class GridSpec:
    dims: Tuple[int, int, int]
    voxel_size: float
    origin: Tuple[float, float, float]
    frame: str
    name: str

    @property
    def n_voxels(self) -> int:
        return int(self.dims[0] * self.dims[1] * self.dims[2])

    @staticmethod
    def of(voxel_grid) -> "GridSpec":
        return GridSpec(tuple(int(d) for d in voxel_grid.dims), float(voxel_grid.voxel_size), tuple(float(o) for o in voxel_grid.origin),
                        str(voxel_grid.frame), str(voxel_grid.name))


def ijk_of(grid: GridSpec, flat: torch.Tensor) -> torch.Tensor:
    X, Y, Z = grid.dims
    return torch.stack([flat // (Y * Z), (flat // Z) % Y, flat % Z], -1)


def voxel_centres(grid: GridSpec, flat: torch.Tensor) -> torch.Tensor:
    """[M] flat -> [M, 3] float32 metric centres (``delayed_semantic.model.voxel_centres`` with ``grid``)."""
    ijk = ijk_of(grid, flat).float()
    return (ijk + 0.5) * grid.voxel_size + torch.as_tensor(grid.origin, device=flat.device, dtype=torch.float32)


def world_keys(grid: GridSpec, T_grid_to_world: np.ndarray, device) -> torch.Tensor:
    """Mapper world key of every grid voxel centre, flat C order (``semantic_completion.evidence.lift.grid_world_keys`` with ``grid``)."""
    X, Y, Z = grid.dims
    ix, iy, iz = torch.meshgrid(torch.arange(X, device=device), torch.arange(Y, device=device), torch.arange(Z, device=device), indexing="ij")
    c = (torch.stack([ix, iy, iz], -1).reshape(-1, 3).double() + 0.5) * grid.voxel_size + torch.as_tensor(grid.origin, dtype=torch.float64, device=device)
    T = torch.as_tensor(np.asarray(T_grid_to_world, np.float64), device=device)
    w = c @ T[:3, :3].T + T[:3, 3]
    return pack(torch.floor(w / MAPPER_KEY_VOXEL_M).to(torch.int64))


def subvoxels(native_flat: torch.Tensor, native: GridSpec, fine: GridSpec, ratio: int) -> torch.Tensor:
    """[M] native flat -> [M, ratio^3] fine flat (ratio 1 -> the voxel itself)."""
    ijk = ijk_of(native, native_flat) * ratio
    o = torch.stack(torch.meshgrid(*[torch.arange(ratio, device=native_flat.device)] * 3, indexing="ij"), -1).reshape(-1, 3)
    s = ijk[:, None, :] + o[None]
    X, Y, Z = fine.dims
    return (s[..., 0] * Y + s[..., 1]) * Z + s[..., 2]


def reduce_any(mask_fine: torch.Tensor, fine: GridSpec, ratio: int) -> torch.Tensor:
    """Fine flat bool -> native flat bool, occupied iff any subvoxel (the frozen Occ3D rule)."""
    if ratio == 1:
        return mask_fine
    X, Y, Z = fine.dims
    return mask_fine.reshape(X // ratio, ratio, Y // ratio, ratio, Z // ratio, ratio).amax(dim=(1, 3, 5)).reshape(-1)


def reduce_min(values_fine: torch.Tensor, fine: GridSpec, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return values_fine
    X, Y, Z = fine.dims
    return values_fine.reshape(X // ratio, ratio, Y // ratio, ratio, Z // ratio, ratio).amin(dim=(1, 3, 5)).reshape(-1)


__all__ = ["MAPPER_KEY_VOXEL_M", "GridSpec", "ijk_of", "voxel_centres", "world_keys", "subvoxels", "reduce_any", "reduce_min"]
