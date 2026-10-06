"""Fixed (unlearned) completion rules sharing the observed-evidence path.

B0  = Gate 15 nearest observed evidence voxel, 3D Euclidean (``tools/semantic_completion/evaluation/baseline_nearest``, reused).
B0g = geodesic nearest observed evidence: labels spread from evidence voxels through 6-connected voxels that are
      predicted occupied (or evidence) -- one voxel per iteration, fixed direction order (+x, -x, +y, -y, +z, -z) --
      so evidence does not jump across free space to a disconnected surface. Voxels the spread never reaches keep B0.
"""
from __future__ import annotations

import torch

from tools.delayed_semantic import config as C


@torch.no_grad()
def geodesic_label(evidence_label: torch.Tensor, evidence_mask: torch.Tensor, passable: torch.Tensor, fallback: torch.Tensor,
                   max_iter: int = 1024) -> tuple:
    """``(label [N_VOXELS] long, reached [N_VOXELS] bool)``; all inputs flat on one device."""
    X, Y, Z = C.GRID_DIMS
    lab = torch.where(evidence_mask, evidence_label, torch.zeros_like(evidence_label)).reshape(X, Y, Z)
    reached = evidence_mask.reshape(X, Y, Z).clone()
    pas = (passable | evidence_mask).reshape(X, Y, Z)
    for _ in range(max_iter):
        new_lab = torch.zeros_like(lab); got = torch.zeros_like(reached)
        for dim in range(3):
            for sgn in (1, -1):
                src_r = torch.zeros_like(reached); src_l = torch.zeros_like(lab)
                sl_dst = [slice(None)] * 3; sl_src = [slice(None)] * 3
                if sgn == 1:
                    sl_dst[dim], sl_src[dim] = slice(1, None), slice(None, -1)
                else:
                    sl_dst[dim], sl_src[dim] = slice(None, -1), slice(1, None)
                src_r[tuple(sl_dst)] = reached[tuple(sl_src)]; src_l[tuple(sl_dst)] = lab[tuple(sl_src)]
                take = src_r & ~got & ~reached & pas
                new_lab = torch.where(take, src_l, new_lab); got |= take
        if not got.any():
            break
        lab = torch.where(got, new_lab, lab); reached |= got
    reached = reached.reshape(-1)
    return torch.where(reached, lab.reshape(-1), fallback), reached


__all__ = ["geodesic_label"]
