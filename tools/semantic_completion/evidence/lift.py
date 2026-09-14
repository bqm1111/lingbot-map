"""Lift the five causal Trident feature maps onto the grid at the frozen mapper's own surface voxels.

Association rule (the simplest one the frozen geometry supports, stated once):

* a ray is a lattice pixel the mapper **accepts** (conf >= 1.5, 1 m < s*z < 60 m) -- the exact
  ``IncrementalMapper._integrate`` test on the same canonical depth, the same clip scale ``s`` and the
  same anchor-identity pose ``inv(P_4) P_q`` with translation scaled by ``s``;
* its surface point is the ray endpoint ``p = dir * s*z`` in the anchor-camera world frame (metres),
  keyed exactly like the mapper: ``pack(floor(p / 0.2))``;
* a grid voxel (flat C-order index on the 256x256x32 velodyne grid) is looked up through its centre,
  exactly like ``IncrementalMapper.query``;
* **evidence voxel** = a grid voxel whose world key received at least one endpoint AND that the frozen
  hard map marks occupied (``S_H``). Nothing else carries semantic evidence -- in particular no free,
  carved, completed or future voxel.

Aggregation: per world key, the count-weighted mean of the unit-norm teacher features of its endpoints
(one ray, weight 1), then L2-normalised. No learned fusion, no confidence weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from core.mapping import depth as D7, rays as RY
from core.mapping.mapper import pack
from tools.baseline_reconstruction.current.geometry import LocalClip, frame_inputs
from tools.semantic_completion import config as C
from tools.semantic_completion.evidence.trident import sample_at_lattice


@dataclass
class VoxelEvidence:
    """Sparse semantic evidence of one anchor.

    flat_index: [N] int64, C-order index on the 256x256x32 grid, ascending, all inside S_H
    feature:    [N, 1024] float32, unit L2 norm (teacher space)
    count:      [N] int32, number of causal rays whose endpoint landed in the voxel's world key
    """
    flat_index: torch.Tensor
    feature: torch.Tensor
    count: torch.Tensor

    def dense_mask(self) -> torch.Tensor:
        m = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=self.flat_index.device)
        m[self.flat_index] = True
        return m


def grid_world_keys(T_grid_to_world: np.ndarray, device) -> torch.Tensor:
    """World key of every grid voxel centre, flat C order -- the lookup ``IncrementalMapper.query`` performs."""
    X, Y, Z = C.GRID_DIMS
    ix, iy, iz = torch.meshgrid(torch.arange(X, device=device), torch.arange(Y, device=device), torch.arange(Z, device=device), indexing="ij")
    c = (torch.stack([ix, iy, iz], -1).reshape(-1, 3).double() + 0.5) * C.VOXEL_SIZE_M + torch.as_tensor(C.GRID_ORIGIN_M, dtype=torch.float64, device=device)
    T = torch.as_tensor(np.asarray(T_grid_to_world, np.float64), device=device)
    w = c @ T[:3, :3].T + T[:3, 3]
    return pack(torch.floor(w / C.VOXEL_SIZE_M).to(torch.int64))


def ray_endpoints(clip: LocalClip, device):
    """Per frame: ``(world keys [n], accepted lattice pixel flat indices [n])`` of the mapper's accepted rays."""
    s = float(clip.scale_m_per_unit)
    out = []
    for f in frame_inputs(clip, device):
        d_m = s * f.depth_canonical.to(torch.float64)
        acc = (f.conf.to(torch.float64) >= D7.CONF_THRESHOLD) & torch.isfinite(d_m) & (d_m > D7.MIN_DEPTH_M) & (d_m < D7.MAX_DEPTH_M)
        T = np.asarray(f.pose_c2w_canonical, np.float64).copy(); T[:3, 3] *= s        # the mapper's single scale on translation
        Tw = torch.as_tensor(T, device=device)
        sel = acc.reshape(-1).nonzero(as_tuple=True)[0]
        dirs = RY.pixel_rays(f.K, d_m.shape, device).reshape(-1, 3)[sel]
        p = dirs * d_m.reshape(-1)[sel].unsqueeze(1)
        pw = p @ Tw[:3, :3].T + Tw[:3, 3]
        out.append((pack(torch.floor(pw / C.VOXEL_SIZE_M).to(torch.int64)), sel))
    return out


def lift_clip_evidence(clip: LocalClip, token_maps: Sequence[torch.Tensor], S_occupied: torch.Tensor,
                       T_cam_to_grid: np.ndarray, device) -> VoxelEvidence:
    """Evidence voxels of one clip. ``token_maps[q]``: ``[1024, 34, 128]`` of frame ``t-4+q``; ``S_occupied``: ``[N_VOXELS]`` bool."""
    assert len(token_maps) == len(clip.stream_indices)
    lattice_hw = clip.depth_canonical.shape[1:]
    keys, feats = [], []
    for (k, pix), tokens in zip(ray_endpoints(clip, device), token_maps):
        keys.append(k); feats.append(sample_at_lattice(tokens, lattice_hw).reshape(-1, C.TEACHER_DIM)[pix])
    keys, feats = torch.cat(keys), torch.cat(feats)
    uq, inv = torch.unique(keys, return_inverse=True)
    fsum = torch.zeros(len(uq), C.TEACHER_DIM, device=device).index_add_(0, inv, feats)
    cnt = torch.zeros(len(uq), dtype=torch.int32, device=device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.int32))
    gk = grid_world_keys(np.linalg.inv(np.asarray(T_cam_to_grid, np.float64)), device)
    pos = torch.searchsorted(uq, gk).clamp_(max=len(uq) - 1)
    hit = (uq[pos] == gk) & S_occupied.to(device)
    flat = hit.nonzero(as_tuple=True)[0]
    row = pos[flat]
    feature = torch.nn.functional.normalize(fsum[row], dim=-1)
    return VoxelEvidence(flat, feature, cnt[row])


def save_evidence(ev: VoxelEvidence, path: str, meta: Dict) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, flat_index=ev.flat_index.cpu().numpy().astype(np.int32), feature=ev.feature.cpu().numpy().astype(np.float16),
             count=ev.count.cpu().numpy().astype(np.int32), **meta)
    os.replace(tmp, path)


def load_evidence(path: str, device) -> VoxelEvidence:
    with np.load(path) as z:
        flat = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device)
        feat = torch.nn.functional.normalize(torch.from_numpy(z["feature"].astype(np.float32)).to(device), dim=-1)  # fp16 storage -> renormalise
        cnt = torch.from_numpy(z["count"]).to(device)
    return VoxelEvidence(flat, feat, cnt)


__all__ = ["VoxelEvidence", "grid_world_keys", "ray_endpoints", "lift_clip_evidence", "save_evidence", "load_evidence"]
