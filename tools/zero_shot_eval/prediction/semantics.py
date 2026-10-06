"""B0 and B3 semantic features on any grid.

* ``nearest_evidence``: B0's rule (3D Euclidean distance transform on the grid, scipy index order), plus the distance in metres.
* ``association_forward``: the frozen ``EvidenceAssociation.forward`` with the grid constants taken from ``grid``. The query
  height feature is (z - grid origin_z) / (grid height extent), exactly the frozen definition applied to the target grid.
* ``b3_features``: chunked scorer over query voxels (16,384 per chunk, 16 nearest evidence candidates), returning 1024-d features.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from tools.delayed_semantic.model import EvidenceAssociation, nearest_candidates
from tools.zero_shot_eval.prediction.grid import GridSpec, voxel_centres

CHUNK = 16384


def nearest_evidence(evidence_dense: torch.Tensor, grid: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """``(nearest evidence flat [N] int64, distance_m [N] float32)`` (``semantic_completion/evaluation/baseline_nearest`` with ``grid``)."""
    m = evidence_dense.reshape(grid.dims).cpu().numpy()
    if not m.any():
        return np.full(grid.n_voxels, -1, np.int64), np.full(grid.n_voxels, np.inf, np.float32)
    dist, idx = ndimage.distance_transform_edt(~m, return_indices=True)
    flat = np.ravel_multi_index((idx[0], idx[1], idx[2]), grid.dims).reshape(-1)
    return flat.astype(np.int64), (dist.reshape(-1) * grid.voxel_size).astype(np.float32)


@torch.no_grad()
def path_fractions(q_xyz: torch.Tensor, e_xyz: torch.Tensor, masks: torch.Tensor, samples: int, grid: GridSpec) -> torch.Tensor:
    s = (torch.arange(samples, device=q_xyz.device, dtype=torch.float32) + 0.5) / samples
    p = q_xyz[:, None, None, :] + s[None, None, :, None] * (e_xyz - q_xyz[:, None, :])[:, :, None, :]
    origin = torch.as_tensor(grid.origin, device=p.device, dtype=torch.float32)
    ijk = torch.floor((p - origin) / grid.voxel_size).long()
    for a, dim in enumerate(grid.dims):
        ijk[..., a].clamp_(0, dim - 1)
    X, Y, Z = grid.dims
    flat = (ijk[..., 0] * Y + ijk[..., 1]) * Z + ijk[..., 2]
    return masks.reshape(len(masks), -1)[:, flat].float().mean(-1).permute(1, 2, 0)


@torch.no_grad()
def association_forward(model: EvidenceAssociation, grid: GridSpec, q_trunk, q_xyz, cand, ev, masks):
    K = cand.shape[1]; r = model.cfg.range_norm_m
    e_xyz = ev["xyz"][cand]
    delta = (e_xyz - q_xyz[:, None]) / r
    sem = model.project(ev["feature"][cand]) if model.project is not None else q_trunk.new_zeros(*cand.shape, model.cfg.evidence_proj)
    edge = torch.cat([ev["trunk"][cand], sem, delta, delta.norm(dim=-1, keepdim=True),
                      (torch.log1p(ev["count"].float()) / 5)[cand, None], ev["current"].float()[cand, None], (ev["n_frames"].float() / 5)[cand, None],
                      path_fractions(q_xyz, e_xyz, masks, model.cfg.path_samples, grid)], -1)
    query = torch.cat([q_trunk, ((q_xyz[:, 2:] - grid.origin[2]) / (grid.dims[2] * grid.voxel_size))], -1)
    logits = model.score(torch.cat([query[:, None].expand(-1, K, -1), edge], -1)).squeeze(-1)
    w = torch.softmax(logits, -1)
    return F.normalize((w[..., None] * ev["feature"][cand]).sum(1), dim=-1), w


@torch.no_grad()
def b3_features(model: EvidenceAssociation, grid: GridSpec, query_flat: torch.Tensor, ev: dict, trunk_flat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """[Q, 1024] unit features for ``query_flat`` (none of which is an evidence voxel)."""
    out = []
    for s in range(0, len(query_flat), CHUNK):
        q = query_flat[s:s + CHUNK]; qxyz = voxel_centres(grid, q)
        idx, _ = nearest_candidates(qxyz, ev["xyz"], model.cfg.k_candidates)
        f, _ = association_forward(model, grid, trunk_flat[:, q].T, qxyz, idx, ev, masks)
        out.append(f)
    return torch.cat(out) if out else torch.zeros(0, ev["feature"].shape[1], device=query_flat.device)


__all__ = ["CHUNK", "nearest_evidence", "path_fractions", "association_forward", "b3_features"]
