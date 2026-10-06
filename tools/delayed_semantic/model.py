"""Learned evidence association (B1 / B2): each query voxel reads a weighted mix of nearby OBSERVED evidence.

For a query voxel q (a voxel that is not a student evidence voxel) and its K nearest observed evidence
voxels e_1..e_K (3D Euclidean, all from H_t):

    score_k = MLP([ trunk(q) 24, height(q) 1,                                   # frozen geometry at q
                    trunk(e_k) 24, P f(e_k) 16,                                  # geometry at e_k, vocabulary-free projection
                    (e_k - q)/6 m 3, |e_k - q|/6 m 1,
                    log1p(count(e_k))/5 1, observed in frame t 1, frames observing e_k / 5 1,   # provenance
                    fraction of the q->e_k segment in hard-map free F, surface S, predicted occupied O  3 ])
    out(q)  = normalize( sum_k softmax(score)_k f(e_k) )        [1024], language-compatible teacher space

Shapes: Q queries, K candidates, N evidence voxels. ``f`` = unit Trident-H evidence feature [N, 1024].
The output is a convex combination of observed features: a class that no candidate carries cannot be
expressed (stated limitation). Parameters ~26 k; no convolution over the grid, no attention matrix beyond Q x K.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.delayed_semantic import config as C

N_EDGE_FEATURES = 24 + 16 + 3 + 1 + 1 + 1 + 1 + 3
N_QUERY_FEATURES = 24 + 1


def voxel_centres(flat: torch.Tensor) -> torch.Tensor:
    """[M] flat C-order index -> [M, 3] metric centres in the anchor grid frame."""
    X, Y, Z = C.GRID_DIMS
    ijk = torch.stack([flat // (Y * Z), (flat // Z) % Y, flat % Z], -1).float()
    return (ijk + 0.5) * C.VOXEL_SIZE_M + torch.as_tensor(C.GRID_ORIGIN_M, device=flat.device, dtype=torch.float32)


@torch.no_grad()
def nearest_candidates(query_xyz: torch.Tensor, evidence_xyz: torch.Tensor, k: int, chunk: int = 8192):
    """``(index [Q, k] into evidence rows, distance [Q, k] m)``, ascending distance."""
    k = min(k, len(evidence_xyz))
    idx, dist = [], []
    for s in range(0, len(query_xyz), chunk):
        d, i = torch.cdist(query_xyz[s:s + chunk], evidence_xyz).topk(k, largest=False)
        idx.append(i); dist.append(d)
    return torch.cat(idx), torch.cat(dist)


@torch.no_grad()
def path_fractions(q_xyz: torch.Tensor, e_xyz: torch.Tensor, masks: torch.Tensor, samples: int) -> torch.Tensor:
    """Fraction of ``samples`` points on each segment q->e inside each dense mask. masks [M, X, Y, Z] bool -> [Q, K, M]."""
    s = (torch.arange(samples, device=q_xyz.device, dtype=torch.float32) + 0.5) / samples
    p = q_xyz[:, None, None, :] + s[None, None, :, None] * (e_xyz - q_xyz[:, None, :])[:, :, None, :]       # [Q, K, S, 3]
    origin = torch.as_tensor(C.GRID_ORIGIN_M, device=p.device, dtype=torch.float32)
    ijk = torch.floor((p - origin) / C.VOXEL_SIZE_M).long()
    for a, dim in enumerate(C.GRID_DIMS):
        ijk[..., a].clamp_(0, dim - 1)
    X, Y, Z = C.GRID_DIMS
    flat = (ijk[..., 0] * Y + ijk[..., 1]) * Z + ijk[..., 2]
    return masks.reshape(len(masks), -1)[:, flat].float().mean(-1).permute(1, 2, 0)                         # [Q, K, M]


class EvidenceAssociation(nn.Module):
    def __init__(self, cfg: C.ModelConfig = C.ModelConfig(), semantic_projection: bool = True):
        """``semantic_projection=False`` (Gate 16 B3geom): the scorer sees zeros instead of the candidate's semantic projection;
        the output mixture still uses the semantic features."""
        super().__init__()
        self.cfg = cfg
        self.semantic_projection = semantic_projection
        self.project = nn.Linear(C.TEACHER_DIM, cfg.evidence_proj) if semantic_projection else None
        self.score = nn.Sequential(nn.Linear(N_QUERY_FEATURES + N_EDGE_FEATURES, cfg.hidden), nn.GELU(),
                                   nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))

    def forward(self, q_trunk, q_xyz, cand, ev: Dict[str, torch.Tensor], masks: torch.Tensor, return_logits: bool = False):
        """q_trunk [Q, 24], q_xyz [Q, 3], cand [Q, K] rows into ``ev``; ev: feature [N, 1024], trunk [N, 24], xyz [N, 3],
        count [N], current [N] bool, n_frames [N]; masks [3, X, Y, Z] = (F, S, O).
        Returns (out [Q, 1024], weights [Q, K]) and, with ``return_logits``, the scorer logits [Q, K]."""
        K = cand.shape[1]; r = self.cfg.range_norm_m
        e_xyz = ev["xyz"][cand]
        delta = (e_xyz - q_xyz[:, None]) / r
        sem = self.project(ev["feature"][cand]) if self.project is not None else q_trunk.new_zeros(*cand.shape, self.cfg.evidence_proj)
        edge = torch.cat([ev["trunk"][cand], sem, delta, delta.norm(dim=-1, keepdim=True),
                          (torch.log1p(ev["count"].float()) / 5)[cand, None], ev["current"].float()[cand, None], (ev["n_frames"].float() / 5)[cand, None],
                          path_fractions(q_xyz, e_xyz, masks, self.cfg.path_samples)], -1)
        query = torch.cat([q_trunk, ((q_xyz[:, 2:] - C.GRID_ORIGIN_M[2]) / (C.GRID_DIMS[2] * C.VOXEL_SIZE_M))], -1)
        logits = self.score(torch.cat([query[:, None].expand(-1, K, -1), edge], -1)).squeeze(-1)
        w = torch.softmax(logits, -1)
        out = F.normalize((w[..., None] * ev["feature"][cand]).sum(1), dim=-1)
        return (out, w, logits) if return_logits else (out, w)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict:
        return asdict(self.cfg)


def teacher_distribution(feature: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """softmax(40 * cos(f, text)) over the 18 frozen prompts. feature [..., 1024] unit norm -> [..., 18]."""
    return torch.softmax(C.LOGIT_SCALE * (F.normalize(feature, dim=-1) @ text.T), -1)


def kl_to_target(target_prob: torch.Tensor, feature: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """mean_q KL(target || student): the student's class distribution should match the (soft) target distribution."""
    logp = torch.log_softmax(C.LOGIT_SCALE * (feature @ text.T), -1)
    t = target_prob.clamp_min(1e-8)
    return (t * (t.log() - logp)).sum(-1).mean()


__all__ = ["voxel_centres", "nearest_candidates", "path_fractions", "EvidenceAssociation", "teacher_distribution", "kl_to_target"]
