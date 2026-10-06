"""B4 inference: full-grid hidden field once, 1024-d semantic features at query voxels.

Inputs come from ``tools/delayed_semantic/gate16/context.student_context`` (causal, PREDICT_FORBIDDEN audit):
trunk [24, X, Y, Z], hard map S / F and occupancy O [N] bool, observed evidence (flat, unit features, counts).
"""
from __future__ import annotations

from typing import Optional

import torch

from tools.semantic_field.data.semantic_evidence import SparseSemanticEvidence, dense_semantic_channels
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead

CHUNK = 262144


def load_b4(path: str, device) -> SemanticFieldHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SemanticFieldHead(); model.load_state_dict(ck["state_dict"]); return model.to(device).eval()


@torch.no_grad()
def hidden_field(model: SemanticFieldHead, ctx: dict, evidence: SparseSemanticEvidence, mu: torch.Tensor, grid_dims) -> torch.Tensor:
    """``[32, N]`` hidden field for the given (possibly masked) evidence."""
    X, Y, Z = grid_dims
    geo = torch.stack([ctx["S"].reshape(X, Y, Z), ctx["F"].reshape(X, Y, Z), ctx["O"].reshape(X, Y, Z)]).float()
    dense = torch.cat([ctx["trunk"].float(), geo, dense_semantic_channels(evidence, model.project_evidence, mu, grid_dims)], 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.hidden_field(dense[None])
    return h[0].float().reshape(h.shape[1], -1)


@torch.no_grad()
def b4_features(model: SemanticFieldHead, ctx: dict, query_flat: torch.Tensor, mu: torch.Tensor, grid_dims,
                evidence: Optional[SparseSemanticEvidence] = None) -> torch.Tensor:
    """[Q, 1024] unit features. ``evidence`` defaults to all observed evidence of the anchor."""
    if evidence is None:
        ev = ctx["ev"]; evidence = SparseSemanticEvidence(ev.flat_index, ev.feature, ev.count)
    h = hidden_field(model, ctx, evidence, mu, grid_dims)
    return torch.cat([model.semantic_features(h[:, query_flat[s:s + CHUNK]].T) for s in range(0, len(query_flat), CHUNK)]) if len(query_flat) \
        else torch.zeros(0, 1024, device=h.device)


__all__ = ["load_b4", "hidden_field", "b4_features"]
