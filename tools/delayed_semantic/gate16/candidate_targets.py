"""Candidate-level delayed targets: which observed candidate best matches what was later seen at the query voxel.

Inputs are frozen and label-free: candidate readout distributions p_i [Q, K, 18] and the delayed target distribution
q_v [Q, 18] (mean of the voxel's later camera views). No semantic GT is accepted by any function here.

    compatibility  r_i = <p_i, q_v>                in [0, 1] (both are probability vectors)
    distribution   target_i = softmax(r_i / tau)   sums to 1 over the K candidates; tau = 1/40
    best_index     argmax_i r_i                    (first index on ties); diagnostic hard target only
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.delayed_semantic.gate16 import config as G


@dataclass
class CandidateTarget:
    compatibility: torch.Tensor   # [Q, K] float in [0, 1]
    distribution: torch.Tensor    # [Q, K] float, rows sum to 1
    best_index: torch.Tensor      # [Q] int64


def build_target(semantic_probs: torch.Tensor, delayed_prob: torch.Tensor, valid: torch.Tensor | None = None,
                 tau: float = G.TAU_TARGET) -> CandidateTarget:
    assert semantic_probs.dim() == 3 and delayed_prob.dim() == 2 and semantic_probs.shape[-1] == delayed_prob.shape[-1]
    r = (semantic_probs.float() * delayed_prob.float()[:, None, :]).sum(-1)
    logits = r / tau
    if valid is not None:
        logits = logits.masked_fill(~valid, float("-inf"))
    return CandidateTarget(r, torch.softmax(logits, -1), r.masked_fill(~valid, -1.0).argmax(-1) if valid is not None else r.argmax(-1))


__all__ = ["CandidateTarget", "build_target"]
