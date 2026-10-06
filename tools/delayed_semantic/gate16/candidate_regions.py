"""EVALUATION ONLY: candidate-hit / candidate-miss and the oracle candidate selection. Every function takes semantic GT.

Nothing in training imports this module (tests/delayed_semantic/test_gate16.py checks the import closure).

    hit[q]      some candidate's readout class equals GT        miss = ~hit
    oracle[q]   GT where hit; otherwise the readout class of the candidate with the largest p_i[GT]
"""
from __future__ import annotations

import torch

from tools.delayed_semantic.gate16.candidates import CandidateSet


def candidate_hit(cs: CandidateSet, gt: torch.Tensor) -> torch.Tensor:
    """gt [Q] int64 in 1..18 -> [Q] bool."""
    return ((cs.classes == gt[:, None]) & cs.valid).any(-1)


def oracle_label(cs: CandidateSet, gt: torch.Tensor) -> torch.Tensor:
    """[Q] int64: the best label any selection among the candidates can produce by picking one candidate."""
    p_gt = cs.semantic_probs.gather(-1, (gt - 1)[:, None, None].expand(-1, cs.semantic_probs.shape[1], 1)).squeeze(-1)
    pick = p_gt.masked_fill(~cs.valid, -1.0).argmax(-1)
    fallback = cs.classes.gather(1, pick[:, None]).squeeze(1)
    return torch.where(candidate_hit(cs, gt), gt, fallback)


__all__ = ["candidate_hit", "oracle_label"]
