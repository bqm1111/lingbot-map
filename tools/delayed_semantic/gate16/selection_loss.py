"""The B3 delayed term: KL(candidate target || softmax(scorer logits)), averaged over queries."""
from __future__ import annotations

import torch


def selection_kl(target_distribution: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """target_distribution [Q, K] (rows sum to 1), logits [Q, K] -> scalar mean KL."""
    logq = torch.log_softmax(logits.float(), -1)
    t = target_distribution.float()
    return torch.where(t > 0, t * (torch.log(t.clamp_min(1e-12)) - logq), torch.zeros_like(t)).sum(-1).mean()


@torch.no_grad()
def selection_stats(target_best: torch.Tensor, target_distribution: torch.Tensor, weights: torch.Tensor) -> dict:
    """Per-batch diagnostics: top-1 / top-3 agreement with argmax compatibility, mean weight on the best candidate, KL."""
    top3 = weights.topk(min(3, weights.shape[1]), -1).indices
    return {"top1": float((weights.argmax(-1) == target_best).float().mean()),
            "top3": float((top3 == target_best[:, None]).any(-1).float().mean()),
            "p_best": float(weights.gather(1, target_best[:, None]).mean()),
            "kl": float(selection_kl(target_distribution, torch.log(weights.clamp_min(1e-12))))}


__all__ = ["selection_kl", "selection_stats"]
