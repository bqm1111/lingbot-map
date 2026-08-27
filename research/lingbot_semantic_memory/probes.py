"""Parameter-matched semantic probes.

Every representation gets the *same* architecture -- LayerNorm, linear, GELU, linear,
L2 normalise -- but the representations differ in width (1024 for the pre-GCT encoder
tokens, 2048 for the ``frame ‖ global`` GCT block outputs).  Holding the hidden width
fixed would hand the GCT probes ~60 % more parameters than the encoder probe and
confound the comparison, so the hidden width is instead **solved per input dimension**
to hit a shared parameter budget.  The resulting counts agree to well under 0.1 %.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def probe_param_count(d_in: int, hidden: int, d_out: int) -> int:
    """Exact trainable-parameter count of :class:`SemanticProbe`."""
    return 2 * d_in + (d_in * hidden + hidden) + (hidden * d_out + d_out)


def solve_hidden(d_in: int, d_out: int, target_params: int) -> int:
    """Largest hidden width whose probe stays within ``target_params``."""
    per_unit = d_in + 1 + d_out
    hidden = (target_params - 2 * d_in - d_out) // per_unit
    if hidden < 1:
        raise ValueError(f"target_params={target_params} too small for d_in={d_in}")
    return int(hidden)


class SemanticProbe(nn.Module):
    """Small sidecar decoding a frozen representation into the teacher's feature space.

    Args:
        d_in: Width of the frozen representation.
        d_out: Teacher feature width.
        hidden: Hidden width; solve it with :func:`solve_hidden` for a matched budget.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.d_in, self.d_out, self.hidden = d_in, d_out, hidden
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., d_in]`` to L2-normalised ``[..., d_out]``."""
        y = self.fc2(self.act(self.fc1(self.norm(x))))
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_matched_probe(d_in: int, d_out: int, target_params: int,
                        max_params: int) -> Tuple[SemanticProbe, int]:
    """Build a probe whose parameter count matches ``target_params`` as closely as possible."""
    hidden = solve_hidden(d_in, d_out, target_params)
    probe = SemanticProbe(d_in, d_out, hidden)
    if probe.num_params > max_params:
        raise ValueError(f"probe has {probe.num_params} params, over the {max_params} cap")
    return probe, hidden


def cosine_distillation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``1 - cos(pred, target)`` averaged over tokens; both inputs are L2-normalised."""
    t = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (1.0 - (pred * t).sum(-1)).mean()
