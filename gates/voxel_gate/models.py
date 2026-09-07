"""The minimal learned visible-voxel corrector.

Three 3x3x3 convolutions at width 16 over the six C3 evidence channels, then a
1x1x1 zero-initialised head that emits a **residual logit**::

    z = prior(C3 occupancy) + delta_theta(features)
    p = sigmoid(z)

``prior`` is +/- ``PRIOR_LOGIT``, so with the zero-initialised head ``p`` is
sigmoid(+-4) and thresholding at 0.5 returns C3 occupancy exactly -- the model starts
as the identity on V0 and has to earn every change. The residual is unbounded in sign,
so a voxel can be deleted (large negative delta on an occupied voxel) or added (large
positive delta on an empty one).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PRIOR_LOGIT = 4.0


class VoxelCorrector3D(nn.Module):
    def __init__(self, in_ch: int = 6, ch: int = 16, n_blocks: int = 3, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        layers, c = [], in_ch
        for _ in range(n_blocks):
            layers += [nn.Conv3d(c, ch, kernel, padding=pad),
                       nn.GroupNorm(4, ch), nn.GELU()]
            c = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv3d(ch, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x[:, 0]`` must be the binary C3 occupancy channel. Returns a logit volume."""
        prior = (2.0 * x[:, 0:1] - 1.0) * PRIOR_LOGIT
        return prior + self.head(self.body(x))


def apply_region(pred: torch.Tensor, c3_occupied: torch.Tensor,
                 region: torch.Tensor) -> torch.Tensor:
    """Enforce the visible-correction protocol.

    Inside the correction region the model decides; outside it the frozen C3 occupancy is
    preserved verbatim. Because the region is a dilation of C3 occupancy it contains every
    C3-occupied voxel, so the second term is empty and "preserve C3" and "force empty"
    coincide -- both are written out here so the guarantee is explicit and testable.
    """
    return (pred & region) | (c3_occupied & ~region)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
