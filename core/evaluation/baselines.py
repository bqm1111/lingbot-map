# extracted from gate8a/baselines.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Non-learned reference predictions, all defined on the evaluation grid.

Two of them exist specifically to make over-prediction visible. Gate 8's completion beat
the incremental mapper on every benchmark and still lost to "declare everything occupied"
on the held-out set, so a completion is only credited here if it also beats

* ``all_valid_occupied`` -- every valid voxel occupied;
* ``editable_fill``      -- every *editable* voxel occupied, protected mapper cells kept;
* ``random_editable``    -- editable voxels flipped at random to the learned model's own
  predicted density, protected cells kept, several fixed seeds.

The last two are the honest null for a residual module: they spend exactly the freedom the
residual has, and exactly as much of it, without looking at the input.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def all_valid_occupied(valid: torch.Tensor) -> torch.Tensor:
    return valid.clone()


def editable_fill(base_pred: torch.Tensor, editable: torch.Tensor) -> torch.Tensor:
    """Occupy every editable voxel; every protected voxel keeps the mapper's answer."""
    return base_pred | editable


def matched_density_q(n_target: int, base_pred: torch.Tensor, editable: torch.Tensor,
                      forced: torch.Tensor, valid: torch.Tensor) -> float:
    """Bernoulli rate that makes the random baseline predict ``n_target`` occupied cells."""
    fixed = int((base_pred & ~editable & valid).sum()) + int((forced & editable & valid).sum())
    free = int((editable & ~forced & valid).sum())
    if free == 0:
        return 0.0
    return float(np.clip((n_target - fixed) / free, 0.0, 1.0))


def random_editable(base_pred: torch.Tensor, editable: torch.Tensor, forced: torch.Tensor,
                    valid: torch.Tensor, q: float, seed: int, offset: int = 0
                    ) -> torch.Tensor:
    """Protected cells keep the mapper's answer; forced-occupied cells stay occupied;
    the remaining editable cells are occupied independently with probability ``q``."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(offset))
    r = torch.rand(base_pred.numel(), generator=g).to(base_pred.device)
    coin = (r < q) | forced
    return torch.where(editable & valid, coin, base_pred)


def check_protected(pred: torch.Tensor, base_pred: torch.Tensor, editable: torch.Tensor,
                    valid: torch.Tensor) -> bool:
    """A baseline may never change a voxel the residual cannot change."""
    m = valid & ~editable
    return bool(torch.equal(pred[m], base_pred[m]))


def binary_counts(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Tuple[int, int, int]:
    p = pred & valid; g = gt & valid
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


__all__ = ["all_valid_occupied", "editable_fill", "matched_density_q", "random_editable",
           "check_protected", "binary_counts"]
