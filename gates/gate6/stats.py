"""Paired bootstrap over the benchmark's own resampling unit.

The unit is a *scene* on Occ3D-nuScenes and a *contiguous block of clips* on the two
single-sequence benchmarks. Blocks from one sequence are not independent scenes and are
never described as such; they are the coarsest honest unit available when the whole
validation set is one drive.

Resampling is over units, and each resample re-derives the metric from **summed integer
counts**, so a unit with many occupied voxels carries its true weight.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def contiguous_blocks(n: int, size: int) -> np.ndarray:
    """Block id per item; any final remainder is appended to the last full block."""
    if n == 0:
        return np.zeros(0, np.int64)
    n_full = max(n // size, 1)
    return np.minimum(np.arange(n) // size, n_full - 1)


def paired_bootstrap(unit_ids: Sequence, per_unit: Dict[str, Dict[str, np.ndarray]],
                     metric: Callable[[Dict[str, np.ndarray]], float],
                     n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Paired CI for ``metric(B) - metric(A)`` and for each condition separately.

    ``per_unit[condition][unit]`` holds that unit's summed count block; a resample sums the
    drawn units' blocks (with multiplicity) and evaluates the metric once.
    """
    units = list(unit_ids)
    U = len(units)
    conds = list(per_unit)
    keys = list(per_unit[conds[0]][units[0]])
    stacks = {c: {k: np.stack([per_unit[c][u][k] for u in units]) for k in keys}
              for c in conds}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, U, size=(n_boot, U))
    out = {c: np.empty(n_boot) for c in conds}
    diff = np.empty(n_boot)
    for b in range(n_boot):
        d = draws[b]
        vals = {}
        for c in conds:
            vals[c] = metric({k: stacks[c][k][d].sum(axis=0) for k in keys})
            out[c][b] = vals[c]
        diff[b] = vals[conds[1]] - vals[conds[0]]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    res = {"n_units": U, "n_boot": n_boot, "seed": seed,
           "point": {c: metric({k: stacks[c][k].sum(axis=0) for k in keys}) for c in conds},
           "ci": {c: [float(np.percentile(out[c], lo)), float(np.percentile(out[c], hi))]
                  for c in conds}}
    a, b_ = conds
    res["paired_difference"] = {
        "of": f"{b_} - {a}",
        "point": res["point"][b_] - res["point"][a],
        "ci": [float(np.percentile(diff, lo)), float(np.percentile(diff, hi))],
        "excludes_zero": bool(np.percentile(diff, lo) > 0 or np.percentile(diff, hi) < 0)}
    return res


__all__ = ["contiguous_blocks", "paired_bootstrap"]
