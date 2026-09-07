"""The two completion envelopes, and the counting that separates them.

* the **oracle** may add only voxels the ground truth says are occupied. It is an
  optimistic ceiling for any model confined to that local correction region, and it is not
  deployable: the target decides membership;
* the **morphology** baseline adds every valid voxel inside the radius. It is deployable,
  non-learned, and it pays the precision that the oracle does not.

Both are supersets of their base at every radius, and the oracle's false-positive count is
identical to the base's at every radius by construction -- asserted here, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import distance as _dist


@dataclass
class CountBlock:
    """One (condition, construction, radius) result for one clip, as integer counts."""
    btp: int = 0
    bfp: int = 0
    bfn: int = 0
    tp: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    fp: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    fn: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    n_base: int = 0
    n_added: int = 0
    n_added_tp: int = 0
    n_added_fp: int = 0


def count(t_ch: np.ndarray, p_ch: np.ndarray, C: int) -> CountBlock:
    """Binary and per-class TP/FP/FN over the valid voxels of one clip.

    ``t_ch`` / ``p_ch`` carry the teacher/benchmark channel index, ``-1`` for empty.
    """
    b = CountBlock()
    vt, vp = t_ch >= 0, p_ch >= 0
    b.btp = int((vt & vp).sum())
    b.bfp = int((~vt & vp).sum())
    b.bfn = int((vt & ~vp).sum())
    both = vt & vp
    b.tp = np.bincount(t_ch[both & (t_ch == p_ch)], minlength=C).astype(np.int64)
    b.fp = np.bincount(p_ch[vp], minlength=C).astype(np.int64) - b.tp
    b.fn = np.bincount(t_ch[vt], minlength=C).astype(np.int64) - b.tp
    return b


def build(base_occ: np.ndarray, base_ch: np.ndarray, prop_ch: np.ndarray,
          d2: np.ndarray, gt_occ: np.ndarray, voxel_size: float, radius_m: float,
          construction: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(p_ch, added)`` for one construction at one radius, over the valid voxels.

    ``prop_ch`` is the propagated teacher channel of every voxel in range; it is read only
    where a voxel is actually added. ``base_ch`` is never overwritten.
    """
    inr = _dist.within_radius(d2, radius_m, voxel_size) & ~base_occ
    if construction == "oracle":
        added = inr & gt_occ
    elif construction == "morph":
        added = inr
    else:
        raise KeyError(construction)
    p_ch = np.where(base_occ, base_ch, np.where(added, prop_ch, -1)).astype(np.int32)
    return p_ch, added


def evaluate(base_occ: np.ndarray, base_ch: np.ndarray, prop_ch: np.ndarray,
             d2: np.ndarray, t_ch: np.ndarray, voxel_size: float,
             radii_m: Sequence[float], C: int) -> Dict[str, Dict[float, CountBlock]]:
    """Every construction at every radius, for one clip.

    All arrays are already restricted to the clip's **valid** voxels, so nothing here has
    to know about evaluation masks.
    """
    gt_occ = t_ch >= 0
    n_base = int(base_occ.sum())
    out: Dict[str, Dict[float, CountBlock]] = {}
    for constr in ("oracle", "morph"):
        per_r: Dict[float, CountBlock] = {}
        for r in radii_m:
            p_ch, added = build(base_occ, base_ch, prop_ch, d2, gt_occ, voxel_size,
                                float(r), constr)
            b = count(t_ch, p_ch, C)
            b.n_base = n_base
            b.n_added = int(added.sum())
            b.n_added_tp = int((added & gt_occ).sum())
            b.n_added_fp = b.n_added - b.n_added_tp
            per_r[float(r)] = b
        out[constr] = per_r
    return out


def assert_invariants(blocks: Dict[str, Dict[float, CountBlock]], base: CountBlock,
                      radii_m: Sequence[float]) -> None:
    """The four structural guarantees, checked on every clip rather than argued."""
    rs = sorted(float(r) for r in radii_m)
    for constr in ("oracle", "morph"):
        prev = None
        for r in rs:
            b = blocks[constr][r]
            # r = 0 reproduces the base exactly
            if r == 0.0:
                assert (b.btp, b.bfp, b.bfn) == (base.btp, base.bfp, base.bfn)
                assert b.n_added == 0
            # supersets: recall never falls
            assert b.btp >= base.btp and b.bfn <= base.bfn
            if prev is not None:
                assert b.btp >= prev.btp and b.bfn <= prev.bfn
            prev = b
        if constr == "oracle":
            for r in rs:                       # the oracle adds no false positive, ever
                assert blocks["oracle"][r].bfp == base.bfp
                assert blocks["oracle"][r].n_added_fp == 0


__all__ = ["CountBlock", "count", "build", "evaluate", "assert_invariants"]
