"""Paired, scene-aware bootstrap over **Gate 6's own resampling units**.

Reused wholesale from Gate 8 (``tools/gate8/aggregate.py``) so that every interval printed
since Gate 6 resamples the same blocks with the same seed and the same 10 000 draws. The
only addition is that Gate 8A's per-clip counts may come from a stored *score histogram*
evaluated at a threshold, so the comparison functions take plain ``[n_clip, 3]`` arrays and
a clip-id list rather than a Gate-6 count block.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

from gates.gate7a import stats as S7

N_BOOT, SEED = 10000, 0
_UNITS: Dict[str, tuple] = {}


def gate6_units(dataset: str, repo_root: str):
    """``(clip_id -> unit, unit_name)``, read from Gate 6's own count block."""
    if dataset not in _UNITS:
        from tools.gate6.analyze import unit_ids
        z = np.load(os.path.join(repo_root, "artifacts", "gate6",
                                 f"counts_{dataset}_B-D.npz"), allow_pickle=False)
        cid = [str(c) for c in z["clip_id"]]
        u, name = unit_ids(dataset, cid, [str(g) for g in z["group"]])
        _UNITS[dataset] = (dict(zip(cid, u)), name)
    return _UNITS[dataset]


def paired_binary(dataset: str, repo_root: str, clips_a: Sequence[str], a: np.ndarray,
                  clips_b: Sequence[str], b: np.ndarray,
                  pc_a: Optional[np.ndarray] = None, pc_b: Optional[np.ndarray] = None):
    """Paired difference in pooled binary IoU (and SSC mIoU when per-class counts given).

    ``a``/``b`` are ``[n_clip, 3]`` TP/FP/FN; ``pc_a``/``pc_b`` are ``[n_clip, C, 3]``.
    """
    ka = {str(c): i for i, c in enumerate(clips_a)}
    kb = {str(c): i for i, c in enumerate(clips_b)}
    umap, uname = gate6_units(dataset, repo_root)
    common = sorted(set(ka) & set(kb) & set(umap))
    if not common:
        return None
    ia = np.array([ka[c] for c in common]); ib = np.array([kb[c] for c in common])
    units = [umap[c] for c in common]
    uids = sorted(set(units)); ui = {u: i for i, u in enumerate(uids)}
    row = np.array([ui[u] for u in units])

    def fold(x, idx):
        out = np.zeros((len(uids),) + x.shape[1:], np.float64)
        np.add.at(out, row, x[idx].astype(np.float64))
        return out

    ua, ub = fold(a, ia), fold(b, ib)
    mult = S7.multiplicities(len(uids), N_BOOT, SEED)
    sa, sb = S7.binary_iou(S7.resample(ua, mult)), S7.binary_iou(S7.resample(ub, mult))
    pa, pb = float(S7.binary_iou(ua.sum(0))), float(S7.binary_iou(ub.sum(0)))
    out = {"n_clips": len(common), "unit": uname, "n_units": len(uids),
           "binary_iou": S7.paired(pa, pb, sa, sb)}
    if pc_a is not None and pc_b is not None:
        qa, qb = fold(pc_a, ia), fold(pc_b, ib)
        ma, mb = S7.miou(S7.resample(qa, mult)), S7.miou(S7.resample(qb, mult))
        out["ssc_miou"] = S7.paired(float(S7.miou(qa.sum(0))), float(S7.miou(qb.sum(0))), ma, mb)
    return out


def counts_from_gate6_block(z) -> tuple:
    """``(clip_ids, [n,3] binary, [n,C,3] per-class)`` from a Gate-6/8 counts npz."""
    cid = [str(c) for c in z["clip_id"]]
    binary = np.asarray(z["binary"])[:, :3].astype(np.int64)
    pc = np.stack([z["tp"], z["fp"], z["fn"]], -1).astype(np.int64)
    return cid, binary, pc


__all__ = ["gate6_units", "paired_binary", "counts_from_gate6_block", "N_BOOT", "SEED"]
