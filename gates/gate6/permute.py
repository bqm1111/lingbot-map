"""The vocabulary-permutation negative control, computed in closed form.

The control asks: does the *assignment* of teacher channels to benchmark classes carry
information, holding the geometry and the teacher's spatial partition fixed? Permuting the
class-to-text assignment without recomputing anything is exactly that question, because a
permutation changes only which class name a channel answers to -- the score vectors, the
argmax channel and the predicted occupancy are untouched.

That makes the control exact and free. Everything the two reported metrics need is already
in the per-clip count blocks:

    M[g, ch]   valid voxels with ground-truth class ``g`` and predicted channel ``ch``,
               both occupied                                   (``conf``)
    miss[g]    valid voxels with ground-truth class ``g``, prediction empty (``decomp[:,0]``)
    fpe[ch]    valid voxels with ground truth empty, predicted channel ``ch`` (``fp_empty``)

Under a permutation whose inverse is ``inv`` (class ``c`` is now answered by channel
``inv[c]``)::

    TP_c = M[c, inv[c]]
    FP_c = (M[:, inv[c]].sum() + fpe[inv[c]]) - TP_c
    FN_c = (M[c, :].sum() + miss[c]) - TP_c

``tests/gate6`` asserts that the identity permutation reproduces the evaluator's own
numbers bit-for-bit, which is what licenses the closed form.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def metrics_under(M: np.ndarray, miss: np.ndarray, fpe: np.ndarray,
                  inv: np.ndarray) -> Tuple[float, float]:
    """``(full SSC mIoU, TP-conditioned balanced recall)`` for one permutation."""
    tp = M[np.arange(len(inv)), inv]
    pred_total = M.sum(axis=0)[inv] + fpe[inv]
    gt_total = M.sum(axis=1) + miss
    fp = pred_total - tp
    fn = gt_total - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    miou = float(np.nanmean(iou[den > 0])) if np.any(den > 0) else 0.0
    rows = M.sum(axis=1)
    rec = np.where(rows > 0, tp / np.maximum(rows, 1), np.nan)
    bal = float(np.nanmean(rec[rows > 0])) if np.any(rows > 0) else 0.0
    return miou, bal


def permutations(n_classes: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` deterministic permutations of ``n_classes`` channels."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_classes) for _ in range(n)])


def run_control(M: np.ndarray, miss: np.ndarray, fpe: np.ndarray, n: int = 100,
                seed: int = 0) -> Dict[str, object]:
    C = M.shape[0]
    real = metrics_under(M, miss, fpe, np.arange(C))
    perms = permutations(C, n, seed)
    vals = np.array([metrics_under(M, miss, fpe, np.argsort(p)) for p in perms])
    out = {}
    for i, name in enumerate(("ssc_miou", "tp_balanced_recall")):
        col = vals[:, i]
        out[name] = {
            "real": real[i],
            "permutation_mean": float(col.mean()),
            "permutation_std": float(col.std(ddof=1)),
            "permutation_p50": float(np.percentile(col, 50)),
            "permutation_p95": float(np.percentile(col, 95)),
            "permutation_max": float(col.max()),
            "real_exceeds_p95": bool(real[i] > float(np.percentile(col, 95))),
            "real_percentile": float(100.0 * (col < real[i]).mean()),
            "n_permutations_beating_real": int((col >= real[i]).sum()),
        }
    out["n_permutations"] = int(n)
    out["seed"] = int(seed)
    out["values"] = vals.tolist()
    return out


__all__ = ["metrics_under", "permutations", "run_control"]
