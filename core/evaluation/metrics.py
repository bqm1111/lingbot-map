# extracted from gate6/metrics.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Semantic-occupancy metrics, per clip, in a form that sums across clips.

Every quantity is stored as **integer counts**, never as a per-clip ratio, so a dataset
result is the sum of its clips and a bootstrap resample is the sum of its units. Averaging
per-clip IoUs would silently weight a clip with 200 occupied voxels like one with 80,000.

Three families:

* official SSC -- per-class TP/FP/FN over the official valid mask, plus binary occupancy;
* TP-conditioned -- a ``[C, C]`` confusion matrix restricted to voxels where prediction
  **and** target are occupied. This is what says whether the teacher's semantics survived
  lifting, independently of how much geometry was reconstructed;
* the coverage/naming decomposition -- every valid ground-truth occupied voxel falls in
  exactly one of three bins, so the three counts sum to the number of such voxels by
  construction, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DIST_BANDS = ((0, 10), (10, 20), (20, 30), (30, 40))
HEIGHT_BANDS = ((-2.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 6.0))


def band_masks(grid, dist_bands=DIST_BANDS, height_bands=HEIGHT_BANDS):
    """Flat voxel-index masks for horizontal-range and height bands of a grid."""
    ix, iy, iz = np.meshgrid(*[np.arange(d) for d in grid.dims], indexing="ij")
    c = (np.stack([ix, iy, iz], -1).astype(np.float64) + 0.5) * grid.voxel_size \
        + np.asarray(grid.origin, float)
    rng = np.linalg.norm(c[..., :2], axis=-1).reshape(-1)
    z = c[..., 2].reshape(-1)
    return ([(f"{int(lo)}-{int(hi)}m", (rng >= lo) & (rng < hi)) for lo, hi in dist_bands],
            [(f"z{lo:g}_{hi:g}", (z >= lo) & (z < hi)) for lo, hi in height_bands])


@dataclass
class ClipCounts:
    """All counts of one clip under one occupancy condition."""
    clip_id: str
    group: str
    condition: str
    n_valid: int = 0
    n_gt_occupied: int = 0
    n_pred_occupied: int = 0
    btp: int = 0
    bfp: int = 0
    bfn: int = 0
    tp: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    fp: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    fn: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    fp_empty: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    conf: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    conf_support: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    conf_dilonly: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    decomp: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    decomp_dist: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    decomp_height: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    dist_tpfpfn: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), np.int64))
    dist_conf_hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))


def clip_counts(pred_label: np.ndarray, pred_support: Optional[np.ndarray],
                target: np.ndarray, keep: np.ndarray, labels: Sequence[int],
                empty_label: int, dist: Sequence, height: Sequence,
                clip_id: str, group: str, condition: str) -> ClipCounts:
    """Count everything for one clip.

    ``pred_label`` is a flat volume carrying the benchmark's label ids, ``empty_label``
    where the frozen occupancy says empty. ``pred_support`` is 1 on raw-reconstruction
    voxels, 0 on dilation-only ones and ignored where the prediction is empty.
    """
    C = len(labels)
    lab = np.asarray(labels, np.int32)
    chan_of = np.full(int(max(lab.max(), empty_label)) + 2, -1, np.int32)
    chan_of[lab] = np.arange(C, dtype=np.int32)

    k = keep.reshape(-1)
    t = target.reshape(-1)[k]
    p = pred_label.reshape(-1)[k]
    sup = (pred_support.reshape(-1)[k] if pred_support is not None
           else np.ones(len(p), np.uint8))

    gt_occ = t != empty_label
    pr_occ = p != empty_label
    c = ClipCounts(clip_id=clip_id, group=group, condition=condition,
                   n_valid=int(k.sum()), n_gt_occupied=int(gt_occ.sum()),
                   n_pred_occupied=int(pr_occ.sum()))
    c.btp = int((gt_occ & pr_occ).sum())
    c.bfp = int((~gt_occ & pr_occ).sum())
    c.bfn = int((gt_occ & ~pr_occ).sum())

    ti = np.where(gt_occ, chan_of[np.clip(t, 0, len(chan_of) - 1)], -1)
    pi = np.where(pr_occ, chan_of[np.clip(p, 0, len(chan_of) - 1)], -1)
    valid_t, valid_p = ti >= 0, pi >= 0

    c.tp = np.bincount(ti[valid_t & (ti == pi)], minlength=C).astype(np.int64)
    c.fp = (np.bincount(pi[valid_p], minlength=C).astype(np.int64) - c.tp)
    c.fn = (np.bincount(ti[valid_t], minlength=C).astype(np.int64) - c.tp)
    # predictions landing on GT-empty voxels, per channel: the permutation control needs
    # them separately from predictions landing on a *different* occupied class
    c.fp_empty = np.bincount(pi[valid_p & ~valid_t], minlength=C).astype(np.int64)

    both = valid_t & valid_p
    c.conf = np.bincount(ti[both] * C + pi[both], minlength=C * C).reshape(C, C).astype(np.int64)
    sb = both & (sup == 1)
    db = both & (sup == 0)
    c.conf_support = np.bincount(ti[sb] * C + pi[sb], minlength=C * C).reshape(C, C).astype(np.int64)
    c.conf_dilonly = np.bincount(ti[db] * C + pi[db], minlength=C * C).reshape(C, C).astype(np.int64)

    # coverage / naming / correct, over every valid GT-occupied voxel
    miss = valid_t & ~valid_p
    corr = both & (ti == pi)
    name = both & (ti != pi)
    c.decomp = np.stack([np.bincount(ti[miss], minlength=C),
                         np.bincount(ti[name], minlength=C),
                         np.bincount(ti[corr], minlength=C)], axis=1).astype(np.int64)

    def by_band(bands):
        out = np.zeros((len(bands), 3), np.int64)
        for bi, (_n, m) in enumerate(bands):
            mm = m[k]
            out[bi] = (int((miss & mm).sum()), int((name & mm).sum()), int((corr & mm).sum()))
        return out

    c.decomp_dist = by_band(dist)
    c.decomp_height = by_band(height)

    # per-distance-band per-class TP/FP/FN, for semantic mIoU by distance
    dt = np.zeros((len(dist), C, 3), np.int64)
    for bi, (_n, m) in enumerate(dist):
        mm = m[k]
        tb, pb = ti[mm], pi[mm]
        vt, vp = tb >= 0, pb >= 0
        tpb = np.bincount(tb[vt & (tb == pb)], minlength=C)
        dt[bi, :, 0] = tpb
        dt[bi, :, 1] = np.bincount(pb[vp], minlength=C) - tpb
        dt[bi, :, 2] = np.bincount(tb[vt], minlength=C) - tpb
    c.dist_tpfpfn = dt
    return c


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(counts: Sequence[ClipCounts]) -> Dict[str, np.ndarray]:
    """Sum a set of clips into one count block."""
    if not counts:
        return {}
    out = {"n_clips": len(counts)}
    for f in ("n_valid", "n_gt_occupied", "n_pred_occupied", "btp", "bfp", "bfn"):
        out[f] = int(sum(getattr(c, f) for c in counts))
    for f in ("tp", "fp", "fn", "fp_empty", "conf", "conf_support", "conf_dilonly", "decomp",
              "decomp_dist", "decomp_height", "dist_tpfpfn"):
        out[f] = np.sum([getattr(c, f) for c in counts], axis=0)
    return out


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def summarize(agg: Dict[str, np.ndarray], names: Sequence[str]) -> Dict[str, object]:
    """Count block -> the reported metrics."""
    if not agg:
        return {}
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    present = den > 0
    conf = agg["conf"]
    both = int(conf.sum())
    diag = np.diag(conf)
    gt_rows = conf.sum(axis=1)
    rec = np.where(gt_rows > 0, diag / np.maximum(gt_rows, 1), np.nan)
    dec = agg["decomp"].sum(axis=0)

    def cond(cm):
        n = int(cm.sum())
        r = cm.sum(axis=1)
        d = np.diag(cm)
        rr = np.where(r > 0, d / np.maximum(r, 1), np.nan)
        return {"n": n, "top1_accuracy": _safe_div(int(d.sum()), n),
                "balanced_recall": (float(np.nanmean(rr)) if np.any(r > 0) else 0.0),
                "n_classes_present": int((r > 0).sum())}

    total_gt = int(dec.sum())
    out = {
        "n_clips": agg["n_clips"], "n_valid": agg["n_valid"],
        "n_gt_occupied": agg["n_gt_occupied"], "n_pred_occupied": agg["n_pred_occupied"],
        "binary_iou": _safe_div(agg["btp"], agg["btp"] + agg["bfp"] + agg["bfn"]),
        "binary_precision": _safe_div(agg["btp"], agg["btp"] + agg["bfp"]),
        "binary_recall": _safe_div(agg["btp"], agg["btp"] + agg["bfn"]),
        "empty_prediction_rate": _safe_div(agg["n_valid"] - agg["n_pred_occupied"],
                                           agg["n_valid"]),
        "ssc_miou": float(np.nanmean(iou[present])) if present.any() else 0.0,
        "n_classes_scored": int(present.sum()),
        "per_class": {names[i]: {"iou": (None if not present[i] else float(iou[i])),
                                 "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                                 "gt_voxels": int(tp[i] + fn[i]),
                                 "tp_conditioned_recall":
                                     (None if gt_rows[i] == 0 else float(rec[i])),
                                 "tp_conditioned_gt_voxels": int(gt_rows[i])}
                      for i in range(len(names))},
        "tp_conditioned": cond(conf),
        "tp_conditioned_reconstruction_support": cond(agg["conf_support"]),
        "tp_conditioned_dilation_only": cond(agg["conf_dilonly"]),
        "decomposition": {
            "n_valid_gt_occupied": total_gt,
            "coverage_miss": int(dec[0]), "naming_error": int(dec[1]),
            "correct": int(dec[2]),
            "coverage_miss_fraction": _safe_div(dec[0], total_gt),
            "naming_error_fraction": _safe_div(dec[1], total_gt),
            "correct_fraction": _safe_div(dec[2], total_gt),
            "sums_to_one": abs(_safe_div(dec.sum(), total_gt) - 1.0) < 1e-12
                           if total_gt else True},
    }
    return out


__all__ = ["ClipCounts", "clip_counts", "aggregate", "summarize", "band_masks",
           "DIST_BANDS", "HEIGHT_BANDS"]
