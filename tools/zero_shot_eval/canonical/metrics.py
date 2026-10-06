"""Metric primitives shared by Track P and Track O. Pure numpy; no dataset knowledge.

Confusion convention: rows = GT label id, columns = predicted label id, ``n_classes`` includes the empty class. All metrics
are computed from confusions pooled over samples (never averaged per sample).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


def confusion(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """[n_classes, n_classes] int64 of the voxels passed in (callers select the region first)."""
    return np.bincount(gt.astype(np.int64) * n_classes + pred.astype(np.int64), minlength=n_classes * n_classes).reshape(n_classes, n_classes)


@dataclass(frozen=True)
class OccupancyCounts:
    tp: int
    fp: int
    fn: int

    @staticmethod
    def of(pred_occ: np.ndarray, gt_occ: np.ndarray) -> "OccupancyCounts":
        return OccupancyCounts(int((pred_occ & gt_occ).sum()), int((pred_occ & ~gt_occ).sum()), int((~pred_occ & gt_occ).sum()))

    def __add__(self, o: "OccupancyCounts") -> "OccupancyCounts":
        return OccupancyCounts(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.tp, self.fp, self.fn], np.int64)


def occupancy_scores(counts: np.ndarray) -> Dict[str, float]:
    tp, fp, fn = (float(x) for x in counts)
    return {"iou": tp / max(tp + fp + fn, 1.0), "precision": tp / max(tp + fp, 1.0), "recall": tp / max(tp + fn, 1.0), "tp": tp, "fp": fp, "fn": fn}


def per_class_iou(conf: np.ndarray, classes: Iterable[int]) -> Dict[int, Optional[float]]:
    """IoU per class; None where the class has no GT voxel (Track P convention)."""
    tp = np.diag(conf).astype(np.float64); gt = conf.sum(1); pr = conf.sum(0)
    return {c: (tp[c] / (gt[c] + pr[c] - tp[c]) if gt[c] > 0 else None) for c in classes}


def semantic_summary(conf: np.ndarray, classes: List[int], empty_class: int) -> Dict:
    """Track P semantic metrics: mIoU over classes with GT support, accuracies, coverage, prediction frequencies."""
    iou = per_class_iou(conf, classes)
    sup = [c for c in classes if iou[c] is not None]
    gt = conf.sum(1); pr = conf.sum(0); tp = np.diag(conf)
    sem_gt = sum(int(gt[c]) for c in classes)
    pred_sem = np.asarray([pr[c] for c in classes], np.float64)
    return {"miou": float(np.mean([iou[c] for c in sup])) if sup else None,
            "per_class_iou": {int(c): iou[c] for c in classes},
            "overall_accuracy": float(sum(tp[c] for c in classes) / max(sem_gt, 1)),
            "mean_class_accuracy": float(np.mean([tp[c] / gt[c] for c in sup])) if sup else None,
            "gt_classes": len(sup), "gt_classes_predicted": int(sum(1 for c in sup if pr[c] > 0)),
            "predicted_frequency": {int(c): float(pr[c] / max(pred_sem.sum(), 1)) for c in classes},
            "gt_frequency": {int(c): float(gt[c] / max(sem_gt, 1)) for c in classes},
            "gt_voxels": sem_gt, "empty_class": empty_class}


def occany_ssc_miou(conf: np.ndarray, n_classes: int, empty_class: int, other_class: int, ignore_other: bool) -> Dict:
    """OccAny ``SSCMetrics.get_stats`` (occany/metrics/ssc.py): IoU = TP / (TP + FP + FN + 1e-5) for every non-empty class,
    absent classes included (IoU 0), mean over them."""
    tp = np.diag(conf).astype(np.float64); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-5)
    valid = [i for i in range(n_classes) if i != empty_class and (not ignore_other or i != other_class)]
    return {"miou": float(np.mean(iou[valid])), "per_class_iou": {int(i): float(iou[i]) for i in valid}}


def superclass_confusion(conf: np.ndarray, mapping: List[int], n_super: int) -> np.ndarray:
    m = np.asarray(mapping, np.int64)
    out = np.zeros((n_super, n_super), np.int64)
    np.add.at(out, (m[:, None].repeat(len(m), 1), m[None, :].repeat(len(m), 0)), conf)
    return out


__all__ = ["confusion", "OccupancyCounts", "occupancy_scores", "per_class_iou", "semantic_summary", "occany_ssc_miou", "superclass_confusion"]
