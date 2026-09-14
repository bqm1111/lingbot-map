"""Confusion-matrix semantics. Label space: 0 = empty, 1..18 = classes, 255 = ignore (never counted).

Two kinds of region, never mixed:
* **semantic-only** regions contain GT-occupied labelled voxels only; the matrix is over classes 1..18 and
  answers "given the voxel is occupied, is the class right?" (oracle-occupancy / completed-region diagnostics);
* **SSC** regions contain every labelled voxel (empty included) with the prediction set to 0 wherever the
  frozen occupancy says empty; the 19x19 matrix gives the scene-completion mIoU over classes 1..18.
mIoU averages the classes with GT support in the region (the empty class is scored in SSC but excluded).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

NC = 19          # 0..18


def confusion(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """``[19, 19]`` int64, rows = GT label, cols = predicted label, over ``mask``."""
    g, p = gt[mask].long(), pred[mask].long()
    assert (g.numel() == 0 or int(g.max()) < NC) and (p.numel() == 0 or int(p.max()) < NC)
    return torch.bincount(g * NC + p, minlength=NC * NC).reshape(NC, NC).cpu().numpy().astype(np.int64)


def summarize(conf: np.ndarray, semantic_only: bool) -> Dict:
    """IoU / accuracy per class, mIoU, mean class accuracy, coverage and collapse diagnostics."""
    cm = conf.astype(np.float64)
    classes = range(1, NC)
    tp = np.diag(cm); gt_n = cm.sum(1); pred_n = cm.sum(0)
    iou = {c: (tp[c] / (gt_n[c] + pred_n[c] - tp[c])) if gt_n[c] > 0 else None for c in classes}
    acc = {c: (tp[c] / gt_n[c]) if gt_n[c] > 0 else None for c in classes}
    supported = [c for c in classes if gt_n[c] > 0]
    pred_cls = pred_n[1:] if semantic_only else pred_n[1:]
    total_pred = pred_cls.sum()
    freq_pred = pred_cls / max(total_pred, 1); freq_gt = gt_n[1:] / max(gt_n[1:].sum(), 1)
    nz = freq_pred[freq_pred > 0]
    return {"miou": float(np.mean([iou[c] for c in supported])) if supported else None,
            "mean_class_accuracy": float(np.mean([acc[c] for c in supported])) if supported else None,
            "overall_accuracy": float(tp[1:].sum() / max(gt_n[1:].sum(), 1)),
            "per_class_iou": {c: iou[c] for c in classes}, "per_class_accuracy": {c: acc[c] for c in classes},
            "n_gt_classes": len(supported),
            "gt_classes_predicted": int(sum(1 for c in supported if pred_n[c] > 0)),
            "class_coverage": float(np.mean([pred_n[c] > 0 for c in supported])) if supported else None,
            "n_classes_ge_1pct_of_predictions": int((freq_pred >= 0.01).sum()),
            "dominant_pred_class": int(1 + freq_pred.argmax()), "dominant_pred_fraction": float(freq_pred.max()),
            "dominant_gt_class": int(1 + freq_gt.argmax()), "dominant_gt_fraction": float(freq_gt.max()),
            "pred_entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "gt_frequency": freq_gt.tolist(), "pred_frequency": freq_pred.tolist(), "n_voxels": int(gt_n[1:].sum() if semantic_only else gt_n.sum())}


__all__ = ["NC", "confusion", "summarize"]
