"""Evaluation metrics for observed-surface semantic reconstruction.

Deliberately **not** semantic scene completion: only surfaces the cameras actually
saw are scored.  Two mIoU variants are reported side by side because they answer
different questions:

* ``end_to_end_miou`` counts a visible GT point that landed in no populated voxel as
  a miss (an *unknown* prediction), so coverage failures are penalised;
* ``matched_miou`` restricts to covered points, so it isolates semantic quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class Confusion:
    """Confusion matrix with class 0 reserved for *ignore* / *unknown*."""

    def __init__(self, num_classes: int, device: Optional[torch.device] = None) -> None:
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device or "cpu")

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        keep = (gt >= 0) & (gt < self.n) & (pred >= 0) & (pred < self.n)
        idx = gt[keep].long() * self.n + pred[keep].long()
        self.mat += torch.bincount(idx, minlength=self.n**2).reshape(self.n, self.n).to(self.mat.device)

    def iou(self) -> torch.Tensor:
        tp = self.mat.diag().float()
        fp = self.mat.sum(0).float() - tp
        fn = self.mat.sum(1).float() - tp
        return tp / (tp + fp + fn).clamp_min(1)

    def present(self) -> torch.Tensor:
        """Classes that occur in the ground truth."""
        return self.mat.sum(1) > 0

    def accuracy(self) -> float:
        total = self.mat.sum()
        return float(self.mat.diag().sum() / total) if total > 0 else 0.0

    def mean_accuracy(self) -> float:
        """Mean per-class recall over classes present in the ground truth."""
        present = self.present().clone()
        present[0] = False
        if not bool(present.any()):
            return 0.0
        recall = self.mat.diag().float() / self.mat.sum(1).clamp_min(1).float()
        return float(recall[present].mean())

    def miou(self) -> float:
        present = self.present().clone()
        present[0] = False
        if not bool(present.any()):
            return 0.0
        return float(self.iou()[present].mean())

    def per_class(self, names: Sequence[str]) -> Dict[str, float]:
        present = self.present().clone()
        present[0] = False
        iou = self.iou()
        return {names[c - 1]: float(iou[c] * 100) for c in range(1, self.n) if present[c]}


@dataclass
class CoverageCounter:
    """Visible GT points vs. points that landed in a populated voxel."""

    total: int = 0
    covered: int = 0

    def update(self, n_total: int, n_covered: int) -> None:
        self.total += int(n_total)
        self.covered += int(n_covered)

    @property
    def ratio(self) -> float:
        return self.covered / max(self.total, 1)


def cross_view_consistency(
    predictions: torch.Tensor, track_id: torch.Tensor
) -> Tuple[float, int]:
    """Mean cosine of each prediction to its track's mean prediction.

    Labels are not involved, so this runs on any unlabelled held-out sequence.
    Returns ``(mean_cosine, n_tracks_used)``; only tracks with >= 2 observations count.
    """
    valid = track_id >= 0
    if not bool(valid.any()):
        return float("nan"), 0
    p = torch.nn.functional.normalize(predictions[valid].float(), dim=-1)
    ids = track_id[valid]
    uniq, inverse = torch.unique(ids, return_inverse=True)
    sums = torch.zeros(uniq.numel(), p.shape[-1], device=p.device)
    sums.index_add_(0, inverse, p)
    counts = torch.zeros(uniq.numel(), device=p.device)
    counts.index_add_(0, inverse, torch.ones(inverse.numel(), device=p.device))
    multi = counts >= 2
    if not bool(multi.any()):
        return float("nan"), 0
    means = torch.nn.functional.normalize(sums / counts.unsqueeze(1), dim=-1)
    keep = multi[inverse]
    cos = (p[keep] * means[inverse][keep]).sum(-1)
    return float(cos.mean()), int(multi.sum())


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares Sim(3) alignment of ``src`` onto ``dst``.

    Geometry only — no semantic information enters the correspondence.

    Returns:
        ``(R, t, s, rmse)`` such that ``dst ≈ s · R · src + t``.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching [N, 3] arrays, got {src.shape} and {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s_c**2).sum() / len(src)
    scale = float((D * np.diag(S)).sum() / var) if with_scale and var > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    residual = dst - (scale * (R @ src.T).T + t)
    rmse = float(np.sqrt((residual**2).sum(1).mean()))
    return R, t, scale, rmse


def geometry_fscore(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float, chunk: int = 8192
) -> Dict[str, float]:
    """Precision / recall / F-score between two point clouds at a distance threshold."""
    if pred.numel() == 0 or gt.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0, "threshold": threshold}

    def _min_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty(a.shape[0], device=a.device)
        for i in range(0, a.shape[0], chunk):
            out[i : i + chunk] = torch.cdist(a[i : i + chunk], b).min(dim=1).values
        return out

    precision = float((_min_dists(pred, gt) < threshold).float().mean())
    recall = float((_min_dists(gt, pred) < threshold).float().mean())
    f = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "fscore": f, "threshold": threshold}


@dataclass
class RunMetrics:
    """The row of the feasibility table for one model."""

    name: str
    end_to_end_miou: float = float("nan")
    matched_miou: float = float("nan")
    mean_accuracy: float = float("nan")
    point_accuracy: float = float("nan")
    coverage: float = float("nan")
    cross_view_consistency: float = float("nan")
    inference_seconds: float = float("nan")
    peak_vram_gb: float = float("nan")
    trainable_params: int = 0
    training_seconds: float = float("nan")
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.name,
            "3D mIoU (end-to-end)": self.end_to_end_miou,
            "matched mIoU": self.matched_miou,
            "mean acc": self.mean_accuracy,
            "coverage": self.coverage,
            "cross-view cons.": self.cross_view_consistency,
            "trainable params": self.trainable_params,
            "train (s)": self.training_seconds,
            "peak VRAM (GB)": self.peak_vram_gb,
        }


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    """Fixed-width markdown table from uniform row dicts."""
    if not rows:
        return "(no rows)"
    headers = list(rows[0].keys())

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "n/a" if np.isnan(v) else f"{v:.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    cells = [[fmt(r.get(h, "")) for h in headers] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in cells:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join(lines)
