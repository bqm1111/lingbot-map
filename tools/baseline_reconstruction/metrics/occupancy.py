"""Binary occupancy metrics: TP / FP / FN counts, IoU, precision, recall, by region and range.

The scene-completion IoU of every benchmark used here is the same quantity:

    SC IoU = TP / (TP + FP + FN)   over the voxels inside the evaluation mask,

with occupied ground truth = "label is not the free class" inside the mask. Counts are
integers so that per-clip results can be summed exactly and re-derived from files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from tools.baseline_reconstruction import config as C


@dataclass(frozen=True)
class OccupancyCounts:
    tp: int
    fp: int
    fn: int

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def n_gt(self) -> int:
        return self.tp + self.fn

    def __add__(self, other: "OccupancyCounts") -> "OccupancyCounts":
        return OccupancyCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_dict(self) -> Dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "iou": self.iou,
                "precision": self.precision, "recall": self.recall, "n_gt": self.n_gt}


ZERO = OccupancyCounts(0, 0, 0)


def counts(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> OccupancyCounts:
    """``pred`` / ``gt`` / ``mask`` are flat boolean tensors on the same grid."""
    p, g = pred & mask, gt & mask
    return OccupancyCounts(int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()))


def counts_by_range(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                    range_bin: torch.Tensor, n_bins: int) -> List[OccupancyCounts]:
    """One :class:`OccupancyCounts` per range bin, plus the total as the last entry."""
    p, g = pred & mask, gt & mask
    out = []
    for sel in (p & g, p & ~g, ~p & g):
        out.append(torch.bincount(range_bin[sel], minlength=n_bins).cpu().numpy())
    tp, fp, fn = out
    per_bin = [OccupancyCounts(int(tp[b]), int(fp[b]), int(fn[b])) for b in range(n_bins)]
    return per_bin + [OccupancyCounts(int(tp.sum()), int(fp.sum()), int(fn.sum()))]


def range_bins_for_grid(dims: Sequence[int], voxel_size: float, origin: Sequence[float],
                        T_grid_to_camera: np.ndarray, edges_m: Sequence[float] = C.RANGE_EDGES_M) -> np.ndarray:
    """Range-bin index of every voxel centre, by its **z in the anchor camera frame**.

    ``T_grid_to_camera`` maps metric grid-frame coordinates to the anchor camera frame
    (metres, OpenCV convention: z forward). Bin ``b`` covers ``edges[b-1] <= z < edges[b]``;
    values below the first edge (including behind the camera) fall in bin 0.
    """
    n = int(np.prod(dims))
    gi = np.stack(np.unravel_index(np.arange(n), tuple(int(d) for d in dims)), 1).astype(np.float64)
    centre = (gi + 0.5) * float(voxel_size) + np.asarray(origin, np.float64)
    T = np.asarray(T_grid_to_camera, np.float64)
    z_cam = (centre @ T[:3, :3].T + T[:3, 3])[:, 2]
    return np.clip(np.digitize(z_cam, np.asarray(edges_m, np.float64)), 0, len(edges_m)).astype(np.int64)


@dataclass
class RegionAccumulator:
    """Sum of counts per (region, range bin) over many clips; the last range entry is the total."""
    regions: Tuple[str, ...]
    n_bins: int
    totals: Dict[str, List[OccupancyCounts]] = field(default_factory=dict)
    n_clips: int = 0

    def __post_init__(self) -> None:
        for r in self.regions:
            self.totals[r] = [ZERO] * (self.n_bins + 1)

    def add(self, region: str, per_bin: Sequence[OccupancyCounts]) -> None:
        assert len(per_bin) == self.n_bins + 1, "expected one entry per bin plus the total"
        self.totals[region] = [a + b for a, b in zip(self.totals[region], per_bin)]

    def summary(self, range_names: Sequence[str] = C.RANGE_NAMES) -> Dict[str, Dict]:
        out = {}
        for r in self.regions:
            tot = self.totals[r][-1]
            out[r] = dict(tot.as_dict(),
                          by_range={name: self.totals[r][b].as_dict() for b, name in enumerate(range_names)})
        return out


def seed_statistics(values: Iterable[float]) -> Dict[str, float]:
    """Mean, median and sample sd over seeds -- both centre statistics are always reported."""
    v = np.asarray(list(values), np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


__all__ = ["OccupancyCounts", "ZERO", "counts", "counts_by_range", "range_bins_for_grid",
           "RegionAccumulator", "seed_statistics"]
