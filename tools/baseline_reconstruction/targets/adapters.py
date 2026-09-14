"""Target-dataset adapters for the zero-shot evaluation. **Evaluation-only** modules.

Everything dataset-specific is delegated to the frozen ``core`` code that produced the
historical numbers (``core.datasets.sources.segments`` for the streams and anchors,
``core.evaluation.official_targets.semantic_target`` for the official labels and masks,
``core.datasets.grids`` for the prediction / evaluation grids and the Occ3D reduction), so
the current baseline is scored on exactly the anchors, masks and grids of record.

Frames and units:

* ``SFrame.T_cam_to_grid`` maps the **anchor camera** (metres, OpenCV axes) to the grid
  frame: the anchor velodyne frame for SemanticKITTI, the anchor ego frame for Occ3D;
* prediction grid = where the map is queried and the network runs; evaluation grid = where
  the official label lives. They coincide for SemanticKITTI; Occ3D predicts at 0.2 m and is
  reduced to 0.4 m by the frozen any-subvoxel rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from core.datasets import grids as G, sources as S
from core.evaluation import official_targets as OT
from core.evaluation.regions import occ_to_eval_grid
from tools.baseline_reconstruction import config as C


@dataclass(frozen=True)
class AnchorRef:
    """One official evaluation anchor inside a stream segment."""
    dataset: str
    segment: str
    anchor_index: int            # position in the segment's stream
    clip_id: str
    image_paths: Tuple[str, ...] # the five images t-4..t
    keys: Tuple[str, ...]
    T_cam_to_grid: np.ndarray    # [4, 4] anchor camera -> grid frame, metres
    gt_ref: dict


class TargetAdapter:
    """Streams, anchors, grids and official targets of one benchmark."""

    def __init__(self, dataset: str, repo_root: str = C.REPO_ROOT):
        assert dataset in C.TARGET_PROTOCOLS, dataset
        self.dataset, self.repo_root = dataset, repo_root
        self.prediction_grid, self.evaluation_grid = G.PREDICTION_GRID[dataset], G.EVAL_GRID[dataset]
        self.protocol = C.TARGET_PROTOCOLS[dataset]
        self._segments: Optional[List[S.Seg]] = None

    # ---- streams and anchors --------------------------------------------------------
    @property
    def segments(self) -> List[S.Seg]:
        if self._segments is None:
            self._segments = S.segments(self.dataset, self.repo_root)
        return self._segments

    def anchors(self) -> Iterator[AnchorRef]:
        """Every official anchor with a complete causal window ``i-4..i`` (the historical past5 rule)."""
        for seg in self.segments:
            for f in seg.frames:
                if f.gt_ref is None or f.index + min(C.CLIP_OFFSETS) < 0:
                    continue
                idx = [f.index + o for o in C.CLIP_OFFSETS]
                yield AnchorRef(self.dataset, seg.name, f.index, str(f.gt_ref["clip_id"]),
                                tuple(seg.frames[j].path for j in idx), tuple(seg.frames[j].key for j in idx),
                                np.asarray(f.T_cam_to_grid, np.float64), f.gt_ref)

    def moge_path(self, key: str) -> str:
        return S.moge_path(self.dataset, key)

    # ---- official target ------------------------------------------------------------
    def official_target(self, ref: AnchorRef, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(gt_occupied, keep)`` flat on the evaluation grid, or ``(None, None)`` if the label is missing."""
        target, keep = OT.semantic_target(self.dataset, ref.gt_ref, self.repo_root)
        if target is None:
            return None, None
        gt = torch.from_numpy(((target != self.evaluation_grid.empty_class) & keep).reshape(-1)).to(device)
        return gt, torch.from_numpy(keep.reshape(-1)).to(device)

    # ---- grids ----------------------------------------------------------------------
    def to_evaluation_grid(self, occ_pred_grid: torch.Tensor) -> torch.Tensor:
        """Boolean occupancy on the prediction grid -> evaluation grid (identity or any-subvoxel)."""
        return occ_to_eval_grid(occ_pred_grid, self.dataset)

    def range_bins(self, ref: AnchorRef, device) -> torch.Tensor:
        """Range bin of every **evaluation-grid** voxel by z in the anchor camera."""
        g = self.evaluation_grid
        X, Y, Z = (int(d) for d in g.dims)
        ix, iy, iz = torch.meshgrid(torch.arange(X, device=device), torch.arange(Y, device=device), torch.arange(Z, device=device), indexing="ij")
        centre = (torch.stack([ix, iy, iz], -1).reshape(-1, 3).double() + 0.5) * float(g.voxel_size) + torch.as_tensor(np.asarray(g.origin, np.float64), device=device)
        T = torch.as_tensor(np.linalg.inv(ref.T_cam_to_grid), device=device)
        z_cam = (centre @ T[:3, :3].T + T[:3, 3])[:, 2].contiguous()
        edges = torch.as_tensor(np.asarray(C.RANGE_EDGES_M, np.float64), device=device)
        return torch.bucketize(z_cam, edges, right=True).clamp_(0, len(C.RANGE_EDGES_M))   # == np.digitize: edges[b-1] <= z < edges[b]


__all__ = ["AnchorRef", "TargetAdapter"]
