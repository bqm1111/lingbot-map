"""Build and read the voxel feature targets -- the C-condition supervision.

Same privileged window, same geometry, same fusion rule as the class-probability target the
gate already uses. The only difference is what rides on the ray: instead of Trident-H's
25-channel class distribution, each accepted pixel carries its frozen SAM-H embedding,
projected through the frozen 64-d basis.

Fusion is therefore literally the same code -- ``IncrementalMapper`` accumulates
``sum(w * v)`` and ``sum(w)`` per voxel with ``w`` the geometry-reliability weight
``W_LINGBOT * clamp(conf / conf_threshold, 0.25, 2)``, never the teacher's own confidence --
so a voxel's feature target is the **confidence-weighted mean** of the per-pixel features of
every future ray whose occupied band reached it, and its support weight is identical to the
class target's. Evidence lands only on the occupied band around a measured surface; free
space carries no feature, and nothing is written behind a surface.

The support actually used in training is the **intersection** of that with the gate's
existing class-probability support (fused weight >= 1, geometrically supported, and the two
halves of the future window agreeing on the top-1 class). Storing the intersection means B
and C are supervised on exactly the same voxels -- the matched control the study needs --
and the per-anchor row counts recorded alongside it say how much that intersection gave up
relative to the historical B configuration.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.mapping.feed import CachedFeed
from tools.feature_study import study as ST
from tools.feature_study.sam_teacher import lattice_grid, to_lattice


class Projection:
    """The frozen mean + basis. Loaded once, applied identically everywhere."""

    def __init__(self, device, path: Optional[str] = None):
        self.path = path or ST.pca_path()
        with np.load(self.path) as z:
            self.mean = torch.from_numpy(z["mean"]).to(device)            # [256]
            self.basis = torch.from_numpy(z["basis"]).to(device)          # [256, 64]
            self.n_cells = int(z["n_cells"])
        self.sha256 = ST.sha256_file(self.path)
        self.device = device

    def apply(self, feat: torch.Tensor) -> torch.Tensor:
        """``[256, H, W]`` -> ``[64, H, W]``. Centring and projection are both linear, so
        doing them before the bilinear resample gives the same field as doing them after."""
        C, H, W = feat.shape
        x = feat.reshape(C, -1).T - self.mean
        return (x @ self.basis).T.reshape(ST.FEATURE_DIM, H, W)


class FeatureFeed(CachedFeed):
    """The gate's frozen feed, with the SAM feature map in place of the class map.

    ``sem_probs`` is the mapper's only teacher input, so substituting it is the whole
    change: geometry, poses, depth, confidence and the metric scale are untouched.
    """

    def __init__(self, seg, device, source: str, proj: Projection, cache_size: int = 32):
        super().__init__(seg, device, with_semantics=False)
        self.source = source
        self.proj = proj
        self._cache: Dict[int, torch.Tensor] = {}
        self._order: list = []
        self._cache_size = int(cache_size)
        self._grid: Optional[torch.Tensor] = None
        self.n_missing_feat = 0

    def feature_map(self, i: int) -> Optional[torch.Tensor]:
        if i in self._cache:
            return self._cache[i]
        key = self.seg.frames[i].key
        p = ST.sam_feature_path(self.source, key)
        if not os.path.exists(p):
            self.n_missing_feat += 1
            return None
        with np.load(p) as z:
            feat = torch.from_numpy(z["feat"].astype(np.float32)).to(self.device)
            native_hw = tuple(int(x) for x in z["native_hw"])
            input_hw = tuple(int(x) for x in z["input_hw"])
        lat_hw = (int(self.dep.shape[1]), int(self.dep.shape[2]))
        if self._grid is None:
            self._grid = lattice_grid(native_hw, input_hw, lat_hw, self.device)
        v = to_lattice(self.proj.apply(feat), self._grid)                 # [64, hp, wp]
        v = v.permute(1, 2, 0).contiguous()                               # [hp, wp, 64]
        self._cache[i] = v
        self._order.append(i)
        while len(self._order) > self._cache_size:
            self._cache.pop(self._order.pop(0), None)
        return v

    def frame(self, i: int):
        fi = super().frame(i)
        fi.sem_probs = self.feature_map(i)
        return fi


def save(path: str, rows: np.ndarray, feat: np.ndarray, w: np.ndarray,
         support: Dict[str, int], dims, stream_index: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, feat_rows=rows.astype(np.int32), feat=feat.astype(np.float16),
             feat_w=w.astype(np.float16), dims=np.asarray(dims, np.int32),
             stream_index=np.int32(stream_index),
             n_class_support=np.int64(support["class"]),
             n_feature_support=np.int64(support["feature"]),
             n_common_support=np.int64(support["common"]),
             feature_dim=np.int32(ST.FEATURE_DIM))
    os.replace(tmp, path)
    return os.path.getsize(path)


def load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {"feat_rows": z["feat_rows"].astype(np.int64),
                "feat": z["feat"].astype(np.float32), "feat_w": z["feat_w"].astype(np.float32),
                "dims": tuple(int(x) for x in z["dims"]),
                "n_class_support": int(z["n_class_support"]),
                "n_feature_support": int(z["n_feature_support"]),
                "n_common_support": int(z["n_common_support"])}


__all__ = ["Projection", "FeatureFeed", "save", "load"]
