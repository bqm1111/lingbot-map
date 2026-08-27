"""Voxel-hashed open-vocabulary feature field.

Fuses per-pixel CLIP features from many views into a sparse voxel grid, then
answers free-text queries against it.

Memory is the whole problem here: a 512-d float32 CLIP vector per voxel costs
2 KB, so a few million voxels would need gigabytes.  Instead the features are
projected onto a low-rank orthonormal basis (top-d right singular vectors of a
sample of features) *before* fusion, so only d floats per voxel are ever
stored.  Because the basis is orthonormal and uncentered, similarity survives
the projection exactly within the subspace:

    f · t  ≈  (Pᵀ P f) · t  =  (P f) · (P t)  =  z · q

so a text query is embedded once, projected once, and compared against the
stored ``z`` directly.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# 21 bits per axis inside an int64 key, centred so negative coordinates work.
_BITS = 21
_OFFSET = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def quantize(points: torch.Tensor, voxel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Map points to packed int64 voxel keys.

    Returns (keys, valid) where ``valid`` masks points that fall outside the
    representable range.
    """
    ijk = torch.floor(points / voxel_size).to(torch.int64) + _OFFSET
    valid = ((ijk >= 0) & (ijk <= _MASK)).all(dim=-1)
    ijk = ijk.clamp_(0, _MASK)
    keys = (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]
    return keys, valid


def unquantize(keys: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Voxel centres for packed keys, [M, 3]."""
    ix = (keys >> (2 * _BITS)) & _MASK
    iy = (keys >> _BITS) & _MASK
    iz = keys & _MASK
    ijk = torch.stack([ix, iy, iz], dim=-1).to(torch.float32) - _OFFSET
    return (ijk + 0.5) * voxel_size


def fit_projection(features: torch.Tensor, dim: int) -> tuple[torch.Tensor, float]:
    """Top-``dim`` orthonormal basis of a feature sample (uncentered SVD).

    Centering is deliberately skipped: with an orthonormal, uncentered basis a
    projected vector's norm is preserved, so cosine similarity can be computed
    directly in the compressed space without reconstructing the mean.

    Returns (components [dim, D], explained_ratio).
    """
    feats = features.float()
    # Full SVD on a tall matrix is wasteful; the Gram matrix is only D x D.
    gram = feats.T @ feats
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    components = evecs[:, :dim].T.contiguous()
    explained = (evals[:dim].sum() / evals.clamp_min(0).sum()).item()
    return components, explained


class FieldAccumulator:
    """Streaming voxel accumulator: sum of projected features + hit counts."""

    def __init__(
        self, voxel_size: float, dim: int, device: torch.device, buffer_points: int = 4_000_000
    ):
        self.voxel_size = voxel_size
        self.dim = dim
        self.device = device
        self.keys = torch.empty(0, dtype=torch.int64, device=device)
        self.sums = torch.empty(0, dim, dtype=torch.float32, device=device)
        self.counts = torch.empty(0, dtype=torch.float32, device=device)
        # Each merge costs O(existing_voxels), so merging once per frame makes
        # long sequences quadratic.  Batch incoming points instead.
        self.buffer_points = buffer_points
        self._buf_keys: list[torch.Tensor] = []
        self._buf_feats: list[torch.Tensor] = []
        self._buffered = 0

    @torch.no_grad()
    def add(self, points: torch.Tensor, feats: torch.Tensor) -> None:
        """Fuse [N, 3] world points carrying [N, dim] projected features."""
        keys, valid = quantize(points, self.voxel_size)
        keys, feats = keys[valid], feats[valid].float()
        if keys.numel() == 0:
            return
        self._buf_keys.append(keys)
        self._buf_feats.append(feats)
        self._buffered += keys.numel()
        if self._buffered >= self.buffer_points:
            self.flush()

    @torch.no_grad()
    def flush(self) -> None:
        """Merge buffered points into the voxel store."""
        if not self._buf_keys:
            return
        keys = torch.cat(self._buf_keys)
        feats = torch.cat(self._buf_feats)
        self._buf_keys, self._buf_feats, self._buffered = [], [], 0

        all_keys = torch.cat([self.keys, keys])
        uniq, inverse = torch.unique(all_keys, return_inverse=True)

        sums = torch.zeros(uniq.numel(), self.dim, device=self.device, dtype=torch.float32)
        sums.index_add_(0, inverse, torch.cat([self.sums, feats]))
        counts = torch.zeros(uniq.numel(), device=self.device, dtype=torch.float32)
        counts.index_add_(
            0, inverse, torch.cat([self.counts, torch.ones(keys.numel(), device=self.device)])
        )
        self.keys, self.sums, self.counts = uniq, sums, counts

    def finalize(self, components: torch.Tensor, min_count: int = 1) -> "FeatureField":
        self.flush()
        keep = self.counts >= min_count
        keys, sums, counts = self.keys[keep], self.sums[keep], self.counts[keep]
        feats = sums / counts.unsqueeze(1)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return FeatureField(
            keys=keys.cpu().numpy(),
            features=feats.half().cpu().numpy(),
            counts=counts.cpu().numpy().astype(np.int32),
            components=components.cpu().numpy().astype(np.float32),
            voxel_size=self.voxel_size,
        )


@dataclass
class FeatureField:
    """A fused, queryable open-vocabulary map."""

    keys: np.ndarray  # [M] int64, sorted
    features: np.ndarray  # [M, d] float16, L2-normalized
    counts: np.ndarray  # [M] int32
    components: np.ndarray  # [d, D] float32, orthonormal rows
    voxel_size: float
    meta: Optional[dict] = None

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            keys=self.keys,
            features=self.features,
            counts=self.counts,
            components=self.components,
            voxel_size=np.float32(self.voxel_size),
            meta=np.array(repr(self.meta or {})),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureField":
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in d:
            try:
                meta = eval(str(d["meta"]), {"__builtins__": {}})  # noqa: S307 - self-written repr
            except Exception:
                meta = {}
        return cls(
            keys=d["keys"],
            features=d["features"],
            counts=d["counts"],
            components=d["components"],
            voxel_size=float(d["voxel_size"]),
            meta=meta,
        )

    # ── query ───────────────────────────────────────────────────────────────

    def to_torch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = torch.from_numpy(self.keys).to(device)
        feats = torch.from_numpy(self.features).to(device).float()
        comps = torch.from_numpy(self.components).to(device)
        return keys, feats, comps

    @torch.no_grad()
    def query(self, text_embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Cosine similarity of every voxel against each text embedding.

        Args:
            text_embeddings: [T, D] L2-normalized, from ``DenseCLIP.encode_text``.

        Returns:
            [M, T] similarity.
        """
        _, feats, comps = self.to_torch(device)
        q = text_embeddings.to(device).float() @ comps.T  # [T, d]
        q = torch.nn.functional.normalize(q, dim=-1)
        return feats @ q.T

    @torch.no_grad()
    def lookup(self, points: torch.Tensor, keys_gpu: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Voxel index for each point, or -1 where the voxel is empty."""
        device = points.device
        keys = keys_gpu if keys_gpu is not None else torch.from_numpy(self.keys).to(device)
        pk, valid = quantize(points, self.voxel_size)
        idx = torch.searchsorted(keys, pk)
        idx = idx.clamp(max=keys.numel() - 1)
        hit = valid & (keys[idx] == pk)
        return torch.where(hit, idx, torch.full_like(idx, -1))

    @property
    def num_voxels(self) -> int:
        return int(self.keys.shape[0])

    def centres(self) -> np.ndarray:
        return unquantize(torch.from_numpy(self.keys), self.voxel_size).numpy()


def unproject(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
) -> torch.Tensor:
    """Depth map to world points, matching lingbot_map.utils.geometry conventions.

    Args:
        depth: [H, W] metric depth.
        intrinsic: [3, 3] pinhole, zero skew.
        extrinsic: [3, 4] world-to-camera (OpenCV convention).

    Returns:
        [H, W, 3] world coordinates.
    """
    H, W = depth.shape
    device = depth.device
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (u - cu) * depth / fu
    y = (v - cv) * depth / fv
    cam = torch.stack([x, y, depth], dim=-1)  # [H, W, 3]

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    # world = R^T (cam - t) for a world-to-camera extrinsic.
    return (cam - t) @ R
