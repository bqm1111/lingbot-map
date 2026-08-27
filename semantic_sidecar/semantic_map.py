"""Sparse open-vocabulary semantic map and text querying.

The map is a voxel hash carrying a normalised semantic embedding per voxel, built by
running the sidecar on every frame, bilinearly upsampling its token-grid output to
pixel resolution and lifting it with LingBot's frozen depth/pose.  Querying encodes
class names with the teacher's *text* encoder, applies the same fixed PCA basis used
for the training targets, and takes cosine similarity — so the teacher's image branch
is never needed at inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from semantic.feature_field import quantize, unquantize
from semantic_sidecar.lingbot_features import unproject_depth

BACKGROUND_PROMPTS = ("object", "things", "stuff", "texture")


@dataclass
class SparseSemanticMap:
    """Voxelised semantic reconstruction of observed surfaces."""

    keys: torch.Tensor          # [M] int64, sorted packed voxel keys
    embeddings: torch.Tensor    # [M, d] float16, L2-normalised
    counts: torch.Tensor        # [M] int32
    dispersion: torch.Tensor    # [M] float32, 1 - mean cos to the voxel mean
    rgb: torch.Tensor           # [M, 3] uint8
    voxel_size: float
    meta: Dict[str, Any]

    @property
    def num_voxels(self) -> int:
        return int(self.keys.numel())

    def centres(self) -> torch.Tensor:
        return unquantize(self.keys, self.voxel_size)

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            keys=self.keys.numpy(),
            embeddings=self.embeddings.numpy(),
            counts=self.counts.numpy(),
            dispersion=self.dispersion.numpy(),
            rgb=self.rgb.numpy(),
            voxel_size=np.float32(self.voxel_size),
            meta=np.array(json.dumps(self.meta, default=str)),
        )

    @classmethod
    def load(cls, path: str) -> "SparseSemanticMap":
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in d:
            try:
                meta = json.loads(str(d["meta"]))
            except json.JSONDecodeError:
                meta = {}
        return cls(
            keys=torch.from_numpy(d["keys"]),
            embeddings=torch.from_numpy(d["embeddings"]),
            counts=torch.from_numpy(d["counts"]),
            dispersion=torch.from_numpy(d["dispersion"]),
            rgb=torch.from_numpy(d["rgb"]),
            voxel_size=float(d["voxel_size"]),
            meta=meta,
        )

    # -- query ------------------------------------------------------------- #
    def similarity(self, text_embeddings: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """``[M, T]`` cosine similarity against already-projected text embeddings."""
        device = device or self.embeddings.device
        emb = self.embeddings.to(device).float()
        q = F.normalize(text_embeddings.to(device).float(), dim=-1)
        return emb @ q.T

    def classify(
        self, text_embeddings: torch.Tensor, temperature: float = 0.05, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-set assignment: ``(labels [M], scores [M])`` with labels in ``0..T-1``."""
        sims = self.similarity(text_embeddings, device)
        probs = torch.softmax(sims / max(temperature, 1e-6), dim=-1)
        scores, labels = probs.max(dim=-1)
        return labels, scores

    def relevancy(
        self, query: torch.Tensor, negatives: torch.Tensor, device=None
    ) -> torch.Tensor:
        """LERF-style relevancy of one query against fixed background prompts."""
        pos = self.similarity(query.reshape(1, -1), device).squeeze(-1)
        neg = self.similarity(negatives, device)
        return torch.exp(pos).unsqueeze(-1) / (torch.exp(pos).unsqueeze(-1) + torch.exp(neg))

    def lookup(self, points: torch.Tensor) -> torch.Tensor:
        """Voxel index per world point, ``-1`` when the voxel is empty."""
        keys = self.keys.to(points.device)
        pk, valid = quantize(points, self.voxel_size)
        idx = torch.searchsorted(keys, pk).clamp(max=max(keys.numel() - 1, 0))
        hit = valid & (keys.numel() > 0) & (keys[idx.clamp(max=max(keys.numel() - 1, 0))] == pk)
        return torch.where(hit, idx, torch.full_like(idx, -1))

    # -- export ------------------------------------------------------------ #
    def export_ply(self, path: str, colours: Optional[np.ndarray] = None) -> None:
        """Write an ASCII PLY of the map, optionally recoloured per voxel."""
        pts = self.centres().numpy().astype(np.float32)
        cols = self.rgb.numpy() if colours is None else colours.astype(np.uint8)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            )
            for p, c in zip(pts, cols):
                fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


class SemanticMapAccumulator:
    """Streaming voxel accumulation of normalised embeddings + colour.

    Tracks the running sum of embeddings (for the mean) and of their squared norms via
    the sum of cosines against the final mean, which yields per-voxel dispersion in a
    single pass without storing observations.
    """

    def __init__(self, voxel_size: float, dim: int, device: torch.device, buffer_points: int = 2_000_000):
        self.voxel_size, self.dim, self.device = voxel_size, dim, device
        self.keys = torch.empty(0, dtype=torch.int64, device=device)
        self.sums = torch.empty(0, dim, dtype=torch.float32, device=device)
        self.counts = torch.empty(0, dtype=torch.float32, device=device)
        self.rgb = torch.empty(0, 3, dtype=torch.float32, device=device)
        self.buffer_points = buffer_points
        self._buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._buffered = 0

    @torch.no_grad()
    def add(self, points: torch.Tensor, feats: torch.Tensor, rgb: torch.Tensor) -> None:
        keys, valid = quantize(points, self.voxel_size)
        if not bool(valid.any()):
            return
        self._buf.append((keys[valid], feats[valid].float(), rgb[valid].float()))
        self._buffered += int(valid.sum())
        if self._buffered >= self.buffer_points:
            self.flush()

    @torch.no_grad()
    def flush(self) -> None:
        if not self._buf:
            return
        keys = torch.cat([b[0] for b in self._buf])
        feats = torch.cat([b[1] for b in self._buf])
        rgb = torch.cat([b[2] for b in self._buf])
        self._buf, self._buffered = [], 0

        all_keys = torch.cat([self.keys, keys])
        uniq, inverse = torch.unique(all_keys, return_inverse=True)
        sums = torch.zeros(uniq.numel(), self.dim, device=self.device)
        sums.index_add_(0, inverse, torch.cat([self.sums, feats]))
        cols = torch.zeros(uniq.numel(), 3, device=self.device)
        cols.index_add_(0, inverse, torch.cat([self.rgb, rgb]))
        counts = torch.zeros(uniq.numel(), device=self.device)
        counts.index_add_(
            0, inverse, torch.cat([self.counts, torch.ones(keys.numel(), device=self.device)])
        )
        self.keys, self.sums, self.counts, self.rgb = uniq, sums, counts, cols

    @torch.no_grad()
    def finalize(self, min_count: int = 1, meta: Optional[Dict[str, Any]] = None) -> SparseSemanticMap:
        self.flush()
        keep = self.counts >= min_count
        keys, sums, counts, cols = self.keys[keep], self.sums[keep], self.counts[keep], self.rgb[keep]
        mean = sums / counts.unsqueeze(1)
        norms = mean.norm(dim=-1)
        # For unit-norm observations, ||mean|| is exactly the mean cosine to the mean
        # direction, so dispersion needs no second pass.
        dispersion = (1.0 - norms).clamp(0.0, 1.0)
        feats = F.normalize(mean, dim=-1)
        return SparseSemanticMap(
            keys=keys.cpu(),
            embeddings=feats.half().cpu(),
            counts=counts.to(torch.int32).cpu(),
            dispersion=dispersion.float().cpu(),
            rgb=(cols / counts.unsqueeze(1)).clamp(0, 255).to(torch.uint8).cpu(),
            voxel_size=self.voxel_size,
            meta=meta or {},
        )


@torch.no_grad()
def embeddings_to_pixels(token_embeddings: torch.Tensor, h: int, w: int, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Upsample ``[h*w, d]`` token predictions to ``[H, W, d]``, renormalised."""
    d = token_embeddings.shape[-1]
    grid = token_embeddings.reshape(h, w, d).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)[0]
    return F.normalize(up.permute(1, 2, 0), dim=-1)


@torch.no_grad()
def patch_rgb_to_pixels(patch_rgb: torch.Tensor, h: int, w: int, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Expand cached patch-grid colour ``[3, h*w]`` uint8 to ``[H, W, 3]`` float 0-255."""
    grid = patch_rgb.float().reshape(3, h, w).unsqueeze(0)
    up = F.interpolate(grid, size=out_hw, mode="nearest")[0]
    return up.permute(1, 2, 0)


@torch.no_grad()
def embedding_diversity(embeddings: torch.Tensor, num_pairs: int = 20000, seed: int = 0) -> float:
    """``1 - mean |cos|`` between random pairs of embeddings.

    A sidecar that collapses to a single direction scores ~0 here while still looking
    perfectly "cross-view consistent", so this is the guard against reading a
    degenerate solution as a good one.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(0, n, (min(num_pairs, n * 4),), generator=g)
    b = torch.randint(0, n, (min(num_pairs, n * 4),), generator=g)
    keep = a != b
    if not bool(keep.any()):
        return float("nan")
    x = F.normalize(embeddings.float(), dim=-1)
    cos = (x[a[keep]] * x[b[keep]]).sum(-1)
    return float(1.0 - cos.abs().mean())


@torch.no_grad()
def lift_frame(
    embeddings: torch.Tensor,
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    rgb: Optional[torch.Tensor],
    pixel_stride: int,
    conf_threshold: float,
    min_depth: float,
    max_depth: Optional[float],
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift a frame's per-pixel embeddings into world space.

    Args:
        embeddings: ``[H, W, d]`` normalised per-pixel semantic vectors.
        rgb: Optional ``[H, W, 3]`` colour in 0-255; mid-grey is used when absent.

    Returns:
        ``(points [N,3], feats [N,d], rgb [N,3])`` after sub-sampling and filtering.
    """
    keep = (depth > min_depth) & (depth_conf >= conf_threshold)
    if max_depth is not None:
        keep &= depth < max_depth
    if valid_mask is not None:
        keep &= valid_mask
    keep = keep[::pixel_stride, ::pixel_stride]

    world = unproject_depth(depth, intrinsic, extrinsic)[::pixel_stride, ::pixel_stride]
    feats = embeddings[::pixel_stride, ::pixel_stride]
    colour = torch.full_like(world, 128.0) if rgb is None else rgb[::pixel_stride, ::pixel_stride]
    return world[keep], feats[keep], colour[keep]
