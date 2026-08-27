"""3D observation tracks: which (frame, token) pairs look at the same surface.

For a voxel centre ``x``, the track is ``O_x = {(t, u) : ||P_t(u) - x|| < eps}`` where
``P_t(u)`` is the world point LingBot predicts for token ``u`` of frame ``t``.  Tracks
are the mechanism that turns inconsistent per-view teacher features into a single
view-consistent 3D target, so their purity matters more than their count.

Storage is CSR (``obs_ptr`` into flat observation arrays), sharded into safetensors
files with a JSON manifest.  Token *features* are never copied here — an observation
is a ``(frame, token)`` index into the stage-1 feature cache, so the two stores cannot
drift apart.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from semantic.feature_field import quantize, unquantize
from semantic_sidecar.config import TrackConfig
from semantic_sidecar.lingbot_features import (
    FeatureCacheReader,
    camera_center,
    patch_centre_pixels,
    project_to_camera,
    unproject_depth,
)

logger = logging.getLogger(__name__)

OBS_PER_SHARD = 2_000_000


# --------------------------------------------------------------------------- #
# Voxelisation
# --------------------------------------------------------------------------- #
def anchor_statistics(
    reader: FeatureCacheReader, num_anchors: int, conf_threshold: float, min_depth: float
) -> Tuple[float, float]:
    """``(median valid depth, median focal length)`` over evenly spaced anchor frames.

    The depth is the scene's characteristic scale — the monocular reconstruction has
    no metric units, so voxel size is expressed relative to it.  The focal length
    converts a patch-token width into a world-space footprint.
    """
    n = reader.num_frames
    idx = np.unique(np.linspace(0, n - 1, min(num_anchors, n)).astype(int))
    samples: List[torch.Tensor] = []
    focals: List[float] = []
    for i in idx:
        g = reader.geometry(int(i))
        depth = g["depth"].float()
        conf = g["depth_conf"].float()
        keep = (depth > min_depth) & (conf >= conf_threshold)
        if keep.any():
            samples.append(depth[keep].flatten())
        focals.append(float(g["intrinsic"][0, 0]))
    if not samples:
        raise ValueError("no valid depth found for the anchor frames")
    return float(torch.cat(samples).median()), float(np.median(focals))


def anchor_median_depth(
    reader: FeatureCacheReader, num_anchors: int, conf_threshold: float, min_depth: float
) -> float:
    """Median valid depth over evenly spaced anchor frames."""
    return anchor_statistics(reader, num_anchors, conf_threshold, min_depth)[0]


def voxel_size_for_scene(
    cfg: TrackConfig,
    median_depth: float,
    focal: Optional[float] = None,
    patch_size: int = 14,
) -> float:
    """Voxel edge for a scene, in the reconstruction's own (arbitrary) units.

    Priority: an explicit metric size, then a size expressed in patch-token
    footprints (``voxel_token_scale``), then a plain fraction of the median depth.
    The token-footprint form is preferred because a voxel narrower than one token's
    world footprint cannot merge observations from two views, which silently starves
    the tracks.
    """
    if cfg.metric_voxel_size is not None:
        return float(cfg.metric_voxel_size)
    if cfg.voxel_token_scale is not None and focal:
        return float(cfg.voxel_token_scale * patch_size * median_depth / focal)
    return float(cfg.rel_voxel_size * median_depth)


# --------------------------------------------------------------------------- #
# Per-frame token observations
# --------------------------------------------------------------------------- #
@dataclass
class FrameObservations:
    """Token-level observations of one frame, already filtered for validity."""

    token_index: torch.Tensor  # [n] int64, index into the h*w token grid
    points: torch.Tensor       # [n, 3] world
    depth: torch.Tensor        # [n]
    conf: torch.Tensor         # [n]
    view_dir: torch.Tensor     # [n, 3] unit vector from surface towards the camera


def frame_observations(
    depth: torch.Tensor,
    conf: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    patch_size: int,
    cfg: TrackConfig,
    median_depth: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> FrameObservations:
    """Sample one world point per patch token and drop unreliable ones.

    Args:
        depth: ``[H, W]``.
        conf: ``[H, W]`` LingBot ``depth_conf``.
        intrinsic: ``[3, 3]``.
        extrinsic: ``[3, 4]`` world-to-camera.
        patch_size: LingBot patch size (14).
        cfg: Track configuration (thresholds).
        median_depth: Scene anchor depth, used for the relative far clip.
        valid_mask: Optional ``[H, W]`` bool, ``True`` where the pixel is usable
            (e.g. not sky).
    """
    H, W = depth.shape
    device = depth.device
    centres = patch_centre_pixels(H, W, patch_size, device=device)
    ys = centres[:, 0].round().long().clamp(0, H - 1)
    xs = centres[:, 1].round().long().clamp(0, W - 1)

    d = depth[ys, xs].float()
    c = conf[ys, xs].float()
    keep = (d > cfg.min_depth) & (d < cfg.max_depth_rel * median_depth) & (c >= cfg.conf_threshold)
    if valid_mask is not None:
        keep &= valid_mask[ys, xs]

    world = unproject_depth(depth.float(), intrinsic.float(), extrinsic.float())
    pts = world[ys, xs]
    cam_c = camera_center(extrinsic.float())
    view = torch.nn.functional.normalize(cam_c.unsqueeze(0) - pts, dim=-1)

    idx = torch.arange(centres.shape[0], device=device)
    return FrameObservations(
        token_index=idx[keep], points=pts[keep], depth=d[keep], conf=c[keep], view_dir=view[keep]
    )


def reprojection_residual(
    centres: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    depth_map: torch.Tensor,
) -> torch.Tensor:
    """Relative depth disagreement of ``centres`` when reprojected into a frame.

    A track centre that a frame really observes should project to a pixel whose
    predicted depth matches the centre's camera-space z.  Returns
    ``|z - d(u,v)| / z``; points that fall behind or outside the image get ``inf``.
    """
    H, W = depth_map.shape
    uv, z = project_to_camera(centres, intrinsic.float(), extrinsic.float())
    u = uv[:, 0].round().long()
    v = uv[:, 1].round().long()
    inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    res = torch.full_like(z, float("inf"))
    if inside.any():
        d = depth_map.float()[v[inside].clamp(0, H - 1), u[inside].clamp(0, W - 1)]
        res[inside] = (z[inside] - d).abs() / z[inside].clamp_min(1e-6)
    return res


# --------------------------------------------------------------------------- #
# Track set
# --------------------------------------------------------------------------- #
@dataclass
class TrackSet:
    """CSR-encoded observation tracks for one scene."""

    centres: torch.Tensor       # [M, 3] float32
    keys: torch.Tensor          # [M] int64 packed voxel keys
    obs_ptr: torch.Tensor       # [M + 1] int64
    obs_frame: torch.Tensor     # [K] int32
    obs_token: torch.Tensor     # [K] int32
    obs_conf: torch.Tensor      # [K] float32
    obs_residual: torch.Tensor  # [K] float32
    obs_cos: torch.Tensor       # [K] float32, |cos| between view dir and mean view dir
    obs_dist: torch.Tensor      # [K] float32, distance from camera
    voxel_size: float
    meta: Dict[str, Any]

    @property
    def num_tracks(self) -> int:
        return int(self.centres.shape[0])

    @property
    def num_observations(self) -> int:
        return int(self.obs_frame.shape[0])

    def observations(self, track: int) -> Dict[str, torch.Tensor]:
        a, b = int(self.obs_ptr[track]), int(self.obs_ptr[track + 1])
        return {
            "frame": self.obs_frame[a:b],
            "token": self.obs_token[a:b],
            "conf": self.obs_conf[a:b],
            "residual": self.obs_residual[a:b],
            "cos": self.obs_cos[a:b],
            "dist": self.obs_dist[a:b],
        }

    def counts(self) -> torch.Tensor:
        return (self.obs_ptr[1:] - self.obs_ptr[:-1]).to(torch.int32)

    # -- persistence ------------------------------------------------------- #
    def tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "centres": self.centres,
            "keys": self.keys,
            "obs_ptr": self.obs_ptr,
            "obs_frame": self.obs_frame,
            "obs_token": self.obs_token,
            "obs_conf": self.obs_conf,
            "obs_residual": self.obs_residual,
            "obs_cos": self.obs_cos,
            "obs_dist": self.obs_dist,
        }

    def save(self, root: str, scene: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """Write sharded safetensors + manifest.  Returns the manifest path."""
        from safetensors.torch import save_file

        out_dir = os.path.join(root, scene)
        os.makedirs(out_dir, exist_ok=True)
        counts = self.counts()
        shards: List[Dict[str, Any]] = []

        start, running, shard_idx = 0, 0, 0
        bounds: List[Tuple[int, int]] = []
        for i in range(self.num_tracks):
            running += int(counts[i])
            if running >= OBS_PER_SHARD:
                bounds.append((start, i + 1))
                start, running = i + 1, 0
        if start < self.num_tracks:
            bounds.append((start, self.num_tracks))
        if not bounds:
            bounds = [(0, 0)]

        for lo, hi in bounds:
            a, b = int(self.obs_ptr[lo]), int(self.obs_ptr[hi])
            ptr = self.obs_ptr[lo : hi + 1] - self.obs_ptr[lo]
            payload = {
                "centres": self.centres[lo:hi].contiguous(),
                "keys": self.keys[lo:hi].contiguous(),
                "obs_ptr": ptr.contiguous(),
                "obs_frame": self.obs_frame[a:b].contiguous(),
                "obs_token": self.obs_token[a:b].contiguous(),
                "obs_conf": self.obs_conf[a:b].contiguous(),
                "obs_residual": self.obs_residual[a:b].contiguous(),
                "obs_cos": self.obs_cos[a:b].contiguous(),
                "obs_dist": self.obs_dist[a:b].contiguous(),
            }
            fname = f"tracks_{shard_idx:05d}.safetensors"
            save_file(payload, os.path.join(out_dir, fname), metadata={"scene": scene})
            shards.append({"index": shard_idx, "file": fname, "tracks": [lo, hi], "observations": b - a})
            shard_idx += 1

        manifest = {
            "version": 1,
            "scene": scene,
            "voxel_size": self.voxel_size,
            "num_tracks": self.num_tracks,
            "num_observations": self.num_observations,
            "shards": shards,
            **self.meta,
            **(extra_meta or {}),
        }
        path = os.path.join(out_dir, "manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, root: str, scene: str) -> "TrackSet":
        from safetensors.torch import load_file

        out_dir = os.path.join(root, scene)
        with open(os.path.join(out_dir, "manifest.json")) as fh:
            manifest = json.load(fh)

        parts = [load_file(os.path.join(out_dir, s["file"])) for s in manifest["shards"]]
        if not parts:
            raise ValueError(f"no shards in {out_dir}")

        centres = torch.cat([p["centres"] for p in parts])
        keys = torch.cat([p["keys"] for p in parts])
        ptrs, offset = [], 0
        for p in parts:
            ptrs.append(p["obs_ptr"][:-1] + offset)
            offset += int(p["obs_ptr"][-1])
        obs_ptr = torch.cat(ptrs + [torch.tensor([offset], dtype=torch.int64)])
        cat = lambda k: torch.cat([p[k] for p in parts])  # noqa: E731
        return cls(
            centres=centres,
            keys=keys,
            obs_ptr=obs_ptr,
            obs_frame=cat("obs_frame"),
            obs_token=cat("obs_token"),
            obs_conf=cat("obs_conf"),
            obs_residual=cat("obs_residual"),
            obs_cos=cat("obs_cos"),
            obs_dist=cat("obs_dist"),
            voxel_size=float(manifest["voxel_size"]),
            meta=manifest,
        )


def tracks_are_cached(root: str, scene: str) -> bool:
    path = os.path.join(root, scene, "manifest.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError:
        return False
    return all(os.path.exists(os.path.join(root, scene, s["file"])) for s in manifest["shards"])


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_tracks_from_observations(
    per_frame: Sequence[Tuple[int, FrameObservations]],
    voxel_size: float,
    cfg: TrackConfig,
    depth_maps: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
) -> TrackSet:
    """Group per-frame observations into tracks and filter them.

    Args:
        per_frame: ``(frame_index, FrameObservations)`` pairs.
        voxel_size: World units per voxel.
        cfg: Thresholds and caps.
        depth_maps: Optional ``frame -> (depth, intrinsic, extrinsic)`` used for the
            cross-view reprojection check.  Without it residuals stay at 0.
        device: Device to run the grouping on.

    Returns:
        A filtered :class:`TrackSet`.
    """
    device = device or (per_frame[0][1].points.device if per_frame else torch.device("cpu"))
    if not per_frame:
        return _empty_trackset(voxel_size, cfg)

    pts = torch.cat([o.points for _, o in per_frame]).to(device)
    if pts.numel() == 0:
        return _empty_trackset(voxel_size, cfg)
    frames = torch.cat(
        [torch.full((o.points.shape[0],), f, dtype=torch.int64) for f, o in per_frame]
    ).to(device)
    tokens = torch.cat([o.token_index for _, o in per_frame]).to(device)
    confs = torch.cat([o.conf for _, o in per_frame]).to(device)
    views = torch.cat([o.view_dir for _, o in per_frame]).to(device)
    dists = torch.cat([o.depth for _, o in per_frame]).to(device)

    keys, valid = quantize(pts, voxel_size)
    keys, pts, frames, tokens, confs, views, dists = (
        keys[valid], pts[valid], frames[valid], tokens[valid], confs[valid], views[valid], dists[valid]
    )
    if keys.numel() == 0:
        return _empty_trackset(voxel_size, cfg)

    # Sort by (key, frame) so every track's observations are contiguous and ordered.
    order = torch.argsort(keys * (int(frames.max()) + 2) + frames)
    keys, pts, frames, tokens, confs, views, dists = (
        keys[order], pts[order], frames[order], tokens[order], confs[order], views[order], dists[order]
    )
    uniq, inverse, counts = torch.unique_consecutive(keys, return_inverse=True, return_counts=True)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)])

    # Track centre = mean of member points (voxel-bounded, so the mean is safe).
    centres = torch.zeros(uniq.numel(), 3, device=device, dtype=torch.float32)
    centres.index_add_(0, inverse, pts)
    centres /= counts.unsqueeze(1).float()

    # -- geometric outlier rejection (MAD about the track centre) ------------ #
    resid3d = (pts - centres[inverse]).norm(dim=-1)
    med = _grouped_median(resid3d, inverse, uniq.numel())
    mad = _grouped_median((resid3d - med[inverse]).abs(), inverse, uniq.numel())
    tol = (med + cfg.outlier_mad_scale * mad.clamp_min(1e-8)).clamp_min(0.5 * voxel_size)
    keep = resid3d <= tol[inverse]

    # -- cross-view depth/reprojection consistency --------------------------- #
    residual = torch.zeros_like(resid3d)
    if depth_maps:
        for f, (depth, intr, extr) in depth_maps.items():
            sel = frames == f
            if not bool(sel.any()):
                continue
            r = reprojection_residual(
                centres[inverse[sel]], intr.to(device), extr.to(device), depth.to(device)
            )
            residual[sel] = torch.nan_to_num(r, posinf=1e3)
        keep &= residual <= cfg.reproj_tol

    # -- view-direction agreement ------------------------------------------- #
    mean_view = torch.zeros_like(centres)
    mean_view.index_add_(0, inverse, views)
    mean_view = torch.nn.functional.normalize(mean_view, dim=-1)
    cos = (views * mean_view[inverse]).sum(-1).clamp(-1.0, 1.0)

    # -- cap observations per track (seeded, deterministic) ------------------ #
    if cfg.max_observations > 0:
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        noise = torch.rand(keep.shape[0], generator=gen).to(device)
        rank = _grouped_rank(torch.where(keep, noise, noise + 10.0), inverse, uniq.numel())
        keep &= rank < cfg.max_observations

    # -- drop short tracks and re-pack --------------------------------------- #
    kept_counts = torch.zeros(uniq.numel(), device=device, dtype=torch.int64)
    kept_counts.index_add_(0, inverse, keep.to(torch.int64))
    good_track = kept_counts >= cfg.min_observations
    keep &= good_track[inverse]

    track_map = torch.full((uniq.numel(),), -1, dtype=torch.int64, device=device)
    kept_tracks = torch.nonzero(good_track, as_tuple=False).squeeze(-1)
    track_map[kept_tracks] = torch.arange(kept_tracks.numel(), device=device)

    sel = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    new_track = track_map[inverse[sel]]
    reorder = torch.argsort(new_track * (int(frames.max()) + 2) + frames[sel])
    sel = sel[reorder]
    new_track = new_track[reorder]

    new_counts = torch.bincount(new_track, minlength=kept_tracks.numel())
    new_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), new_counts.cumsum(0)])

    # Recompute centres from the surviving observations only.
    final_centres = torch.zeros(kept_tracks.numel(), 3, device=device, dtype=torch.float32)
    final_centres.index_add_(0, new_track, pts[sel])
    final_centres /= new_counts.clamp_min(1).unsqueeze(1).float()

    meta = {
        "min_observations": cfg.min_observations,
        "max_observations": cfg.max_observations,
        "conf_threshold": cfg.conf_threshold,
        "reproj_tol": cfg.reproj_tol,
        "outlier_mad_scale": cfg.outlier_mad_scale,
        "seed": cfg.seed,
        "raw_observations": int(keys.numel()),
        "raw_voxels": int(uniq.numel()),
    }
    return TrackSet(
        centres=final_centres.cpu(),
        keys=uniq[kept_tracks].cpu(),
        obs_ptr=new_ptr.cpu(),
        obs_frame=frames[sel].to(torch.int32).cpu(),
        obs_token=tokens[sel].to(torch.int32).cpu(),
        obs_conf=confs[sel].cpu(),
        obs_residual=residual[sel].cpu(),
        obs_cos=cos[sel].cpu(),
        obs_dist=dists[sel].cpu(),
        voxel_size=voxel_size,
        meta=meta,
    )


def _empty_trackset(voxel_size: float, cfg: TrackConfig) -> TrackSet:
    z = lambda dt: torch.zeros(0, dtype=dt)  # noqa: E731
    return TrackSet(
        centres=torch.zeros(0, 3),
        keys=z(torch.int64),
        obs_ptr=torch.zeros(1, dtype=torch.int64),
        obs_frame=z(torch.int32),
        obs_token=z(torch.int32),
        obs_conf=z(torch.float32),
        obs_residual=z(torch.float32),
        obs_cos=z(torch.float32),
        obs_dist=z(torch.float32),
        voxel_size=voxel_size,
        meta={"min_observations": cfg.min_observations, "raw_observations": 0, "raw_voxels": 0},
    )


def _grouped_median(values: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Per-group median of a flat value array (groups need not be contiguous)."""
    out = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    order = torch.argsort(group * (values.max() - values.min() + 1.0) + values)
    sorted_group = group[order]
    sorted_vals = values[order]
    counts = torch.bincount(group, minlength=num_groups)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=values.device), counts.cumsum(0)[:-1]])
    mid = starts + counts // 2
    nonempty = counts > 0
    out[nonempty] = sorted_vals[mid[nonempty]]
    _ = sorted_group
    return out


def _grouped_rank(values: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Rank of each element inside its group, ascending by ``values``."""
    order = torch.argsort(group.double() * (values.max() - values.min() + 1.0).double() + values.double())
    counts = torch.bincount(group, minlength=num_groups)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=values.device), counts.cumsum(0)[:-1]])
    position = torch.arange(values.numel(), device=values.device) - starts[group[order]]
    rank = torch.empty_like(position)
    rank[order] = position
    return rank


def build_scene_tracks(
    reader: FeatureCacheReader,
    cfg: TrackConfig,
    device: torch.device,
    patch_size: int = 14,
    progress: bool = True,
) -> TrackSet:
    """End-to-end track construction for one cached scene."""
    from tqdm.auto import tqdm

    median_depth, focal = anchor_statistics(
        reader, cfg.anchor_frames, cfg.conf_threshold, cfg.min_depth
    )
    voxel_size = voxel_size_for_scene(cfg, median_depth, focal, patch_size)
    logger.info(
        "scene %s: median depth %.4f, focal %.1f -> voxel size %.5f (%.2f token footprints)",
        reader.scene, median_depth, focal, voxel_size,
        voxel_size / max(patch_size * median_depth / max(focal, 1e-6), 1e-9),
    )

    per_frame: List[Tuple[int, FrameObservations]] = []
    depth_maps: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    it = range(reader.num_frames)
    for f in tqdm(it, desc=f"observations[{reader.scene}]", disable=not progress):
        g = reader.geometry(f)
        depth = g["depth"].to(device).float()
        conf = g["depth_conf"].to(device).float()
        intr = g["intrinsic"].to(device).float()
        extr = g["extrinsic"].to(device).float()
        obs = frame_observations(depth, conf, intr, extr, patch_size, cfg, median_depth)
        if obs.points.numel():
            per_frame.append((f, obs))
        depth_maps[f] = (depth, intr, extr)

    tracks = build_tracks_from_observations(per_frame, voxel_size, cfg, depth_maps, device=device)
    tracks.meta.update(
        {
            "median_depth": median_depth,
            "focal": focal,
            "voxel_token_scale": cfg.voxel_token_scale,
            "rel_voxel_size": cfg.rel_voxel_size,
            "metric_voxel_size": cfg.metric_voxel_size,
            "num_frames": reader.num_frames,
            "patch_grid": list(reader.patch_hw),
        }
    )
    return tracks
