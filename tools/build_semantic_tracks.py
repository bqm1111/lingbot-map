#!/usr/bin/env python
"""Build 3D observation tracks and their geometry-consolidated semantic targets.

    python tools/build_semantic_tracks.py --config configs/semantic_sidecar/feasibility.yaml \
        --teacher_density 1.0

Consumes only the frozen caches; LingBot is never re-run.  Teacher availability is
sampled at the **frame** level with the recorded seed, and a track without enough
teacher evidence is marked invalid rather than silently filled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
import torch

import _bootstrap  # noqa: F401

from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.consensus import compute_consensus, segment_ids
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import FeatureCacheReader, manifest_path
from semantic_sidecar.teacher_features import SemanticProjection, sample_teacher_frames
from semantic_sidecar.tracks import TrackSet, build_scene_tracks, tracks_are_cached

from cache_teacher_features import load_teacher_features

logger = logging.getLogger("build_tracks")


def gather_observation_features(
    tracks: TrackSet, teacher: torch.Tensor, projection: SemanticProjection, num_tokens: int
) -> torch.Tensor:
    """Project each observation's teacher feature into the compressed space.

    Args:
        teacher: ``[S, n_tokens, D]`` cached teacher features.
    """
    frames = tracks.obs_frame.long()
    tokens = tracks.obs_token.long()
    if tokens.numel() and int(tokens.max()) >= num_tokens:
        raise ValueError(
            f"observation token index {int(tokens.max())} exceeds the scene's {num_tokens} tokens"
        )
    flat = teacher.reshape(-1, teacher.shape[-1])
    picked = flat[frames * num_tokens + tokens].float()
    return projection.project(picked, normalize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--teacher_density", type=float, default=None,
                    help="override train.teacher_density; also names the output subdirectory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device(args.device)

    density = args.teacher_density if args.teacher_density is not None else cfg.train.teacher_density
    tag = f"density{int(round(density * 100)):03d}"
    cache_root = cfg.paths.cache_root
    teacher_root = os.path.join(cache_root, "teacher")
    tracks_root = os.path.join(cfg.paths.tracks_root, tag)

    scenes = expand_scenes(cfg.scenes)
    if args.roles:
        scenes = [s for s in scenes if s.role in args.roles]
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in scenes if s.name in wanted or s.name.split("_c")[0] in wanted]
    scenes = [s for s in scenes if os.path.exists(manifest_path(cache_root, s.name))]
    if not scenes:
        raise SystemExit("no cached scenes found")

    projection = SemanticProjection.load(os.path.join(cache_root, "pca.safetensors"))
    logger.info("projection: %dd basis, explains %.1f%%", projection.dim, 100 * projection.explained_ratio)
    provenance = collect_provenance(cfg, tool="build_semantic_tracks", teacher_density=density)

    stats: List[Dict[str, object]] = []
    for i, spec in enumerate(scenes, 1):
        if tracks_are_cached(tracks_root, spec.name) and not args.overwrite:
            logger.info("[%d/%d] %s: tracks cached, skipping", i, len(scenes), spec.name)
            continue
        reader = FeatureCacheReader(cache_root, spec.name)
        h, w = reader.patch_hw
        t0 = time.time()
        tracks = build_scene_tracks(reader, cfg.tracks, device, reader.manifest["patch_size"])
        logger.info(
            "[%d/%d] %s: %d tracks / %d observations from %d frames (%.1fs)",
            i, len(scenes), spec.name, tracks.num_tracks, tracks.num_observations,
            reader.num_frames, time.time() - t0,
        )
        if tracks.num_tracks == 0:
            logger.warning("%s produced no tracks; skipping", spec.name)
            continue

        teacher_feats = load_teacher_features(teacher_root, spec.name)
        obs_feats = gather_observation_features(tracks, teacher_feats, projection, h * w)
        del teacher_feats

        frame_mask = sample_teacher_frames(reader.num_frames, density, cfg.train.teacher_density_seed)
        has_teacher = torch.from_numpy(frame_mask)[tracks.obs_frame.long()]

        targets = compute_consensus(
            obs_features=obs_feats,
            obs_ptr=tracks.obs_ptr,
            obs_conf=tracks.obs_conf,
            obs_residual=tracks.obs_residual,
            obs_cos=tracks.obs_cos,
            obs_has_teacher=has_teacher,
            cfg=cfg.consensus,
            min_teacher_obs=cfg.tracks.min_teacher_obs,
        )

        tracks.save(
            tracks_root,
            spec.name,
            extra_meta={
                "teacher_density": density,
                "teacher_density_seed": cfg.train.teacher_density_seed,
                "teacher_frames": int(frame_mask.sum()),
                "provenance": provenance,
            },
        )
        targets.save(
            tracks_root,
            spec.name,
            extra={"teacher_density": density, "scene": spec.name, "projection_dim": projection.dim},
        )
        # Observation-level teacher features live next to the tracks: training needs
        # them for the pixelwise baseline and they are cheap at 64-d.
        from safetensors.torch import save_file

        save_file(
            {
                "obs_features": obs_feats.half().cpu().contiguous(),
                "obs_has_teacher": has_teacher.to(torch.uint8).contiguous(),
                "frame_teacher_mask": torch.from_numpy(frame_mask).to(torch.uint8).contiguous(),
            },
            os.path.join(tracks_root, spec.name, "obs_features.safetensors"),
            metadata={"scene": spec.name, "density": str(density)},
        )

        valid = int(targets.valid.sum())
        stats.append(
            {
                "scene": spec.name,
                "frames": reader.num_frames,
                "tracks": tracks.num_tracks,
                "observations": tracks.num_observations,
                "valid_targets": valid,
                "valid_fraction": valid / max(tracks.num_tracks, 1),
                "mean_obs_per_track": tracks.num_observations / max(tracks.num_tracks, 1),
                "mean_dispersion": float(targets.dispersion[targets.valid].mean()) if valid else float("nan"),
                "voxel_size": tracks.voxel_size,
                "median_depth": tracks.meta.get("median_depth"),
                "teacher_frames": int(frame_mask.sum()),
            }
        )
        logger.info(
            "    valid targets %d/%d (%.1f%%), mean obs/track %.2f, mean dispersion %.4f",
            valid, tracks.num_tracks, 100 * valid / max(tracks.num_tracks, 1),
            stats[-1]["mean_obs_per_track"], stats[-1]["mean_dispersion"],
        )

    if stats:
        write_json(
            os.path.join(tracks_root, "tracks_summary.json"),
            {"provenance": provenance, "teacher_density": density, "scenes": stats},
        )
    logger.info("tracks written to %s", tracks_root)


if __name__ == "__main__":
    main()
