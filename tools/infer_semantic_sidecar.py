#!/usr/bin/env python
"""Run a sidecar (or a baseline) over cached scenes and build sparse semantic maps.

    python tools/infer_semantic_sidecar.py --config configs/semantic_sidecar/feasibility.yaml \
        --run_dir output/semantic_sidecar/runs/mlp_consensus --roles eval qualitative

Baselines that need no training:
  --baseline teacher_direct   frozen teacher features pooled onto the LingBot token
                              grid and projected with the same PCA basis — the exact
                              ceiling the sidecar is trying to match, since the
                              sidecar also predicts per token
  --baseline teacher_fullres  the same teacher at its own dense resolution, resampled
                              straight to pixels — the *unconstrained* teacher, which
                              shows what the token-grid quantisation costs
  --baseline lingbot_raw      raw LingBot tokens compressed to the same width — a
                              text-unaligned control, expected to be near chance

Map voxel size is derived from geometry alone (scene anchor median depth), so every
model in the comparison is voxelised identically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import FeatureCacheReader, manifest_path
from semantic_sidecar.metrics import cross_view_consistency
from semantic_sidecar.models import build_sidecar
from semantic_sidecar.semantic_map import (
    SemanticMapAccumulator,
    embedding_diversity,
    embeddings_to_pixels,
    lift_frame,
    patch_rgb_to_pixels,
)
from semantic_sidecar.teacher_features import SemanticProjection
from semantic_sidecar.tracks import TrackSet, anchor_statistics, tracks_are_cached, voxel_size_for_scene

logger = logging.getLogger("infer_sidecar")
BASELINES = ("teacher_direct", "teacher_fullres", "lingbot_raw")


def load_sidecar(run_dir: str, device: torch.device):
    """Load ``best.pt`` and rebuild the architecture from its stored config."""
    from semantic_sidecar.config import ModelConfig

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["config"]["model"])
    num_layers = len(ckpt["config"]["lingbot"]["layers"])
    model = build_sidecar(mcfg, num_layers=num_layers)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def token_embeddings_for_scene(
    mode: str,
    reader: FeatureCacheReader,
    device: torch.device,
    model=None,
    teacher_features: Optional[torch.Tensor] = None,
    projection: Optional[SemanticProjection] = None,
    token_basis: Optional[torch.Tensor] = None,
    batch: int = 16,
) -> torch.Tensor:
    """Per-frame token embeddings ``[S, n_tokens, d]`` for the chosen model/baseline."""
    n = reader.num_frames
    out: List[torch.Tensor] = []
    for start in range(0, n, batch):
        frames = range(start, min(start + batch, n))
        if mode in ("teacher_direct", "teacher_fullres"):
            feats = teacher_features[list(frames)].to(device).float()
            emb = projection.to(device).project(feats, normalize=True)
        else:
            toks = torch.stack([reader.tokens(f) for f in frames]).to(device).float()  # [B,L,N,C]
            toks = toks.permute(0, 2, 1, 3).contiguous()  # [B,N,L,C]
            if mode == "sidecar":
                emb = model(toks)
            else:  # lingbot_raw
                flat = toks.flatten(-2)
                emb = F.normalize(F.normalize(flat, dim=-1) @ token_basis.to(device).T, dim=-1)
        out.append(emb.half().cpu())
    return torch.cat(out)


@torch.no_grad()
def teacher_pixel_embeddings(
    teacher, reader: FeatureCacheReader, frames: Sequence[int], projection: SemanticProjection,
    out_hw: Tuple[int, int], device: torch.device,
) -> torch.Tensor:
    """Teacher features at full pixel resolution for a batch of frames, ``[B, H, W, d]``.

    Bypasses the token grid entirely, so the difference against ``teacher_direct``
    isolates what the 14-pixel token quantisation costs.
    """
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(
        [reader.image_path(f) for f in frames],
        mode="crop",
        image_size=reader.manifest["preprocess"]["image_size"],
        patch_size=reader.manifest["patch_size"],
    ).to(device)
    dense = teacher.encode_dense(images).permute(0, 3, 1, 2).float()  # [B, D, hd, wd]
    dense = F.interpolate(dense, size=out_hw, mode="bilinear", align_corners=False)
    dense = F.normalize(dense.permute(0, 2, 3, 1), dim=-1)
    return projection.to(device).project(dense, normalize=True)


def fit_token_basis(readers: List[FeatureCacheReader], dim: int, seed: int, max_frames: int = 48) -> torch.Tensor:
    """Orthonormal basis of raw LingBot tokens, for the text-unaligned control."""
    from semantic.feature_field import fit_projection

    rng = np.random.default_rng(seed)
    samples = []
    for reader in readers:
        idx = np.unique(np.linspace(0, reader.num_frames - 1, min(max_frames, reader.num_frames)).astype(int))
        for i in idx:
            t = reader.tokens(int(i)).float()  # [L, N, C]
            t = t.permute(1, 0, 2).flatten(-2)
            sel = rng.choice(t.shape[0], size=min(512, t.shape[0]), replace=False)
            samples.append(F.normalize(t[sel], dim=-1))
    comps, _ = fit_projection(torch.cat(samples), dim)
    return comps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run_dir", default=None, help="a trained sidecar run directory")
    ap.add_argument("--baseline", default=None, choices=BASELINES)
    ap.add_argument("--name", default=None, help="output name (defaults to run/baseline name)")
    ap.add_argument("--roles", nargs="*", default=["eval"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--tracks_tag", default="density100", help="tracks subdir for cross-view metrics")
    ap.add_argument("--export_ply", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if bool(args.run_dir) == bool(args.baseline):
        raise SystemExit("pass exactly one of --run_dir or --baseline")

    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)
    cache_root = cfg.paths.cache_root
    projection = SemanticProjection.load(os.path.join(cache_root, "pca.safetensors"))

    mode = "sidecar" if args.run_dir else args.baseline
    name = args.name or (os.path.basename(os.path.normpath(args.run_dir)) if args.run_dir else args.baseline)
    out_root = os.path.join(cfg.paths.maps_root, name)
    os.makedirs(out_root, exist_ok=True)

    model, ckpt = (None, None)
    if mode == "sidecar":
        model, ckpt = load_sidecar(args.run_dir, device)
        logger.info("loaded sidecar from %s (step %s)", args.run_dir, ckpt.get("step"))

    scenes = [s for s in expand_scenes(cfg.scenes) if s.role in args.roles]
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in scenes if s.name in wanted or s.name.split("_c")[0] in wanted]
    scenes = [s for s in scenes if os.path.exists(manifest_path(cache_root, s.name))]
    if not scenes:
        raise SystemExit("no cached scenes selected")

    live_teacher = None
    if mode == "teacher_fullres":
        from semantic_sidecar.teacher_features import build_teacher

        live_teacher = build_teacher(cfg.teacher)
        logger.info("teacher_fullres uses %s at its native dense resolution", live_teacher.name)

    token_basis = None
    if mode == "lingbot_raw":
        # Fitted on *training* scenes, exactly like the teacher PCA basis, so the
        # control is compressed under the same rules as the real method.
        fit_specs = [s for s in expand_scenes(cfg.scenes)
                     if s.role == "train" and os.path.exists(manifest_path(cache_root, s.name))][:3]
        if not fit_specs:
            raise SystemExit("lingbot_raw needs at least one cached training scene")
        readers = [FeatureCacheReader(cache_root, s.name) for s in fit_specs]
        token_basis = fit_token_basis(readers, cfg.teacher.pca_dim, cfg.seed)
        logger.info(
            "fitted %dd raw-token basis on %s for the control baseline",
            cfg.teacher.pca_dim, [s.name for s in fit_specs],
        )

    provenance = collect_provenance(cfg, tool="infer_semantic_sidecar", mode=mode, name=name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Records for scenes we skip are carried over from the previous summary, so a
    # resumed run never silently drops metrics for work that is already done.
    previous = {r["scene"]: r for r in ((_read_json(os.path.join(out_root, "inference_summary.json")) or {})
                                        .get("scenes") or []) if isinstance(r, dict) and "scene" in r}
    results: List[Dict[str, object]] = []
    t_all = time.time()
    for spec in scenes:
        map_path = os.path.join(out_root, f"{spec.name}.npz")
        if os.path.exists(map_path) and not args.overwrite:
            if spec.name in previous:
                logger.info("%s: map exists, reusing its recorded metrics", spec.name)
                results.append(previous[spec.name])
            else:
                logger.warning("%s: map exists but has no recorded metrics; pass --overwrite", spec.name)
            continue
        reader = FeatureCacheReader(cache_root, spec.name)
        h, w = reader.patch_hw
        H, W = reader.image_hw

        teacher_feats = None
        if mode in ("teacher_direct", "teacher_fullres"):
            from cache_teacher_features import load_teacher_features

            teacher_feats = load_teacher_features(os.path.join(cache_root, "teacher"), spec.name)

        t0 = time.time()
        emb = token_embeddings_for_scene(
            mode, reader, device, model, teacher_feats, projection, token_basis
        )
        infer_seconds = time.time() - t0

        # Voxel size comes from geometry only, so every model in the comparison is
        # voxelised identically and the maps stay directly comparable.
        median_depth, focal = anchor_statistics(
            reader, cfg.tracks.anchor_frames, cfg.tracks.conf_threshold, cfg.tracks.min_depth
        )
        voxel_size = voxel_size_for_scene(
            cfg.tracks, median_depth, focal, reader.manifest["patch_size"]
        )
        accum = SemanticMapAccumulator(voxel_size, emb.shape[-1], device)
        fullres_cache: Dict[int, torch.Tensor] = {}
        for f in tqdm(range(reader.num_frames), desc=f"lift[{spec.name}]", leave=False):
            g = reader.geometry(f)
            if mode == "teacher_fullres":
                if f not in fullres_cache:
                    fullres_cache.clear()
                    block = list(range(f, min(f + 8, reader.num_frames)))
                    dense = teacher_pixel_embeddings(
                        live_teacher, reader, block, projection, (H, W), device
                    )
                    fullres_cache = {i: dense[k] for k, i in enumerate(block)}
                pixels = fullres_cache[f]
            else:
                pixels = embeddings_to_pixels(emb[f].to(device).float(), h, w, (H, W))
            pts, feats, rgb = lift_frame(
                pixels,
                g["depth"].to(device).float(),
                g["depth_conf"].to(device).float(),
                g["intrinsic"].to(device).float(),
                g["extrinsic"].to(device).float(),
                rgb=patch_rgb_to_pixels(g["image_patch_rgb"].to(device), h, w, (H, W)),
                pixel_stride=cfg.evaluation.pixel_stride,
                conf_threshold=cfg.evaluation.conf_threshold,
                min_depth=cfg.tracks.min_depth,
                max_depth=cfg.tracks.max_depth_rel * median_depth,
            )
            if pts.numel():
                accum.add(pts, feats, rgb)
        smap = accum.finalize(
            min_count=cfg.evaluation.min_count,
            meta={
                "scene": spec.name, "mode": mode, "name": name,
                "voxel_size": voxel_size, "median_depth": median_depth,
                "pixel_stride": cfg.evaluation.pixel_stride,
                "min_count": cfg.evaluation.min_count,
                "embedding_dim": int(emb.shape[-1]),
                "projection_explained_ratio": projection.explained_ratio,
                "provenance": provenance,
            },
        )
        smap.save(map_path)
        if args.export_ply:
            smap.export_ply(os.path.join(out_root, f"{spec.name}_rgb.ply"))

        # Cross-view semantic consistency needs tracks, which exist for any scene the
        # track builder was run on; it uses no labels.
        cvc, n_tracks = float("nan"), 0
        tracks_root = os.path.join(cfg.paths.tracks_root, args.tracks_tag)
        if tracks_are_cached(tracks_root, spec.name):
            tracks = TrackSet.load(tracks_root, spec.name)
            flat = emb.reshape(-1, emb.shape[-1])
            idx = tracks.obs_frame.long() * (h * w) + tracks.obs_token.long()
            from semantic_sidecar.consensus import segment_ids

            cvc, n_tracks = cross_view_consistency(flat[idx].float(), segment_ids(tracks.obs_ptr))

        np.save(os.path.join(out_root, f"{spec.name}_token_embeddings.npy"), emb.numpy())
        # Guard against a collapsed sidecar: a constant output would score a perfect
        # cross-view consistency while carrying no information at all.
        diversity = embedding_diversity(smap.embeddings.float())
        record = {
            "scene": spec.name, "role": spec.role,
            "voxels": smap.num_voxels, "voxel_size": voxel_size,
            "frames": reader.num_frames, "inference_seconds": infer_seconds,
            "cross_view_consistency": cvc, "cross_view_tracks": n_tracks,
            "mean_dispersion": float(smap.dispersion.mean()),
            "embedding_diversity": diversity,
        }
        results.append(record)
        logger.info(
            "%s: %d voxels @ %.4f, inference %.1fs, cross-view %.4f (%d tracks), diversity %.4f",
            spec.name, smap.num_voxels, voxel_size, infer_seconds, cvc, n_tracks, diversity,
        )

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    prior_peak = (_read_json(os.path.join(out_root, "inference_summary.json")) or {}).get("peak_vram_gb")
    if isinstance(prior_peak, (int, float)) and not np.isnan(float(prior_peak)):
        peak = max(peak, float(prior_peak))
    write_json(
        os.path.join(out_root, "inference_summary.json"),
        {
            "name": name, "mode": mode, "run_dir": args.run_dir,
            "needs_teacher_at_inference": mode in ("teacher_direct", "teacher_fullres"),
            "scenes": results, "peak_vram_gb": peak,
            "geometry_fingerprint": geometry_fingerprint(cache_root, scenes),
            "total_seconds": time.time() - t_all, "provenance": provenance,
            "parameter_report": (ckpt or {}).get("parameter_report"),
            "training_summary": _read_json(os.path.join(args.run_dir, "summary.json")) if args.run_dir else None,
        },
    )
    logger.info("wrote maps for %d scenes to %s (peak VRAM %.2f GB)", len(results), out_root, peak)


@torch.no_grad()
def geometry_fingerprint(cache_root: str, specs) -> str:
    """SHA-256 over the frozen depth/pose of every selected scene.

    Every model in the comparison must produce the identical value; that is the
    machine-checkable form of "the sidecar does not touch geometry".
    """
    import hashlib

    digest = hashlib.sha256()
    for spec in sorted(specs, key=lambda s: s.name):
        reader = FeatureCacheReader(cache_root, spec.name)
        digest.update(spec.name.encode())
        for f in range(reader.num_frames):
            g = reader.geometry(f)
            for key in ("depth", "extrinsic", "intrinsic"):
                digest.update(g[key].contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: str):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


if __name__ == "__main__":
    main()
