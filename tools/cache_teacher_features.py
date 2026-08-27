#!/usr/bin/env python
"""Cache frozen teacher features on the LingBot token grid, and fit the PCA basis.

    python tools/cache_teacher_features.py --config configs/semantic_sidecar/feasibility.yaml

The PCA basis is fitted **only** on scenes with ``role: train`` and is written once to
``<cache_root>/pca.safetensors`` together with the exact scene list used, so the claim
"no target data, no target taxonomy" is auditable after the fact.  Teacher features
themselves are stored in the teacher's own 512-d space; projection happens at training
time, which keeps the cache reusable if the basis is refitted.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from lingbot_map.utils.load_fn import load_and_preprocess_images
from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import (
    FeatureCacheReader,
    manifest_path,
    save_shard,
    shard_path,
)
from semantic_sidecar.teacher_features import (
    SemanticProjection,
    build_teacher,
    fit_semantic_projection,
)

logger = logging.getLogger("cache_teacher")
SHARD_FRAMES = 64


def teacher_scene_dir(root: str, scene: str) -> str:
    return os.path.join(root, scene)


def teacher_is_cached(root: str, scene: str) -> bool:
    return os.path.exists(os.path.join(root, scene, "teacher_manifest.json"))


@torch.no_grad()
def encode_scene(teacher, reader: FeatureCacheReader, batch_size: int, device: str) -> torch.Tensor:
    """Teacher features for every frame of a scene → ``[S, n_tokens, D]`` fp16."""
    h, w = reader.patch_hw
    paths = [reader.image_path(i) for i in range(reader.num_frames)]
    out: List[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=f"teacher[{reader.scene}]", leave=False):
        batch = paths[start : start + batch_size]
        images = load_and_preprocess_images(
            batch,
            mode="crop",
            image_size=reader.manifest["preprocess"]["image_size"],
            patch_size=reader.manifest["patch_size"],
        ).to(device)
        feats = teacher.encode_patch_grid(images, h, w)  # [B, h*w, D]
        out.append(feats.half().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refit_pca", action="store_true", help="refit the basis even if it exists")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    cache_root = cfg.paths.cache_root
    teacher_root = os.path.join(cache_root, "teacher")
    scenes = expand_scenes(cfg.scenes)
    if args.roles:
        scenes = [s for s in scenes if s.role in args.roles]
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in scenes if s.name in wanted or s.name.split("_c")[0] in wanted]

    available = [s for s in scenes if os.path.exists(manifest_path(cache_root, s.name))]
    missing = [s.name for s in scenes if s not in available]
    if missing:
        logger.warning("%d scenes have no LingBot cache yet, skipping: %s", len(missing), missing[:5])
    if not available:
        raise SystemExit("no cached scenes; run tools/cache_lingbot_features.py first")

    teacher = build_teacher(cfg.teacher)
    logger.info("teacher: %s (D=%d)", teacher.name, teacher.embed_dim)

    provenance = collect_provenance(cfg, tool="cache_teacher_features", teacher=teacher.name)

    # ── encode ──────────────────────────────────────────────────────────── #
    for spec in available:
        if teacher_is_cached(teacher_root, spec.name) and not args.overwrite:
            logger.info("%s: teacher cache present, skipping", spec.name)
            continue
        reader = FeatureCacheReader(cache_root, spec.name)
        t0 = time.time()
        feats = encode_scene(teacher, reader, args.batch_size, cfg.teacher.device)
        out_dir = teacher_scene_dir(teacher_root, spec.name)
        os.makedirs(out_dir, exist_ok=True)
        shards = []
        for idx, start in enumerate(range(0, feats.shape[0], SHARD_FRAMES)):
            block = feats[start : start + SHARD_FRAMES]
            fname = f"teacher_{idx:05d}.safetensors"
            save_shard(os.path.join(out_dir, fname), {"features": block}, {"scene": spec.name})
            shards.append({"index": idx, "file": fname, "frames": [start, start + block.shape[0]]})
        write_json(
            os.path.join(out_dir, "teacher_manifest.json"),
            {
                "scene": spec.name,
                "teacher": teacher.name,
                "embed_dim": teacher.embed_dim,
                "input_scale": cfg.teacher.input_scale,
                "patch_grid": list(reader.patch_hw),
                "num_frames": int(feats.shape[0]),
                "shards": shards,
                "provenance": provenance,
            },
        )
        logger.info("%s: %d frames in %.1fs", spec.name, feats.shape[0], time.time() - t0)

    # ── PCA basis (training scenes only) ────────────────────────────────── #
    pca_path = os.path.join(cache_root, "pca.safetensors")
    if os.path.exists(pca_path) and not args.refit_pca:
        proj = SemanticProjection.load(pca_path)
        logger.info(
            "PCA basis exists: %dd, explains %.1f%% (fit on %d scenes)",
            proj.dim, 100 * proj.explained_ratio, len(proj.meta.get("fit_scenes", [])),
        )
        return

    train_scenes = [s for s in available if s.role == "train"]
    if not train_scenes:
        raise SystemExit("no training scenes available to fit the PCA basis on")

    rng = np.random.default_rng(cfg.seed)
    samples: List[torch.Tensor] = []
    per_scene = max(1, cfg.teacher.pca_frames // len(train_scenes))
    for spec in train_scenes:
        reader = FeatureCacheReader(cache_root, spec.name)
        feats = load_teacher_features(teacher_root, spec.name)
        idx = np.unique(np.linspace(0, feats.shape[0] - 1, min(per_scene, feats.shape[0])).astype(int))
        for i in idx:
            f = feats[int(i)].float()
            n = min(cfg.teacher.pca_samples_per_frame, f.shape[0])
            sel = rng.choice(f.shape[0], size=n, replace=False)
            samples.append(F.normalize(f[sel], dim=-1))
        del feats

    mat = torch.cat(samples)
    proj = fit_semantic_projection(
        mat,
        cfg.teacher.pca_dim,
        meta={
            "teacher": teacher.name,
            "fit_scenes": [s.name for s in train_scenes],
            "fit_datasets": sorted({s.dataset for s in train_scenes}),
            "num_samples": int(mat.shape[0]),
            "seed": cfg.seed,
            "uses_target_labels": False,
            "uses_target_class_names": False,
            "provenance": provenance,
        },
    )
    proj.save(pca_path)
    logger.info(
        "PCA basis: %d samples -> %dd, explains %.1f%% of feature energy -> %s",
        mat.shape[0], proj.dim, 100 * proj.explained_ratio, pca_path,
    )


def load_teacher_features(teacher_root: str, scene: str) -> torch.Tensor:
    """Concatenated teacher features for a scene, ``[S, n_tokens, D]`` fp16."""
    import json

    from safetensors.torch import load_file

    out_dir = os.path.join(teacher_root, scene)
    with open(os.path.join(out_dir, "teacher_manifest.json")) as fh:
        manifest = json.load(fh)
    return torch.cat([load_file(os.path.join(out_dir, s["file"]))["features"] for s in manifest["shards"]])


if __name__ == "__main__":
    main()
