"""On-disk cache of frozen LingBot representations, teacher features and geometry.

One ``.npz`` per chunk holds everything downstream stages need, so probe training and
evaluation never re-run LingBot or DINO.  Nothing cached carries a gradient: hooks
detach, both models run under ``torch.no_grad``, and arrays are stored as numpy.

Cache key includes the chunk identity, the representation list, the LingBot checkpoint
hash and the teacher identity, so changing any of them produces a different file rather
than silently reusing stale features.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from research.lingbot_semantic_memory.config import (
    ChunkSpec, Phase1Config, REPO_ROOT, write_json,
)
from research.lingbot_semantic_memory.dataset_adapter import (
    assert_pure_resize, get_backend, load_chunk_images, scale_intrinsics,
)
from research.lingbot_semantic_memory.hooks import FrozenDinoTeacher, FrozenLingBot
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


@dataclass
class CachedChunk:
    """Everything one chunk contributes to the experiment."""

    name: str
    role: str
    sequence: str
    start: int
    grid_hw: Tuple[int, int]
    image_hw: Tuple[int, int]
    reps: Dict[str, np.ndarray]      # name -> [S, P, C] float16
    teacher: np.ndarray              # [S, P, 768] float16
    pred_depth: np.ndarray           # [S, H, W] float32 (arbitrary scale)
    pred_conf: np.ndarray            # [S, H, W] float32 (expp1 confidence, >= 1)
    pred_extrinsic: np.ndarray       # [S, 3, 4] float32, world-to-camera
    pred_intrinsic: np.ndarray       # [3, 3] float32
    oracle_depth: Optional[np.ndarray]
    oracle_extrinsic: Optional[np.ndarray]
    oracle_intrinsic: Optional[np.ndarray]
    labels: Optional[np.ndarray]     # [S, gh, gw] uint8, evaluation only

    def rep(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.reps[name])


def cache_path(cfg: Phase1Config, chunk: ChunkSpec) -> str:
    return os.path.join(REPO_ROOT, cfg.cache_dir, f"{chunk.name}.npz")


def save_chunk(path: str, c: CachedChunk) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "teacher": c.teacher, "pred_depth": c.pred_depth, "pred_conf": c.pred_conf,
        "pred_extrinsic": c.pred_extrinsic, "pred_intrinsic": c.pred_intrinsic,
        "grid_hw": np.asarray(c.grid_hw), "image_hw": np.asarray(c.image_hw),
        "meta": np.asarray(json.dumps({"name": c.name, "role": c.role,
                                       "sequence": c.sequence, "start": c.start})),
    }
    for k, v in c.reps.items():
        payload[f"rep::{k}"] = v
    for k in ("oracle_depth", "oracle_extrinsic", "oracle_intrinsic", "labels"):
        v = getattr(c, k)
        if v is not None:
            payload[k] = v
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_chunk(path: str) -> CachedChunk:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    reps = {k[5:]: z[k] for k in z.files if k.startswith("rep::")}
    opt = lambda k: z[k] if k in z.files else None
    return CachedChunk(
        name=meta["name"], role=meta["role"], sequence=meta["sequence"], start=meta["start"],
        grid_hw=tuple(int(x) for x in z["grid_hw"]), image_hw=tuple(int(x) for x in z["image_hw"]),
        reps=reps, teacher=z["teacher"], pred_depth=z["pred_depth"], pred_conf=z["pred_conf"],
        pred_extrinsic=z["pred_extrinsic"], pred_intrinsic=z["pred_intrinsic"],
        oracle_depth=opt("oracle_depth"), oracle_extrinsic=opt("oracle_extrinsic"),
        oracle_intrinsic=opt("oracle_intrinsic"), labels=opt("labels"),
    )


@torch.no_grad()
def build_chunk(
    cfg: Phase1Config, chunk: ChunkSpec, lingbot: FrozenLingBot,
    teacher: FrozenDinoTeacher, with_oracle: bool, with_labels: bool,
    timings: Optional[Dict[str, float]] = None,
) -> CachedChunk:
    """Run both frozen models over one chunk and assemble everything for the cache."""
    images = load_chunk_images(cfg.data, chunk)
    S, _, H, W = images.shape
    gh, gw = H // cfg.data.patch_size, W // cfg.data.patch_size

    t0 = time.time()
    preds, reps = lingbot.run_chunk(images, capture=True)
    torch.cuda.synchronize()
    t_ling = time.time() - t0

    t0 = time.time()
    teach = teacher.encode(images)
    torch.cuda.synchronize()
    t_teach = time.time() - t0
    if timings is not None:
        timings["lingbot_s"] = timings.get("lingbot_s", 0.0) + t_ling
        timings["teacher_s"] = timings.get("teacher_s", 0.0) + t_teach
        timings["frames"] = timings.get("frames", 0.0) + S

    if teach.shape[1] != gh * gw:
        raise RuntimeError(f"teacher grid {teach.shape[1]} != LingBot grid {gh*gw}")

    extr, intr = pose_encoding_to_extri_intri(preds["pose_enc"].unsqueeze(0), (H, W))
    pred_extr = extr[0].float().numpy()               # world-to-camera
    pred_intr = intr[0, 0].float().numpy()

    oracle_depth = oracle_extr = oracle_intr = None
    labels = None
    if with_oracle or with_labels:
        backend = get_backend(cfg.data)
        orig_hw = backend.original_hw(chunk.sequence)
        assert_pure_resize(cfg.data, orig_hw, (H, W))
        if with_oracle:
            oracle_depth = backend.oracle_depth(chunk, (H, W))
            oracle_extr = backend.oracle_poses_w2c(chunk).astype(np.float32)
            oracle_intr = scale_intrinsics(
                backend.intrinsics(chunk.sequence), orig_hw, (H, W)).astype(np.float32)
        if with_labels and backend.has_labels:
            labels = backend.patch_labels(chunk, (H, W), (gh, gw))

    return CachedChunk(
        name=chunk.name, role=chunk.role, sequence=chunk.sequence, start=chunk.start,
        grid_hw=(gh, gw), image_hw=(H, W),
        reps={k: v.numpy() for k, v in reps.items()},
        teacher=teach.numpy(),
        pred_depth=preds["depth"][..., 0].float().numpy(),
        pred_conf=preds["depth_conf"].float().numpy(),
        pred_extrinsic=pred_extr, pred_intrinsic=pred_intr,
        oracle_depth=oracle_depth, oracle_extrinsic=oracle_extr,
        oracle_intrinsic=oracle_intr, labels=labels,
    )


def ensure_cache(
    cfg: Phase1Config, chunks: List[ChunkSpec], lingbot: FrozenLingBot,
    teacher: FrozenDinoTeacher, with_oracle: bool, with_labels: bool,
    timings: Optional[Dict[str, float]] = None, force: bool = False,
) -> List[str]:
    """Build any missing chunk caches; return the list of paths in order."""
    paths = []
    for ch in chunks:
        p = cache_path(cfg, ch)
        if force or not os.path.exists(p):
            print(f"  [cache] building {ch.name} ({ch.role})", flush=True)
            save_chunk(p, build_chunk(cfg, ch, lingbot, teacher, with_oracle, with_labels, timings))
        else:
            print(f"  [cache] hit {ch.name}", flush=True)
        paths.append(p)
    return paths


def cache_size_bytes(cfg: Phase1Config) -> int:
    d = os.path.join(REPO_ROOT, cfg.cache_dir)
    if not os.path.isdir(d):
        return 0
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".npz"))
