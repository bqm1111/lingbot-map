"""Frozen LingBot-MAP wrapper: geometry passthrough + detached intermediate tokens.

The model is loaded exactly as ``demo.load_model`` does, every parameter is set to
``requires_grad=False``, and inference runs under ``torch.no_grad``.  Intermediate
tokens are read with a **forward hook on ``model.aggregator``**, which cannot alter
the module's return value — verified bit-exactly by
``tests/semantic_sidecar/test_frozen_geometry.py``.

Coordinate convention (measured, see ``docs/semantic_sidecar_repo_audit.md`` §3):
the ``extrinsic`` written by ``demo.postprocess`` is consumed as **world-to-camera**,
so ``world = Rᵀ (cam − t)``.  The world frame is the first frame's camera frame and
the scale is arbitrary (monocular).
"""

from __future__ import annotations

import collections
import contextlib
import glob
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.geometry import closed_form_inverse_se3
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

from semantic_sidecar.config import (
    AGGREGATOR_SELECTED_BLOCKS,
    PATCH_START_IDX,
    TOKEN_CHANNELS,
    LingBotConfig,
    SceneSpec,
    collect_provenance,
    file_sha256,
)

logger = logging.getLogger(__name__)

SHARD_FRAMES = 32
COORDINATE_SYSTEM = "first_camera_w2c_opencv"


# --------------------------------------------------------------------------- #
# Geometry helpers (world-to-camera convention)
# --------------------------------------------------------------------------- #
def unproject_depth(
    depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor
) -> torch.Tensor:
    """Unproject a depth map to world coordinates.

    Args:
        depth: ``[H, W]`` predicted depth (camera z, arbitrary scale).
        intrinsic: ``[3, 3]`` pinhole matrix in the *processed* pixel grid.
        extrinsic: ``[3, 4]`` **world-to-camera** rigid transform.

    Returns:
        ``[H, W, 3]`` world points.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H, W], got {tuple(depth.shape)}")
    H, W = depth.shape
    device = depth.device
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cam = torch.stack([(u - cu) * depth / fu, (v - cv) * depth / fv, depth], dim=-1)
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return (cam - t) @ R


def camera_center(extrinsic: torch.Tensor) -> torch.Tensor:
    """Camera centre in world coordinates for a ``[3, 4]`` world-to-camera matrix."""
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)


def project_to_camera(
    points: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points into a camera.

    Args:
        points: ``[N, 3]`` world coordinates.
        intrinsic: ``[3, 3]``.
        extrinsic: ``[3, 4]`` world-to-camera.

    Returns:
        ``(uv [N, 2], z [N])`` where ``z`` is the camera-space depth.
    """
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam = points @ R.transpose(-1, -2) + t
    z = cam[:, 2]
    uv = torch.stack(
        [
            cam[:, 0] / z.clamp_min(1e-8) * intrinsic[0, 0] + intrinsic[0, 2],
            cam[:, 1] / z.clamp_min(1e-8) * intrinsic[1, 1] + intrinsic[1, 2],
        ],
        dim=-1,
    )
    return uv, z


def patch_grid(height: int, width: int, patch_size: int = 14) -> Tuple[int, int]:
    """Token grid size for a processed image, ``(h, w)``."""
    return height // patch_size, width // patch_size


def patch_centre_pixels(
    height: int, width: int, patch_size: int = 14, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Centre pixel of every patch token, row-major, ``[h*w, 2]`` as ``(y, x)``."""
    h, w = patch_grid(height, width, patch_size)
    ys = torch.arange(h, device=device, dtype=torch.float32) * patch_size + (patch_size - 1) / 2
    xs = torch.arange(w, device=device, dtype=torch.float32) * patch_size + (patch_size - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)


def pool_to_patch_grid(dense: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Area-pool a dense ``[C, H, W]`` map onto an ``h x w`` token grid → ``[h*w, C]``."""
    pooled = F.adaptive_avg_pool2d(dense.unsqueeze(0), (h, w))[0]
    return pooled.reshape(pooled.shape[0], h * w).transpose(0, 1).contiguous()


# --------------------------------------------------------------------------- #
# Frozen model wrapper
# --------------------------------------------------------------------------- #
@dataclass
class FrameFeatures:
    """One frame's frozen outputs.

    ``tokens`` is ``[n_layers, n_tokens, 2048]`` (patch tokens only, special tokens
    dropped) and is always detached and on CPU.
    """

    frame_index: int
    image_path: str
    tokens: Optional[torch.Tensor]
    depth: torch.Tensor
    depth_conf: torch.Tensor
    intrinsic: torch.Tensor
    extrinsic: torch.Tensor
    pose_enc: torch.Tensor
    image_patch_rgb: torch.Tensor


class FrozenLingBot:
    """Loads LingBot-MAP frozen and optionally taps its intermediate tokens.

    Args:
        cfg: Model/runtime configuration.
        capture_tokens: When ``False`` the hook is never registered, which is the
            reference path used to prove the hook is side-effect free.
    """

    def __init__(self, cfg: LingBotConfig, capture_tokens: bool = True) -> None:
        self.cfg = cfg
        self.capture_tokens = capture_tokens
        self.device = torch.device(cfg.device)
        self.layers = list(cfg.layers)
        unknown = set(self.layers) - set(AGGREGATOR_SELECTED_BLOCKS)
        if unknown:
            raise ValueError(
                f"layers {sorted(unknown)} are not produced by the aggregator; "
                f"available: {list(AGGREGATOR_SELECTED_BLOCKS)}"
            )
        self._layer_slots = [AGGREGATOR_SELECTED_BLOCKS.index(i) for i in self.layers]

        self.model = self._build()
        self.checkpoint_sha256 = file_sha256(cfg.model_path) if os.path.exists(cfg.model_path) else "missing"
        self._buffer: List[torch.Tensor] = []
        self._handle = None
        self._last_tokens_per_frame: Optional[int] = None

    # -- construction ------------------------------------------------------ #
    def _build(self) -> torch.nn.Module:
        from lingbot_map.models.gct_stream import GCTStream

        cfg = self.cfg
        model = GCTStream(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            enable_3d_rope=cfg.enable_3d_rope,
            max_frame_num=cfg.max_frame_num,
            kv_cache_sliding_window=cfg.kv_cache_sliding_window,
            kv_cache_scale_frames=cfg.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=cfg.use_sdpa,
            camera_num_iterations=cfg.camera_num_iterations,
        )
        if cfg.model_path:
            ckpt = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning("checkpoint: %d missing, %d unexpected keys", len(missing), len(unexpected))
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # -- parameter accounting ---------------------------------------------- #
    def parameter_report(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"lingbot_total": total, "lingbot_frozen": total - trainable, "lingbot_trainable": trainable}

    def assert_frozen(self) -> None:
        """Raise if any LingBot parameter is trainable or carries a gradient."""
        bad = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"{len(bad)} LingBot parameters are trainable, e.g. {bad[:3]}")
        grads = [n for n, p in self.model.named_parameters() if p.grad is not None]
        if grads:
            raise RuntimeError(f"{len(grads)} LingBot parameters carry a gradient, e.g. {grads[:3]}")

    # -- hook -------------------------------------------------------------- #
    def _hook(self, _module, _inputs, output):
        tokens_list, patch_start_idx = output
        if patch_start_idx != PATCH_START_IDX:
            raise RuntimeError(
                f"unexpected patch_start_idx {patch_start_idx} (expected {PATCH_START_IDX})"
            )
        picked = [tokens_list[s][:, :, patch_start_idx:, :].detach() for s in self._layer_slots]
        # [B, S, P, C] x L -> [S, L, P, C] with B == 1
        stacked = torch.stack(picked, dim=2)[0]
        self._buffer.append(stacked.to(torch.float16).cpu())
        return None  # never modify the module's output

    @contextlib.contextmanager
    def _capturing(self) -> Iterator[None]:
        if not self.capture_tokens:
            yield
            return
        self._buffer = []
        self._handle = self.model.aggregator.register_forward_hook(self._hook)
        try:
            yield
        finally:
            self._handle.remove()
            self._handle = None

    # -- inference --------------------------------------------------------- #
    @torch.no_grad()
    def run_sequence(self, images: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Run frozen streaming inference over ``[S, 3, H, W]`` images in ``[0, 1]``.

        Returns:
            ``(predictions, tokens)`` where ``predictions`` holds the unchanged model
            outputs (on CPU, batch dim squeezed) and ``tokens`` is
            ``[S, n_layers, n_tokens, 2048]`` fp16 or ``None`` when capture is off.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [S, 3, H, W], got {tuple(images.shape)}")
        self._reset_kv_manager_if_resolution_changed(images.shape[-2], images.shape[-1])
        dtype = getattr(torch, self.cfg.autocast_dtype)
        autocast = (
            torch.amp.autocast("cuda", dtype=dtype)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with self._capturing(), autocast:
            preds = self.model.inference_streaming(
                images,
                num_scale_frames=self.cfg.num_scale_frames,
                keyframe_interval=self.cfg.keyframe_interval,
                output_device=torch.device("cpu"),
            )

        tokens = None
        if self.capture_tokens:
            tokens = torch.cat(self._buffer, dim=0)
            self._buffer = []
            if tokens.shape[0] != images.shape[0]:
                raise RuntimeError(
                    f"captured {tokens.shape[0]} token frames for {images.shape[0]} images"
                )

        out = {k: (v[0] if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == 1 else v)
               for k, v in preds.items()}
        return out, tokens

    def _reset_kv_manager_if_resolution_changed(self, height: int, width: int) -> None:
        """Drop the paged KV cache when the token count changes.

        ``AggregatorStream._get_flashinfer_manager`` sizes the paged cache lazily on
        the first forward and never resizes it, so processing a 518x518 scene after a
        294x518 one otherwise trips
        ``assert patch_k.shape[0] == self.patches_per_frame``.  Setting the attribute
        back to ``None`` is exactly what the lazy initialiser expects, and touches no
        LingBot source.
        """
        h, w = patch_grid(height, width, self.cfg.patch_size)
        tokens_per_frame = h * w + PATCH_START_IDX
        if self._last_tokens_per_frame not in (None, tokens_per_frame):
            agg = self.model.aggregator
            if getattr(agg, "kv_cache_manager", None) is not None:
                logger.info(
                    "resolution changed (%d -> %d tokens/frame); rebuilding the paged KV cache",
                    self._last_tokens_per_frame, tokens_per_frame,
                )
                agg.kv_cache_manager = None
            if self.model.camera_head is not None and hasattr(self.model.camera_head, "clean_kv_cache"):
                self.model.camera_head.clean_kv_cache()
        self._last_tokens_per_frame = tokens_per_frame

    @staticmethod
    def to_extrinsics(pose_enc: torch.Tensor, image_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[S, 9]`` pose encoding to ``(extrinsic [S,3,4] w2c, intrinsic [S,3,3])``.

        Mirrors ``demo.postprocess`` exactly, including its inversion, so cached
        geometry is identical to what ``batch_demo.py --save_predictions`` writes.
        """
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), image_hw)
        e4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), dtype=extrinsic.dtype)
        e4[..., :3, :4] = extrinsic
        e4[..., 3, 3] = 1.0
        e4 = closed_form_inverse_se3(e4.reshape(-1, 4, 4)).reshape(e4.shape)
        return e4[0, ..., :3, :4], intrinsic[0]


# --------------------------------------------------------------------------- #
# Scene discovery + shard IO
# --------------------------------------------------------------------------- #
def list_scene_images(spec: SceneSpec) -> List[str]:
    """Sorted, filtered frame paths for a scene spec.

    Zero-byte files are dropped: the TartanAir mirror on this machine contains 155
    truncated PNGs out of 55 000, and a single unreadable frame would otherwise abort
    a whole scene.  Dropping one frame of a video only widens the local baseline
    slightly, so the sequence stays usable.
    """
    pattern = os.path.join(spec.image_folder, f"*{spec.image_ext}")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {spec.image_ext} files under {spec.image_folder}")
    usable = [p for p in paths if os.path.getsize(p) > 0]
    if len(usable) != len(paths):
        logger.warning(
            "%s: dropping %d unreadable (zero-byte) frames", spec.image_folder, len(paths) - len(usable)
        )
    paths = usable[spec.start :: spec.stride]
    if spec.max_frames is not None:
        paths = paths[: spec.max_frames]
    return paths


def shard_path(root: str, scene: str, index: int) -> str:
    return os.path.join(root, scene, f"shard_{index:05d}.safetensors")


def manifest_path(root: str, scene: str) -> str:
    return os.path.join(root, scene, "manifest.json")


def save_shard(path: str, tensors: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    """Write a safetensors shard, creating parents."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=meta)


def load_shard(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path, device=device)


class FeatureCacheWriter:
    """Accumulates frames and flushes fixed-size safetensors shards."""

    def __init__(self, root: str, scene: str, shard_frames: int = SHARD_FRAMES) -> None:
        self.root, self.scene, self.shard_frames = root, scene, shard_frames
        self.frames: List[FrameFeatures] = []
        self.shards: List[Dict[str, Any]] = []
        os.makedirs(os.path.join(root, scene), exist_ok=True)

    def add(self, frame: FrameFeatures) -> None:
        self.frames.append(frame)
        if len(self.frames) >= self.shard_frames:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        idx = len(self.shards)
        path = shard_path(self.root, self.scene, idx)
        tensors = {
            "depth": torch.stack([f.depth for f in self.frames]),
            "depth_conf": torch.stack([f.depth_conf for f in self.frames]),
            "intrinsic": torch.stack([f.intrinsic for f in self.frames]),
            "extrinsic": torch.stack([f.extrinsic for f in self.frames]),
            "pose_enc": torch.stack([f.pose_enc for f in self.frames]),
            "image_patch_rgb": torch.stack([f.image_patch_rgb for f in self.frames]),
            "frame_index": torch.tensor([f.frame_index for f in self.frames], dtype=torch.int64),
        }
        if self.frames[0].tokens is not None:
            tensors["tokens"] = torch.stack([f.tokens for f in self.frames])
        save_shard(path, tensors, {"scene": self.scene, "shard": str(idx)})
        self.shards.append(
            {
                "index": idx,
                "file": os.path.basename(path),
                "frames": [f.frame_index for f in self.frames],
                "image_paths": [f.image_path for f in self.frames],
            }
        )
        self.frames = []

    def write_manifest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Flush any pending frames and write the manifest; returns the written dict."""
        self.flush()
        payload = dict(payload)
        payload["shards"] = self.shards
        payload["num_frames"] = sum(len(s["frames"]) for s in self.shards)
        with open(manifest_path(self.root, self.scene), "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return payload


#: Process-wide LRU over loaded shards.  A token shard is ~350 MB, and training keeps
#: two dozen scene readers alive at once, so a per-reader cache would need tens of GB
#: of RAM.  Bounding it globally keeps residency at ``_SHARD_CACHE_LIMIT`` shards no
#: matter how many scenes are open.
_SHARD_CACHE: "OrderedDict[Tuple[str, str, int], Dict[str, torch.Tensor]]" = OrderedDict()
_SHARD_CACHE_LIMIT = 4


def set_shard_cache_limit(limit: int) -> None:
    """Set how many feature shards may stay resident in RAM (>= 1)."""
    global _SHARD_CACHE_LIMIT
    _SHARD_CACHE_LIMIT = max(1, int(limit))
    while len(_SHARD_CACHE) > _SHARD_CACHE_LIMIT:
        _SHARD_CACHE.popitem(last=False)


def clear_shard_cache() -> None:
    _SHARD_CACHE.clear()


class FeatureCacheReader:
    """Random access over a cached scene, with lazy, globally-bounded shard loading."""

    def __init__(self, root: str, scene: str, device: str = "cpu") -> None:
        self.root, self.scene, self.device = root, scene, device
        with open(manifest_path(root, scene)) as fh:
            self.manifest: Dict[str, Any] = json.load(fh)
        self._index: List[Tuple[int, int]] = []
        for shard in self.manifest["shards"]:
            for local, _ in enumerate(shard["frames"]):
                self._index.append((shard["index"], local))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_frames(self) -> int:
        return len(self._index)

    @property
    def patch_hw(self) -> Tuple[int, int]:
        return tuple(self.manifest["patch_grid"])  # type: ignore[return-value]

    @property
    def image_hw(self) -> Tuple[int, int]:
        return tuple(self.manifest["processed_hw"])  # type: ignore[return-value]

    def _shard(self, index: int) -> Dict[str, torch.Tensor]:
        key = (self.root, self.scene, index)
        shard = _SHARD_CACHE.get(key)
        if shard is None:
            shard = load_shard(shard_path(self.root, self.scene, index), device=self.device)
            _SHARD_CACHE[key] = shard
            while len(_SHARD_CACHE) > _SHARD_CACHE_LIMIT:
                _SHARD_CACHE.popitem(last=False)
        else:
            _SHARD_CACHE.move_to_end(key)
        return shard

    def get(self, frame: int, keys: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        shard_idx, local = self._index[frame]
        shard = self._shard(shard_idx)
        keys = keys or list(shard.keys())
        return {k: shard[k][local] for k in keys if k in shard}

    def geometry(self, frame: int) -> Dict[str, torch.Tensor]:
        return self.get(frame, ["depth", "depth_conf", "intrinsic", "extrinsic", "image_patch_rgb"])

    def tokens(self, frame: int) -> torch.Tensor:
        got = self.get(frame, ["tokens"])
        if "tokens" not in got:
            raise KeyError(f"scene {self.scene} was cached without tokens")
        return got["tokens"]

    def image_path(self, frame: int) -> str:
        shard_idx, local = self._index[frame]
        return self.manifest["shards"][shard_idx]["image_paths"][local]


def scene_is_cached(root: str, scene: str) -> bool:
    """True when a complete manifest and all its shards exist."""
    mpath = manifest_path(root, scene)
    if not os.path.exists(mpath):
        return False
    try:
        with open(mpath) as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError:
        return False
    return all(
        os.path.exists(os.path.join(root, scene, s["file"])) for s in manifest.get("shards", [])
    )


def cache_scene(
    model: FrozenLingBot,
    spec: SceneSpec,
    root: str,
    provenance: Optional[Dict[str, Any]] = None,
    batch_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the frozen model over one scene and write its feature shards.

    Returns the manifest that was written.
    """
    paths = batch_paths if batch_paths is not None else list_scene_images(spec)
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=model.cfg.image_size, patch_size=model.cfg.patch_size
    )
    S, _, H, W = images.shape
    h, w = patch_grid(H, W, model.cfg.patch_size)

    preds, tokens = model.run_sequence(images)
    extrinsic, intrinsic = FrozenLingBot.to_extrinsics(preds["pose_enc"], (H, W))

    writer = FeatureCacheWriter(root, spec.name)
    for i in range(S):
        rgb = pool_to_patch_grid(images[i], h, w).transpose(0, 1).contiguous()  # [3, h*w]
        writer.add(
            FrameFeatures(
                frame_index=i,
                image_path=paths[i],
                tokens=None if tokens is None else tokens[i],
                depth=preds["depth"][i, ..., 0].to(torch.float16),
                depth_conf=preds["depth_conf"][i].to(torch.float16),
                intrinsic=intrinsic[i].float(),
                extrinsic=extrinsic[i].float(),
                pose_enc=preds["pose_enc"][i].float(),
                image_patch_rgb=(rgb * 255).round().clamp(0, 255).to(torch.uint8),
            )
        )

    from PIL import Image as _Image

    native = _Image.open(paths[0]).size  # (width, height)
    manifest = {
        "scene": spec.name,
        "dataset": spec.dataset,
        "role": spec.role,
        "image_folder": os.path.abspath(spec.image_folder),
        "native_wh": list(native),
        "processed_hw": [H, W],
        "patch_grid": [h, w],
        "patch_size": model.cfg.patch_size,
        "num_tokens": h * w,
        "token_channels": TOKEN_CHANNELS,
        "layers": model.layers,
        "layer_slots": model._layer_slots,
        "patch_start_idx": PATCH_START_IDX,
        "preprocess": {
            "mode": "crop",
            "image_size": model.cfg.image_size,
            "resize_rule": "width->image_size, height->round(h*(image_size/w)/patch)*patch, "
            "centre-crop height to image_size when larger",
        },
        "coordinate_system": COORDINATE_SYSTEM,
        "inference": {
            "mode": model.cfg.mode,
            "num_scale_frames": model.cfg.num_scale_frames,
            "keyframe_interval": model.cfg.keyframe_interval,
            "backend": "sdpa" if model.cfg.use_sdpa else "flashinfer",
            "autocast_dtype": model.cfg.autocast_dtype,
        },
        "checkpoint": os.path.abspath(model.cfg.model_path),
        "checkpoint_sha256": model.checkpoint_sha256,
        "spec": {
            "stride": spec.stride,
            "start": spec.start,
            "max_frames": spec.max_frames,
            "chunk_size": spec.chunk_size,
        },
        "provenance": provenance or collect_provenance(),
    }
    return writer.write_manifest(manifest)
