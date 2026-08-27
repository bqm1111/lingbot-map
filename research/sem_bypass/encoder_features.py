"""Frozen LingBot restricted to its **pre-GCT** encoder tokens.

The previous study cached aggregator blocks ``[11, 23]`` (mid-GCT + final-GCT,
2 x 1024 = 2048 channels per layer).  SemBypass is defined on the *frame-independent*
DINO encoder output instead -- ``aggregator.patch_embed`` -> ``x_norm_patchtokens``,
1024 channels, computed before any cross-frame attention.

Rather than fork ``semantic_sidecar``, this subclasses :class:`FrozenLingBot` and swaps
only the capture hook, so the model construction, freezing, geometry post-processing,
shard writer and manifest schema are the *same code* the earlier results came from.
The hook is a forward hook returning ``None``: it cannot alter the module's output, and
``tests/test_geometry_invariance.py`` asserts depth / pose / confidence are bit-exact
against the existing ``[11, 23]`` cache.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import torch

from semantic_sidecar.config import PATCH_START_IDX, LingBotConfig, SceneSpec
from semantic_sidecar.lingbot_features import (
    FeatureCacheWriter, FrameFeatures, FrozenLingBot, list_scene_images, patch_grid,
    pool_to_patch_grid,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images

logger = logging.getLogger("sem_bypass.encoder")

#: Width of ``x_norm_patchtokens`` for LingBot's ``dinov2_vitl14_reg`` patch embedder.
ENCODER_CHANNELS = 1024
#: Manifest tag distinguishing this cache from the aggregator-block caches.
ENCODER_LAYER_TAG = "patch_embed"


class FrozenLingBotEncoder(FrozenLingBot):
    """:class:`FrozenLingBot` that captures pre-GCT encoder tokens instead of blocks.

    The captured tensor is ``[S, 1, P, 1024]`` -- one "layer" so the unmodified
    :class:`~semantic_sidecar.models.semantic_sidecar.SemanticSidecar` consumes it with
    ``num_layers=1, in_channels=1024``.
    """

    def __init__(self, cfg: LingBotConfig, capture_tokens: bool = True) -> None:
        # Bypass the parent's aggregator-block validation: `layers` is not a block list
        # here.  Everything else (build, freeze, hashing) is the parent's.
        cfg = LingBotConfig(**{**cfg.__dict__, "layers": [23]})   # placeholder, unused
        super().__init__(cfg, capture_tokens=capture_tokens)
        self.layers = [ENCODER_LAYER_TAG]
        self._layer_slots = [0]

    # -- hook: patch_embed instead of the aggregator ------------------------ #
    def _encoder_hook(self, _module, _inputs, output):
        t = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        if t.shape[-1] != ENCODER_CHANNELS:
            raise RuntimeError(f"expected {ENCODER_CHANNELS} encoder channels, got {t.shape[-1]}")
        # [B*S, P, C] with B == 1 -> [S, 1, P, C]
        self._buffer.append(t.detach().unsqueeze(1).to(torch.float16).cpu())
        return None  # never modify the module output

    @contextlib.contextmanager
    def _capturing(self) -> Iterator[None]:
        if not self.capture_tokens:
            yield
            return
        self._buffer = []
        self._handle = self.model.aggregator.patch_embed.register_forward_hook(self._encoder_hook)
        try:
            yield
        finally:
            self._handle.remove()
            self._handle = None


def cache_scene_encoder(
    model: FrozenLingBotEncoder, spec: SceneSpec, root: str,
    provenance: Optional[Dict[str, Any]] = None,
    batch_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Cache one scene's pre-GCT tokens + geometry, in the existing shard format.

    Mirrors ``semantic_sidecar.lingbot_features.cache_scene`` exactly except for the
    token source and the ``token_channels`` / ``layers`` manifest fields, so
    ``FeatureCacheReader``, the track builder and the evaluator all work unchanged.
    """
    paths = batch_paths if batch_paths is not None else list_scene_images(spec)
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=model.cfg.image_size, patch_size=model.cfg.patch_size)
    S, _, H, W = images.shape
    h, w = patch_grid(H, W, model.cfg.patch_size)

    preds, tokens = model.run_sequence(images)
    extrinsic, intrinsic = FrozenLingBot.to_extrinsics(preds["pose_enc"], (H, W))
    if tokens is None or tokens.shape[1:] != (1, h * w, ENCODER_CHANNELS):
        raise RuntimeError(f"unexpected encoder token shape {None if tokens is None else tuple(tokens.shape)}")

    writer = FeatureCacheWriter(root, spec.name)
    for i in range(S):
        rgb = pool_to_patch_grid(images[i], h, w).transpose(0, 1).contiguous()
        writer.add(FrameFeatures(
            frame_index=i, image_path=paths[i], tokens=tokens[i],
            depth=preds["depth"][i, ..., 0].to(torch.float16),
            depth_conf=preds["depth_conf"][i].to(torch.float16),
            intrinsic=intrinsic[i].float(), extrinsic=extrinsic[i].float(),
            pose_enc=preds["pose_enc"][i].float(),
            image_patch_rgb=(rgb * 255).round().clamp(0, 255).to(torch.uint8),
        ))

    from PIL import Image as _Image
    native = _Image.open(paths[0]).size
    manifest = {
        "scene": spec.name, "dataset": spec.dataset, "role": spec.role,
        "image_folder": os.path.abspath(spec.image_folder),
        "native_wh": list(native), "processed_hw": [H, W], "patch_grid": [h, w],
        "patch_size": model.cfg.patch_size, "num_tokens": h * w,
        "token_channels": ENCODER_CHANNELS,
        "layers": [ENCODER_LAYER_TAG], "layer_slots": [0],
        "representation": "pre_gct_encoder:aggregator.patch_embed.x_norm_patchtokens",
        "patch_start_idx": PATCH_START_IDX,
        "preprocess": {"mode": "crop", "image_size": model.cfg.image_size,
                       "patch_size": model.cfg.patch_size},
        "checkpoint": model.cfg.model_path, "checkpoint_sha256": model.checkpoint_sha256,
        "inference": {"mode": model.cfg.mode, "num_scale_frames": model.cfg.num_scale_frames,
                      "keyframe_interval": model.cfg.keyframe_interval,
                      "autocast_dtype": model.cfg.autocast_dtype},
        "coordinate_system": "first_camera_w2c_opencv",
        "spec": spec.__dict__,
    }
    if provenance:
        manifest["provenance"] = provenance
    return writer.write_manifest(manifest)
