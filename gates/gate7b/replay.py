"""Native direct-mode streaming replay of the frozen LingBot-Map model.

The five-frame clip protocol of Gates 6 and 7A resets the model 163/1182/1753 times.
Here the model is reset **only at a genuine boundary** -- a new nuScenes scene, or the
start of a KITTI sequence -- and otherwise carries its own anchor context, its sliding
pose-reference window and its trajectory tokens across the whole stream.

Two properties of the frozen model shape what is possible, and both are read off the code
rather than assumed:

* ``aggregator.stream`` advances ``total_frames_processed`` -- the **global RoPE frame
  index** -- only for frames whose KV is appended. A non-keyframe therefore consumes no
  RoPE slot. This is the mechanism that lets a stream longer than ``max_frame_num`` run at
  all, and it is used through one global rule (:func:`keyframe_interval_for`), never tuned
  per benchmark;
* the KV cache is a sliding window of 64 blocks with the scale frames pinned, so memory is
  bounded and the native context is a *recent* window plus the initial anchors -- not the
  whole past. The persistent map, not the KV cache, is what carries long-horizon evidence.

Nothing is trained here. The model is loaded frozen, every parameter has
``requires_grad_(False)``, and the constructor asserts it.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # gates/<pkg>/ -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: Frozen LingBot inference settings, identical to the ones every previous gate used.
INFERENCE_RESOLUTION = 518
PATCH_SIZE = 14
NUM_SCALE_FRAMES = 5
KV_CACHE_SLIDING_WINDOW = 64
AUTOCAST_DTYPE = "bfloat16"
MAX_FRAME_NUM = 1024
#: Slots held back from the RoPE table so a stream never lands on its last index.
ROPE_MARGIN = 8
CHECKPOINT = "checkpoints/lingbot-map/204754b/lingbot-map.pt"


def keyframe_interval_for(n_frames: int, max_frame_num: int = MAX_FRAME_NUM,
                          scale_frames: int = NUM_SCALE_FRAMES,
                          margin: int = ROPE_MARGIN) -> int:
    """The one global rule that keeps a stream inside the frozen RoPE table.

    A non-keyframe consumes no global frame index, so a stream of ``n_frames`` needs
    ``scale_frames + ceil((n_frames - scale_frames) / k)`` slots. This returns the
    **smallest** ``k`` that fits, which is 1 for every stream short enough to need
    nothing. It is a capacity rule, not a tuning knob: it depends only on the stream
    length and the frozen model's table size, never on a benchmark score.
    """
    budget = max_frame_num - margin - scale_frames
    if budget <= 0:
        raise ValueError("no RoPE budget left for streamed frames")
    rest = max(int(n_frames) - int(scale_frames), 0)
    k = 1
    while -(-rest // k) > budget:            # ceil division
        k += 1
    return k


def rope_slots(n_frames: int, k: int, scale_frames: int = NUM_SCALE_FRAMES) -> int:
    rest = max(int(n_frames) - int(scale_frames), 0)
    return int(scale_frames + -(-rest // k))


def build_model(device, max_frame_num: int = MAX_FRAME_NUM):
    """The frozen model, in the same configuration every previous gate used."""
    from lingbot_map.models.gct_stream import GCTStream
    from gates.scale_gate.config import REPO_ROOT
    m = GCTStream(img_size=INFERENCE_RESOLUTION, patch_size=PATCH_SIZE,
                  enable_3d_rope=True, max_frame_num=max_frame_num,
                  kv_cache_sliding_window=KV_CACHE_SLIDING_WINDOW,
                  kv_cache_scale_frames=NUM_SCALE_FRAMES,
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(os.path.join(REPO_ROOT, CHECKPOINT), map_location="cpu",
                    weights_only=False)
    missing, unexpected = m.load_state_dict(ck.get("model", ck), strict=False)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(f"{len(trainable)} LingBot parameters are trainable")
    return m, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def load_images(paths: Sequence[str]) -> torch.Tensor:
    """The frozen preprocessing: ``mode='crop'`` at 518 px, patch 14. CPU tensor."""
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    return load_and_preprocess_images(list(paths), mode="crop",
                                      image_size=INFERENCE_RESOLUTION,
                                      patch_size=PATCH_SIZE)


def frame_index_trace(model) -> int:
    """The model's own global RoPE frame counter. Used by the causality tests."""
    return int(getattr(model.aggregator, "total_frames_processed", -1))


@torch.inference_mode()
def replay_segment(model, images_cpu: torch.Tensor, keyframe_interval: int,
                   device) -> Dict[str, np.ndarray]:
    """One unbroken causal pass over a whole segment.

    ``model.clean_kv_cache()`` is called exactly once, at the start -- this is the only
    reset, and it is what makes the boundary a boundary.
    """
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    if images_cpu.dim() == 4:
        images_cpu = images_cpu.unsqueeze(0)
    S = images_cpu.shape[1]
    model.clean_kv_cache()
    assert frame_index_trace(model) == 0, "state did not reset at the segment boundary"
    with torch.amp.autocast("cuda", dtype=getattr(torch, AUTOCAST_DTYPE)):
        preds = model.inference_streaming(
            images_cpu, num_scale_frames=min(NUM_SCALE_FRAMES, S),
            keyframe_interval=keyframe_interval,
            output_device=torch.device("cpu"))
    pose_enc = preds["pose_enc"][0].float().cpu()
    extr, intr = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0),
                                              tuple(images_cpu.shape[-2:]))
    c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
    c2w[:, :3, :4] = extr[0].double().numpy()
    return {
        "pred_pose_c2w": c2w.astype(np.float32),        # camera_to_world, CANONICAL scale
        "pred_K": intr[0].float().cpu().numpy().astype(np.float32),
        "pred_depth": preds["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
        "pred_depth_conf": preds["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        "pose_enc": pose_enc.numpy().astype(np.float32),
        "rope_slots_used": np.int64(frame_index_trace(model)),
        "keyframe_interval": np.int64(keyframe_interval),
        "proc_hw": np.asarray(images_cpu.shape[-2:], np.int64),
    }


__all__ = ["build_model", "load_images", "replay_segment", "keyframe_interval_for",
           "rope_slots", "frame_index_trace", "INFERENCE_RESOLUTION", "PATCH_SIZE",
           "NUM_SCALE_FRAMES", "KV_CACHE_SLIDING_WINDOW", "MAX_FRAME_NUM", "CHECKPOINT",
           "AUTOCAST_DTYPE", "ROPE_MARGIN"]
