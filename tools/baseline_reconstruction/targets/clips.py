"""Local-reset five-frame LingBot clips on a target stream, with the causal MoGe-2 clip scale.

Identical procedure to ``tools/local_registration_audit/run_local_lingbot.py`` (which made the
KITTI-360 clips the baseline was trained on): the frozen LingBot-Map checkpoint on the five
images ``t-4..t`` alone -- ``model.clean_kv_cache()`` before every clip, RoPE counter asserted
at 0, keyframe interval 1 -- and the metric scale ``exp(nanmedian_q log_s[q])`` from the frozen
per-frame MoGe-2 (B variant) candidates on the clip's own depth. The stream cache, the stream
scale and any frame before ``t-4`` are never read. Output npz fields match ``LocalClip.load``.

Runs only after the source freeze (``require_source_freeze``), because it opens target images.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import torch

from core.mapping import depth as D7
from gates.gate7b import replay, scale as SC
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.targets.adapters import AnchorRef, TargetAdapter


def _moge(adapter: TargetAdapter, key: str):
    p = adapter.moge_path(key)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return z["moge_depth"].astype(np.float32), D7.unpack_mask(z["moge_mask"], z["mask_shape"])


def infer_clip(model, adapter: TargetAdapter, ref: AnchorRef, device) -> Dict[str, np.ndarray]:
    images = replay.load_images(list(ref.image_paths))
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "RoPE frame counter must be 0 at the start of every clip"
    out = replay.replay_segment(model, images, 1, device)
    depth, conf = out["pred_depth"].astype(np.float32), out["pred_depth_conf"].astype(np.float32)
    log_s = []
    for q, key in enumerate(ref.keys):
        mo = _moge(adapter, key)
        log_s.append(float("nan") if mo is None else SC.frame_candidate(mo[0], mo[1], depth[q], conf[q])["log_s"])
    log_s = np.asarray(log_s, np.float64)
    scale = float(np.exp(np.nanmedian(log_s))) if np.isfinite(log_s).any() else float("nan")
    return {"t": np.int64(ref.anchor_index), "stream_indices": np.asarray([ref.anchor_index + o for o in C.CLIP_OFFSETS], np.int32),
            "keys": np.asarray(ref.keys), "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"],
            "pred_pose_c2w": out["pred_pose_c2w"], "pred_K": out["pred_K"], "proc_hw": out["proc_hw"],
            "log_s_frames": log_s, "scale": np.float64(scale), "rope_slots_used": np.int64(out["rope_slots_used"]),
            "clip_id": ref.clip_id, "segment": ref.segment}


def build_clips(dataset: str, device: str, paths: C.CurrentPaths = C.CurrentPaths(), shard: int = 0, shards: int = 1,
                limit: int | None = None) -> Dict[str, int]:
    dev = torch.device(device); torch.cuda.set_device(dev)
    adapter = TargetAdapter(dataset)
    refs = list(adapter.anchors())[shard::shards][:limit]
    model, info = replay.build_model(dev)
    n_written = n_present = n_no_scale = 0; t0 = time.time()
    for ref in refs:
        op = paths.target_clip_path(dataset, ref.segment, ref.anchor_index)
        if os.path.exists(op):
            n_present += 1; continue
        os.makedirs(os.path.dirname(op), exist_ok=True)
        rec = infer_clip(model, adapter, ref, dev)
        n_no_scale += int(not np.isfinite(float(rec["scale"])))
        tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); n_written += 1
        if n_written % 50 == 0:
            print(f"  [{dataset} s{shard}] {n_written}/{len(refs)} clips, {(time.time()-t0)/n_written:.2f}s each", flush=True)
    print(f"[{dataset} s{shard}] {n_written} written (+{n_present} present), {n_no_scale} without a causal scale, "
          f"{(time.time()-t0)/60:.1f} min; model {info}", flush=True)
    return {"n_anchors": len(refs), "n_written": n_written, "n_present": n_present, "n_no_scale": n_no_scale}


__all__ = ["infer_clip", "build_clips"]
