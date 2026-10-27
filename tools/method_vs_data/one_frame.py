"""Phase A — a TRUE one-frame causal clip, with no five-frame information in it.

The frozen five-frame clip (`tools/occany_comparison/clips.py::infer_clip`) is not usable for
this ablation by slicing: its per-frame depth comes from a LingBotMap pass that attended across
all five images, and its metric scale is ``exp(nanmedian_q log_s[q])`` over five MoGe-B
candidates. Taking slot 4 out of that clip would be five-frame geometry wearing a one-frame
label. The brief forbids exactly that.

So this module re-runs the pipeline on the anchor image alone:

* **LingBotMap** is re-run with a single image, KV cache cleared, RoPE counter asserted at 0.
  ``inference_streaming`` already clamps ``num_scale_frames`` to the sequence length, so a
  one-image pass is a supported configuration, not a hack.
* **Scale** is ``exp(log_s)`` of that one frame's own MoGe-B candidate, through the identical
  ``gate7b.scale.frame_candidate`` call. With one candidate the frozen
  ``exp(nanmedian(log_s))`` reduces to this, so the rule is unchanged, not relaxed.
* The only cached artifact reused is the **per-frame MoGe-B depth**, which is computed one
  frame at a time and carries no temporal content. That is stated in the record.

Leakage is asserted, not hoped for: the written record must contain exactly one slot, one
log_s, and the anchor's own frame key.

Clips go to their own root; the frozen five-frame cache is never written.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import torch

from core.mapping import depth as D7
from gates.gate7b import replay, scale as SC
from tools.occany_comparison import config as C
from tools.occany_comparison.moge_extra import resolve_moge_path
from tools.occany_comparison.samples import Sample, build_samples

#: separate from ``C.CLIPS_ROOT``; SSD1 is nearly full, so these live on the 7.3 TB volume
ONE_FRAME_ROOT = os.environ.get(
    "ONE_FRAME_CLIPS_ROOT", "/media/minh/MINH1/lingbot_one_frame/clips")
PROTOCOL = "causal1"


def clip_path(dataset: str, sample_id: str) -> str:
    return os.path.join(ONE_FRAME_ROOT, dataset, PROTOCOL, f"{sample_id}.npz")


def infer_one_frame(model, sample: Sample, device) -> Dict[str, np.ndarray]:
    """The frozen procedure, run on the anchor image only."""
    causal = C.PROTOCOLS["causal"]
    j = sample.window(causal)[causal.anchor_slot]          # the anchor's stream index
    path, key = sample.frame_paths[j], sample.frame_keys[j]

    images = replay.load_images([path])
    assert images.shape[0] == 1 or images.shape[:2] == (1, 1), images.shape
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "RoPE frame counter must be 0 at clip start"
    out = replay.replay_segment(model, images, 1, device)
    assert out["pred_depth"].shape[0] == 1, out["pred_depth"].shape

    p = resolve_moge_path(sample.dataset, key)
    if not os.path.exists(p):
        raise FileNotFoundError(f"MoGe-B frame missing for {sample.sample_id}: {p}")
    with np.load(p) as z:
        md, mm = z["moge_depth"].astype(np.float32), D7.unpack_mask(z["moge_mask"], z["mask_shape"])
    log_s = np.asarray([SC.frame_candidate(md, mm, out["pred_depth"][0].astype(np.float32),
                                           out["pred_depth_conf"][0].astype(np.float32))["log_s"]],
                       np.float64)
    scale = float(np.exp(np.nanmedian(log_s))) if np.isfinite(log_s).any() else float("nan")

    return {"sample_id": sample.sample_id, "protocol": PROTOCOL,
            "anchor_slot": np.int64(0), "positions": np.asarray([j], np.int32),
            "keys": np.asarray([key]),
            "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"],
            "pred_pose_c2w": out["pred_pose_c2w"], "pred_K": out["pred_K"],
            "proc_hw": out["proc_hw"], "log_s_frames": log_s, "scale": np.float64(scale),
            "rope_slots_used": np.int64(out["rope_slots_used"]),
            "n_input_frames": np.int64(1),
            "reused_cache": np.asarray(["moge_b_per_frame_depth_only"])}


def build(dataset: str, device: str, shard: int = 0, shards: int = 1,
          limit: int | None = None) -> Dict[str, int]:
    dev = torch.device(device); torch.cuda.set_device(dev)
    causal = C.PROTOCOLS["causal"]
    samples = [s for s in build_samples(dataset) if s.protocol_ok(causal)][shard::shards][:limit]
    model, info = replay.build_model(dev)
    n_written = n_present = n_no_scale = 0
    t0 = time.time()
    for s in samples:
        op = clip_path(dataset, s.sample_id)
        if os.path.exists(op):
            n_present += 1
            continue
        os.makedirs(os.path.dirname(op), exist_ok=True)
        rec = infer_one_frame(model, s, dev)
        n_no_scale += int(not np.isfinite(float(rec["scale"])))
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, **rec); os.replace(tmp, op)
        n_written += 1
        if n_written % 200 == 0:
            print(f"  [{dataset} s{shard}] {n_written + n_present}/{len(samples)} "
                  f"{(time.time() - t0) / n_written:.2f}s each", flush=True)
    print(f"[{dataset} s{shard}] {n_written} written (+{n_present} present) of {len(samples)}, "
          f"{n_no_scale} without a scale, {(time.time() - t0) / 60:.1f} min; model {info}",
          flush=True)
    return {"dataset": dataset, "n_samples": len(samples), "n_written": n_written,
            "n_present": n_present, "n_no_scale": n_no_scale,
            "clips_root": ONE_FRAME_ROOT, "model": str(info)}


__all__ = ["ONE_FRAME_ROOT", "PROTOCOL", "clip_path", "infer_one_frame", "build"]
