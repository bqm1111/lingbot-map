"""Five-frame local-reset LingBot clips for both protocols, with the frozen clip-scale rule.

Identical procedure to the frozen ``tools/baseline_reconstruction/targets/clips.py::infer_clip``
(KV cache cleared, RoPE counter asserted at 0, keyframe interval 1, scale
``exp(nanmedian_q log_s[q])`` from the clip's own five MoGe-B candidates). The only generalisation is
the window: ``protocol.offsets`` instead of the fixed ``t-4..t``. Every clip must have all five MoGe
frames -- a missing one is an error here, never a silent four-frame scale.
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


def infer_clip(model, sample: Sample, protocol: C.Protocol, device) -> Dict[str, np.ndarray]:
    idx = sample.window(protocol)
    images = replay.load_images([sample.frame_paths[j] for j in idx])
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "RoPE frame counter must be 0 at the start of every clip"
    out = replay.replay_segment(model, images, 1, device)
    depth, conf = out["pred_depth"].astype(np.float32), out["pred_depth_conf"].astype(np.float32)
    log_s = []
    for q, j in enumerate(idx):
        p = resolve_moge_path(sample.dataset, sample.frame_keys[j])
        if not os.path.exists(p):
            raise FileNotFoundError(f"MoGe-B frame missing for {sample.sample_id} slot {q}: {p} (run cli/build_moge.py)")
        with np.load(p) as z:
            md, mm = z["moge_depth"].astype(np.float32), D7.unpack_mask(z["moge_mask"], z["mask_shape"])
        log_s.append(SC.frame_candidate(md, mm, depth[q], conf[q])["log_s"])
    log_s = np.asarray(log_s, np.float64)
    scale = float(np.exp(np.nanmedian(log_s))) if np.isfinite(log_s).any() else float("nan")
    return {"sample_id": sample.sample_id, "protocol": protocol.name, "anchor_slot": np.int64(protocol.anchor_slot),
            "positions": np.asarray(idx, np.int32), "keys": np.asarray([sample.frame_keys[j] for j in idx]),
            "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"], "pred_pose_c2w": out["pred_pose_c2w"],
            "pred_K": out["pred_K"], "proc_hw": out["proc_hw"], "log_s_frames": log_s, "scale": np.float64(scale),
            "rope_slots_used": np.int64(out["rope_slots_used"])}


def build_clips(dataset: str, device: str, shard: int = 0, shards: int = 1, limit: int | None = None) -> Dict[str, int]:
    dev = torch.device(device); torch.cuda.set_device(dev)
    samples = build_samples(dataset)[shard::shards][:limit]
    jobs = [(s, p) for s in samples for p in C.PROTOCOLS.values() if s.protocol_ok(p)]
    model, info = replay.build_model(dev)
    n_written = n_present = n_no_scale = 0; t0 = time.time()
    for s, p in jobs:
        op = C.clip_path(dataset, p.name, s.sample_id)
        if os.path.exists(op):
            n_present += 1; continue
        os.makedirs(os.path.dirname(op), exist_ok=True)
        rec = infer_clip(model, s, p, dev)
        n_no_scale += int(not np.isfinite(float(rec["scale"])))
        tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); n_written += 1
        if n_written % 200 == 0:
            print(f"  [{dataset} s{shard}] {n_written + n_present}/{len(jobs)} clips, {(time.time()-t0)/n_written:.2f}s each", flush=True)
    print(f"[{dataset} s{shard}] {n_written} written (+{n_present} present) of {len(jobs)}, {n_no_scale} without a scale, "
          f"{(time.time()-t0)/60:.1f} min; model {info}", flush=True)
    return {"n_jobs": len(jobs), "n_written": n_written, "n_present": n_present, "n_no_scale": n_no_scale}


__all__ = ["infer_clip", "build_clips"]
