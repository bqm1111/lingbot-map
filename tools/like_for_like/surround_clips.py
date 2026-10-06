"""Six-view local-reset LingBot clips: the six surround cameras of ONE timestep, in ring order.

Same procedure as ``tools/occany_comparison/clips.infer_clip`` -- KV cache cleared, RoPE counter asserted at 0, keyframe
interval 1, clip scale ``exp(nanmedian_q log_s[q])`` over the clip's own MoGe candidates. The only difference is what the
window holds: instead of five frames of one camera over time, six cameras at one instant, ``CAM_FRONT`` first so the map's
origin is the same camera the benchmark grid is defined against. No rig calibration is fed to the model; it registers the
views from image content alone, as ``surround_probe.py`` measured.

    PYTHONPATH=$PWD python -m tools.like_for_like.surround_clips --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.mapping import depth as D7
from gates.gate7b import replay, scale as SC
from tools.like_for_like import config as LC, surround as SR
from tools.occany_comparison.moge_extra import resolve_moge_path

ANCHOR_SLOT = 0                      # CAM_FRONT, the camera the benchmark placement is defined against


def clip_path(sample_id: str) -> str:
    return os.path.join(SR.CLIP_ROOT, f"{sample_id}.npz")


def infer_surround_clip(model, sample, device) -> Dict[str, np.ndarray]:
    views = SR.ring_views(SR.sample_token(sample))
    images = replay.load_images([v["path"] for v in views])
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "RoPE frame counter must be 0 at the start of every clip"
    out = replay.replay_segment(model, images, 1, device)
    depth, conf = out["pred_depth"].astype(np.float32), out["pred_depth_conf"].astype(np.float32)
    log_s = []
    for q, v in enumerate(views):
        p = resolve_moge_path("occ3d", v["key"])
        if not os.path.exists(p):
            raise FileNotFoundError(f"MoGe frame missing for {sample.sample_id} slot {q} ({v['camera']}): {p}")
        with np.load(p) as z:
            md, mm = z["moge_depth"].astype(np.float32), D7.unpack_mask(z["moge_mask"], z["mask_shape"])
        log_s.append(SC.frame_candidate(md, mm, depth[q], conf[q])["log_s"])
    log_s = np.asarray(log_s, np.float64)
    scale = float(np.exp(np.nanmedian(log_s))) if np.isfinite(log_s).any() else float("nan")
    return {"sample_id": sample.sample_id, "protocol": "surround", "anchor_slot": np.int64(ANCHOR_SLOT),
            "cameras": np.asarray(SR.RING), "keys": np.asarray([v["key"] for v in views]),
            "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"], "pred_pose_c2w": out["pred_pose_c2w"],
            "pred_K": out["pred_K"], "proc_hw": out["proc_hw"], "log_s_frames": log_s, "scale": np.float64(scale),
            "rope_slots_used": np.int64(out["rope_slots_used"])}


def build_clips(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    samples = SR.samples()[shard::shards][:limit]
    os.makedirs(SR.CLIP_ROOT, exist_ok=True)
    model, info = replay.build_model(dev)
    n_written = n_present = n_no_scale = 0; t0 = time.time()
    for s in samples:
        op = clip_path(s.sample_id)
        if os.path.exists(op):
            n_present += 1; continue
        rec = infer_surround_clip(model, s, dev)
        n_no_scale += int(not np.isfinite(float(rec["scale"])))
        tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); n_written += 1
        if n_written % 100 == 0:
            print(f"  [surround s{shard}] {n_written + n_present}/{len(samples)} clips, {(time.time()-t0)/n_written:.2f}s each", flush=True)
    rec = {"setting": "surround", "shard": shard, "shards": shards, "n_samples": len(samples), "n_written": n_written,
           "n_present": n_present, "n_no_scale": n_no_scale, "seconds": time.time() - t0, "model": info}
    os.makedirs(LC.LOG_DIR, exist_ok=True)
    json.dump(rec, open(os.path.join(LC.LOG_DIR, f"surround_clips_s{shard}of{shards}.json"), "w"), indent=1, default=str)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(build_clips(a.device, a.shard, a.shards, a.limit), indent=1, default=str))


if __name__ == "__main__":
    main()
