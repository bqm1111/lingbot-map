#!/usr/bin/env python
"""Native direct-mode LingBot replay over the chronological streams.

One unbroken causal pass per segment. ``clean_kv_cache()`` is called exactly once per
segment -- at a genuine boundary and nowhere else -- so anchor context, the sliding
pose-reference window and the trajectory tokens persist across the whole stream.

One model instance per dataset: the FlashInfer KV manager binds to the first frame
geometry it sees and is never rebuilt, so a 294x518 stream and a 154x518 stream cannot
share a process. Verified in Phase 0 by the assertion it raises.

    python tools/gate7b/stream_lingbot.py --dataset kitti360 --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate7b import replay, streams                                               # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")
STREAM_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream"


def seg_path(dataset: str, name: str) -> str:
    return os.path.join(STREAM_ROOT, dataset, f"{name}.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["semantickitti", "occ3d", "kitti360"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-segments", type=int, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    ds = a.dataset
    os.makedirs(os.path.join(STREAM_ROOT, ds), exist_ok=True)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.cuda.init()
    model, ckinfo = replay.build_model(dev)

    segs = streams.build(ds, REPO_ROOT)
    if a.limit_segments:
        segs = segs[:a.limit_segments]
    torch.cuda.reset_peak_memory_stats(dev)
    t0, rows, n_frames = time.time(), [], 0
    for si, seg in enumerate(segs):
        out_p = seg_path(ds, seg.name)
        frames = seg.frames[:a.limit_frames] if a.limit_frames else seg.frames
        if os.path.exists(out_p) and not a.overwrite:
            rows.append({"segment": seg.name, "n_frames": len(frames), "cached": True})
            continue
        k = replay.keyframe_interval_for(len(seg))     # rule sees the FULL segment length
        imgs = replay.load_images([f.path for f in frames])
        ts = time.time()
        out = replay.replay_segment(model, imgs, k, dev)
        dt = time.time() - ts
        np.savez_compressed(
            out_p, keys=np.asarray([f.key for f in frames]),
            order=np.asarray([f.order for f in frames], np.int64),
            pred_depth=out["pred_depth"], pred_depth_conf=out["pred_depth_conf"],
            pred_pose_c2w=out["pred_pose_c2w"], pred_K=out["pred_K"],
            pose_enc=out["pose_enc"], proc_hw=out["proc_hw"],
            keyframe_interval=out["keyframe_interval"],
            rope_slots=np.int64(replay.rope_slots(len(frames), k)))
        n_frames += len(frames)
        rows.append({"segment": seg.name, "n_frames": len(frames),
                     "keyframe_interval": int(k),
                     "rope_slots": int(replay.rope_slots(len(frames), k)),
                     "seconds": dt, "fps": len(frames) / max(dt, 1e-9), "cached": False})
        if (si + 1) % 10 == 0 or len(segs) < 10:
            print(f"  [{ds}] {si+1}/{len(segs)} segments, {n_frames} frames, "
                  f"{rows[-1]['fps']:.1f} FPS", flush=True)

    meta = {"dataset": ds, "n_segments": len(segs),
            "n_frames_streamed": n_frames,
            "seconds": time.time() - t0,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
            "checkpoint": replay.CHECKPOINT, "checkpoint_load": ckinfo,
            "inference": {"resolution": replay.INFERENCE_RESOLUTION,
                          "patch_size": replay.PATCH_SIZE,
                          "num_scale_frames": replay.NUM_SCALE_FRAMES,
                          "kv_cache_sliding_window": replay.KV_CACHE_SLIDING_WINDOW,
                          "max_frame_num": replay.MAX_FRAME_NUM,
                          "autocast": replay.AUTOCAST_DTYPE},
            "boundary_rule": streams.BOUNDARY_RULE[ds],
            "segments": rows, "root": os.path.join(STREAM_ROOT, ds)}
    write_json(os.path.join(ART, f"stream_{ds}.json"), meta)
    fps = [r["fps"] for r in rows if not r.get("cached")]
    print(f"[{ds}] {n_frames} frames, {meta['seconds']/60:.1f} min, "
          f"median {np.median(fps) if fps else 0:.1f} FPS, "
          f"peak {meta['peak_gpu_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
