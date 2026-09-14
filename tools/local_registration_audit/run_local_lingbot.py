#!/usr/bin/env python
"""Condition L: frozen LingBot-Map on each five-frame causal clip alone.

Reset = ``gates.gate7b.replay.replay_segment`` on the five images: ``model.clean_kv_cache()``
clears every KV cache and resets the aggregator's global RoPE frame counter
(``total_frames_processed``, asserted 0 before every clip; a five-frame clip runs entirely as
the batched scale block, which does not advance it) and the cached 3D positions; frames are numbered 0..4
inside the clip (t-4 = 0, so the clip's own first frame is the canonical origin, the same
convention the stream uses at its start); no map, trajectory token or pose reference from
before t-4 exists. Same weights, preprocessing, resolution, autocast and five scale frames
as the stream; keyframe interval 1. Metric scale: the frozen per-frame MoGe-2 candidate
(``gates.gate7b.scale.frame_candidate``) on the clip's own depth, median over its five
frames -- the stream's own five-frame rule applied to the clip.

    python tools/local_registration_audit/run_local_lingbot.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from gates.gate7b import replay, scale as SC, depth as D7G                                # noqa: E402
from core.datasets import sources as S, kitti360_splits as SRC                            # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    anchors = [x.stream_index for x in SRC.anchors(a.drive, REPO_ROOT)[::LR.ANCHOR_STRIDE]][a.shard::a.shards][:a.limit]
    model, info = replay.build_model(dev)
    n_no_moge = 0
    if src == "kitti360":                                         # validation drive: clip-keyed MoGe-2 caches
        mindex = D7.moge_frame_index("kitti360", REPO_ROOT)
        def moge(key):
            r = D7.load_moge_frame("kitti360", key, mindex)
            return None if r is None else (r[0], r[1])
    else:
        def moge(key):
            p = S.moge_path(src, key)
            if not os.path.exists(p):
                return None
            with np.load(p) as z:
                return z["moge_depth"].astype(np.float32), D7.unpack_mask(z["moge_mask"], z["mask_shape"])
    wsha = replay.CHECKPOINT
    os.makedirs(os.path.dirname(LR.clip_path(a.drive, 0)), exist_ok=True); t0 = time.time(); nw = 0
    for t in anchors:
        op = LR.clip_path(a.drive, t)
        if os.path.exists(op):
            continue
        idx = [t + o for o in LR.OFFSETS]; paths = [seg.frames[j].path for j in idx]
        imgs = replay.load_images(paths)
        model.clean_kv_cache(); assert replay.frame_index_trace(model) == 0
        out = replay.replay_segment(model, imgs, 1, dev)
        slots = int(out["rope_slots_used"])      # five scale frames run as one batched block and do not advance the streaming counter
        dep = out["pred_depth"].astype(np.float32); cnf = out["pred_depth_conf"].astype(np.float32); logs = []
        for q, j in enumerate(idx):
            mo = moge(seg.frames[j].key)
            if mo is None:
                logs.append(float("nan")); n_no_moge += 1; continue
            md, mm = mo
            logs.append(SC.frame_candidate(md, mm, dep[q], cnf[q])["log_s"])
        logs = np.asarray(logs, np.float64); ls = float(np.nanmedian(logs)) if np.isfinite(logs).any() else float("nan")
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, t=np.int64(t), stream_indices=np.asarray(idx, np.int32), keys=np.asarray([seg.frames[j].key for j in idx]),
                            pred_depth=out["pred_depth"], pred_depth_conf=out["pred_depth_conf"], pred_pose_c2w=out["pred_pose_c2w"], pred_K=out["pred_K"],
                            proc_hw=out["proc_hw"], log_s_frames=logs, scale=np.float64(np.exp(ls)), rope_slots_used=np.int64(slots))
        os.replace(tmp, op); nw += 1
        if nw % 50 == 0:
            print(f"  [{a.drive[17:21]} s{a.shard}] {nw}/{len(anchors)} clips, {(time.time()-t0)/nw:.2f}s each", flush=True)
    print(f"[{a.drive} s{a.shard}] {nw} clips in {(time.time()-t0)/60:.1f} min; frames without MoGe {n_no_moge}; load info {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
