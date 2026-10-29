#!/usr/bin/env python
"""Build the ONE shared occupancy-map cache per (benchmark, inference mode).

Both modes write the *same* object for every anchor: the hard-map channels ``(S, F)`` on the
prediction grid, in the anchor camera's frame, plus the gauge and the frames that produced
them. Every completion head and every seed is then a forward pass over exactly these maps,
so the upstream input cannot depend on the training source or the completion seed. That is
the only way the source comparison of this phase is a comparison of heads.

Mode A (clip-reset reference)
    Read the frozen five-frame clip from the cache ``build_clips.py`` wrote, which already
    carries that clip's own LingBot depth/pose and its own five-frame scale, and run the
    unchanged mapper. Nothing is recomputed, so mode A reproduces the historical procedure
    by construction -- and ``--verify-canonical`` proves it, by running the frozen
    KITTI-360 seed-0 head over these maps and requiring the result to be *bit-identical*
    to the stored canonical prediction.

Mode B (persistent state, initialization-only scale)
    One ``replay_segment`` pass per physical segment: ``clean_kv_cache()`` once, at the
    segment start, then every frame of the segment in chronological order. The metric scale
    is estimated from the segment's first five frames and frozen; no MoGe call and no scale
    refresh happens afterwards, which is a property of the code path, not a claim -- the
    MoGe reader is called exactly ``N_INIT_FRAMES`` times per segment and the counter is
    recorded. Each anchor's map is then built from the SAME five causal frames the
    reference uses, with the frozen scalar forced into the mapper.

    The explicit occupancy-map window is therefore identical between the modes. What
    differs is (a) the depth and pose of those five frames, which now come from a model
    that has seen the whole earlier stream, and (b) the gauge, which is now the segment's
    startup scale instead of the clip's own. These change together and the reports never
    attribute the difference to one of them alone.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/build_maps.py \\
        --dataset semantickitti --mode B --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                            # noqa: E402
from core.datasets import grids as CG                                                   # noqa: E402
from gates.gate7b import replay as RP, scale as SC                                       # noqa: E402
from tools.method_vs_data.streaming import config as SV, streams as ST, trace as TR       # noqa: E402
from tools.occany_comparison import config as OC                                         # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels      # noqa: E402
from tools.occany_comparison.moge_extra import resolve_moge_path                          # noqa: E402
from tools.zero_shot_eval import samples as ZS                                            # noqa: E402

CAUSAL = OC.PROTOCOLS["causal"]


def read_moge(dataset: str, key: str):
    """The frozen MoGe-2 frame cache. Used for ACCURACY only; runtime calls the model."""
    with np.load(resolve_moge_path(dataset, key)) as z:
        d = np.asarray(z["moge_depth"], np.float32)
        shp = tuple(int(x) for x in z["mask_shape"])
        m = np.unpackbits(z["moge_mask"])[: int(np.prod(shp))].astype(bool).reshape(shp)
        return d, m, float(z["fov_x_deg"])


def save_map(path: str, S_: torch.Tensor, F_: torch.Tensor, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, S=np.packbits(S_.cpu().numpy()), F=np.packbits(F_.cpu().numpy()),
                        n_voxels=np.int64(S_.numel()), meta=json.dumps(meta, default=float))
    os.replace(tmp, path)


def load_map(dataset: str, mode: str, sample_id: str):
    with np.load(SV.map_cache_path(dataset, mode, sample_id)) as z:
        n = int(z["n_voxels"])
        S_ = np.unpackbits(z["S"])[:n].astype(bool)
        F_ = np.unpackbits(z["F"])[:n].astype(bool)
        return S_, F_, json.loads(str(z["meta"]))


# --------------------------------------------------------------------------- mode A
def build_mode_a(dataset: str, device: str, limit: Optional[int] = None,
                 overwrite: bool = False) -> dict:
    grid = CG.PREDICTION_GRID[dataset]
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    sids = sorted(by_sid)[:limit]
    rows, t0 = [], time.time()
    for i, sid in enumerate(sids):
        out = SV.map_cache_path(dataset, "A", sid)
        if os.path.exists(out) and not overwrite:
            continue
        s = by_sid[sid]
        clip = WindowClip.load(OC.clip_path(dataset, "causal", sid))
        assert clip.anchor_slot == CAUSAL.anchor_slot, sid
        T = np.asarray(s.T_cam_to_grid_ours, np.float64)
        mapper = build_mapper(clip, dev)
        S_, F_ = hard_channels(mapper, grid, T)
        n_rows = len(mapper.table)
        del mapper
        meta = {"dataset": dataset, "mode": "A", "sample_id": sid, "segment": s.segment,
                "target_pos": int(s.target_pos),
                "map_frame_positions": [int(s.target_pos + o) for o in CAUSAL.offsets],
                "map_frame_keys": [s.frame_keys[s.target_pos + o] for o in CAUSAL.offsets],
                "anchor_slot": int(clip.anchor_slot),
                "scale_m_per_unit": float(clip.scale_m_per_unit),
                "scale_policy": "clip's own five frames (reference)",
                "T_cam_to_grid": np.asarray(T).tolist(),
                "n_occupied": int(S_.sum()), "n_free": int(F_.sum()),
                "n_table_rows": int(n_rows), "grid": grid.name,
                "dims": list(grid.dims), "voxel_size": grid.voxel_size}
        save_map(out, S_, F_, meta)
        rows.append({"sample_id": sid, "scale": float(clip.scale_m_per_unit),
                     "n_occupied": int(S_.sum()), "n_free": int(F_.sum())})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sids)}  {(time.time()-t0)/(i+1):.3f}s each", flush=True)
    return {"dataset": dataset, "mode": "A", "n_built": len(rows), "n_total": len(sids),
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------- mode B
@torch.no_grad()
def build_mode_b(dataset: str, device: str, segments: Optional[Sequence[str]] = None,
                 limit_frames: Optional[int] = None, overwrite: bool = False,
                 keyframe_interval: Optional[int] = None) -> dict:
    grid = CG.PREDICTION_GRID[dataset]
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    model, info = RP.build_model(dev)
    assert model.pred_normalization is False, \
        "pred_normalization normalises across the whole sequence and would be non-causal"
    segs = ST.build(dataset)
    if segments:
        want = set(segments)
        segs = [s for s in segs if s.name in want]
    logs, t0 = [], time.time()
    for si, seg in enumerate(segs):
        n_frames = len(seg) if limit_frames is None else min(len(seg), limit_frames)
        anchors = [(sid, p) for sid, p in seg.anchors if p < n_frames]
        if not anchors:
            continue
        todo = [a for a in anchors
                if overwrite or not os.path.exists(SV.map_cache_path(dataset, "B", a[0]))]
        if not todo:
            continue
        ts = time.time()
        k = keyframe_interval if keyframe_interval is not None else RP.keyframe_interval_for(n_frames)
        images = RP.load_images(list(seg.frame_paths[:n_frames]))
        t_load = time.time() - ts

        tr = time.time()
        with TR.StateTrace(model) as st:
            out = RP.replay_segment(model, images, k, dev)   # ONE reset, at the start
        t_replay = time.time() - tr
        assert not st.mid_stream_resets(), f"{seg.name}: state reset mid-stream"
        assert st.peak_frames == RP.rope_slots(n_frames, k), \
            f"{seg.name}: {st.peak_frames} RoPE slots used, {RP.rope_slots(n_frames, k)} predicted"
        depth = out["pred_depth"].astype(np.float32)
        conf = out["pred_depth_conf"].astype(np.float32)
        pose = out["pred_pose_c2w"].astype(np.float64)
        K = out["pred_K"].astype(np.float64)
        assert len(depth) == n_frames

        # ---- initialization-only gauge: exactly N_INIT_FRAMES MoGe reads, then frozen
        n_moge, cands = 0, []
        for p in range(min(SV.N_INIT_FRAMES, n_frames)):
            md, mm, _fov = read_moge(dataset, seg.frame_keys[p])
            n_moge += 1
            c = SC.frame_candidate(md, mm, depth[p], conf[p])
            cands.append({"position": p, "key": seg.frame_keys[p], **c})
        fin = [c["log_s"] for c in cands if np.isfinite(c["log_s"])]
        s_frozen = float(np.exp(np.median(fin))) if fin else float("nan")
        assert n_moge == min(SV.N_INIT_FRAMES, n_frames)

        rows = []
        if np.isfinite(s_frozen) and s_frozen > 0:
            for sid, p in todo:
                idx = [p + o for o in CAUSAL.offsets]
                assert max(idx) <= p and min(idx) >= 0, (sid, idx)
                sample = _sample(dataset, sid)
                T = np.asarray(sample.T_cam_to_grid_ours, np.float64)
                clip = WindowClip(CAUSAL.anchor_slot, depth[idx], conf[idx], pose[idx],
                                  K[idx], s_frozen)
                mapper = build_mapper(clip, dev)
                S_, F_ = hard_channels(mapper, grid, T)
                n_rows_tbl = len(mapper.table)
                del mapper
                meta = {"dataset": dataset, "mode": "B", "sample_id": sid,
                        "segment": seg.name, "target_pos": int(p),
                        "map_frame_positions": [int(j) for j in idx],
                        "map_frame_keys": [seg.frame_keys[j] for j in idx],
                        "anchor_slot": int(CAUSAL.anchor_slot),
                        "scale_m_per_unit": float(s_frozen),
                        "scale_policy": ("segment startup scale: median of the first "
                                         f"{SV.N_INIT_FRAMES} frames' log candidates, frozen"),
                        "scale_frozen_at_position": int(min(SV.N_INIT_FRAMES, n_frames) - 1),
                        "n_moge_calls_in_segment": int(n_moge),
                        "T_cam_to_grid": np.asarray(T).tolist(),
                        "n_occupied": int(S_.sum()), "n_free": int(F_.sum()),
                        "n_table_rows": int(n_rows_tbl), "grid": grid.name,
                        "dims": list(grid.dims), "voxel_size": grid.voxel_size,
                        "segment_n_frames_processed": int(n_frames),
                        "lingbot_keyframe_interval": int(k),
                        "lingbot_resets_in_segment": 1,
                        "time_s_in_segment": float(seg.times_s[p])}
                save_map(SV.map_cache_path(dataset, "B", sid), S_, F_, meta)
                rows.append({"sample_id": sid, "position": int(p),
                             "time_s": float(seg.times_s[p]),
                             "n_occupied": int(S_.sum()), "n_free": int(F_.sum())})
        logs.append({
            "segment": seg.name, "n_frames_processed": int(n_frames),
            "duration_s": float(seg.times_s[n_frames - 1]),
            "keyframe_interval": int(k),
            "rope_slots_used": int(st.peak_frames),
            "state_trace": st.summary(),
            "lingbot_resets": 1, "reset_positions": [0],
            "n_moge_calls": int(n_moge),
            "scale_m_per_unit": float(s_frozen),
            "scale_candidates": cands,
            "scale_valid": bool(np.isfinite(s_frozen) and s_frozen > 0),
            "n_anchors_written": len(rows), "anchors": rows,
            "seconds": {"load_images": t_load, "replay": t_replay,
                        "total": time.time() - ts}})
        print(f"  [{si+1}/{len(segs)}] {seg.name}: {n_frames} frames, rope {st.peak_frames}, "
              f"scale {s_frozen:.4f}, {len(rows)} anchors, {time.time()-ts:.1f}s", flush=True)
        del out, depth, conf, pose, K, images
        torch.cuda.empty_cache()
    bad = [l["segment"] for l in logs if not l["scale_valid"]]
    return {"dataset": dataset, "mode": "B", "n_segments": len(logs),
            "n_anchors_written": int(sum(l["n_anchors_written"] for l in logs)),
            "segments_without_a_valid_startup_scale": bad,
            "lingbot_load": info, "keyframe_interval_rule": "gate7b.replay.keyframe_interval_for",
            "moge_source": "frozen gate-7B frame cache (accuracy run); runtime calls the model",
            "seconds": time.time() - t0, "segment_logs": logs}


_SAMPLES: Dict[str, Dict] = {}


def _sample(dataset: str, sid: str):
    if dataset not in _SAMPLES:
        _SAMPLES[dataset] = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    return _SAMPLES[dataset][sid]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SV.DATASETS))
    ap.add_argument("--mode", required=True, choices=list(SV.MODES))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--segments", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="mode A: first N anchors")
    ap.add_argument("--limit-frames", type=int, default=None, help="mode B: first N frames")
    ap.add_argument("--keyframe-interval", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SV.ensure_dirs()
    if a.mode == "A":
        r = build_mode_a(a.dataset, a.device, a.limit, a.overwrite)
    else:
        r = build_mode_b(a.dataset, a.device, a.segments, a.limit_frames, a.overwrite,
                         a.keyframe_interval)
    tag = f"build_maps_{a.dataset}_mode{a.mode}"
    out = a.out or os.path.join(SV.LOG_DIR, f"{tag}.json")
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in r.items() if k != "segment_logs"}, indent=1, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
