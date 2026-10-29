#!/usr/bin/env python
"""Continuous-runtime cost of the persistent-state pipeline, measured as a robot would pay it.

This is NOT the clip-reset benchmark with MoGe subtracted. It is a frame-by-frame streaming
loop: the state is reset once, the first five frames pay the startup cost including their
real MoGe-2 forwards, and every later frame pays an update; frames that are evaluation
anchors additionally pay a map build, a CompletionUNet forward and the read-out. Nothing is
read from a cache: LingBot, MoGe-2, the metric scale, the hard map and the completion all
run inside the timed region.

Timing boundary
    start  decoded RGB frames and ordinary calibration metadata in host memory
    end    the completed occupancy grid on the evaluation grid, as a host numpy array
           (the same evaluator-ready boundary the accuracy run scores)

Included: image preprocessing, host-to-device transfers, every LingBot forward, the startup
MoGe-2 forwards, the metric-scale estimation, the mapper, the map query, the CompletionUNet
forward, and the geometry read-out onto the evaluation grid.

Excluded, and stated: image decode (outside the declared boundary, measured separately),
ground-truth loading, scoring, visualisation, prediction serialisation, and model
construction / checkpoint loading -- which is reported on its own as a startup-once cost.

Three figures are kept apart because they answer different questions:
    * **per input update** -- what every frame costs, whether or not it emits an output;
    * **per completed prediction** -- what an emitted occupancy grid costs on top of that;
    * **sequence average per prediction** -- total sequence computation, startup and all
      intervening updates included, divided by the number of grids emitted. This is the
      honest robot figure and it is never computed by dividing a latency by a frame count.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/runtime_stream.py \\
        --dataset semantickitti --mode stream --device cuda:3
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
from tools.method_vs_data.runtime import config as RC                                   # noqa: E402
from tools.method_vs_data.streaming import (build_maps as BM, config as SV, heads as HD,  # noqa: E402
                                            streams as ST, trace as TR)
from tools.occany_comparison import config as OC                                         # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels      # noqa: E402
from tools.zero_shot_eval import samples as ZS                                            # noqa: E402

CAUSAL = OC.PROTOCOLS["causal"]
UPDATE_STAGES = ("preprocess", "transfer_and_lingbot", "pose_decode")
STARTUP_STAGES = ("preprocess", "lingbot_scale_block", "moge", "scale")
PREDICT_STAGES = ("mapper", "map_query", "unet", "readout")
#: three contiguous streams per benchmark, evenly spaced over the available frames, and the
#: target length. SemanticKITTI's single 815-frame segment affords three 200-frame streams;
#: an Occ3D scene is ~40 samples, so the 200-frame target is NOT available there and the
#: reduction is reported beside every amortised figure.
TARGET_STREAM_FRAMES = 200
N_STREAMS = 3
REPS = 3


class Clock:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.t: Dict[str, float] = {}
        self._k, self._t0 = None, 0.0

    def __call__(self, k):
        self._k = k
        return self

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.enabled:
            torch.cuda.synchronize()
            self.t[self._k] = self.t.get(self._k, 0.0) + (time.perf_counter() - self._t0)
        return False


def _store(t: torch.Tensor) -> np.ndarray:
    """The deployed storage precision for depth and confidence.

    ``gates/gate7b/replay.replay_segment`` writes both as float16 and every consumer -- the
    clip cache, the mode-A maps, the mode-B maps -- reads that float16 back as float32. The
    driver does the same, so the timed computation is the scored computation and not a
    higher-precision variant of it.
    """
    return t.float().cpu().numpy().astype(np.float16).astype(np.float32)


class StreamDriver:
    """Frame-by-frame LingBot streaming, replicating ``inference_streaming``'s two phases.

    Phase 1 is one forward over the ``n_scale`` initialization frames with
    ``num_frame_per_block = n_scale`` (the model's own bidirectional scale block); phase 2 is
    one forward per later frame with ``num_frame_per_block = 1``. Same arguments, same
    autocast, same ``cudagraph_mark_step_begin``, so the driver is the deployed path split
    open rather than a second implementation -- which ``--verify-driver`` proves by requiring
    bit-identical depth, confidence, pose and intrinsics against ``replay_segment``.
    """

    def __init__(self, model, device, n_scale: int = RP.NUM_SCALE_FRAMES):
        self.m, self.dev, self.n_scale = model, device, int(n_scale)
        self.t = 0

    def reset(self) -> None:
        self.m.clean_kv_cache()
        self.t = 0

    def _decode(self, pose_enc: torch.Tensor, hw):
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        pe = pose_enc.float().cpu()
        extr, intr = pose_encoding_to_extri_intri(pe.unsqueeze(0), tuple(hw))
        c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
        c2w[:, :3, :4] = extr[0].double().numpy()
        return c2w, intr[0].float().cpu().numpy()

    @torch.inference_mode()
    def _fwd(self, images_cpu: torch.Tensor, per_block: int):
        imgs = images_cpu.to(self.dev, non_blocking=True)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast("cuda", dtype=getattr(torch, RP.AUTOCAST_DTYPE)):
            o = self.m.forward(imgs, num_frame_for_scale=self.n_scale,
                               num_frame_per_block=per_block, causal_inference=True)
        return o, imgs.shape[-2:]

    def scale_block(self, images_cpu: torch.Tensor, clock: Clock):
        """``images_cpu``: ``[1, n_scale, 3, H, W]``. Returns per-frame geometry."""
        with clock("lingbot_scale_block"):
            o, hw = self._fwd(images_cpu, self.n_scale)
            d = _store(o["depth"][0, ..., 0])
            c = _store(o["depth_conf"][0])
            pe = o["pose_enc"][0]
        c2w, K = self._decode(pe, hw)
        self.t = self.n_scale
        return d, c, c2w, K

    def update(self, image_cpu: torch.Tensor, clock: Clock):
        """One later frame. ``image_cpu``: ``[1, 1, 3, H, W]``."""
        with clock("transfer_and_lingbot"):
            o, hw = self._fwd(image_cpu, 1)
            d = _store(o["depth"][0, 0, ..., 0])
            c = _store(o["depth_conf"][0, 0])
            pe = o["pose_enc"][0]
        with clock("pose_decode"):
            c2w, K = self._decode(pe, hw)
        self.t += 1
        return d, c, c2w[0], K[0]


class StreamPipeline:
    """Resident models plus the deployed five-frame ring buffer."""

    def __init__(self, dataset: str, device: str):
        from gates.scale_gate.config import load_config
        from moge_gauge.calibrated import CalibratedMoGe
        self.dataset = dataset
        self.dev = torch.device(device)
        torch.cuda.set_device(self.dev)
        t0 = time.perf_counter()
        self.lingbot, self.lingbot_info = RP.build_model(self.dev)
        t1 = time.perf_counter()
        g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
        self.moge = CalibratedMoGe(self.dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                                   revision=g5.moge.hf_revision)
        t2 = time.perf_counter()
        self.net = HD.load_head("k360", 0, self.dev)
        t3 = time.perf_counter()
        self.rd = HD.Readout(dataset, self.dev)
        self.grid_obj = CG.PREDICTION_GRID[dataset]
        self.driver = StreamDriver(self.lingbot, self.dev)
        self.load_seconds = {"lingbot_map": t1 - t0, "moge2_vitl": t2 - t1,
                             "completion_unet": t3 - t2, "total": t3 - t0}

    def params(self) -> dict:
        def count(m):
            ps = list(m.parameters())
            return (int(sum(p.numel() for p in ps)),
                    int(sum(p.numel() for p in ps if p.requires_grad)))
        mm = next((getattr(self.moge, a) for a in ("model", "moge", "_model")
                   if isinstance(getattr(self.moge, a, None), torch.nn.Module)), None)
        out = {"lingbot_map": count(self.lingbot), "completion_unet": count(self.net),
               "moge2_vitl": count(mm) if mm is not None else None}
        return {"per_module_total_and_requires_grad": out,
                "total_inference_parameters": sum(v[0] for v in out.values() if v),
                "task_specific_trained_parameters": out["completion_unet"][0]}

    def predict(self, ring: List[dict], scale: float, T: np.ndarray, clock: Clock):
        """One completed occupancy grid from the five buffered frames and the frozen scale."""
        with clock("mapper"):
            clip = WindowClip(CAUSAL.anchor_slot,
                              np.stack([r["d"] for r in ring]).astype(np.float32),
                              np.stack([r["c"] for r in ring]).astype(np.float32),
                              np.stack([r["P"] for r in ring]).astype(np.float64),
                              np.stack([r["K"] for r in ring]).astype(np.float64), scale)
            mapper = build_mapper(clip, self.dev)
        with clock("map_query"):
            S_, F_ = hard_channels(mapper, self.grid_obj, T)
        del mapper
        with clock("unet"):
            from core.model.net import apply_residual
            x = torch.stack([S_.reshape(self.rd.fine.dims),
                             torch.zeros(self.rd.fine.dims, dtype=torch.bool, device=self.dev),
                             F_.reshape(self.rd.fine.dims)]).float()[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res, _ = self.net(x)
            r = res[:, 0].float()
            occ = apply_residual(torch.zeros_like(r), r).reshape(1, -1) > 0
        with clock("readout"):
            from tools.zero_shot_eval.prediction.grid import reduce_any
            grid = reduce_any(occ[0], self.rd.fine, self.rd.ratio).cpu().numpy()
        return grid


def select_streams(dataset: str) -> List[dict]:
    """Three contiguous streams per benchmark, chosen by a declared rule, not by a score."""
    segs = ST.build(dataset)
    out = []
    if dataset == "semantickitti":
        seg = segs[0]
        n = min(TARGET_STREAM_FRAMES, len(seg))
        starts = np.linspace(0, len(seg) - n, N_STREAMS).astype(int)
        for st in starts:
            out.append({"segment": seg.name, "start": int(st), "n_frames": int(n),
                        "rule": f"{N_STREAMS} evenly spaced {n}-frame windows of segment "
                                f"{seg.name} ({len(seg)} frames)"})
    else:
        # the scene whose length is the median, and its two neighbours in that ranking
        order = sorted(segs, key=lambda s: (len(s), s.name))
        mid = len(order) // 2
        for s in order[max(mid - 1, 0): mid + 2][:N_STREAMS]:
            out.append({"segment": s.name, "start": 0, "n_frames": len(s),
                        "rule": ("the median-length validation scene and its neighbours in "
                                 "the length ranking; the 200-frame target is NOT available "
                                 "for Occ3D, whose scenes are ~40 samples")})
    return out


def run_stream(pipe: StreamPipeline, seg, start: int, n_frames: int, anchors: Dict[str, int],
               stage_mode: bool, live_moge: bool = True) -> dict:
    """One stream from a clean state. Returns every timing the protocol asks for."""
    dataset = pipe.dataset
    clk = Clock(stage_mode)
    by_pos = {p: sid for sid, p in seg.anchors}
    T_of = {sid: np.asarray(s.T_cam_to_grid_ours, np.float64)
            for sid, s in anchors.items()}
    positions = list(range(start, start + n_frames))
    paths = [seg.frame_paths[p] for p in positions]
    keys = [seg.frame_keys[p] for p in positions]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(pipe.dev)
    pipe.driver.reset()
    with TR.StateTrace(pipe.lingbot) as tr:
        # ---------- startup: the five initialization frames, with their real MoGe forwards
        torch.cuda.synchronize()
        t_start0 = time.perf_counter()
        with clk("preprocess"):
            imgs0 = RP.load_images(paths[:SV.N_INIT_FRAMES]).unsqueeze(0)
        d0, c0, P0, K0 = pipe.driver.scale_block(imgs0, clk)
        fovs, mds, mms = [], [], []
        with clk("moge"):
            for q in range(SV.N_INIT_FRAMES):
                if live_moge:
                    fov = _fov_of(dataset, keys[q])
                    o = pipe.moge.infer_calibrated(imgs0[0, q].cpu().numpy(), fov_x_deg=fov)
                    mds.append(np.asarray(o.depth_z, np.float32))
                    mms.append(np.asarray(o.mask, bool))
                else:
                    md, mm, fov = BM.read_moge(dataset, keys[q])
                    mds.append(md); mms.append(mm)
                fovs.append(fov)
        with clk("scale"):
            logs = np.asarray([SC.frame_candidate(mds[q], mms[q], d0[q], c0[q])["log_s"]
                               for q in range(SV.N_INIT_FRAMES)], np.float64)
            scale = float(np.exp(np.nanmedian(logs))) if np.isfinite(logs).any() else float("nan")
        torch.cuda.synchronize()
        t_startup = time.perf_counter() - t_start0
        mem_startup = {"allocated_bytes": int(torch.cuda.max_memory_allocated(pipe.dev)),
                       "reserved_bytes": int(torch.cuda.max_memory_reserved(pipe.dev))}

        ring = [{"d": d0[q], "c": c0[q], "P": P0[q], "K": K0[q]}
                for q in range(SV.N_INIT_FRAMES)]
        torch.cuda.reset_peak_memory_stats(pipe.dev)

        # ---------- the first prediction, from the state the startup left behind
        first_pos = positions[SV.N_INIT_FRAMES - 1]
        t_first = float("nan")
        n_pred, updates, preds, grids = 0, [], [], {}
        if first_pos in by_pos:
            sid = by_pos[first_pos]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = pipe.predict(ring, scale, T_of[sid], clk)
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
            grids[sid] = g
            n_pred += 1
        t_first_total = t_startup + (0.0 if np.isnan(t_first) else t_first)

        # ---------- steady state
        torch.cuda.synchronize()
        t_steady0 = time.perf_counter()
        for i in range(SV.N_INIT_FRAMES, n_frames):
            torch.cuda.synchronize()
            tu = time.perf_counter()
            with clk("preprocess"):
                im = RP.load_images([paths[i]]).unsqueeze(0)
            d, c, P, K = pipe.driver.update(im, clk)
            torch.cuda.synchronize()
            updates.append(time.perf_counter() - tu)
            ring = ring[1:] + [{"d": d, "c": c, "P": P, "K": K}]
            p = positions[i]
            if p in by_pos:
                sid = by_pos[p]
                torch.cuda.synchronize()
                tp = time.perf_counter()
                g = pipe.predict(ring, scale, T_of[sid], clk)
                torch.cuda.synchronize()
                preds.append(time.perf_counter() - tp)
                if len(grids) < 8:
                    grids[sid] = g
                n_pred += 1
        torch.cuda.synchronize()
        t_steady = time.perf_counter() - t_steady0
    mem_steady = {"allocated_bytes": int(torch.cuda.max_memory_allocated(pipe.dev)),
                  "reserved_bytes": int(torch.cuda.max_memory_reserved(pipe.dev))}
    total = t_startup + (0.0 if np.isnan(t_first) else t_first) + t_steady
    gaps = np.diff(seg.times_s[positions])
    return {
        "segment": seg.name, "start_position": start, "n_frames_processed": n_frames,
        "duration_s": float(seg.times_s[positions[-1]] - seg.times_s[positions[0]]),
        "n_predictions_emitted": n_pred,
        "output_cadence": {"predictions_per_processed_frame": n_pred / n_frames,
                           "rule": ("every processed frame at or after position 4 of the "
                                    "segment that is a canonical anchor emits a grid")},
        "startup_s": t_startup,
        "startup_includes": {"lingbot_scale_block_frames": SV.N_INIT_FRAMES,
                             "moge2_forwards": SV.N_INIT_FRAMES,
                             "live_moge": bool(live_moge)},
        "startup_scale_m_per_unit": scale,
        "first_prediction_latency_s": {
            "compute_after_last_init_frame_arrives": t_first_total,
            "prediction_step_only": t_first,
            "frame_acquisition_waiting_s": float(seg.times_s[positions[SV.N_INIT_FRAMES - 1]]
                                                 - seg.times_s[positions[0]]),
            "note": ("waiting is the wall-clock time for the four later initialization frames "
                     "to arrive at the sensor cadence; it is NOT computation and is never "
                     "added into a latency figure")},
        "per_input_update_s": RC.summarise(updates) if updates else None,
        "per_prediction_s": RC.summarise(preds) if preds else None,
        "steady_state_s": t_steady,
        "total_sequence_compute_s": total,
        "sequence_average_per_prediction_s": total / max(n_pred, 1),
        "measured_predictions_per_second": n_pred / total if total > 0 else None,
        "measured_input_updates_per_second": n_frames / total if total > 0 else None,
        "peak_memory_startup": mem_startup,
        "peak_memory_sustained": mem_steady,
        "state_trace": tr.summary(),
        "frame_gap_s": {"median": float(np.median(gaps)), "max": float(gaps.max())},
        "stages_s": dict(clk.t) if stage_mode else None,
        "_grids": grids,
    }


_FOV: Dict[tuple, float] = {}


def _fov_of(dataset: str, key: str) -> float:
    """The calibrated horizontal FOV: a deterministic function of ordinary calibration."""
    if (dataset, key) not in _FOV:
        _, _, fov = BM.read_moge(dataset, key)
        _FOV[(dataset, key)] = fov
    return _FOV[(dataset, key)]


def bench(dataset: str, device: str, stage_mode: bool = False, reps: int = REPS,
          limit_streams: Optional[int] = None, n_frames: Optional[int] = None) -> dict:
    pipe = StreamPipeline(dataset, device)
    segs = {s.name: s for s in ST.build(dataset)}
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    specs = select_streams(dataset)[:limit_streams]
    if n_frames:
        for sp in specs:
            sp["n_frames"] = min(n_frames, len(segs[sp["segment"]]))

    # warm up software and kernels on a stream that is NOT measured, then reset state
    w = specs[0]
    run_stream(pipe, segs[w["segment"]], w["start"], min(12, w["n_frames"]), by_sid, False)
    torch.cuda.synchronize()

    runs = []
    for sp in specs:
        for rep in range(reps):
            r = run_stream(pipe, segs[sp["segment"]], sp["start"], sp["n_frames"],
                           by_sid, stage_mode)
            r.pop("_grids")
            r["rep"] = rep
            r["selection_rule"] = sp["rule"]
            runs.append(r)
            print(f"  {sp['segment']}@{sp['start']} n={sp['n_frames']} rep{rep}: "
                  f"startup {r['startup_s']*1e3:.0f} ms, update "
                  f"{r['per_input_update_s']['mean_ms']:.1f} ms, predict "
                  f"{r['per_prediction_s']['mean_ms']:.1f} ms, seq/pred "
                  f"{r['sequence_average_per_prediction_s']*1e3:.1f} ms", flush=True)

    agg = lambda f: RC.summarise([f(r) for r in runs])                         # noqa: E731
    upd = [u for r in runs for u in [r["per_input_update_s"]["mean_ms"] / 1e3]]
    return {
        "method": "ours_streaming", "dataset": dataset, "device": device,
        "mode": "stages" if stage_mode else "e2e",
        "n_streams": len(specs), "reps": reps, "streams": specs,
        "startup_s": agg(lambda r: r["startup_s"]),
        "first_prediction_compute_s": agg(
            lambda r: r["first_prediction_latency_s"]["compute_after_last_init_frame_arrives"]),
        "frame_acquisition_waiting_s": agg(
            lambda r: r["first_prediction_latency_s"]["frame_acquisition_waiting_s"]),
        "per_input_update_s": RC.summarise(upd),
        "per_input_update_pooled_s": RC.summarise(
            [x / 1e3 for r in runs for x in [r["per_input_update_s"]["mean_ms"]]]),
        "per_prediction_s": RC.summarise(
            [r["per_prediction_s"]["mean_ms"] / 1e3 for r in runs]),
        "sequence_average_per_prediction_s": agg(
            lambda r: r["sequence_average_per_prediction_s"]),
        "measured_predictions_per_second": agg(lambda r: r["measured_predictions_per_second"]),
        "total_sequence_compute_s": agg(lambda r: r["total_sequence_compute_s"]),
        "peak_memory_startup_bytes": {
            "allocated": max(r["peak_memory_startup"]["allocated_bytes"] for r in runs),
            "reserved": max(r["peak_memory_startup"]["reserved_bytes"] for r in runs)},
        "peak_memory_sustained_bytes": {
            "allocated": max(r["peak_memory_sustained"]["allocated_bytes"] for r in runs),
            "reserved": max(r["peak_memory_sustained"]["reserved_bytes"] for r in runs)},
        "model_load_seconds_reported_separately": pipe.load_seconds,
        "parameters": pipe.params(),
        "configuration": {
            "lingbot_checkpoint": RP.CHECKPOINT, "lingbot_load": pipe.lingbot_info,
            "lingbot_resolution": RP.INFERENCE_RESOLUTION,
            "lingbot_autocast": RP.AUTOCAST_DTYPE,
            "kv_cache_sliding_window_blocks": RP.KV_CACHE_SLIDING_WINDOW,
            "moge_repo": "Ruicheng/moge-2-vitl",
            "moge_revision": pipe.moge.provenance().get("hf_revision"),
            "moge_calls": f"{SV.N_INIT_FRAMES} at startup, none afterwards",
            "completion_source": "k360", "completion_seed": 0,
            "prediction_grid": pipe.rd.fine.name, "prediction_dims": list(pipe.rd.fine.dims),
            "eval_grid": pipe.rd.native.name, "eval_dims": list(pipe.rd.native.dims),
            "batch": "one frame per update, one anchor per prediction",
            "state_policy": "persistent within the stream; reset only at the stream start",
            "resident_models": ["LingBotMap GCTStream", "MoGe-2 ViT-L", "CompletionUNet"],
            "future_frames_required": 0, "generated_views": 0},
        "boundary": {
            "start": "decoded RGB frames + ordinary calibration in host memory",
            "end": "completed occupancy grid on the evaluation grid, host numpy array",
            "excluded": ["image decode", "ground-truth loading", "scoring", "visualisation",
                         "prediction serialisation",
                         "model construction and checkpoint loading (reported separately)"],
            "no_cached_quantities": ("depth, point maps, poses, scale and hard maps are all "
                                     "computed inside the timed region; the only cached read "
                                     "is the calibrated horizontal FOV, a deterministic "
                                     "function of ordinary calibration")},
        "machine": RC.machine(device),
        "runs": runs,
    }


def run_clipreset(pipe: StreamPipeline, seg, start: int, n_frames: int, anchors: Dict[str, int],
                  stage_mode: bool, live_moge: bool = True) -> dict:
    """The clip-reset reference as an operating mode, timed on the SAME anchors.

    Every anchor pays the whole cost from scratch: a state reset, five image loads, one
    five-frame LingBot block, five MoGe-2 forwards, a scale, a map, a completion and the
    read-out. This is the mode every historical accuracy number was produced in, and timing it
    here -- same process, same device, same anchors, same boundary -- is what makes the A/B
    runtime comparison a comparison rather than two measurements.
    """
    clk = Clock(stage_mode)
    by_pos = {p_: sid for sid, p_ in seg.anchors}
    positions = [p_ for p_ in range(start, start + n_frames) if p_ in by_pos]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(pipe.dev)
    lat, scales = [], []
    for p_ in positions:
        sid = by_pos[p_]
        idx = [p_ + o for o in CAUSAL.offsets]
        paths = [seg.frame_paths[j] for j in idx]
        keys = [seg.frame_keys[j] for j in idx]
        T = np.asarray(anchors[sid].T_cam_to_grid_ours, np.float64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.driver.reset()                       # the reset IS the mode
        with clk("preprocess"):
            imgs = RP.load_images(paths).unsqueeze(0)
        d, c, P, K = pipe.driver.scale_block(imgs, clk)
        mds, mms = [], []
        with clk("moge"):
            for q in range(len(paths)):
                if live_moge:
                    o = pipe.moge.infer_calibrated(imgs[0, q].cpu().numpy(),
                                                   fov_x_deg=_fov_of(pipe.dataset, keys[q]))
                    mds.append(np.asarray(o.depth_z, np.float32))
                    mms.append(np.asarray(o.mask, bool))
                else:
                    md, mm, _f = BM.read_moge(pipe.dataset, keys[q])
                    mds.append(md); mms.append(mm)
        with clk("scale"):
            logs = np.asarray([SC.frame_candidate(mds[q], mms[q], d[q], c[q])["log_s"]
                               for q in range(len(paths))], np.float64)
            sc_ = float(np.exp(np.nanmedian(logs))) if np.isfinite(logs).any() else float("nan")
        ring = [{"d": d[q], "c": c[q], "P": P[q], "K": K[q]} for q in range(len(paths))]
        pipe.predict(ring, sc_, T, clk)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        scales.append(sc_)
    return {
        "segment": seg.name, "start_position": start, "n_frames_window": n_frames,
        "n_predictions_emitted": len(lat),
        "latency_per_completed_occupancy_grid": RC.summarise(lat),
        "per_input_update_s": None,
        "sequence_average_per_prediction_s": float(np.sum(lat)) / max(len(lat), 1),
        "measured_predictions_per_second": len(lat) / float(np.sum(lat)) if lat else None,
        "total_sequence_compute_s": float(np.sum(lat)),
        "resets": len(lat),
        "moge_forwards": len(lat) * SV.N_INIT_FRAMES,
        "peak_memory_sustained": {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(pipe.dev)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(pipe.dev))},
        "scale_distinct_values": len(set(np.round(scales, 10))),
        "stages_s": dict(clk.t) if stage_mode else None,
    }


def bench_clipreset(dataset: str, device: str, stage_mode: bool = False, reps: int = REPS,
                    limit_streams: Optional[int] = None,
                    n_frames: Optional[int] = None) -> dict:
    pipe = StreamPipeline(dataset, device)
    segs = {s.name: s for s in ST.build(dataset)}
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    specs = select_streams(dataset)[:limit_streams]
    if n_frames:
        for sp in specs:
            sp["n_frames"] = min(n_frames, len(segs[sp["segment"]]))
    w = specs[0]
    run_clipreset(pipe, segs[w["segment"]], w["start"], min(12, w["n_frames"]), by_sid, False)
    torch.cuda.synchronize()
    runs = []
    for sp in specs:
        for rep in range(reps):
            r = run_clipreset(pipe, segs[sp["segment"]], sp["start"], sp["n_frames"],
                              by_sid, stage_mode)
            r["rep"] = rep; r["selection_rule"] = sp["rule"]
            runs.append(r)
            print(f"  {sp['segment']}@{sp['start']} rep{rep}: per-grid "
                  f"{r['latency_per_completed_occupancy_grid']['mean_ms']:.1f} ms "
                  f"({r['n_predictions_emitted']} grids)", flush=True)
    return {
        "method": "ours_clip_reset", "dataset": dataset, "device": device,
        "mode": "stages" if stage_mode else "e2e",
        "n_streams": len(specs), "reps": reps, "streams": specs,
        "latency_per_completed_occupancy_grid": RC.summarise(
            [r["latency_per_completed_occupancy_grid"]["mean_ms"] / 1e3 for r in runs]),
        "sequence_average_per_prediction_s": RC.summarise(
            [r["sequence_average_per_prediction_s"] for r in runs]),
        "measured_predictions_per_second": RC.summarise(
            [r["measured_predictions_per_second"] for r in runs]),
        "peak_memory_sustained_bytes": {
            "allocated": max(r["peak_memory_sustained"]["allocated_bytes"] for r in runs),
            "reserved": max(r["peak_memory_sustained"]["reserved_bytes"] for r in runs)},
        "resets_per_prediction": 1,
        "moge_forwards_per_prediction": SV.N_INIT_FRAMES,
        "model_load_seconds_reported_separately": pipe.load_seconds,
        "parameters": pipe.params(),
        "configuration": {
            "lingbot_checkpoint": RP.CHECKPOINT,
            "lingbot_resolution": RP.INFERENCE_RESOLUTION,
            "lingbot_autocast": RP.AUTOCAST_DTYPE,
            "moge_repo": "Ruicheng/moge-2-vitl",
            "moge_revision": pipe.moge.provenance().get("hf_revision"),
            "moge_calls": f"{SV.N_INIT_FRAMES} per anchor",
            "completion_source": "k360", "completion_seed": 0,
            "prediction_grid": pipe.rd.fine.name, "eval_grid": pipe.rd.native.name,
            "batch": "one evaluation anchor",
            "state_policy": "reset at every anchor",
            "resident_models": ["LingBotMap GCTStream", "MoGe-2 ViT-L", "CompletionUNet"],
            "future_frames_required": 0, "generated_views": 0},
        "boundary": {"start": "decoded RGB frames + ordinary calibration in host memory",
                     "end": "completed occupancy grid on the evaluation grid, host numpy array",
                     "excluded": ["image decode", "ground-truth loading", "scoring",
                                  "visualisation", "prediction serialisation",
                                  "model construction and checkpoint loading"]},
        "machine": RC.machine(device),
        "runs": runs,
    }


def verify_driver(dataset: str, device: str, n_frames: int = 40) -> dict:
    """Does the frame-by-frame driver reproduce ``replay_segment`` exactly?"""
    pipe = StreamPipeline(dataset, device)
    seg = ST.build(dataset)[0]
    paths = list(seg.frame_paths[:n_frames])
    imgs = RP.load_images(paths)
    ref = RP.replay_segment(pipe.lingbot, imgs, 1, pipe.dev)
    clk = Clock(False)
    pipe.driver.reset()
    d0, c0, P0, K0 = pipe.driver.scale_block(imgs[:SV.N_INIT_FRAMES].unsqueeze(0), clk)
    D, C, P, K = list(d0), list(c0), list(P0), list(K0)
    for i in range(SV.N_INIT_FRAMES, n_frames):
        d, c, p, k = pipe.driver.update(imgs[i:i + 1].unsqueeze(0), clk)
        D.append(d); C.append(c); P.append(p); K.append(k)
    out = {}
    for name, got, want in (("depth", np.stack(D), ref["pred_depth"].astype(np.float32)),
                            ("depth_conf", np.stack(C), ref["pred_depth_conf"].astype(np.float32)),
                            ("pose_c2w", np.stack(P), ref["pred_pose_c2w"].astype(np.float64)),
                            ("K", np.stack(K), ref["pred_K"].astype(np.float64))):
        got = got.astype(np.float64)
        want = np.asarray(want, np.float64)
        out[name] = {"identical": bool(np.array_equal(got, want)),
                     "max_abs_difference": float(np.abs(got - want).max())}
    return {"dataset": dataset, "n_frames": n_frames, "fields": out,
            "all_identical": all(v["identical"] for v in out.values()),
            "note": ("the timed driver and the scored accuracy pass are the same computation; "
                     "if this is identical, runtime measured here belongs to the accuracy "
                     "reported from the mode-B maps")}


def verify_predictions(dataset: str, device: str, n_frames: int = 60) -> dict:
    """Do the LIVE-MoGe timed predictions match the cached-MoGe scored mode-B maps?"""
    pipe = StreamPipeline(dataset, device)
    seg = ST.build(dataset)[0]
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    r = run_stream(pipe, seg, 0, n_frames, by_sid, False, live_moge=True)
    cached_scale = json.load(open(os.path.join(
        SV.LOG_DIR, f"build_maps_{dataset}_modeB.json")))["segment_logs"]
    cs = next(l["scale_m_per_unit"] for l in cached_scale if l["segment"] == seg.name)
    rows = []
    for sid, g in r["_grids"].items():
        S_, F_, meta = BM.load_map(dataset, "B", sid)
        ref = pipe.rd.completion(pipe.net, S_, F_)
        inter, union = int((g & ref).sum()), int((g | ref).sum())
        rows.append({"sample_id": sid, "identical": bool(np.array_equal(g, ref)),
                     "agreement_iou": 100 * inter / max(union, 1),
                     "n_disagreeing": int((g != ref).sum()), "n_voxels": int(g.size)})
    return {"dataset": dataset, "n_frames": n_frames, "n_checked": len(rows), "rows": rows,
            "timed_startup_scale_live_moge": r["startup_scale_m_per_unit"],
            "scored_startup_scale_cached_moge": cs,
            "scale_relative_difference": abs(r["startup_scale_m_per_unit"] - cs) / cs,
            "all_identical": all(x["identical"] for x in rows),
            "min_agreement_iou": min(x["agreement_iou"] for x in rows) if rows else None,
            "note": ("the runtime pass calls MoGe-2 live while the accuracy pass read the "
                     "frozen MoGe frame cache; identical startup scales and identical grids "
                     "mean the two differ in nothing that reaches a number")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SV.DATASETS))
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--mode", default="e2e", choices=["e2e", "stages"])
    ap.add_argument("--operating-mode", default="stream", choices=["stream", "clipreset"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--streams", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--verify-driver", type=int, default=0)
    ap.add_argument("--verify-predictions", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SV.ensure_dirs()
    if a.verify_driver:
        r = verify_driver(a.dataset, a.device, a.verify_driver); tag = f"verify_driver_{a.dataset}"
    elif a.verify_predictions:
        r = verify_predictions(a.dataset, a.device, a.verify_predictions)
        tag = f"verify_stream_predictions_{a.dataset}"
    elif a.operating_mode == "clipreset":
        r = bench_clipreset(a.dataset, a.device, a.mode == "stages", a.reps, a.streams, a.frames)
        tag = f"clipreset_{a.dataset}_{a.mode}"
    else:
        r = bench(a.dataset, a.device, a.mode == "stages", a.reps, a.streams, a.frames)
        tag = f"stream_{a.dataset}_{a.mode}"
    out = a.out or os.path.join(SV.RUNTIME_DIR, f"{tag}.json")
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in r.items() if k not in ("runs", "rows")},
                     indent=1, default=float)[:3000])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
