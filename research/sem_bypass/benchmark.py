#!/usr/bin/env python
"""Gate 2: online batch-one latency of the five inference pipelines.

Pipelines
---------
``lingbot_only``        frozen LingBot geometry, no semantics.
``maskclip_only``       frozen MaskCLIP dense encoder, no geometry.
``lingbot_maskclip``    geometry + MaskCLIP run per frame (what SemBypass replaces).
``lingbot_sembypass``   geometry + the sidecar on already-computed encoder tokens.
``sidecar_only``        the incremental sidecar cost alone.

Protocol: batch size 1, identical input resolution, real inference precision
(bf16 autocast for LingBot exactly as in caching, fp16 for MaskCLIP as in the teacher
cacher), >= 50 warm-up iterations, >= 300 measured frames, ``torch.cuda.synchronize()``
around every timed region, and **no disk I/O inside the timed section** -- every frame is
preloaded to GPU first.

Streaming note: LingBot is causal with a KV cache, so per-frame latency depends on cache
occupancy. Frames are pushed through one at a time in a single streaming session, which
is the real online cost; the cache is reset only between pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from typing import Callable, Dict, List, Optional

import torch

from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import list_scene_images
from lingbot_map.utils.load_fn import load_and_preprocess_images

from research.sem_bypass.encoder_features import FrozenLingBotEncoder
from research.sem_bypass.model import build_sidecar

logger = logging.getLogger("sem_bypass.bench")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_stream(step: Callable[[torch.Tensor], None], frames: List[torch.Tensor],
                warmup: int, measure: int, reset: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Time ``step`` once per frame; return latency statistics in milliseconds."""
    if reset:
        reset()
    for i in range(warmup):
        step(frames[i % len(frames)])
    _sync()
    if reset:
        reset()
    lat: List[float] = []
    for i in range(measure):
        f = frames[i % len(frames)]
        _sync()
        t0 = time.perf_counter()
        step(f)
        _sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat_sorted = sorted(lat)
    return {
        "median_ms": statistics.median(lat),
        "mean_ms": statistics.fmean(lat),
        "p90_ms": lat_sorted[int(0.90 * (len(lat_sorted) - 1))],
        "p99_ms": lat_sorted[int(0.99 * (len(lat_sorted) - 1))],
        "fps": 1000.0 / statistics.median(lat),
        "frames": len(lat),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="research/sem_bypass/configs/gate1.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measure", type=int, default=300)
    ap.add_argument("--pool", type=int, default=64, help="distinct frames preloaded to GPU")
    ap.add_argument("--output-dir", default="research/sem_bypass/outputs/gate2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(a.config, [])
    set_seed(a.seed)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)

    # -- preload frames (outside every timed region) ------------------------ #
    scene = [s for s in expand_scenes(cfg.scenes) if s.role == "eval"][0]
    paths = list_scene_images(scene)[: a.pool]
    images = load_and_preprocess_images(paths, mode="crop", image_size=cfg.lingbot.image_size,
                                        patch_size=cfg.lingbot.patch_size)
    H, W = images.shape[-2:]
    frames = [images[i: i + 1].to(dev).contiguous() for i in range(images.shape[0])]
    logger.info("preloaded %d frames at %dx%d on %s", len(frames), W, H, dev)

    results: Dict[str, Dict] = {}
    params: Dict[str, int] = {}

    # -- frozen models ------------------------------------------------------ #
    lcfg = cfg.lingbot
    lcfg.device = str(dev)
    lingbot = FrozenLingBotEncoder(lcfg, capture_tokens=True)
    lingbot.assert_frozen()
    params["lingbot"] = sum(p.numel() for p in lingbot.model.parameters())

    sidecar = build_sidecar(out_dim=cfg.teacher.pca_dim).to(dev).eval()
    for p in sidecar.parameters():
        p.requires_grad_(False)
    params["sidecar"] = sum(p.numel() for p in sidecar.parameters())

    from semantic.dense_clip import DenseCLIP
    clip = DenseCLIP(model_name=cfg.teacher.model_name, pretrained=cfg.teacher.pretrained,
                     device=str(dev), variant=cfg.teacher.variant)
    params["maskclip"] = sum(p.numel() for p in clip.model.parameters())
    for p in clip.model.parameters():
        p.requires_grad_(False)
    scale = cfg.teacher.input_scale
    th, tw = int(round(H * scale)), int(round(W * scale))
    ps = clip.patch_size
    th, tw = (th // ps) * ps, (tw // ps) * ps
    logger.info("MaskCLIP input %dx%d (scale %.1f, patch %d)", tw, th, scale, ps)

    amp = getattr(torch, cfg.lingbot.autocast_dtype)

    # -- pipeline steps ----------------------------------------------------- #
    def reset_lingbot():
        lingbot.model.clean_kv_cache()

    @torch.no_grad()
    def lingbot_step(f: torch.Tensor, capture: bool = False):
        lingbot.capture_tokens = capture
        with torch.amp.autocast("cuda", dtype=amp):
            return lingbot.model.inference_streaming(f, num_scale_frames=1, keyframe_interval=1)

    @torch.no_grad()
    def maskclip_step(f: torch.Tensor):
        x = torch.nn.functional.interpolate(f, size=(th, tw), mode="bilinear", align_corners=False)
        return clip.encode_dense(x)

    tok = torch.randn(1, 407, 1, 1024, device=dev)

    @torch.no_grad()
    def sidecar_step(_f=None):
        return sidecar(tok)

    logger.info("== lingbot_only ==")
    results["lingbot_only"] = time_stream(lambda f: lingbot_step(f, False), frames,
                                          a.warmup, a.measure, reset_lingbot)
    logger.info("== maskclip_only ==")
    results["maskclip_only"] = time_stream(maskclip_step, frames, a.warmup, a.measure)
    logger.info("== sidecar_only ==")
    results["sidecar_only"] = time_stream(sidecar_step, frames, max(a.warmup, 200), max(a.measure, 1000))

    @torch.no_grad()
    def combo_maskclip(f: torch.Tensor):
        lingbot_step(f, False)
        maskclip_step(f)

    @torch.no_grad()
    def combo_sembypass(f: torch.Tensor):
        lingbot.capture_tokens = True
        with torch.amp.autocast("cuda", dtype=amp):
            lingbot.model.inference_streaming(f, num_scale_frames=1, keyframe_interval=1)
        t = lingbot._buffer[-1] if lingbot._buffer else tok
        sidecar(t.to(dev).float() if t.device.type != "cuda" else t.float())
        lingbot._buffer = []

    logger.info("== lingbot_maskclip ==")
    results["lingbot_maskclip"] = time_stream(combo_maskclip, frames, a.warmup, a.measure, reset_lingbot)
    logger.info("== lingbot_sembypass ==")
    torch.cuda.reset_peak_memory_stats(dev)
    results["lingbot_sembypass"] = time_stream(combo_sembypass, frames, a.warmup, a.measure, reset_lingbot)

    # -- derived quantities ------------------------------------------------- #
    g = results["lingbot_only"]["median_ms"]
    sb = results["lingbot_sembypass"]["median_ms"]
    mc = results["lingbot_maskclip"]["median_ms"]
    inc_sb = sb - g
    inc_mc = results["maskclip_only"]["median_ms"]
    derived = {
        "sembypass_added_latency_ms": inc_sb,
        "sembypass_added_latency_pct": 100.0 * inc_sb / g,
        "maskclip_added_latency_ms": mc - g,
        "maskclip_vs_sidecar_speed_ratio": inc_mc / max(results["sidecar_only"]["median_ms"], 1e-9),
        "end_to_end_speedup_pct": 100.0 * (mc - sb) / mc,
        "end_to_end_speedup_x": mc / sb,
    }
    gates = {
        "sembypass_under_5pct_latency": derived["sembypass_added_latency_pct"] < 5.0,
        "maskclip_at_least_3x_slower_than_sidecar": derived["maskclip_vs_sidecar_speed_ratio"] >= 3.0,
        "measurable_end_to_end_advantage": (mc - sb) > 0,
    }
    gates["GATE2_PASS"] = all(gates.values())
    gates["strong"] = bool(derived["maskclip_vs_sidecar_speed_ratio"] >= 5.0
                           and derived["end_to_end_speedup_pct"] >= 15.0)

    payload = {
        "provenance": collect_provenance(cfg, tool="sem_bypass.benchmark"),
        "settings": {"device": str(dev), "gpu": torch.cuda.get_device_name(dev),
                     "warmup": a.warmup, "measure": a.measure, "batch_size": 1,
                     "lingbot_input_hw": [H, W], "maskclip_input_hw": [th, tw],
                     "lingbot_precision": cfg.lingbot.autocast_dtype,
                     "maskclip_precision": "float16", "frames_preloaded": len(frames)},
        "parameters": params,
        "parameter_memory_mb": {k: v * 2 / 2 ** 20 for k, v in params.items()},
        "latency": results, "derived": derived, "gates": gates,
        "peak_gpu_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
    }
    out = os.path.join(a.output_dir, "latency.json")
    write_json(out, payload)
    for k, v in results.items():
        logger.info("%-20s median %7.2f ms  mean %7.2f  p90 %7.2f  fps %6.1f",
                    k, v["median_ms"], v["mean_ms"], v["p90_ms"], v["fps"])
    logger.info("derived: %s", json.dumps({k: round(v, 3) for k, v in derived.items()}))
    logger.info("gates:   %s", json.dumps(gates))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
