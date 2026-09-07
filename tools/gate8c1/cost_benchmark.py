#!/usr/bin/env python
"""Gate 8C-1: per-frame inference cost, ours against the released OccAny checkpoint.

Every number here is wall-clock on **this** machine (RTX PRO 6000 Blackwell, sm_120), one
GPU, batch 1, fp32 heads / bf16 aggregator exactly as deployed. Nothing is extrapolated
from FLOP counts.

The honest difficulty is that our evaluation runs from cached frozen-model outputs, so a
naive timing of ``eval_target.py`` measures only the mapper and the 0.99 M completion
network and silently omits the three large frozen models that produced the cache. This
script times the frozen stack too, stage by stage, so the totals are comparable:

    LingBot-Map   per frame, streaming (the deployed path, KV cache warm)
    MoGe-2        first five frames of a sequence only -- amortised over the sequence
    Trident-H     per frame, open-vocabulary segmentation
    mapper.step   per frame, ray casting and log-odds integration (no parameters)
    query + net   per *output*: dense export onto the benchmark grid + CompletionUNet

OccAny's cost is read from its own run on this machine: its loader emits one prediction per
sample and its tqdm rate is seconds per sample, so its per-output cost is directly
comparable to ours. Its five input frames are consumed inside that same second.

    python tools/gate8c1/cost_benchmark.py --dataset semantickitti --device cuda:2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, vocab                                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import to_eval_grid                                          # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402

OCCANY_LOGS = {"semantickitti": "/media/SSD1/MINH_DATASETS/occany_out/logs_kitti",
               "occ3d": "/media/SSD1/MINH_DATASETS/occany_out/logs"}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n, warmup=2):
    """Median and IQR of ``n`` timed calls, in milliseconds."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p10_ms": ts[max(0, int(.1 * len(ts)) - 1)],
            "p90_ms": ts[min(len(ts) - 1, int(.9 * len(ts)))], "n": n}


def occany_seconds_per_sample(ds):
    """Seconds per emitted prediction, scraped from OccAny's own tqdm output."""
    d = OCCANY_LOGS.get(ds)
    if not d or not os.path.isdir(d):
        return None
    rates = []
    for f in sorted(glob.glob(os.path.join(d, "extract_pid*.log"))):
        txt = open(f, errors="ignore").read()
        for m in re.finditer(r"(\d+\.\d+)s/it", txt):
            rates.append(float(m.group(1)))
        for m in re.finditer(r"(\d+\.\d+)it/s", txt):
            v = float(m.group(1))
            if v > 0:
                rates.append(1.0 / v)
    if not rates:
        return None
    rates = rates[len(rates) // 4:]                     # drop the warm-up quarter
    return {"median_s_per_sample": statistics.median(rates),
            "p10_s": sorted(rates)[int(.1 * len(rates))],
            "p90_s": sorted(rates)[int(.9 * len(rates))],
            "n_observations": len(rates)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="semantickitti",
                    choices=["semantickitti", "occ3d"])
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--skip-frozen", action="store_true",
                    help="time only the mapper and the completion network")
    a = ap.parse_args()

    ds = a.dataset
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    fz = json.load(open(MANIFEST))["seeds"]["0"]
    tau = float(fz["occupancy_threshold"])
    C = len(vocab.load(ds))
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    into = V8.into_matrix(ds)
    comp = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    seg = S.segments(ds, REPO_ROOT)[0]
    i = sorted(seg.anchors)[len(seg.anchors) // 2]
    feed = CachedFeed(seg, dev)
    with np.load(S.scale_path(ds, seg.name)) as z:
        scale = float(np.exp(np.median(z["log_s"][:5])))

    out = {"dataset": ds, "device": torch.cuda.get_device_name(dev),
           "params": {"completion_unet_trained": sum(p.numel()
                                                     for p in comp.net.parameters())},
           "stages_ms": {}, "notes": {}}
    print(f"device: {out['device']}\ndataset: {ds}\n", flush=True)

    # ---- the part we trained --------------------------------------------------------
    m, _ = window_map(feed, seg, i, PAST5_OFFSETS, scale, into, C, dev)
    f = seg.frames[i]
    P = feed.pose[i].copy()
    P[:3, 3] *= scale
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))

    def do_query():
        q = m.query(MAP, Tgw)
        q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        return q

    out["stages_ms"]["map_query_dense_export"] = timeit(do_query, a.reps)
    q = do_query()

    def do_net():
        fin, _ = comp.raw(q, MAP)
        return to_eval_grid(fin, q["logodds"], ds)[0] >= tau

    out["stages_ms"]["completion_unet_forward"] = timeit(do_net, a.reps)

    # ---- the mapper: integrating one frame -------------------------------------------
    # a fresh map per rep would dominate the timing; integrating one more frame into the
    # existing map is exactly what the streaming system does at run time
    j = min(i + 1, len(seg) - 1)

    def do_integrate():
        m.step(feed.frame(j))

    out["stages_ms"]["mapper_step_one_frame"] = timeit(do_integrate, max(4, a.reps // 2))
    del m
    torch.cuda.empty_cache()

    # ---- the frozen stack -------------------------------------------------------------
    if not a.skip_frozen:
        try:
            from gates.gate7b import replay
            model, _ck = replay.build_model(dev)
            imgs = replay.load_images([seg.frames[k].path for k in range(i - 4, i + 1)])

            def do_lingbot():
                replay.replay_segment(model, imgs, 1, dev)

            r = timeit(do_lingbot, max(3, a.reps // 4), warmup=1)
            r = {k: (v / 5 if k.endswith("_ms") else v) for k, v in r.items()}
            out["stages_ms"]["lingbot_map_per_frame"] = r
            out["notes"]["lingbot"] = ("timed over a 5-frame streaming replay and divided "
                                       "by 5; keyframe_interval=1")
            del model
            torch.cuda.empty_cache()
            print("  lingbot ok", flush=True)
        except Exception as e:                                     # noqa: BLE001
            out["notes"]["lingbot"] = f"NOT TIMED: {type(e).__name__}: {e}"
            print("  lingbot failed:", e, flush=True)

        try:
            from moge_gauge.calibrated import CalibratedMoGe
            from gates.scale_gate.config import load_config
            g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
            moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo)
            from PIL import Image
            rgb = (np.asarray(Image.open(seg.frames[i].path).convert("RGB"),
                                  dtype=np.float32) / 255.0).transpose(2, 0, 1)
            out["stages_ms"]["moge2_per_frame"] = timeit(
                lambda: moge.infer_calibrated(rgb, fov_x_deg=70.0), max(4, a.reps // 2))
            out["notes"]["moge"] = ("runs on the first 5 frames of a sequence only, then "
                                    "ScaleState freezes; amortised cost per frame over an "
                                    "N-frame sequence is 5/N of this")
            del moge
            torch.cuda.empty_cache()
            print("  moge ok", flush=True)
        except Exception as e:                                     # noqa: BLE001
            out["notes"]["moge"] = f"NOT TIMED: {type(e).__name__}: {e}"
            print("  moge failed:", e, flush=True)

        out["notes"]["trident"] = ("Trident-H runs in its own conda environment "
                                   "(tools/gate6/cache_semantics.py) and is not importable "
                                   "here; see the cache build log for its rate")

    oc = occany_seconds_per_sample(ds)
    if oc:
        out["occany_per_output_sample"] = oc
        out["notes"]["occany"] = (
            "OccAny emits one prediction per sample and consumes its five frames inside "
            "that sample, so seconds/sample is its per-output cost. Measured from its own "
            "run on this machine (MUSt3R + generative novel-view rendering, "
            "--gen --batch_gen_view 12).")

    with open(os.path.join(ART, f"cost_benchmark_{ds}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{'stage':34s} {'median ms':>10} {'p10':>9} {'p90':>9}")
    for k, v in out["stages_ms"].items():
        print(f"{k:34s} {v['median_ms']:10.1f} {v['p10_ms']:9.1f} {v['p90_ms']:9.1f}")
    if oc:
        print(f"\nOccAny: {oc['median_s_per_sample']:.2f} s per emitted prediction "
              f"(p10 {oc['p10_s']:.2f}, p90 {oc['p90_s']:.2f}, "
              f"n={oc['n_observations']})")
    print(f"\nwrote artifacts/gate8c1/cost_benchmark_{ds}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
