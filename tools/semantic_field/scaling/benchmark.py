"""Throughput of the disk-backed training path on real files (step B).

Measures, on distinct current-pool anchors (never a handful repeated):
  per-anchor host load (np.load + decode) cold and warm -- "cold" = the files' page cache dropped with
  posix_fadvise(DONTNEED) (unprivileged, only our own reads are affected), "warm" = read again immediately;
  host->device transfer; full B4 and B3 training steps (forward + backward + optimizer) with the host cache disabled,
  so every step pays its reads; peak host RSS and GPU memory.
The models are disposable; no score is computed.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.benchmark --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from typing import Dict, List

import numpy as np
import torch

from tools.semantic_field.scaling import loaders
from tools.semantic_field.scaling.anchor_store import tensor_bytes
from tools.semantic_field.scaling.config import ARTIFACT_DIR, CheckpointConfig, LoaderConfig
from tools.semantic_field.scaling.run import train_on_anchors
from tools.semantic_field.scaling.source_pool import current_pool


def drop_page_cache(files: List[str]) -> None:
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def time_host_loads(method: str, keys, device) -> Dict:
    tiers = loaders.delayed_tiers(); load = loaders.host_loader(method, tiers); files = loaders.files_of(method)
    drop_page_cache([f for k in keys for f in files(k)])
    rows = []
    for k in keys:
        t0 = time.perf_counter(); a = load(k); t_cold = time.perf_counter() - t0
        t0 = time.perf_counter(); load(k); t_warm = time.perf_counter() - t0
        torch.cuda.synchronize(device); t0 = time.perf_counter()
        _ = {n: v.to(device) for n, v in a.items() if isinstance(v, torch.Tensor)}; torch.cuda.synchronize(device)
        rows.append((t_cold, t_warm, time.perf_counter() - t0, tensor_bytes(a), sum(os.path.getsize(f) for f in files(k))))
    r = np.asarray(rows)
    q = lambda c: {"mean": float(r[:, c].mean()), "p50": float(np.median(r[:, c])), "p90": float(np.percentile(r[:, c], 90))}
    return {"anchors": len(keys), "cold_load_s": q(0), "warm_load_s": q(1), "host_to_device_s": q(2),
            "decoded_mb_mean": float(r[:, 3].mean() / 1e6), "file_mb_mean": float(r[:, 4].mean() / 1e6)}


def time_training(method: str, keys, device, steps: int, tmp: str, seed: int = 0, drop_cache: bool = True) -> Dict:
    files = loaders.files_of(method)
    if drop_cache:
        drop_page_cache([f for k in keys for f in files(k)])
    t0 = time.time()
    run = train_on_anchors(method, keys, seed, steps, device, os.path.join(tmp, f"bench_{method}_seed{seed}_{os.getpid()}.pt"), LoaderConfig(0),
                           CheckpointConfig(resume_every_steps=10 ** 9), tag=f"bench_{method}_seed{seed}")
    wall = time.time() - t0
    return {"steps": steps, "distinct_anchors_available": len(keys), "anchor_reads": run.anchors.stats.misses,
            "seconds_in_loop": run.state.seconds, "seconds_per_step": run.state.seconds / steps, "wall_incl_setup_s": wall,
            "peak_gpu_bytes": run.peak_gpu_bytes, "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def main() -> None:
    """Default: load timings + one B4 and one B3 training benchmark. ``--only B4|B3`` runs a single training benchmark
    (used to launch several processes on one GPU, each with its own ``--seed`` and ``--no-drop-cache``)."""
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--b4-steps", type=int, default=300); ap.add_argument("--b3-steps", type=int, default=300)
    ap.add_argument("--only", choices=["B4", "B3"], default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-drop-cache", action="store_true"); ap.add_argument("--out", default="throughput.json")
    a = ap.parse_args()
    pool = list(current_pool().anchors)
    out_dir = os.path.join(ARTIFACT_DIR, "benchmark"); tmp = os.path.join(out_dir, "tmp"); os.makedirs(tmp, exist_ok=True)
    out: Dict = {}
    if a.only is None:
        rng = np.random.default_rng(0); load_keys = [pool[i] for i in rng.choice(len(pool), 120, replace=False)]
        out = {"host_load_B4": time_host_loads("B4", load_keys[:60], a.device), "host_load_B3": time_host_loads("B3", load_keys[60:], a.device)}
    for method in ([a.only] if a.only else ["B4", "B3"]):
        steps = a.b4_steps if method == "B4" else a.b3_steps
        out[f"train_{method}"] = time_training(method, pool, a.device, steps, tmp, a.seed, not a.no_drop_cache)
    for f in os.listdir(tmp):
        if f.endswith(f"_{os.getpid()}.pt"):
            os.remove(os.path.join(tmp, f))
    out["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    json.dump(out, open(os.path.join(out_dir, a.out), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
