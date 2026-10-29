"""Shared, recorded setup for the runtime comparison: the anchor sample, and the machine.

The anchor sample is written to disk once and read by BOTH benchmarks, including the OccAny
one, which runs under a different interpreter. Neither benchmark may choose its own anchors.

Selection rule (recorded, not tuned): the canonical anchor list of the dataset is cut into
``n`` contiguous strata and one anchor is drawn uniformly from each with ``numpy.random.
default_rng(seed)``. On SemanticKITTI there is a single sequence, so the strata spread the
sample over the drive; on Occ3D the strata fall across scenes in scene order.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "artifacts", "method_vs_data", "runtime")
ARTIFACT_DIR = os.path.abspath(ARTIFACT_DIR)
N_ANCHORS = 50
SEED = 0
WARMUP = 3
PASSES = 3
#: the GPU every timed configuration must use, so no two numbers come from different silicon
BENCH_DEVICE = "cuda:3"


def anchor_file(dataset: str) -> str:
    return os.path.join(ARTIFACT_DIR, f"anchors_{dataset}.json")


def select(sample_ids, n: int = N_ANCHORS, seed: int = SEED):
    ids = list(sample_ids)
    assert len(ids) >= n, f"{len(ids)} anchors available, {n} requested"
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(ids), n + 1).astype(int)
    return [ids[int(rng.integers(a, b))] for a, b in zip(edges[:-1], edges[1:])]


def write_anchors(dataset: str, sample_ids, n: int = N_ANCHORS, seed: int = SEED) -> dict:
    picked = select(sample_ids, n, seed)
    rec = {"dataset": dataset, "n_population": len(list(sample_ids)), "n_selected": len(picked),
           "seed": seed, "rule": "one uniform draw from each of n contiguous strata of the "
                                 "canonical anchor list", "anchors": picked}
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    json.dump(rec, open(anchor_file(dataset), "w"), indent=1)
    return rec


def read_anchors(dataset: str) -> list:
    return json.load(open(anchor_file(dataset)))["anchors"]


def machine(device: str) -> dict:
    """Everything a reader needs to know the numbers came from one machine in one state."""
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:                                           # pragma: no cover
            return f"unavailable: {e}"
    import torch
    i = int(device.split(":")[-1]) if ":" in device else 0
    p = torch.cuda.get_device_properties(i) if torch.cuda.is_available() else None
    return {
        "gpu_index": i,
        "gpu_name": p.name if p else None,
        "gpu_total_memory_bytes": int(p.total_memory) if p else None,
        "gpu_capability": f"{p.major}.{p.minor}" if p else None,
        "nvidia_smi_during_setup": sh("nvidia-smi",
                                      "--query-gpu=index,utilization.gpu,memory.used",
                                      "--format=csv,noheader"),
        "cpu_model": next((l.split(":", 1)[1].strip()
                           for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
        "cpu_count": os.cpu_count(),
        "loadavg": os.getloadavg(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    }


def summarise(samples) -> dict:
    """mean / median / p95 / min / max in milliseconds, from a list of seconds."""
    a = np.asarray(sorted(samples), np.float64) * 1e3
    return {"n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "max_ms": float(a.max()), "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "predictions_per_second": float(1.0 / (a.mean() / 1e3))}


__all__ = ["ARTIFACT_DIR", "N_ANCHORS", "SEED", "WARMUP", "PASSES", "BENCH_DEVICE",
           "anchor_file", "select", "write_anchors", "read_anchors", "machine", "summarise"]
