"""MoGe-B depth for the frames our frozen cache lacks, with the exact gate-7B recipe.

The frozen cache (``tools/gate7b/cache_moge_b.py``) covers every frame of our streams. OccAny's forward
windows reach up to eight samples past the target, which on the 109 Occ3D scenes with 41 samples needs
sample 40 -- outside our 40-frame stream. Those frames are computed here, once, with the same frozen
model revision, the same processed lattice and the same calibrated horizontal FOV rule
(``fx_native * W_proc / W_native``), into a separate root so the frozen cache is never written to.
The recipe is first re-run on frames the frozen cache already holds and compared.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from core.datasets import sources as S
from core.mapping import depth as D7
from gates.gate7b import replay
from tools.occany_comparison import config as C
from tools.occany_comparison.samples import Sample

NATIVE_WIDTH = {"occ3d": 1600}
MOGE_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"


def resolve_moge_path(dataset: str, key: str) -> str:
    """The frozen gate-7B frame if it exists, otherwise this experiment's extra frame."""
    p = S.moge_path(dataset, key)
    return p if os.path.exists(p) else os.path.join(C.MOGE_EXTRA_ROOT, dataset, f"{key}.npz")


def missing_frames(samples: List[Sample]) -> Dict[str, Tuple[Sample, int]]:
    todo: Dict[str, Tuple[Sample, int]] = {}
    for s in samples:
        for p in C.PROTOCOLS.values():
            if s.protocol_ok(p):
                for j in s.window(p):
                    if not os.path.exists(S.moge_path(s.dataset, s.frame_keys[j])):
                        todo.setdefault(s.frame_keys[j], (s, j))
    return todo


def _fov(sample: Sample, j: int, proc_width: int) -> float:
    from moge_gauge.calibration import calibrated_fov_x_deg
    return calibrated_fov_x_deg(float(sample.K_native[j, 0, 0]) * proc_width / NATIVE_WIDTH[sample.dataset], proc_width)


def _infer(moge, sample: Sample, j: int) -> Dict[str, np.ndarray]:
    rgb = replay.load_images([sample.frame_paths[j]])[0].numpy()
    fov = _fov(sample, j, rgb.shape[-1])
    o = moge.infer_calibrated(rgb, fov_x_deg=fov)
    mk = np.asarray(o.mask, bool)
    return {"moge_depth": np.asarray(o.depth_z, np.float32).astype(np.float16), "moge_mask": np.packbits(mk.reshape(-1)),
            "mask_shape": np.asarray(mk.shape, np.int32), "fov_x_deg": np.float32(fov), "fov_out_deg": np.float32(o.fov_x_deg)}


def build_missing(dataset: str, samples: List[Sample], device: str, n_reproduce: int = 3) -> Dict:
    from gates.scale_gate.config import load_config
    from moge_gauge.calibrated import CalibratedMoGe
    todo = missing_frames(samples)
    record = {"dataset": dataset, "n_missing": len(todo), "root": os.path.join(C.MOGE_EXTRA_ROOT, dataset)}
    if not todo:
        return record
    dev = torch.device(device); torch.cuda.set_device(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo, revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == MOGE_REVISION

    # the recipe must reproduce frames the frozen cache already has
    repro = []
    for s in [s for s in samples if s.target_pos == 0][:n_reproduce]:
        with np.load(S.moge_path(dataset, s.frame_keys[0])) as z:
            ref = {k: z[k] for k in ("moge_depth", "moge_mask", "fov_x_deg")}
        new = _infer(moge, s, 0)
        d_ref, d_new = ref["moge_depth"].astype(np.float32), new["moge_depth"].astype(np.float32)
        fin = np.isfinite(d_ref) & np.isfinite(d_new)
        repro.append({"key": s.frame_keys[0], "fov_abs_diff": float(abs(ref["fov_x_deg"] - new["fov_x_deg"])),
                      "mask_equal": bool(np.array_equal(ref["moge_mask"], new["moge_mask"])),
                      "depth_max_rel_diff": float(np.max(np.abs(d_ref[fin] - d_new[fin]) / np.maximum(d_ref[fin], 1e-3)))})
    record["reproduction_on_frozen_frames"] = repro

    out_dir = record["root"]; os.makedirs(out_dir, exist_ok=True)
    t0, written = time.time(), 0
    for key, (s, j) in sorted(todo.items()):
        op = os.path.join(out_dir, f"{key}.npz")
        if os.path.exists(op):
            continue
        tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **_infer(moge, s, j)); os.replace(tmp, op); written += 1
    record.update({"n_written": written, "seconds": time.time() - t0, "moge_provenance": prov,
                   "variant": "G51-B (full lattice, calibrated FOV), same as tools/gate7b/cache_moge_b.py"})
    os.makedirs(C.LOG_DIR, exist_ok=True)
    json.dump(record, open(os.path.join(C.LOG_DIR, f"moge_extra_{dataset}.json"), "w"), indent=1, default=str)
    return record


__all__ = ["resolve_moge_path", "missing_frames", "build_missing"]
