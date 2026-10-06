"""Frozen-geometry features for training (geometry seed 0), one no-grad bf16 forward per anchor.

    student/<drive>/<t>.npz   student-side (H_t only): S, F (hard map) and O (frozen occupancy) packbits,
                              trunk_E [N, 24] fp16 = CompletionUNet decoder feature at the evidence voxels
    query/<drive>/<t>.npz     training-only: trunk_D [M, 24] fp16 at the delayed target voxels (same order as
                              the target file). Only the *location* comes from the target; the feature is the
                              student's own frozen geometry at t.

Evaluation never reads this cache: it recomputes the forward on the full grid.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.semantic_completion.evidence import build as G15B
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk, state_digest

CACHE_ROOT = os.path.join(C.DATA_ROOT, "train_cache")


def student_path(drive: str, t: int) -> str:
    return os.path.join(CACHE_ROOT, "student", drive, f"{t:05d}.npz")


def query_path(drive: str, t: int, variant: str = "") -> str:
    """variant '' = the predeclared 10 s targets; '4s' = addendum A sensitivity targets."""
    return os.path.join(CACHE_ROOT, "query" + (f"_{variant}" if variant else ""), drive, f"{t:05d}.npz")


def target_paths(variant: str = "") -> C.Paths:
    return C.Paths(targets_root=os.path.join(C.DATA_ROOT, "delayed_targets" + (f"_{variant}" if variant else "")))


VARIANT_CONFIG = {"": C.TargetConfig(), "4s": C.TargetConfig(horizon_frames=8, horizon_s=4.0)}


def _save(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays); os.replace(tmp, path)


def build_drive(drive: str, device: str, shard: int = 0, shards: int = 1, variants=("", "4s"), paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15); digest = state_digest(net)
    ids = G15B.anchor_ids(drive, paths.g15)[shard::shards]; t0 = time.time(); n = 0
    with FileAudit(patterns=C.TRAIN_FORBIDDEN, strict=True) as audit:
        for t in ids:
            if os.path.exists(student_path(drive, t)) and all(os.path.exists(query_path(drive, t, v)) for v in variants):
                continue
            S, F_ = load_hard_map(drive, t, dev, paths.g15)
            occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_))
            tr = trunk[0].reshape(C.ModelConfig().trunk_channels, -1)
            with np.load(paths.provenance_path(drive, t)) as z:
                flat_E = torch.from_numpy(z["flat_index"].astype(np.int64)).to(dev)
            _save(student_path(drive, t), S=np.packbits(S.cpu().numpy()), F=np.packbits(F_.cpu().numpy()), O=np.packbits(occ[0].cpu().numpy()),
                  trunk_E=tr[:, flat_E].T.half().cpu().numpy(), t=np.int64(t))
            for v in variants:
                with np.load(target_paths(v).target_path(drive, t)) as z:
                    flat_D = torch.from_numpy(z["flat_index"].astype(np.int64)).to(dev)
                _save(query_path(drive, t, v), trunk_D=tr[:, flat_D].T.half().cpu().numpy(), t=np.int64(t))
            n += 1
    assert state_digest(net) == digest
    return {"drive": drive, "shard": shard, "written": n, "seconds": time.time() - t0, "geometry_digest": digest,
            "firewall": {"opened": len(audit.opened), "violations": audit.violations}}


__all__ = ["CACHE_ROOT", "student_path", "query_path", "build_drive"]
