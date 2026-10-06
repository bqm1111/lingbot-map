"""Per-anchor host loaders for B4 and B3: the original loaders called with ``device="cpu"``, plus the file lists used in errors.

B4: ``trainer.load_training_anchor``  -> flat [N] int64, feature [N, 1024] fp16, count [N] int32, S_packed / F_packed [32768] uint8,
    D_flat [M] int64, D_prob [M, 18] fp16 (delayed, TRAINING-ONLY), drive, t.
B3: pilot ``train.load_anchor``        -> flat, feature, count, current [N] bool, n_frames [N], trunk [N, 24] fp16,
    masks_packed [3, 32768] uint8 (F, S, O), D_flat, D_prob, D_trunk [M, 24] fp16, drive, t.
Delayed tiers come from the frozen pilot reliability record (tiers [1, 2]); both methods read the same record.
"""
from __future__ import annotations

import json
import os
from typing import List

from tools.delayed_semantic import config as PILOT, train_cache as TC
from tools.delayed_semantic.train import load_anchor as load_b3_anchor
from tools.semantic_field.scaling.anchor_store import AnchorKey, HostLoader
from tools.semantic_field.training.trainer import load_training_anchor as load_b4_anchor


def delayed_tiers() -> List[int]:
    gate = json.load(open(os.path.join(PILOT.Paths().reliability, "reliability.json")))["gate"]
    if not gate["pass"]:
        raise RuntimeError("the frozen reliability gate did not pass; delayed targets may not be used")
    return list(gate["delayed_loss_tiers"])


def b4_files(key: AnchorKey) -> List[str]:
    d, t = key; p = PILOT.Paths()
    return [p.g15.evidence_path(d, t), p.g15.input_path(d, t), p.target_path(d, t)]


def b3_files(key: AnchorKey) -> List[str]:
    d, t = key; p = PILOT.Paths()
    return [p.g15.evidence_path(d, t), p.provenance_path(d, t), TC.student_path(d, t), p.target_path(d, t), TC.query_path(d, t)]


def host_loader(method: str, tiers: List[int]) -> HostLoader:
    if method == "B4":
        return lambda key: load_b4_anchor(key[0], key[1], "cpu", tiers)
    if method == "B3":
        return lambda key: load_b3_anchor(key[0], key[1], "cpu", tiers, "")
    raise ValueError(f"unknown method {method!r}")


def files_of(method: str):
    return {"B4": b4_files, "B3": b3_files}[method]


__all__ = ["delayed_tiers", "b4_files", "b3_files", "host_loader", "files_of"]
