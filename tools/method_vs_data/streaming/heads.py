"""Completion heads over a shared map cache, plus the map-only arm.

Every arm in this phase is a function of the SAME cached ``(S, F)``: the completion heads
are one ``CompletionUNet`` forward on ``[S, 0, F]`` followed by the frozen read-out, and the
map-only arm is the read-out with the head removed. The read-out, the grid, the channel
order, the bf16 autocast and the occupancy decision ``apply_residual(0, residual) > 0`` are
the frozen baseline's, imported rather than restated.

The KITTI-360 arm is additionally loaded through ``load_frozen_geometry``, which sha256-checks
the checkpoint against ``artifacts/baseline_reconstruction/code_freeze.json`` before use.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from core.datasets import grids as CG
from core.model.net import CompletionUNet, apply_residual, load_checkpoint
from tools.method_vs_data.streaming import config as SV
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_head(source: str, seed: int, device) -> CompletionUNet:
    path = SV.checkpoint_path(source, seed)
    if source == "k360":
        from tools.semantic_completion.frozen_geometry import load_frozen_geometry
        net = load_frozen_geometry(seed, device)          # sha256-checked against the freeze
    else:
        net = load_checkpoint(path, device).net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    assert net.enc1[0].in_channels == 3 and net.n_params() == 967_322, (source, seed)
    return net


class Readout:
    """The frozen prediction-grid -> evaluation-grid read-out of one benchmark."""

    def __init__(self, dataset: str, device):
        self.dataset = dataset
        self.dev = torch.device(device)
        self.fine = GridSpec.of(CG.PREDICTION_GRID[dataset])
        self.native = GridSpec.of(CG.EVAL_GRID[dataset])
        self.ratio = int(round(self.native.voxel_size / self.fine.voxel_size))

    def inputs(self, S_: np.ndarray, F_: np.ndarray) -> torch.Tensor:
        s = torch.from_numpy(S_).to(self.dev).reshape(self.fine.dims)
        f = torch.from_numpy(F_).to(self.dev).reshape(self.fine.dims)
        return torch.stack([s, torch.zeros_like(s), f]).float()[None]

    @torch.no_grad()
    def completion(self, net: CompletionUNet, S_: np.ndarray, F_: np.ndarray) -> np.ndarray:
        x = self.inputs(S_, F_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res, _ = net(x)
        r = res[:, 0].float()
        occ = apply_residual(torch.zeros_like(r), r).reshape(1, -1) > 0
        return reduce_any(occ[0], self.fine, self.ratio).cpu().numpy()

    @torch.no_grad()
    def map_only(self, S_: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(S_).to(self.dev).reshape(self.fine.dims).reshape(-1)
        return reduce_any(s, self.fine, self.ratio).cpu().numpy()


def checkpoint_manifest() -> Dict[str, Dict]:
    out = {}
    for src in SV.SOURCES:
        for seed in SV.SEEDS:
            p = SV.checkpoint_path(src, seed)
            out[f"{src}_seed{seed}"] = {
                "path": os.path.relpath(p, SV.REPO_ROOT),
                "sha256": sha256_file(p) if os.path.exists(p) else None,
                "bytes": os.path.getsize(p) if os.path.exists(p) else None,
                "exploratory": src in SV.EXPLORATORY}
    return out


__all__ = ["load_head", "Readout", "checkpoint_manifest", "sha256_file"]
