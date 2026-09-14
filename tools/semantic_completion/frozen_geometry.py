"""The frozen Current Geometry-Only Baseline, loaded read-only, with the two quantities Gate 15 needs.

    occupancy(x) -> [N_VOXELS] bool   exactly the baseline decision: apply_residual(0, residual) > 0
    trunk(x)     -> [B, 24, X, Y, Z]  the decoder feature d1 the occupancy head reads

Both are produced by ONE forward under ``torch.no_grad`` and bf16 autocast (the baseline's inference
convention), so the semantic head can neither change nor receive gradient from the geometry network.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Tuple

import numpy as np
import torch

from core.model.net import CompletionUNet, apply_residual, load_checkpoint
from tools.semantic_completion import config as C


def state_digest(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_geometry(seed: int, device, paths: C.Paths = C.Paths()) -> CompletionUNet:
    """The seed's final checkpoint, sha256-checked against the baseline's code_freeze.json, eval mode, no grad."""
    path = paths.geometry_checkpoint(seed)
    frozen = json.load(open(paths.geometry_freeze))["sha256"][os.path.relpath(path, C.REPO_ROOT)]
    assert sha256_file(path) == frozen, f"{path} differs from the frozen geometry baseline"
    net = load_checkpoint(path, device).net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    assert net.enc1[0].in_channels == 3 and net.n_params() == 967_322
    return net


def geometry_input(S: torch.Tensor, F_: torch.Tensor) -> torch.Tensor:
    """``[1, 3, X, Y, Z]`` = [S, 0, F] from flat bool tensors (the baseline's network input)."""
    S3, F3 = S.reshape(C.GRID_DIMS), F_.reshape(C.GRID_DIMS)
    return torch.stack([S3, torch.zeros_like(S3), F3]).float()[None]


@torch.no_grad()
def occupancy_and_trunk(net: CompletionUNet, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(occupied [B, N_VOXELS] bool, trunk [B, 24, X, Y, Z] float32)`` from ``x [B, 3, X, Y, Z]``."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res, _, d1 = net(x, return_trunk=True)
    residual = res[:, 0].float()
    occ = apply_residual(torch.zeros_like(residual), residual).reshape(len(x), -1) > 0
    return occ, d1.float()


def load_hard_map(drive: str, t: int, device, paths: C.Paths = C.Paths()) -> Tuple[torch.Tensor, torch.Tensor]:
    with np.load(paths.input_path(drive, t)) as z:
        S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(device)
        F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:C.N_VOXELS].astype(bool)).to(device)
    return S, F_


__all__ = ["state_digest", "sha256_file", "load_frozen_geometry", "geometry_input", "occupancy_and_trunk", "load_hard_map"]
