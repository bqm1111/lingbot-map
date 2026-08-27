"""Shared fixtures: a synthetic two-camera rig with exactly known geometry."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_intrinsic(fx: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def w2c(R: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """World-to-camera ``[3, 4]`` for a camera with rotation ``R`` at ``centre``."""
    E = torch.zeros(3, 4)
    E[:3, :3] = R
    E[:3, 3] = -(R @ centre)
    return E


@pytest.fixture
def rig():
    """Two cameras looking down +z at a fronto-parallel plane at z = 4.

    The baseline (1.4) is exactly one 14-pixel patch footprint at this depth
    (4 / 40 * 14 = 1.4), so the two views' patch-token centres land on the *same*
    world points and multi-view tracks are expected rather than accidental.
    """
    H, W = 28, 42
    fx, cx, cy = 40.0, W / 2.0, H / 2.0
    K = make_intrinsic(fx, cx, cy)
    R = torch.eye(3)
    cams = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.4, 0.0, 0.0])]
    extr = [w2c(R, c) for c in cams]
    return {"H": H, "W": W, "K": K, "extrinsics": extr, "centres": cams, "plane_z": 4.0}


@pytest.fixture
def torch_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    return 0
