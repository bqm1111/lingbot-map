"""The lightweight completion module: a small dense 3D U-Net over the exported map crop.

Dense rather than sparse on purpose: completion has to predict in *unknown* space, which
is not sparse, and the benchmark boxes are small enough (2.1 M / 5.1 M voxels at 0.2 m)
that a 3-level U-Net with 24/48/96 channels runs on the whole grid at inference.

Outputs are **residuals** on top of the frozen map, under one explicit, testable rule
(:func:`apply_residual`): a voxel whose log-odds already carry at least ``LOCK_LOGODDS``
of direct evidence is never changed, and the semantic distribution of a voxel that
already holds teacher evidence is never overwritten. The network can therefore only add
occupancy where the map is unknown or weakly supported, and only name what it adds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate8 import vocab as V8
from gate8.targets import N_INPUT_CHANNELS

#: |log-odds| at or above which a voxel is locked against the residual (~3 agreeing rays).
LOCK_LOGODDS = 2.0


def _block(cin, cout, padding_mode: str = "zeros"):
    """Two padded 3x3x3 convolutions.

    ``padding_mode`` matters more here than it usually does. Channels 2 and 3 of the input
    are ``observed`` and ``1 - observed``; they sum to 1 in every voxel that can occur. With
    ``"zeros"`` the convolution pads the array boundary with a vector where BOTH are 0 -- a
    state absent from the training data -- and the network's response to it propagates
    inward. The grid is only 32 voxels deep in z, so that response reaches a large share of
    the ceiling and the floor. ``"replicate"`` extends the volume with a copy of the edge
    voxel, which is the truthful statement "outside the grid looks like its edge".
    """
    k = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
    return nn.Sequential(nn.Conv3d(cin, cout, **k), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv3d(cout, cout, **k), nn.GroupNorm(8, cout), nn.GELU())


class CompletionUNet(nn.Module):
    def __init__(self, cin: int = N_INPUT_CHANNELS, width: int = 24, n_sem: int = V8.U,
                 padding_mode: str = "zeros"):
        super().__init__()
        w = width
        self.padding_mode = padding_mode
        _b = lambda a, b: _block(a, b, padding_mode)
        self.enc1 = _b(cin, w)
        self.down1 = nn.Conv3d(w, 2 * w, 2, stride=2)
        self.enc2 = _b(2 * w, 2 * w)
        self.down2 = nn.Conv3d(2 * w, 4 * w, 2, stride=2)
        self.enc3 = _b(4 * w, 4 * w)
        self.up2 = nn.ConvTranspose3d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = _b(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose3d(2 * w, w, 2, stride=2)
        self.dec1 = _b(2 * w, w)
        self.head_occ = nn.Conv3d(w, 1, 1)
        self.head_sem = nn.Conv3d(w, n_sem, 1)
        nn.init.zeros_(self.head_occ.weight); nn.init.zeros_(self.head_occ.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head_occ(d1), self.head_sem(d1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def apply_residual(base_logodds: torch.Tensor, occ_res: torch.Tensor) -> torch.Tensor:
    """Final occupancy log-odds. Locked voxels keep the map's value exactly."""
    gate = (base_logodds.abs() < LOCK_LOGODDS).to(occ_res.dtype)
    return base_logodds + occ_res * gate


class Completer:
    """Inference wrapper used by the evaluator: ``complete(query, grid)``."""

    def __init__(self, net: CompletionUNet, device):
        self.net = net.to(device).eval()
        self.device = device

    #: Channels 2 and 3 are ``observed`` and ``1 - observed``: they sum to 1 in every voxel
    #: the network ever saw in training. ``nn.Conv3d(padding=1)`` pads with ZEROS, so at the
    #: array boundary it manufactures ``observed = 0`` AND ``unobserved = 0`` -- a state that
    #: cannot occur in the data. The response to that impossible input leaks inward from the
    #: boundary, and because the grid is only 32 voxels deep in z the leak reaches a large
    #: fraction of the ceiling. ``pad_z`` moves the array boundary away from the real volume
    #: and fills the margin with the CORRECT encoding of "outside the grid": unobserved.
    #: Default 0 keeps every earlier gate's numbers bit-identical.
    PAD_UNOBSERVED_CHANNEL = 3

    def _pad_z(self, inp: torch.Tensor, pad: int) -> torch.Tensor:
        n, c, X, Y, Z = inp.shape
        out = inp.new_zeros((n, c, X, Y, Z + 2 * pad))
        out[:, self.PAD_UNOBSERVED_CHANNEL] = 1.0
        out[..., pad:pad + Z] = inp
        return out

    @torch.no_grad()
    def raw(self, q: Dict[str, torch.Tensor], grid, pad_z: int = 0):
        """``(final_logodds [n], probs [n, U])`` before any thresholding.

        ``complete`` is this plus the frozen ``> 0`` decision; Gate 8A needs the
        continuous score, so the two share one implementation and cannot drift apart.
        """
        X, Y, Z = (int(d) for d in grid.dims)
        n = X * Y * Z
        sem = q["sem"]; semw = q["sem_w"]
        obs = q["observed"].float()
        age = q.get("age", torch.full((n,), -1.0, device=self.device)).float()
        base = q["logodds"].reshape(X, Y, Z)
        inp = torch.stack([base / 4.0, q["w_free"].reshape(X, Y, Z).clamp(max=20) / 20.0,
                           obs.reshape(X, Y, Z), 1.0 - obs.reshape(X, Y, Z),
                           q["n_obs"].float().reshape(X, Y, Z).clamp(max=50) / 50.0,
                           (age.clamp(min=0) / 100.0 * (age >= 0)).reshape(X, Y, Z),
                           semw.reshape(X, Y, Z).clamp(max=20) / 20.0] +
                          [(sem[:, u] / semw.clamp_min(1e-9)).reshape(X, Y, Z)
                           for u in range(V8.U)], 0)[None]
        if pad_z:
            Zin = inp.shape[-1]
            inp = self._pad_z(inp, pad_z)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ_res, sem_log = self.net(inp)
        if pad_z:
            occ_res = occ_res[..., pad_z:pad_z + Zin]
            sem_log = sem_log[..., pad_z:pad_z + Zin]
        occ_res = occ_res[0, 0].float(); sem_log = sem_log[0].float()
        final = apply_residual(base, occ_res).reshape(-1)
        map_probs = sem / semw.clamp_min(1e-9).unsqueeze(1)
        net_probs = F.softmax(sem_log.reshape(V8.U, -1).T, dim=1)
        probs = torch.where((semw > 0).unsqueeze(1), map_probs, net_probs)
        return final, probs

    @torch.no_grad()
    def complete(self, q: Dict[str, torch.Tensor], grid):
        final, probs = self.raw(q, grid)
        return final > 0, probs


def save_checkpoint(net: CompletionUNet, path: str, meta: dict) -> None:
    torch.save({"state_dict": net.state_dict(), "meta": meta,
                "cin": net.enc1[0].in_channels, "width": net.enc1[0].out_channels,
                "padding_mode": getattr(net, "padding_mode", "zeros")}, path)


def load_checkpoint(path: str, device) -> Completer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CompletionUNet(cin=ck["cin"], width=ck["width"],
                         padding_mode=ck.get("padding_mode", "zeros"))
    net.load_state_dict(ck["state_dict"])
    return Completer(net, device)


__all__ = ["CompletionUNet", "Completer", "apply_residual", "save_checkpoint",
           "load_checkpoint", "LOCK_LOGODDS"]
