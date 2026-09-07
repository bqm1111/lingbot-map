"""Depth-refinement heads. LingBot is never touched; these predict a log residual.

    D_refined = D_base * exp(r_theta),  D_base = s_hat * D_lingbot

``r_theta`` is ``max_log_residual * tanh(raw)``, which bounds the correction and makes
the refined depth finite and strictly positive by construction rather than by clamping
after the fact.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_inputs(batch: Dict[str, torch.Tensor], stats: Dict[str, float],
                 use_rgb: bool) -> torch.Tensor:
    """Assemble the network input. Only tensors the model is allowed to see."""
    base = batch["base_depth"]
    v = batch["valid_lingbot_mask"].float()
    ld = (torch.log(base.clamp_min(1e-6)) - stats["log_depth_mean"]) / stats["log_depth_std"]
    cf = (batch["lingbot_confidence"] - stats["conf_mean"]) / stats["conf_std"]
    B, _, H, W = base.shape
    ys = torch.linspace(-1, 1, H, device=base.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(-1, 1, W, device=base.device).view(1, 1, 1, W).expand(B, 1, H, W)
    chans = [ld * v, cf, v, xs, ys]
    if use_rgb:
        chans.append(batch["rgb"] * 2.0 - 1.0)
    return torch.cat(chans, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class IdentityRefiner(nn.Module):
    """Baseline 0: returns a zero residual, i.e. the unrefined depth."""

    def __init__(self, **_):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return torch.zeros_like(x[:, :1])


class DepthCNN(nn.Module):
    """Baseline 1: small residual CNN over depth/confidence/mask/coordinates."""

    def __init__(self, in_ch: int, base: int = 32, max_log_residual: float = 0.7):
        super().__init__()
        self.max_r = max_log_residual
        self.stem = ConvBlock(in_ch, base)
        self.body = nn.Sequential(ConvBlock(base, base), ConvBlock(base, base))
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.stem(x)
        h = h + self.body(h)
        return self.max_r * torch.tanh(self.head(h))


class RGBDUNet(nn.Module):
    """Baseline 2: lightweight U-Net over RGB + depth channels."""

    def __init__(self, in_ch: int, base: int = 32, max_log_residual: float = 0.7):
        super().__init__()
        self.max_r = max_log_residual
        b = base
        self.e1, self.e2, self.e3 = ConvBlock(in_ch, b), ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.d2 = ConvBlock(4 * b + 2 * b, 2 * b)
        self.d1 = ConvBlock(2 * b + b, b)
        self.head = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        u2 = F.interpolate(s3, size=s2.shape[-2:], mode="nearest")
        d2 = self.d2(torch.cat([u2, s2], 1))
        u1 = F.interpolate(d2, size=s1.shape[-2:], mode="nearest")
        d1 = self.d1(torch.cat([u1, s1], 1))
        return self.max_r * torch.tanh(self.head(d1))


ARCHS = {"identity": IdentityRefiner, "depth_cnn": DepthCNN, "rgbd_unet": RGBDUNet}


def build_model(arch: str, in_ch: int, base: int = 32, max_log_residual: float = 0.7):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHS)}")
    if arch == "identity":
        return IdentityRefiner()
    return ARCHS[arch](in_ch=in_ch, base=base, max_log_residual=max_log_residual)


def refine(base_depth: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``D_base * exp(r)`` -- finite and positive whenever ``base_depth`` is."""
    return base_depth * torch.exp(r)
