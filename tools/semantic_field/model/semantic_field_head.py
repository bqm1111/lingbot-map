"""B4: a generative, language-aligned 3D semantic field on frozen geometry and sparse observed semantics.

Dense input ``[B, 61, X, Y, Z]`` (grid C order, velodyne frame of t, 0.2 m voxels):
    trunk        24  frozen CompletionUNet decoder feature d1
    geometry      3  hard map S, hard map F, frozen occupancy O (0/1)
    evidence     32  Linear(1024 -> 32) of the CENTERED evidence feature at evidence voxels, 0 elsewhere
    evidence_mask 1  1 at visible evidence voxels
    count         1  log1p(ray count) / 5 at visible evidence voxels
Hidden field ``[B, 32, X, Y, Z]`` from a residual 3D UNet (1, 1/2, 1/4 resolution). The semantic feature of any voxel is
``normalize(Linear(32 -> 1024)(hidden[v]))`` in the RAW teacher space: a free vector, not a combination of input features.
X, Y must be divisible by 4 and Z by 4 (256x256x32, 128x128x32 and 400x400x32 all are).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.semantic_field.config.schema import FeatureSpace, HeadConfig

INPUT_CHANNELS_FIXED = 1 + 1          # evidence mask + count


def conv3(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, padding_mode="replicate")


class ResidualBlock(nn.Module):
    def __init__(self, width: int, groups: int, dilation: int = 1):
        super().__init__()
        self.body = nn.Sequential(conv3(width, width, dilation=dilation), nn.GroupNorm(groups, width), nn.GELU(),
                                  conv3(width, width, dilation=dilation), nn.GroupNorm(groups, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.body(x))


class SemanticFieldHead(nn.Module):
    def __init__(self, cfg: HeadConfig = HeadConfig(), space: FeatureSpace = FeatureSpace()):
        super().__init__()
        self.cfg, self.space = cfg, space
        w1, w2, w3 = cfg.widths; g = cfg.groups
        cin = cfg.trunk_channels + cfg.geometry_channels + cfg.evidence_projection + INPUT_CHANNELS_FIXED
        self.project_evidence = nn.Linear(space.teacher_dim, cfg.evidence_projection)
        self.stem = nn.Sequential(conv3(cin, w1), nn.GroupNorm(g, w1), nn.GELU(), ResidualBlock(w1, g))
        self.down1 = nn.Sequential(conv3(w1, w2, stride=2), nn.GroupNorm(g, w2), nn.GELU(), ResidualBlock(w2, g), ResidualBlock(w2, g))
        self.down2 = nn.Sequential(conv3(w2, w3, stride=2), nn.GroupNorm(g, w3), nn.GELU(),
                                   *[ResidualBlock(w3, g, d) for d in cfg.coarse_dilations])
        self.lateral2 = nn.Conv3d(w3, w2, 1); self.fuse2 = nn.Sequential(conv3(2 * w2, w2), nn.GroupNorm(g, w2), nn.GELU(), ResidualBlock(w2, g))
        self.lateral1 = nn.Conv3d(w2, w1, 1); self.fuse1 = nn.Sequential(conv3(2 * w1, w1), nn.GroupNorm(g, w1), nn.GELU(), ResidualBlock(w1, g))
        self.semantic_head = nn.Linear(w1, space.teacher_dim)

    def hidden_field(self, dense_input: torch.Tensor) -> torch.Tensor:
        """``[B, 61, X, Y, Z]`` -> ``[B, 32, X, Y, Z]``."""
        s1 = self.stem(dense_input)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        u2 = self.fuse2(torch.cat([F.interpolate(self.lateral2(s3), size=s2.shape[-3:], mode="trilinear", align_corners=False), s2], 1))
        return self.fuse1(torch.cat([F.interpolate(self.lateral1(u2), size=s1.shape[-3:], mode="trilinear", align_corners=False), s1], 1))

    def semantic_features(self, hidden_at_voxels: torch.Tensor) -> torch.Tensor:
        """``[N, 32]`` hidden vectors -> ``[N, 1024]`` unit features in the raw teacher space."""
        return F.normalize(self.semantic_head(hidden_at_voxels), dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict:
        return {"head": asdict(self.cfg), "feature_space": asdict(self.space)}


__all__ = ["ResidualBlock", "SemanticFieldHead"]
