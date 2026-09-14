"""B1: a small semantic propagation head on top of the FROZEN geometry-completion network.

Inputs (all on the same crop or full grid, [B, C, X, Y, Z], C-order voxels of the velodyne grid):
    trunk          [B, 24, X, Y, Z]  frozen CompletionUNet decoder feature ``d1`` (computed under no_grad)
    evidence       [B, 1024, ...]    is never densified: the sparse unit-norm teacher features are
                                     projected to ``evidence_dim`` first and scattered, zeros elsewhere
    evidence_mask  [B, 1, X, Y, Z]   1 where the voxel's evidence is given as input

Output: [B, 1024, X, Y, Z] unit-norm language-aligned feature for every voxel; classification is
``argmax cos(f, text)`` downstream, so the head never sees class ids.

Size is deliberately a fraction of the 967k geometry network: no down-sampling path, only four
residual blocks of dilated 3x3x3 convolutions (dilations 1, 2, 4, 8) at width 32, which reach
30 voxels (6 m) in every direction -- the propagation range this gate measures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.semantic_completion import config as C


class DilatedResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        k = dict(kernel_size=3, padding=dilation, dilation=dilation, padding_mode="replicate")
        self.body = nn.Sequential(nn.Conv3d(width, width, **k), nn.GroupNorm(8, width), nn.GELU(),
                                  nn.Conv3d(width, width, **k), nn.GroupNorm(8, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.body(x))


class PropagationHead(nn.Module):
    def __init__(self, cfg: C.HeadConfig = C.HeadConfig()):
        super().__init__()
        self.cfg = cfg
        self.project_evidence = nn.Linear(C.TEACHER_DIM, cfg.evidence_dim)
        self.stem = nn.Conv3d(cfg.trunk_channels + cfg.evidence_dim + 1, cfg.width, 1)
        self.blocks = nn.Sequential(*[DilatedResidualBlock(cfg.width, d) for d in cfg.dilations])
        self.out = nn.Conv3d(cfg.width, C.TEACHER_DIM, 1)

    def scatter_evidence(self, index: torch.Tensor, feature: torch.Tensor, dims) -> torch.Tensor:
        """Sparse ``feature [N, 1024]`` at flat ``index [N]`` of one sample -> dense ``[evidence_dim, X, Y, Z]``."""
        X, Y, Z = (int(d) for d in dims)
        dense = torch.zeros(X * Y * Z, self.cfg.evidence_dim, device=feature.device, dtype=feature.dtype)
        if index.numel():
            dense[index] = self.project_evidence(feature)
        return dense.T.reshape(self.cfg.evidence_dim, X, Y, Z)

    def forward(self, trunk: torch.Tensor, evidence_dense: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
        x = self.stem(torch.cat([trunk, evidence_dense, evidence_mask], 1))
        return F.normalize(self.out(self.blocks(x)), dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["DilatedResidualBlock", "PropagationHead"]
