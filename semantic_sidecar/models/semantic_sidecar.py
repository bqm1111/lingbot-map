"""Lightweight semantic sidecar over detached LingBot tokens.

Per-layer projection → fusion → two pre-norm residual MLP blocks → normalised
64/128-d head.  Kept deliberately small: the research question is whether the frozen
tokens already carry the signal, and a large transformer here would confound that.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_sidecar.config import ModelConfig, TOKEN_CHANNELS
from semantic_sidecar.models.linear_sidecar import LinearSidecar


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP: ``x + W2 GELU(W1 LN(x))``."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class SemanticSidecar(nn.Module):
    """Maps ``[..., num_layers, 2048]`` frozen tokens to a normalised semantic vector.

    Args:
        num_layers: How many aggregator layers are fed in.
        in_channels: Channels per layer (2048 for LingBot: frame ‖ global attention).
        hidden_dim: Width of the trunk.
        num_blocks: Residual MLP blocks.
        mlp_ratio: Hidden expansion inside each block.
        out_dim: Output dimensionality (must match the PCA basis).
        fuse: ``"concat"`` or ``"sum"`` over the per-layer projections.
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int = TOKEN_CHANNELS,
        hidden_dim: int = 768,
        num_blocks: int = 2,
        mlp_ratio: float = 2.0,
        out_dim: int = 64,
        fuse: str = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fuse not in ("concat", "sum"):
            raise ValueError(f"unknown fuse mode {fuse!r}")
        self.num_layers, self.in_channels, self.fuse = num_layers, in_channels, fuse
        self.out_dim = out_dim

        self.layer_norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(num_layers)])
        self.layer_proj = nn.ModuleList([nn.Linear(in_channels, hidden_dim) for _ in range(num_layers)])
        fused_dim = hidden_dim * num_layers if fuse == "concat" else hidden_dim
        self.fuse_proj = nn.Linear(fused_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, mlp_ratio, dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Args:
            tokens: ``[..., num_layers, in_channels]`` detached frozen features.
        """
        if tokens.shape[-2:] != (self.num_layers, self.in_channels):
            raise ValueError(
                f"expected [..., {self.num_layers}, {self.in_channels}], got {tuple(tokens.shape)}"
            )
        x = tokens.float()
        parts = [proj(norm(x[..., i, :])) for i, (norm, proj) in enumerate(zip(self.layer_norms, self.layer_proj))]
        fused = torch.cat(parts, dim=-1) if self.fuse == "concat" else torch.stack(parts).sum(0)
        h = self.fuse_proj(fused)
        for block in self.blocks:
            h = block(h)
        out = self.head(self.out_norm(h))
        return F.normalize(out, dim=-1) if normalize else out


class LinearSidecarWrapper(nn.Module):
    """Adapts :class:`LinearSidecar` to the ``[..., L, C]`` input signature."""

    def __init__(self, num_layers: int, in_channels: int = TOKEN_CHANNELS, out_dim: int = 64) -> None:
        super().__init__()
        self.num_layers, self.in_channels, self.out_dim = num_layers, in_channels, out_dim
        self.linear = LinearSidecar(num_layers * in_channels, out_dim)

    def forward(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_layers, self.in_channels):
            raise ValueError(
                f"expected [..., {self.num_layers}, {self.in_channels}], got {tuple(tokens.shape)}"
            )
        return self.linear(tokens.flatten(-2), normalize=normalize)


def build_sidecar(cfg: ModelConfig, num_layers: int, in_channels: int = TOKEN_CHANNELS) -> nn.Module:
    """Instantiate the configured sidecar."""
    if cfg.arch == "linear":
        return LinearSidecarWrapper(num_layers, in_channels, cfg.out_dim)
    if cfg.arch == "mlp":
        return SemanticSidecar(
            num_layers=num_layers,
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
            mlp_ratio=cfg.mlp_ratio,
            out_dim=cfg.out_dim,
            fuse=cfg.fuse,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown sidecar arch {cfg.arch!r}")


def parameter_report(
    sidecar: nn.Module, lingbot_total: Optional[int] = None
) -> Dict[str, Any]:
    """Trainable-parameter accounting, printed and logged by every training run."""
    trainable = sum(p.numel() for p in sidecar.parameters() if p.requires_grad)
    total_sidecar = sum(p.numel() for p in sidecar.parameters())
    report: Dict[str, Any] = {
        "sidecar_total": total_sidecar,
        "sidecar_trainable": trainable,
    }
    if lingbot_total is not None:
        report["lingbot_total"] = lingbot_total
        report["lingbot_frozen"] = lingbot_total
        report["trainable_fraction_pct"] = 100.0 * trainable / max(lingbot_total + trainable, 1)
    return report


def format_parameter_report(report: Dict[str, Any]) -> str:
    lines = [
        f"  LingBot total      : {report.get('lingbot_total', 0):,}",
        f"  LingBot frozen     : {report.get('lingbot_frozen', 0):,}",
        f"  Sidecar trainable  : {report['sidecar_trainable']:,}",
    ]
    if "trainable_fraction_pct" in report:
        lines.append(f"  Trainable fraction : {report['trainable_fraction_pct']:.4f} %")
    return "\n".join(lines)
