"""Linear bridge from frozen LingBot tokens to the compressed teacher space.

Deliberately the smallest thing that could work: if a single matrix already recovers
open-vocabulary signal, the tokens themselves carry it and the sidecar is a bridge,
not a semantic model.  Supports ordinary gradient training and a closed-form ridge
solution accumulated in one streaming pass.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSidecar(nn.Module):
    """``[..., L*C] -> [..., out_dim]``, L2-normalised.

    Args:
        in_dim: Concatenated input width (``num_layers * 2048``).
        out_dim: Compressed semantic dimension (64 or 128).
        bias: Include a bias term (off by default so the closed-form ridge solution
            and the gradient solution parameterise the same function).
    """

    def __init__(self, in_dim: int, out_dim: int = 64, bias: bool = False) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.normal_(self.proj.weight, std=in_dim**-0.5)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Args:
            tokens: ``[..., num_layers, C]`` or ``[..., L*C]`` frozen features.
        """
        if tokens.shape[-1] != self.in_dim:
            tokens = tokens.flatten(-2)
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(f"expected input width {self.in_dim}, got {tokens.shape[-1]}")
        out = self.proj(tokens.float())
        return F.normalize(out, dim=-1) if normalize else out

    @torch.no_grad()
    def load_ridge(self, weight: torch.Tensor) -> None:
        """Install a closed-form solution ``[out_dim, in_dim]``."""
        if tuple(weight.shape) != (self.out_dim, self.in_dim):
            raise ValueError(f"expected weight {(self.out_dim, self.in_dim)}, got {tuple(weight.shape)}")
        self.proj.weight.copy_(weight.to(self.proj.weight.dtype))


class RidgeAccumulator:
    """Streaming ``XᵀX`` / ``XᵀY`` accumulation for closed-form ridge regression.

    Memory is ``O(in_dim²)`` regardless of the number of samples, which for
    ``in_dim = 4096`` is 128 MB in float64 — affordable and exact.
    """

    def __init__(self, in_dim: int, out_dim: int, device: str = "cuda", dtype=torch.float64) -> None:
        self.in_dim, self.out_dim = in_dim, out_dim
        self.device, self.dtype = torch.device(device), dtype
        self.xtx = torch.zeros(in_dim, in_dim, device=self.device, dtype=dtype)
        self.xty = torch.zeros(in_dim, out_dim, device=self.device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> None:
        """Accumulate a batch of ``[N, in_dim]`` inputs and ``[N, out_dim]`` targets."""
        x = x.reshape(-1, self.in_dim).to(self.device, self.dtype)
        y = y.reshape(-1, self.out_dim).to(self.device, self.dtype)
        if weight is not None:
            w = weight.reshape(-1, 1).to(self.device, self.dtype)
            xw = x * w
        else:
            xw = x
        self.xtx += xw.T @ x
        self.xty += xw.T @ y
        self.n += x.shape[0]

    @torch.no_grad()
    def solve(self, ridge: float = 1e-2) -> torch.Tensor:
        """Return ``W`` of shape ``[out_dim, in_dim]`` minimising ``||XW^T - Y||² + λ||W||²``."""
        if self.n == 0:
            raise RuntimeError("no samples accumulated")
        scale = self.xtx.diagonal().mean().clamp_min(1e-12)
        a = self.xtx + ridge * scale * torch.eye(self.in_dim, device=self.device, dtype=self.dtype)
        w = torch.linalg.solve(a, self.xty)  # [in_dim, out_dim]
        return w.T.float().cpu()
