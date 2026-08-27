"""A small recurrent causal corrector.

It never sees RGB and never touches LingbotMap: it reads the causal feature
vector from :mod:`prompted_lingbot.features` and emits, per frame, an
*incremental* Sim(3) update

    delta_t = (exp(dlog_s_t), Rodrigues(w_t), dt_t)

composed onto a running correction.  With ``base="anchor"`` the increment is
applied on top of the strongest training-free anchor instead of on top of its own
previous output, which turns the model into a residual on that baseline and makes
"does learning add anything?" a direct question.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn


def axis_angle_to_matrix_t(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues for ``(..., 3)`` torch tensors."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = w / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], -1),
        torch.stack([z, zero, -x], -1),
        torch.stack([-y, x, zero], -1),
    ], dim=-2)
    th = theta.unsqueeze(-1)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


@dataclass
class CorrectorConfig:
    n_features: int = 24
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.0
    max_log_scale_step: float = 0.15
    max_rot_step_rad: float = 0.02
    max_trans_step_m: float = 0.50
    predict_confidence: bool = True
    base: str = "anchor"          # "anchor" | "identity"

    def to_dict(self) -> dict:
        return asdict(self)


class CausalCorrector(nn.Module):
    """GRU over causal features -> per-frame Sim(3) increment."""

    def __init__(self, cfg: CorrectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_norm = nn.LayerNorm(cfg.n_features)
        self.gru = nn.GRU(cfg.n_features, cfg.hidden, cfg.layers,
                          batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        head_out = 7 + (1 if cfg.predict_confidence else 0)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, head_out))
        # Start as the identity increment so an untrained model is a no-op.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, feats: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """``feats (B, T, F)`` -> increments and optional per-frame log-variance."""
        h, hidden = self.gru(self.in_norm(feats), hidden)
        out = self.head(h)
        c = self.cfg
        dlog_s = torch.tanh(out[..., 0:1]) * c.max_log_scale_step
        w = torch.tanh(out[..., 1:4]) * c.max_rot_step_rad
        dt = torch.tanh(out[..., 4:7]) * c.max_trans_step_m
        logvar = out[..., 7:8] if c.predict_confidence else None
        return dlog_s, w, dt, logvar, hidden


def rollout(
    model: CausalCorrector,
    feats: torch.Tensor,
    base_s: torch.Tensor,
    base_R: torch.Tensor,
    base_t: torch.Tensor,
):
    """Compose the model's increments into a per-frame Sim(3).

    Args:
        feats: ``(B, T, F)``.
        base_s / base_R / base_t: the base anchor's per-frame correction,
            ``(B, T)`` / ``(B, T, 3, 3)`` / ``(B, T, 3)``.  Pass the identity to
            make the model fully standalone.

    Returns:
        ``(s, R, t, logvar, increments)`` with ``s (B, T)``, ``R (B, T, 3, 3)``,
        ``t (B, T, 3)``.
    """
    dlog_s, w, dt, logvar, _ = model(feats)
    dR = axis_angle_to_matrix_t(w)
    B, T = feats.shape[:2]

    if model.cfg.base == "anchor":
        # C_t = delta_t o B_t   (residual on the causal baseline)
        s = base_s * torch.exp(dlog_s.squeeze(-1))
        R = dR @ base_R
        t = torch.einsum("btij,btj->bti", dR, base_t * torch.exp(dlog_s)) + dt
    else:
        # C_t = delta_t o C_{t-1}  (fully learned, recurrent composition)
        s_list, R_list, t_list = [], [], []
        s_p = torch.ones(B, device=feats.device, dtype=feats.dtype)
        R_p = torch.eye(3, device=feats.device, dtype=feats.dtype).expand(B, 3, 3)
        t_p = torch.zeros(B, 3, device=feats.device, dtype=feats.dtype)
        for i in range(T):
            e = torch.exp(dlog_s[:, i, 0])
            s_p = s_p * e
            t_p = torch.einsum("bij,bj->bi", dR[:, i], e[:, None] * t_p) + dt[:, i]
            R_p = dR[:, i] @ R_p
            s_list.append(s_p); R_list.append(R_p); t_list.append(t_p)
        s = torch.stack(s_list, 1)
        R = torch.stack(R_list, 1)
        t = torch.stack(t_list, 1)
    return s, R, t, logvar, (dlog_s, w, dt)
