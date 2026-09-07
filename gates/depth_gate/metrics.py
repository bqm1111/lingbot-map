"""Depth metrics, evaluated on valid projected-LiDAR pixels only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    """AbsRel, log RMSE, RMSE (m), delta-1, median absolute error, pixel count."""
    p = pred[valid].double()
    g = gt[valid].double()
    ok = torch.isfinite(p) & torch.isfinite(g) & (p > 0) & (g > 0)
    p, g = p[ok], g[ok]
    if p.numel() == 0:
        return {k: float("nan") for k in
                ("abs_rel", "log_rmse", "rmse_m", "delta1", "median_abs_err_m")} | {"n_valid": 0}
    ratio = torch.maximum(p / g, g / p)
    return {
        "abs_rel": float(((p - g).abs() / g).mean()),
        "log_rmse": float(torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean())),
        "rmse_m": float(torch.sqrt(((p - g) ** 2).mean())),
        "delta1": float((ratio < 1.25).double().mean()),
        "median_abs_err_m": float((p - g).abs().median()),
        "n_valid": int(p.numel()),
    }


def binned_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                   bins: Sequence[Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    """Same metrics split by ground-truth distance."""
    out = {}
    for lo, hi in bins:
        m = valid & (gt >= lo) & (gt < hi)
        out[f"{int(lo)}-{int(hi)}m"] = depth_metrics(pred, gt, m)
    return out
