"""Geometry-consolidated semantic targets.

For each track, the per-view teacher features are fused into one view-consistent
target

    ȳ_x = normalize( Σ_{(t,u) ∈ O_x} w_{t,u} · y_t(u) ).

The weights use **only class-agnostic** quantities — LingBot's ``depth_conf``, the
cross-view reprojection residual, the viewing angle, and (for the robust estimator)
each observation's agreement with the track's own spherical median.  No ground-truth
label and no target taxonomy is involved anywhere.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from semantic_sidecar.config import ConsensusConfig
from semantic_sidecar.tracks import TrackSet

logger = logging.getLogger(__name__)

ESTIMATORS = ("mean", "geometric", "robust")


def observation_weights(
    conf: torch.Tensor,
    residual: torch.Tensor,
    cos_angle: torch.Tensor,
    cfg: ConsensusConfig,
) -> torch.Tensor:
    """Class-agnostic geometric weight for each observation.

    Args:
        conf: LingBot ``depth_conf`` (``expp1`` activation, so ``> 1``).
        residual: Relative cross-view depth disagreement (0 is perfect).
        cos_angle: Cosine between this view's direction and the track's mean view.
    """
    conf_term = (conf - 1.0).clamp_min(0.0).pow(cfg.conf_power)
    residual_term = torch.exp(-residual.clamp_min(0.0) / max(cfg.residual_tau, 1e-6))
    angle_term = cos_angle.clamp_min(0.0).pow(cfg.angle_power)
    return (conf_term * residual_term * angle_term).clamp_min(1e-6)


def _segment_sum(values: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Sum ``values`` (``[K, ...]``) over CSR segments defined by ``ptr``."""
    num = ptr.numel() - 1
    seg = torch.repeat_interleave(
        torch.arange(num, device=values.device), (ptr[1:] - ptr[:-1]).to(values.device)
    )
    out = torch.zeros(num, *values.shape[1:], device=values.device, dtype=values.dtype)
    out.index_add_(0, seg, values)
    return out


def segment_ids(ptr: torch.Tensor, device=None) -> torch.Tensor:
    """Per-observation track id from a CSR pointer array."""
    device = device or ptr.device
    num = ptr.numel() - 1
    return torch.repeat_interleave(
        torch.arange(num, device=device), (ptr[1:] - ptr[:-1]).to(device)
    )


@dataclass
class ConsensusTargets:
    """Per-track consensus features and their diagnostics."""

    features: torch.Tensor       # [M, d] float32, L2-normalised (zeros where invalid)
    valid: torch.Tensor          # [M] bool
    n_obs: torch.Tensor          # [M] int32, observations entering the estimate
    n_teacher_obs: torch.Tensor  # [M] int32, observations with a teacher feature
    total_weight: torch.Tensor   # [M] float32
    dispersion: torch.Tensor     # [M] float32, 1 - mean cos(y_i, consensus)
    n_rejected: torch.Tensor     # [M] int32
    estimator: str
    meta: Dict[str, Any]

    def save(self, root: str, scene: str, extra: Optional[Dict[str, Any]] = None) -> str:
        from safetensors.torch import save_file

        out_dir = os.path.join(root, scene)
        os.makedirs(out_dir, exist_ok=True)
        save_file(
            {
                "features": self.features.half().contiguous(),
                "valid": self.valid.to(torch.uint8).contiguous(),
                "n_obs": self.n_obs.contiguous(),
                "n_teacher_obs": self.n_teacher_obs.contiguous(),
                "total_weight": self.total_weight.contiguous(),
                "dispersion": self.dispersion.contiguous(),
                "n_rejected": self.n_rejected.contiguous(),
            },
            os.path.join(out_dir, "consensus.safetensors"),
            metadata={"estimator": self.estimator},
        )
        meta = {"estimator": self.estimator, **self.meta, **(extra or {})}
        path = os.path.join(out_dir, "consensus.json")
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, root: str, scene: str) -> "ConsensusTargets":
        from safetensors import safe_open

        out_dir = os.path.join(root, scene)
        with safe_open(os.path.join(out_dir, "consensus.safetensors"), framework="pt", device="cpu") as fh:
            data = {k: fh.get_tensor(k) for k in fh.keys()}
            estimator = (fh.metadata() or {}).get("estimator", "unknown")
        meta_path = os.path.join(out_dir, "consensus.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return cls(
            features=data["features"].float(),
            valid=data["valid"].bool(),
            n_obs=data["n_obs"],
            n_teacher_obs=data["n_teacher_obs"],
            total_weight=data["total_weight"],
            dispersion=data["dispersion"],
            n_rejected=data["n_rejected"],
            estimator=estimator,
            meta=meta,
        )


def compute_consensus(
    obs_features: torch.Tensor,
    obs_ptr: torch.Tensor,
    obs_conf: torch.Tensor,
    obs_residual: torch.Tensor,
    obs_cos: torch.Tensor,
    obs_has_teacher: torch.Tensor,
    cfg: ConsensusConfig,
    min_teacher_obs: int = 1,
    pixel_fallback: Optional[torch.Tensor] = None,
) -> ConsensusTargets:
    """Fuse per-observation teacher features into one target per track.

    Args:
        obs_features: ``[K, d]`` L2-normalised teacher features per observation.
            Rows where ``obs_has_teacher`` is ``False`` are ignored (their content is
            irrelevant and may be zero).
        obs_ptr: ``[M + 1]`` CSR pointer.
        obs_conf / obs_residual / obs_cos: ``[K]`` class-agnostic reliability signals.
        obs_has_teacher: ``[K]`` bool, teacher availability (sparse-teacher support).
        cfg: Estimator selection and weight shaping.
        min_teacher_obs: Tracks with fewer teacher observations are invalid.
        pixel_fallback: Optional ``[M, d]`` fallback used only when
            ``cfg.fallback == "pixelwise"``.

    Returns:
        :class:`ConsensusTargets`; invalid tracks carry a zero feature and
        ``valid == False`` — they are never silently filled.
    """
    if cfg.estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {cfg.estimator!r}, expected one of {ESTIMATORS}")
    device = obs_features.device
    num_tracks = obs_ptr.numel() - 1
    dim = obs_features.shape[1]

    has = obs_has_teacher.to(device).float()
    feats = F.normalize(obs_features.float(), dim=-1) * has.unsqueeze(1)

    if cfg.estimator == "mean":
        weights = has.clone()
    else:
        weights = observation_weights(
            obs_conf.to(device).float(), obs_residual.to(device).float(), obs_cos.to(device).float(), cfg
        ) * has

    n_teacher = _segment_sum(has, obs_ptr).to(torch.int32)
    rejected = torch.zeros(num_tracks, device=device)

    def _fuse(w: torch.Tensor) -> torch.Tensor:
        return F.normalize(_segment_sum(feats * w.unsqueeze(1), obs_ptr), dim=-1)

    consensus = _fuse(weights)

    if cfg.estimator == "robust":
        seg = segment_ids(obs_ptr, device)
        agree = (feats * consensus[seg]).sum(-1)
        drop = (agree < cfg.agreement_floor) & (has > 0)
        rejected = _segment_sum(drop.float(), obs_ptr)
        weights = weights * (~drop).float() * agree.clamp_min(0.0)
        refused = _segment_sum(weights, obs_ptr) <= 0
        consensus_r = _fuse(weights)
        # A track whose observations all disagree keeps the first-round estimate
        # rather than collapsing to a zero vector.
        consensus = torch.where(refused.unsqueeze(1), consensus, consensus_r)

    total_weight = _segment_sum(weights, obs_ptr)
    seg = segment_ids(obs_ptr, device)
    cosines = (feats * consensus[seg]).sum(-1) * has
    dispersion = 1.0 - _segment_sum(cosines, obs_ptr) / n_teacher.clamp_min(1).float()
    dispersion = torch.where(n_teacher > 0, dispersion, torch.ones_like(dispersion))

    valid = (n_teacher >= min_teacher_obs) & (consensus.norm(dim=-1) > 1e-6)
    if cfg.fallback == "pixelwise" and pixel_fallback is not None:
        fill = ~valid
        consensus = torch.where(fill.unsqueeze(1), F.normalize(pixel_fallback.float(), dim=-1), consensus)
        valid = valid | (fill & (pixel_fallback.norm(dim=-1) > 1e-6))
    elif cfg.fallback not in ("invalid", "pixelwise"):
        raise ValueError(f"unknown fallback {cfg.fallback!r}")

    consensus = consensus * valid.unsqueeze(1).float()

    return ConsensusTargets(
        features=consensus,
        valid=valid,
        n_obs=(obs_ptr[1:] - obs_ptr[:-1]).to(torch.int32),
        n_teacher_obs=n_teacher,
        total_weight=total_weight,
        dispersion=dispersion,
        n_rejected=rejected.to(torch.int32),
        estimator=cfg.estimator,
        meta={
            "dim": dim,
            "num_tracks": int(num_tracks),
            "conf_power": cfg.conf_power,
            "residual_tau": cfg.residual_tau,
            "angle_power": cfg.angle_power,
            "agreement_floor": cfg.agreement_floor,
            "fallback": cfg.fallback,
            "min_teacher_obs": min_teacher_obs,
        },
    )


def consensus_for_scene(
    tracks: TrackSet,
    obs_features: torch.Tensor,
    obs_has_teacher: torch.Tensor,
    cfg: ConsensusConfig,
    min_teacher_obs: int = 1,
) -> ConsensusTargets:
    """Convenience wrapper binding a :class:`TrackSet` to its observation features."""
    if obs_features.shape[0] != tracks.num_observations:
        raise ValueError(
            f"expected {tracks.num_observations} observation features, got {obs_features.shape[0]}"
        )
    return compute_consensus(
        obs_features=obs_features,
        obs_ptr=tracks.obs_ptr.to(obs_features.device),
        obs_conf=tracks.obs_conf,
        obs_residual=tracks.obs_residual,
        obs_cos=tracks.obs_cos,
        obs_has_teacher=obs_has_teacher,
        cfg=cfg,
        min_teacher_obs=min_teacher_obs,
    )
