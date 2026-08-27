"""Training data and losses for the learned causal corrector.

Nothing here touches LingbotMap: it reads cached predictions only.  Ground truth
is used exclusively to build losses, never as a model input.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn

from .anchors import BASELINES, Prediction
from .conventions import Sim3
from .features import FEATURE_NAMES, N_FEATURES, sequence_features
from .prompts import PromptConfig, training_config_sampler
from .runner import CachedSequence, DepthEvalCache, build_prompts, load_cached

_HAS_DEPTH = FEATURE_NAMES.index("has_depth_prompt")
_DEPTH_LOGRATIO = FEATURE_NAMES.index("depth_prompt_log_ratio")


@dataclass
class Example:
    """One (sequence, prompt-realisation) training example."""

    sequence: str
    config: str
    feats: np.ndarray            # (T, F)
    base_s: np.ndarray           # (T,)
    base_R: np.ndarray           # (T, 3, 3)
    base_t: np.ndarray           # (T, 3)
    pred_c: np.ndarray           # (T, 3) predicted camera centres, model units
    pred_R: np.ndarray           # (T, 3, 3)
    gt_c: np.ndarray             # (T, 3) metric
    gt_R: np.ndarray             # (T, 3, 3)
    log_ratio: np.ndarray        # (T, S) log(gt_depth / pred_depth) samples
    prompt_pose_mask: np.ndarray  # (T,) bool
    prompt_pose_c: np.ndarray    # (T, 3)
    prompt_depth_mask: np.ndarray  # (T,) bool
    prompt_depth_logratio: np.ndarray  # (T,)


def build_examples(
    cache_paths: Sequence[str],
    n_prompt_samples: int,
    base_anchor: str = "depth_scale_pose_se3",
    n_depth_samples: int = 256,
    seed: int = 0,
    configs: Optional[Sequence[PromptConfig]] = None,
) -> List[Example]:
    """Materialise the training pool: every sequence x several prompt regimes."""
    out: List[Example] = []
    for si, path in enumerate(cache_paths):
        seq = load_cached(path)
        if not seq.has_gt_depth:
            continue
        dc = DepthEvalCache.build(seq, n_per_frame=n_depth_samples, seed=seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.log(dc.gt) - np.log(dc.pred)
        lr = np.where(np.isfinite(lr), lr, np.nan).astype(np.float32)
        preds = [seq.prediction(t) for t in range(seq.n_frames)]

        cfg_list = list(configs) if configs is not None else [
            training_config_sampler(seed * 100003 + si * 1009 + k) for k in range(n_prompt_samples)
        ]
        for cfg in cfg_list:
            prompts = build_prompts(seq, cfg)
            feats, base = sequence_features(preds, prompts, BASELINES[base_anchor]())
            T = seq.n_frames
            out.append(Example(
                sequence=seq.name, config=cfg.name, feats=feats,
                base_s=np.array([b.s for b in base], np.float32),
                base_R=np.stack([b.R for b in base]).astype(np.float32),
                base_t=np.stack([b.t for b in base]).astype(np.float32),
                pred_c=seq.pred_pose_c2w[:, :3, 3].astype(np.float32),
                pred_R=seq.pred_pose_c2w[:, :3, :3].astype(np.float32),
                gt_c=seq.gt_pose_c2w[:, :3, 3].astype(np.float32),
                gt_R=seq.gt_pose_c2w[:, :3, :3].astype(np.float32),
                log_ratio=lr,
                prompt_pose_mask=np.array([p.has_pose for p in prompts]),
                prompt_pose_c=np.stack([p.pose_c2w[:3, 3] if p.has_pose else np.zeros(3)
                                        for p in prompts]).astype(np.float32),
                prompt_depth_mask=feats[:, _HAS_DEPTH] > 0.5,
                prompt_depth_logratio=feats[:, _DEPTH_LOGRATIO].astype(np.float32),
            ))
    return out


def collate(batch: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack a batch, right-padding to the longest sequence."""
    T = max(len(b.feats) for b in batch)

    def pad(arr, fill=0.0):
        a = np.asarray(arr)
        out = np.full((len(batch),) + (T,) + a.shape[1:], fill, dtype=np.float32)
        return out

    keys_2d = ["feats", "base_t", "pred_c", "gt_c", "log_ratio", "prompt_pose_c"]
    out: Dict[str, torch.Tensor] = {}
    for k in keys_2d:
        ref = np.asarray(getattr(batch[0], k))
        buf = np.zeros((len(batch), T) + ref.shape[1:], np.float32)
        for i, b in enumerate(batch):
            v = np.asarray(getattr(b, k), np.float32)
            buf[i, :len(v)] = v
        out[k] = torch.from_numpy(buf).to(device)
    for k in ["base_R", "pred_R", "gt_R"]:
        buf = np.tile(np.eye(3, dtype=np.float32), (len(batch), T, 1, 1))
        for i, b in enumerate(batch):
            v = np.asarray(getattr(b, k), np.float32)
            buf[i, :len(v)] = v
        out[k] = torch.from_numpy(buf).to(device)
    buf = np.ones((len(batch), T), np.float32)
    for i, b in enumerate(batch):
        buf[i, :len(b.base_s)] = b.base_s
        buf[i, len(b.base_s):] = b.base_s[-1]
    out["base_s"] = torch.from_numpy(buf).to(device)
    for k in ["prompt_pose_mask", "prompt_depth_mask"]:
        buf = np.zeros((len(batch), T), np.float32)
        for i, b in enumerate(batch):
            buf[i, :len(getattr(b, k))] = np.asarray(getattr(b, k), np.float32)
        out[k] = torch.from_numpy(buf).to(device)
    buf = np.zeros((len(batch), T), np.float32)
    for i, b in enumerate(batch):
        buf[i, :len(b.prompt_depth_logratio)] = b.prompt_depth_logratio
    out["prompt_depth_logratio"] = torch.from_numpy(buf).to(device)
    valid = np.zeros((len(batch), T), np.float32)
    for i, b in enumerate(batch):
        valid[i, :len(b.feats)] = 1.0
    out["valid"] = torch.from_numpy(valid).to(device)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    depth: float = 1.0
    rotation: float = 1.0
    translation: float = 1.0
    smoothness: float = 0.05
    prompt_consistency: float = 0.2
    confidence: float = 0.05
    huber_delta_log: float = 0.1
    huber_delta_m: float = 1.0


def compute_losses(batch, s, R, t, logvar, increments, w: LossWeights):
    """All losses.  Ground truth appears here and nowhere else."""
    valid = batch["valid"]
    n = valid.sum().clamp_min(1.0)

    # -- depth: Huber on log metric depth ---------------------------------- #
    lr = batch["log_ratio"]                       # (B, T, S) = log(gt / pred)
    finite = torch.isfinite(lr)
    lr_f = torch.where(finite, lr, torch.zeros_like(lr))
    resid = torch.log(s.clamp_min(1e-6)).unsqueeze(-1) - lr_f
    hub = Fn.huber_loss(resid, torch.zeros_like(resid), reduction="none",
                        delta=w.huber_delta_log)
    m = finite.float() * valid.unsqueeze(-1)
    loss_depth = (hub * m).sum() / m.sum().clamp_min(1.0)

    # -- corrected poses ---------------------------------------------------- #
    corr_c = torch.einsum("btij,btj->bti", R, s.unsqueeze(-1) * batch["pred_c"]) + t
    corr_R = R @ batch["pred_R"]

    err = corr_c - batch["gt_c"]
    loss_trans = (Fn.huber_loss(err, torch.zeros_like(err), reduction="none",
                                delta=w.huber_delta_m).mean(-1) * valid).sum() / n

    rel = corr_R.transpose(-1, -2) @ batch["gt_R"]
    cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]) - 1.0) / 2.0
    loss_rot = ((torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))) * valid).sum() / n

    # -- temporal smoothness ------------------------------------------------ #
    dlog_s, wvec, dt = increments
    loss_smooth = ((dlog_s.squeeze(-1).abs() + wvec.abs().sum(-1) + dt.abs().sum(-1) * 0.1)
                   * valid).sum() / n

    # -- prompt consistency ------------------------------------------------- #
    pm = batch["prompt_pose_mask"] * valid
    pose_pc = ((corr_c - batch["prompt_pose_c"]).pow(2).sum(-1).clamp_min(1e-12).sqrt() * pm)
    dm = batch["prompt_depth_mask"] * valid
    depth_pc = ((torch.log(s.clamp_min(1e-6)) - batch["prompt_depth_logratio"]).abs() * dm)
    loss_prompt = (pose_pc.sum() / pm.sum().clamp_min(1.0)
                   + depth_pc.sum() / dm.sum().clamp_min(1.0))

    # -- confidence calibration (Gaussian NLL on the position error) --------- #
    if logvar is not None:
        lv = logvar.squeeze(-1).clamp(-8.0, 8.0)
        sq = err.pow(2).sum(-1)
        loss_conf = ((0.5 * torch.exp(-lv) * sq + 0.5 * lv) * valid).sum() / n
    else:
        loss_conf = torch.zeros((), device=s.device)

    total = (w.depth * loss_depth + w.rotation * loss_rot + w.translation * loss_trans
             + w.smoothness * loss_smooth + w.prompt_consistency * loss_prompt
             + w.confidence * loss_conf)
    return total, {
        "loss": float(total.detach()), "depth": float(loss_depth.detach()),
        "rotation": float(loss_rot.detach()), "translation": float(loss_trans.detach()),
        "smoothness": float(loss_smooth.detach()), "prompt": float(loss_prompt.detach()),
        "confidence": float(loss_conf.detach()),
    }


@torch.no_grad()
def validation_metrics(batch, s, R, t):
    """Held-out ATE and log-depth-scale error, in real units."""
    valid = batch["valid"]
    n = valid.sum().clamp_min(1.0)
    corr_c = torch.einsum("btij,btj->bti", R, s.unsqueeze(-1) * batch["pred_c"]) + t
    d = (corr_c - batch["gt_c"]).pow(2).sum(-1)
    ate = torch.sqrt((d * valid).sum() / n)
    lr = batch["log_ratio"]
    finite = torch.isfinite(lr)
    lr_f = torch.where(finite, lr, torch.zeros_like(lr))
    m = finite.float() * valid.unsqueeze(-1)
    med = (lr_f * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    scale_err = ((torch.log(s.clamp_min(1e-6)) - med).abs() * valid).sum() / n
    return {"val_ate_m": float(ate), "val_log_scale_err": float(scale_err)}
