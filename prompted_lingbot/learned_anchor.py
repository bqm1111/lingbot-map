"""Inference-time wrapper that makes the trained corrector a drop-in MetricAnchor.

The GRU is stepped one frame at a time with a carried hidden state, so the
evaluation path is causal in exactly the same sense as the training-free anchors
and is measured by the same causality tests.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from .anchors import BASELINES, MetricAnchor, Prediction
from .conventions import Sim3, axis_angle_to_matrix
from .features import FEATURE_NAMES, N_FEATURES
from .model import CausalCorrector, CorrectorConfig
from .prompts import Prompt


class LearnedCorrectorAnchor(MetricAnchor):
    """Runs :class:`CausalCorrector` frame by frame on top of a base anchor."""

    name = "learned_corrector"

    def __init__(self, model: CausalCorrector, base_anchor_name: str = "depth_scale_pose_se3",
                 device: str = "cpu") -> None:
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.base_anchor_name = base_anchor_name
        self._feature_state = None
        super().__init__()

    # -- lifecycle --------------------------------------------------------- #
    def reset(self) -> None:
        super().reset()
        self._hidden = None
        self._base = BASELINES[self.base_anchor_name]() if self.base_anchor_name else None
        if self._base is not None:
            self._base.reset()
        self._prev_pose: Optional[np.ndarray] = None
        self._last_depth = -1
        self._last_pose = -1
        self._t = 0
        self._running = Sim3.identity()
        self._confidence = 0.0

    # -- one causal step --------------------------------------------------- #
    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        from .features import _depth_stats
        from .anchors import robust_log_scale, sample_predicted_depth
        from .conventions import rotation_geodesic_deg

        if self._base is not None:
            self._base.update(prediction, prompt)
            base = self._base.correction
        else:
            base = Sim3.identity()

        f = np.zeros(N_FEATURES, np.float32)

        def put(name, v):
            f[FEATURE_NAMES.index(name)] = v

        if self._prev_pose is not None:
            put("log_step_translation",
                np.log1p(np.linalg.norm(prediction.centre - self._prev_pose[:3, 3]) * 1e3))
            put("step_rotation_rad", np.radians(rotation_geodesic_deg(
                prediction.pose_c2w[:3, :3], self._prev_pose[:3, :3])))
        med, p10, p90, cmean, cfrac = _depth_stats(prediction.depth, prediction.depth_conf)
        put("log_depth_median", med); put("log_depth_p10", p10); put("log_depth_p90", p90)
        put("depth_conf_mean", cmean); put("depth_conf_frac_gt2", cfrac)

        if prompt.has_depth:
            pd, pc = sample_predicted_depth(prediction, prompt.depth_rows, prompt.depth_cols)
            est = robust_log_scale(prompt.depth_values, pd, weights=pc)
            if est is not None:
                z, disp, n = est
                put("has_depth_prompt", 1.0); put("depth_prompt_log_ratio", z)
                put("depth_prompt_dispersion", disp); put("depth_prompt_log_n", np.log1p(n))
                put("depth_prompt_confidence", prompt.depth_confidence)
                self._last_depth = self._t
        if prompt.has_pose:
            res = base.apply_centres(prediction.centre) - prompt.pose_c2w[:3, 3]
            put("has_pose_prompt", 1.0)
            put("pose_residual_x", res[0]); put("pose_residual_y", res[1])
            put("pose_residual_z", res[2]); put("pose_residual_norm", np.linalg.norm(res))
            put("pose_residual_rot_deg", rotation_geodesic_deg(
                base.apply_rotations(prediction.pose_c2w[:3, :3]), prompt.pose_c2w[:3, :3]))
            put("pose_prompt_confidence", prompt.pose_confidence)
            self._last_pose = self._t

        gap_d = (self._t - self._last_depth) if self._last_depth >= 0 else 1000
        gap_p = (self._t - self._last_pose) if self._last_pose >= 0 else 1000
        put("recency_depth_exp", np.exp(-gap_d / 30.0))
        put("recency_pose_exp", np.exp(-gap_p / 30.0))
        put("log1p_gap_depth", np.log1p(gap_d) / 7.0)
        put("log1p_gap_pose", np.log1p(gap_p) / 7.0)
        put("base_log_scale", np.log(max(base.s, 1e-6)))
        f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)

        with torch.no_grad():
            x = torch.from_numpy(f)[None, None].to(self.device)
            dlog_s, w, dt, logvar, self._hidden = self.model(x, self._hidden)
        dlog_s = float(dlog_s[0, 0, 0]); w = w[0, 0].cpu().numpy().astype(np.float64)
        dt = dt[0, 0].cpu().numpy().astype(np.float64)
        self._confidence = float(np.exp(-0.5 * float(logvar[0, 0, 0]))) if logvar is not None else 0.0
        delta = Sim3(float(np.exp(dlog_s)), axis_angle_to_matrix(w), dt)

        if self.model.cfg.base == "anchor":
            self._correction = delta.compose(base)
        else:
            self._running = delta.compose(self._running)
            self._correction = self._running

        self._prev_pose = np.asarray(prediction.pose_c2w, float).copy()
        self._t += 1
        self._n_updates += 1

    @property
    def confidence(self) -> float:
        return self._confidence

    # -- serialisation ----------------------------------------------------- #
    def state_dict(self) -> dict:
        return dict(super().state_dict(), t=self._t, last_depth=self._last_depth,
                    last_pose=self._last_pose,
                    prev_pose=None if self._prev_pose is None else self._prev_pose.tolist(),
                    hidden=None if self._hidden is None else self._hidden.cpu().tolist(),
                    base=None if self._base is None else self._base.state_dict(),
                    running=self._running.to_dict())

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._t = int(state["t"])
        self._last_depth = int(state["last_depth"])
        self._last_pose = int(state["last_pose"])
        self._prev_pose = (None if state.get("prev_pose") is None
                           else np.asarray(state["prev_pose"], float))
        self._hidden = (None if state["hidden"] is None
                        else torch.tensor(state["hidden"], device=self.device))
        if state["base"] is not None and self._base is not None:
            self._base.load_state_dict(state["base"])
        self._running = Sim3.from_dict(state["running"])


def load_checkpoint(path: str, device: str = "cpu"):
    """Rebuild a :class:`LearnedCorrectorAnchor` factory from a training run."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CorrectorConfig(**ck["model_config"])
    model = CausalCorrector(cfg)
    model.load_state_dict(ck["model_state"])
    base = ck.get("base_anchor", "depth_scale_pose_se3")
    return lambda: LearnedCorrectorAnchor(model, base_anchor_name=base, device=device), ck
