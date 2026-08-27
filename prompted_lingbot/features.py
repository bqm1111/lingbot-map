"""Causal input features for the learned corrector.

Every feature at frame ``t`` is a function of frames ``<= t`` only.  Ground truth
never enters except through a *prompt*, which is the sensor observation the
method is explicitly allowed to consume.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .anchors import MetricAnchor, Prediction, robust_log_scale, sample_predicted_depth
from .conventions import Sim3, rotation_geodesic_deg
from .prompts import Prompt

FEATURE_NAMES = [
    "log_step_translation",     # predicted inter-frame motion, model units
    "step_rotation_rad",
    "log_depth_median",
    "log_depth_p10",
    "log_depth_p90",
    "depth_conf_mean",
    "depth_conf_frac_gt2",
    "has_depth_prompt",
    "depth_prompt_log_ratio",   # robust log(metric / predicted) at this frame
    "depth_prompt_dispersion",
    "depth_prompt_log_n",
    "depth_prompt_confidence",
    "has_pose_prompt",
    "pose_residual_x",          # baseline-corrected centre minus prompted centre
    "pose_residual_y",
    "pose_residual_z",
    "pose_residual_norm",
    "pose_residual_rot_deg",
    "pose_prompt_confidence",
    "recency_depth_exp",        # exp(-gap / 30)
    "recency_pose_exp",
    "log1p_gap_depth",
    "log1p_gap_pose",
    "base_log_scale",           # where the training-free baseline currently sits
]
N_FEATURES = len(FEATURE_NAMES)


def _depth_stats(depth: np.ndarray, conf: np.ndarray, stride: int = 4):
    d = depth[::stride, ::stride].astype(np.float64).ravel()
    c = conf[::stride, ::stride].astype(np.float64).ravel()
    ok = np.isfinite(d) & (d > 1e-6)
    if not ok.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0
    d = d[ok]
    lq = np.log(np.quantile(d, [0.1, 0.5, 0.9]))
    return float(lq[1]), float(lq[0]), float(lq[2]), float(np.mean(c)), float(np.mean(c > 2.0))


def sequence_features(
    predictions: Sequence[Prediction],
    prompts: Sequence[Prompt],
    base_anchor: Optional[MetricAnchor] = None,
) -> Tuple[np.ndarray, List[Sim3]]:
    """``(T, N_FEATURES)`` features plus the base anchor's per-frame correction.

    The base anchor is driven over the same causal stream, so its correction at
    frame ``t`` is itself causal and safe to use as a feature and as the point
    the learned increment is applied on top of.
    """
    T = len(predictions)
    F = np.zeros((T, N_FEATURES), dtype=np.float32)
    base_corrections: List[Sim3] = []
    if base_anchor is not None:
        base_anchor.reset()

    last_depth = last_pose = -1
    for t in range(T):
        pred, pr = predictions[t], prompts[t]
        if base_anchor is not None:
            base_anchor.update(pred, pr)
            base = base_anchor.correction
        else:
            base = Sim3.identity()
        base_corrections.append(base)

        row = {}
        if t > 0:
            dt = pred.centre - predictions[t - 1].centre
            row["log_step_translation"] = float(np.log1p(np.linalg.norm(dt) * 1e3))
            row["step_rotation_rad"] = float(np.radians(rotation_geodesic_deg(
                pred.pose_c2w[:3, :3], predictions[t - 1].pose_c2w[:3, :3])))
        med, p10, p90, cmean, cfrac = _depth_stats(pred.depth, pred.depth_conf)
        row["log_depth_median"], row["log_depth_p10"], row["log_depth_p90"] = med, p10, p90
        row["depth_conf_mean"], row["depth_conf_frac_gt2"] = cmean, cfrac

        if pr.has_depth:
            pd, pc = sample_predicted_depth(pred, pr.depth_rows, pr.depth_cols)
            est = robust_log_scale(pr.depth_values, pd, weights=pc)
            if est is not None:
                z, disp, n = est
                row["has_depth_prompt"] = 1.0
                row["depth_prompt_log_ratio"] = float(z)
                row["depth_prompt_dispersion"] = float(disp)
                row["depth_prompt_log_n"] = float(np.log1p(n))
                row["depth_prompt_confidence"] = float(pr.depth_confidence)
                last_depth = t

        if pr.has_pose:
            corrected_centre = base.apply_centres(pred.centre)
            res = corrected_centre - pr.pose_c2w[:3, 3]
            row["has_pose_prompt"] = 1.0
            row["pose_residual_x"], row["pose_residual_y"], row["pose_residual_z"] = res.tolist()
            row["pose_residual_norm"] = float(np.linalg.norm(res))
            row["pose_residual_rot_deg"] = float(rotation_geodesic_deg(
                base.apply_rotations(pred.pose_c2w[:3, :3]), pr.pose_c2w[:3, :3]))
            row["pose_prompt_confidence"] = float(pr.pose_confidence)
            last_pose = t

        gap_d = (t - last_depth) if last_depth >= 0 else 1000
        gap_p = (t - last_pose) if last_pose >= 0 else 1000
        row["recency_depth_exp"] = float(np.exp(-gap_d / 30.0))
        row["recency_pose_exp"] = float(np.exp(-gap_p / 30.0))
        row["log1p_gap_depth"] = float(np.log1p(gap_d) / 7.0)
        row["log1p_gap_pose"] = float(np.log1p(gap_p) / 7.0)
        row["base_log_scale"] = float(np.log(max(base.s, 1e-6)))

        for k, v in row.items():
            F[t, FEATURE_NAMES.index(k)] = v

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(F, -50.0, 50.0, out=F)
    return F, base_corrections
