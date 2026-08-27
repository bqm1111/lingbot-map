"""Training-free metric anchors.

Every anchor consumes a causal stream of ``(prediction, prompt)`` pairs and
produces a Sim(3) correction ``C_t = (s_t, R_t, t_t)`` applied as::

    x_metric = R_t (s_t x_pred) + t_t
    c_metric = R_t (s_t c_pred) + t_t
    D_metric = s_t D_pred

The world frame is the first camera's frame, so a scale-only correction about
the origin already rescales the whole trajectory correctly -- no recentring is
needed.

All anchors share :class:`MetricAnchor`.  ``update`` may only look at data at or
before the current frame; ``correct`` is a pure function of the state that
``update`` left behind.  :mod:`tests.prompted_lingbot.test_causality` checks this
by construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .conventions import Sim3, rotation_conditioning, sim3_degeneracy, umeyama_sim3
from .prompts import Prompt


@dataclass
class Prediction:
    """One frame of frozen LingbotMap output, in the model's arbitrary scale."""

    frame: int
    pose_c2w: np.ndarray        # (3, 4) camera-to-world
    depth: np.ndarray           # (H, W) Z-depth
    depth_conf: np.ndarray      # (H, W) expp1 confidence, > 1
    K: np.ndarray               # (3, 3)

    @property
    def centre(self) -> np.ndarray:
        return self.pose_c2w[:3, 3]


# --------------------------------------------------------------------------- #
def robust_log_scale(
    metric: np.ndarray,
    predicted: np.ndarray,
    weights: Optional[np.ndarray] = None,
    mad_k: float = 3.0,
    min_samples: int = 8,
) -> Optional[Tuple[float, float, int]]:
    """Weighted, outlier-rejected estimate of ``log(metric / predicted)``.

    Returns ``(log_scale, dispersion, n_inliers)`` or ``None`` if the sample is
    too small or degenerate.  Dispersion is the MAD of the inlier log ratios,
    which is what a downstream estimator should use to weight this observation.
    """
    metric = np.asarray(metric, float).ravel()
    predicted = np.asarray(predicted, float).ravel()
    ok = np.isfinite(metric) & np.isfinite(predicted) & (metric > 1e-6) & (predicted > 1e-6)
    if weights is not None:
        weights = np.asarray(weights, float).ravel()
        ok &= np.isfinite(weights) & (weights > 0)
    if ok.sum() < min_samples:
        return None
    lr = np.log(metric[ok]) - np.log(predicted[ok])
    w = np.ones_like(lr) if weights is None else weights[ok]

    med = float(np.median(lr))
    mad = float(np.median(np.abs(lr - med)))
    scaled_mad = 1.4826 * mad
    if scaled_mad < 1e-9:
        inlier = np.ones_like(lr, dtype=bool)
    else:
        inlier = np.abs(lr - med) <= mad_k * scaled_mad
    if inlier.sum() < min_samples:
        inlier = np.ones_like(lr, dtype=bool)

    lr_i, w_i = lr[inlier], w[inlier]
    order = np.argsort(lr_i)
    lr_s, w_s = lr_i[order], w_i[order]
    cw = np.cumsum(w_s)
    if cw[-1] <= 0:
        return None
    est = float(lr_s[np.searchsorted(cw, 0.5 * cw[-1])])
    disp = float(np.median(np.abs(lr_i - est)) * 1.4826)
    return est, disp, int(inlier.sum())


def sample_predicted_depth(pred: Prediction, rows: np.ndarray, cols: np.ndarray):
    """Predicted depth and confidence at the prompt's sample locations."""
    H, W = pred.depth.shape
    r = np.clip(rows.astype(int), 0, H - 1)
    c = np.clip(cols.astype(int), 0, W - 1)
    return pred.depth[r, c].astype(np.float64), pred.depth_conf[r, c].astype(np.float64)


# --------------------------------------------------------------------------- #
class MetricAnchor:
    """Base class.  Subclasses override :meth:`reset` and :meth:`update`."""

    name = "base"
    is_causal = True

    def __init__(self) -> None:
        self.reset()

    # -- lifecycle --------------------------------------------------------- #
    def reset(self) -> None:
        """Clear all state.  Must be called between sequences."""
        self._correction = Sim3.identity()
        self._n_updates = 0

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        """Fold frame ``prediction.frame``'s observation into the state."""
        raise NotImplementedError

    # -- output ------------------------------------------------------------ #
    @property
    def correction(self) -> Sim3:
        return self._correction

    def correct(self, prediction: Prediction) -> Dict[str, np.ndarray]:
        """Apply the current correction to one prediction."""
        C = self._correction
        return {
            "pose_c2w": C.apply_pose_c2w(prediction.pose_c2w),
            "depth": C.apply_depth(prediction.depth),
            "centre": C.apply_centres(prediction.centre),
            "scale": C.s,
        }

    # -- serialisation ----------------------------------------------------- #
    def state_dict(self) -> dict:
        return {"name": self.name, "correction": self._correction.to_dict(),
                "n_updates": int(self._n_updates)}

    def load_state_dict(self, state: dict) -> None:
        if state["name"] != self.name:
            raise ValueError(f"state is for {state['name']!r}, this is {self.name!r}")
        self._correction = Sim3.from_dict(state["correction"])
        self._n_updates = int(state["n_updates"])


# --------------------------------------------------------------------------- #
# 1. Raw
# --------------------------------------------------------------------------- #
class RawAnchor(MetricAnchor):
    """No correction at all.  The frozen model's own arbitrary scale."""

    name = "raw"

    def update(self, prediction: Prediction, prompt: Prompt) -> None:  # noqa: D102
        return None


# --------------------------------------------------------------------------- #
# 2. First-depth scale
# --------------------------------------------------------------------------- #
class FirstDepthScaleAnchor(MetricAnchor):
    """One robust scale from the first depth prompt, then frozen forever."""

    name = "first_depth_scale"

    def reset(self) -> None:
        super().reset()
        self._locked = False

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        if self._locked or not prompt.has_depth:
            return
        pd, pc = sample_predicted_depth(prediction, prompt.depth_rows, prompt.depth_cols)
        est = robust_log_scale(prompt.depth_values, pd, weights=pc)
        if est is None:
            return
        self._correction = Sim3(float(np.exp(est[0])), np.eye(3), np.zeros(3))
        self._locked = True
        self._n_updates += 1

    def state_dict(self) -> dict:
        return dict(super().state_dict(), locked=self._locked)

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._locked = bool(state["locked"])


# --------------------------------------------------------------------------- #
# 3. Running robust depth scale
# --------------------------------------------------------------------------- #
class RunningDepthScaleAnchor(MetricAnchor):
    """Causal inverse-variance filter on the log scale, with MAD gating.

    Each depth prompt contributes one robust log-ratio observation with its own
    dispersion.  Observations are fused by inverse-variance weighting, with a
    forgetting factor so the estimate can track slow drift, and a gate that
    rejects an observation lying more than ``gate_k`` current standard deviations
    from the running estimate.
    """

    name = "running_depth_scale"

    def __init__(self, forgetting: float = 0.9, gate_k: float = 4.0,
                 min_sigma: float = 0.01, max_consecutive_rejections: int = 3) -> None:
        self.forgetting = float(forgetting)
        self.gate_k = float(gate_k)
        self.min_sigma = float(min_sigma)
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self._mu = 0.0
        self._prec = 0.0          # accumulated precision (1/variance)
        self._rejected = 0
        self._consecutive_rejections = 0

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        if not prompt.has_depth:
            return
        pd, pc = sample_predicted_depth(prediction, prompt.depth_rows, prompt.depth_cols)
        est = robust_log_scale(prompt.depth_values, pd, weights=pc)
        if est is None:
            return
        z, disp, n = est
        sigma = max(disp / np.sqrt(max(n, 1)), self.min_sigma)
        sigma /= max(prompt.depth_confidence, 1e-3) ** 0.5

        if self._prec > 0.0:
            # Confidence decays on every observation, accepted or not, so the
            # filter cannot lock itself out of tracking a genuine change.
            self._prec *= self.forgetting
            cur_sigma = np.sqrt(1.0 / self._prec)
            if abs(z - self._mu) > self.gate_k * np.sqrt(cur_sigma ** 2 + sigma ** 2):
                self._rejected += 1
                self._consecutive_rejections += 1
                if self._consecutive_rejections < self.max_consecutive_rejections:
                    return
                # Several observations in a row agree with each other and not with
                # us: the world changed, we did not.  Restart from this one.
                self._mu, self._prec = z, 0.0
            self._consecutive_rejections = 0

        prec_obs = 1.0 / (sigma ** 2)
        new_prec = self._prec + prec_obs
        self._mu = (self._mu * self._prec + z * prec_obs) / new_prec
        self._prec = new_prec
        self._correction = Sim3(float(np.exp(self._mu)), np.eye(3), np.zeros(3))
        self._n_updates += 1

    def state_dict(self) -> dict:
        return dict(super().state_dict(), mu=self._mu, prec=self._prec,
                    rejected=self._rejected,
                    consecutive_rejections=self._consecutive_rejections)

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._mu = float(state["mu"])
        self._prec = float(state["prec"])
        self._rejected = int(state["rejected"])
        self._consecutive_rejections = int(state.get("consecutive_rejections", 0))


# --------------------------------------------------------------------------- #
# 4/5. Pose-based Sim(3)
# --------------------------------------------------------------------------- #
class _PoseSim3Base(MetricAnchor):
    """Shared machinery for Sim(3) from prompted camera-centre correspondences.

    Rotation acceptance is governed by a **spatial-conditioning** test, not by a
    singular-value ratio.  The ratio test that this replaced never fired on real
    driving data -- measured ratios were 0.033-0.085, always above its 5e-3
    threshold -- while the fitted rotation reached 92 deg of error, because a
    short window of a locally straight trajectory pins the camera *centres*
    without constraining the orientation about the direction of travel.

    A rotation update is accepted only when all of the following hold:

    * enough correspondences (``min_correspondences``);
    * the least-observed extent of the correspondence cloud clears an absolute
      floor (``min_extent_m``) -- this is what a straight run fails;
    * that extent is a non-trivial fraction of the baseline (``min_extent_ratio``);
    * the baseline is a non-trivial fraction of the scene depth the rotation will
      be applied to (``min_baseline_over_depth``);
    * the first-order angular uncertainty, ``fit residual / least extent``, is
      below ``max_rotation_sigma_rad``.

    When the test fails the previous valid rotation is **held** and only the
    observable scale and translation are updated.  A solution is never accepted
    because camera-centre ATE happens to be low.
    """

    min_correspondences = 4
    min_degeneracy = 5e-3               # legacy ratio test, kept as a floor
    min_extent_m = 0.25                 # absolute least-extent floor, metres
    min_extent_ratio = 0.01             # least extent / baseline
    min_baseline_over_depth = 0.02      # baseline / median scene depth
    max_rotation_sigma_rad = 0.05       # ~2.9 deg of angular uncertainty
    min_prompt_sigma_m = 0.05           # noise floor for a "clean" pose prompt

    def __init__(self, window: Optional[int] = None, weighted: bool = True) -> None:
        self.window = window
        self.weighted = weighted
        super().__init__()

    def reset(self) -> None:
        super().reset()
        maxlen = self.window if self.window else None
        self._pred: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._metric: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._weight: Deque[float] = deque(maxlen=maxlen)
        self._sigma: Deque[float] = deque(maxlen=maxlen)
        self._degenerate_skips = 0
        self._rotation_updates = 0
        self._median_pred_depth = 0.0
        self._last_conditioning: Dict[str, float] = {}

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        # Track scene depth in the model's own units; converted to metric with
        # the current scale when the conditioning test needs it.
        d = prediction.depth[::8, ::8]
        d = d[np.isfinite(d) & (d > 1e-6)]
        if d.size:
            self._median_pred_depth = float(np.median(d))
        if not prompt.has_pose:
            return
        self._pred.append(np.asarray(prediction.centre, float).copy())
        self._metric.append(np.asarray(prompt.pose_c2w[:3, 3], float).copy())
        self._weight.append(max(float(prompt.pose_confidence), 1e-3))
        # The sensor's own declared accuracy. pose_confidence is exp(-sigma_m),
        # so this inverts it back to metres. This -- not the fit residual -- is
        # the noise the correspondence extent has to beat.
        conf = min(max(float(prompt.pose_confidence), 1e-6), 1.0)
        self._sigma.append(max(-np.log(conf), self.min_prompt_sigma_m))
        self._refit()

    # -- conditioning ------------------------------------------------------ #
    def rotation_is_observable(self, src: np.ndarray, dst: np.ndarray,
                               candidate: Sim3) -> Tuple[bool, Dict[str, float]]:
        """Spatial-conditioning test for a candidate rotation.

        The angular uncertainty is computed from the **sensor's declared noise**,
        not from the fit residual.  Using the residual would conflate two
        different things: a large residual usually means the frozen model has
        drifted, which is a reason to *apply* a correction, not to refuse one.
        What actually makes a rotation unobservable is a correspondence cloud
        whose least extent is comparable to the noise in the correspondences.
        """
        resid = dst - candidate.apply_points(src)
        residual_rms = float(np.sqrt((resid ** 2).sum(1).mean()))
        sigma_noise = float(np.mean(self._sigma)) if self._sigma else self.min_prompt_sigma_m
        depth_metric = self._median_pred_depth * candidate.s
        cond = rotation_conditioning(dst, residual_rms=sigma_noise,
                                     median_scene_depth=depth_metric)
        cond["residual_rms"] = residual_rms
        cond["prompt_sigma_m"] = sigma_noise
        cond["median_scene_depth_m"] = float(depth_metric)
        ok = (
            cond["n"] >= self.min_correspondences
            and cond["extent"][2] >= self.min_extent_m
            and cond["extent_ratio"] >= self.min_extent_ratio
            and cond["rotation_sigma_rad"] <= self.max_rotation_sigma_rad
            and sim3_degeneracy(src) >= self.min_degeneracy
        )
        if depth_metric > 1e-9:
            ok = ok and cond["baseline_over_depth"] >= self.min_baseline_over_depth
        cond["accepted"] = bool(ok)
        return bool(ok), cond

    def _refit(self) -> None:
        n = len(self._pred)
        if n < self.min_correspondences:
            # Two or three points still pin down scale + translation; fitting a
            # rotation from them is unstable, so leave the correction alone.
            return
        src = np.stack(self._pred)
        dst = np.stack(self._metric)
        if self.weighted:
            w = np.asarray(self._weight, float)
            reps = np.clip(np.round(w / max(w.max(), 1e-9) * 4).astype(int), 1, 4)
            src_w = np.repeat(src, reps, axis=0)
            dst_w = np.repeat(dst, reps, axis=0)
        else:
            src_w, dst_w = src, dst

        candidate = umeyama_sim3(src_w, dst_w)
        ok, cond = self.rotation_is_observable(src, dst, candidate)
        self._last_conditioning = cond
        if not ok:
            # Orientation is not observable from this correspondence set: hold the
            # last valid rotation and update only scale and translation.
            self._degenerate_skips += 1
            self._fit_scale_translation_only(src_w, dst_w)
            return
        self._correction = candidate
        self._rotation_updates += 1
        self._n_updates += 1

    def _fit_scale_translation_only(self, src: np.ndarray, dst: np.ndarray) -> None:
        R = self._correction.R
        rs = src @ R.T
        num = float(((dst - dst.mean(0)) * (rs - rs.mean(0))).sum())
        den = float(((rs - rs.mean(0)) ** 2).sum())
        if den < 1e-12:
            return
        s = num / den
        if not np.isfinite(s) or s <= 0:
            return
        t = dst.mean(0) - s * rs.mean(0)
        self._correction = Sim3(s, R, t)
        self._n_updates += 1

    @property
    def rotation_held_fraction(self) -> float:
        """Fraction of refits where the rotation was held because it was not
        observable.  Report this alongside any ATE number from this anchor."""
        total = self._degenerate_skips + self._rotation_updates
        return float(self._degenerate_skips / total) if total else 0.0

    def state_dict(self) -> dict:
        return dict(super().state_dict(),
                    pred=[p.tolist() for p in self._pred],
                    metric=[m.tolist() for m in self._metric],
                    weight=list(self._weight),
                    sigma=list(self._sigma),
                    degenerate_skips=self._degenerate_skips,
                    rotation_updates=self._rotation_updates,
                    median_pred_depth=self._median_pred_depth)

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        maxlen = self.window if self.window else None
        self._pred = deque([np.asarray(p, float) for p in state["pred"]], maxlen=maxlen)
        self._metric = deque([np.asarray(m, float) for m in state["metric"]], maxlen=maxlen)
        self._weight = deque(state["weight"], maxlen=maxlen)
        self._sigma = deque(state.get("sigma", []), maxlen=maxlen)
        self._degenerate_skips = int(state["degenerate_skips"])
        self._rotation_updates = int(state.get("rotation_updates", 0))
        self._median_pred_depth = float(state.get("median_pred_depth", 0.0))


class CausalPoseSim3Anchor(_PoseSim3Base):
    """Weighted Sim(3) refit from every pose correspondence seen so far."""

    name = "causal_pose_sim3"

    def __init__(self, weighted: bool = True) -> None:
        super().__init__(window=None, weighted=weighted)


class SlidingWindowSim3Anchor(_PoseSim3Base):
    """Weighted Sim(3) from only the most recent ``window`` correspondences."""

    name = "sliding_pose_sim3"

    def __init__(self, window: int = 10, weighted: bool = True) -> None:
        super().__init__(window=window, weighted=weighted)


# --------------------------------------------------------------------------- #
# 7. Combined: depth for scale, pose for the rigid part
# --------------------------------------------------------------------------- #
class DepthScalePoseSE3Anchor(MetricAnchor):
    """Strongest causal training-free combination.

    Scale comes from the running robust depth filter; the rigid part comes from
    an SE(3) fit of the *already rescaled* predicted centres onto the prompted
    metric centres.  Falls back gracefully when only one modality is present.
    """

    name = "depth_scale_pose_se3"

    def __init__(self, window: Optional[int] = None) -> None:
        self.window = window
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self._scale = RunningDepthScaleAnchor()
        self._pose = _PoseSim3Base(window=self.window)
        self._pose.name = "internal_pose"
        self._have_depth_scale = False
        self._rigid_updates = 0
        self._rigid_held = 0
        self._last_conditioning: Dict[str, float] = {}

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        self._scale.update(prediction, prompt)
        if self._scale.correction.s != 1.0:
            self._have_depth_scale = True
        self._pose.update(prediction, prompt)

        s_depth = self._scale.correction.s
        pose_fit = self._pose.correction
        if self._have_depth_scale and len(self._pose._pred) >= self._pose.min_correspondences:
            # Re-fit the rigid part with the scale pinned by the depth prompts.
            # This fit is subject to the SAME spatial-conditioning test as the
            # standalone pose anchors -- without it this path silently reintroduced
            # the unobservable-rotation defect (measured 16.6 deg of orientation
            # error against 3.3 deg for the frozen model) while its own
            # rotation_held_fraction still read 1.00, because the held rotation was
            # being overwritten here.
            src = np.stack(self._pose._pred) * s_depth
            dst = np.stack(self._pose._metric)
            rigid = umeyama_sim3(src, dst, with_scale=False)
            ok, cond = self._pose.rotation_is_observable(
                src, dst, Sim3(1.0, rigid.R, rigid.t))
            self._last_conditioning = cond
            if ok:
                self._correction = Sim3(s_depth, rigid.R, rigid.t)
                self._rigid_updates += 1
            else:
                # Orientation not observable: keep the previous rotation and fit
                # only the translation that goes with the depth-derived scale.
                R = self._correction.R
                t = dst.mean(0) - (src @ R.T).mean(0)
                self._correction = Sim3(s_depth, R, t)
                self._rigid_held += 1
        elif self._have_depth_scale:
            self._correction = Sim3(s_depth, np.eye(3), np.zeros(3))
        else:
            self._correction = pose_fit
        self._n_updates += 1

    @property
    def rotation_held_fraction(self) -> float:
        total = self._rigid_updates + self._rigid_held
        return float(self._rigid_held / total) if total else 0.0

    def state_dict(self) -> dict:
        return dict(super().state_dict(), scale=self._scale.state_dict(),
                    pose=self._pose.state_dict(), have_depth_scale=self._have_depth_scale,
                    rigid_updates=self._rigid_updates, rigid_held=self._rigid_held)

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._scale.load_state_dict(state["scale"])
        self._pose.load_state_dict(state["pose"])
        self._have_depth_scale = bool(state["have_depth_scale"])
        self._rigid_updates = int(state.get("rigid_updates", 0))
        self._rigid_held = int(state.get("rigid_held", 0))


# --------------------------------------------------------------------------- #
# 6. Offline oracle -- NOT a method, a diagnostic upper reference
# --------------------------------------------------------------------------- #
class OfflineOracleSim3Anchor(MetricAnchor):
    """Non-causal Sim(3) fitted to ALL ground-truth camera centres at once.

    This is an upper reference for how much a single global similarity can
    possibly buy.  It sees the future and must never be reported as a method.
    """

    name = "offline_oracle_sim3"
    is_causal = False

    def reset(self) -> None:
        super().reset()
        self._fitted = False

    def fit(self, pred_centres: np.ndarray, gt_centres: np.ndarray) -> None:
        self._correction = umeyama_sim3(np.asarray(pred_centres, float),
                                        np.asarray(gt_centres, float))
        self._fitted = True
        self._n_updates += 1

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        return None

    def state_dict(self) -> dict:
        return dict(super().state_dict(), fitted=self._fitted)

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._fitted = bool(state["fitted"])


# --------------------------------------------------------------------------- #
BASELINES = {
    "raw": lambda: RawAnchor(),
    "first_depth_scale": lambda: FirstDepthScaleAnchor(),
    "running_depth_scale": lambda: RunningDepthScaleAnchor(),
    "causal_pose_sim3": lambda: CausalPoseSim3Anchor(),
    "sliding_pose_sim3": lambda: SlidingWindowSim3Anchor(window=10),
    "depth_scale_pose_se3": lambda: DepthScalePoseSE3Anchor(),
    "offline_oracle_sim3": lambda: OfflineOracleSim3Anchor(),
}

CAUSAL_BASELINES = [k for k, v in BASELINES.items() if v().is_causal]


# --------------------------------------------------------------------------- #
# Diagnostic: the floor a per-frame scale could reach.  Non-causal.
# --------------------------------------------------------------------------- #
class PerFrameOracleScaleAnchor(MetricAnchor):
    """Per-frame oracle depth scale.  Shows the residual model error once scale
    is perfectly known at every frame, i.e. the floor that any scale-correction
    method -- learned or not -- can possibly reach.  Never a method."""

    name = "per_frame_oracle_scale"
    is_causal = False

    def reset(self) -> None:
        super().reset()
        self._schedule: Dict[int, float] = {}

    def set_schedule(self, scales: Dict[int, float]) -> None:
        self._schedule = dict(scales)
        self._fitted = True

    def update(self, prediction: Prediction, prompt: Prompt) -> None:
        s = self._schedule.get(prediction.frame)
        if s is not None and np.isfinite(s) and s > 0:
            self._correction = Sim3(float(s), np.eye(3), np.zeros(3))
            self._n_updates += 1


BASELINES["per_frame_oracle_scale"] = lambda: PerFrameOracleScaleAnchor()
CAUSAL_BASELINES = [k for k, v in BASELINES.items() if v().is_causal]
