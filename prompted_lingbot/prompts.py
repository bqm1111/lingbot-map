"""Causal prompt simulation.

A *prompt* is a sparse metric observation that a real sensor rig could supply at
a given frame: a handful of metric depth samples (a sparse LiDAR return, a ToF
patch) and/or a metric camera pose (a survey-grade GNSS/INS fix, a fiducial).

Two properties are enforced by construction:

* **Causality.** The prompt for frame ``t`` is generated from a per-frame RNG
  seeded by ``(seed, sequence, frame)`` alone.  Nothing about frame ``t`` depends
  on any other frame, so changing a future prompt cannot alter an earlier one.
  ``tests/prompted_lingbot/test_prompts.py`` asserts this.
* **Determinism.** The same ``(PromptConfig, sequence name)`` always yields the
  same prompts, on any machine, regardless of evaluation order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conventions import axis_angle_to_matrix


def _stable_hash(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def _frame_rng(seed: int, sequence: str, frame: int, tag: str) -> np.random.Generator:
    return np.random.default_rng(_stable_hash(seed, sequence, frame, tag))


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromptConfig:
    """Which frames get prompted, and how badly the sensor lies."""

    name: str = "unnamed"

    # -- schedule ---------------------------------------------------------- #
    depth_interval: Optional[int] = None    # depth prompt every K frames
    pose_interval: Optional[int] = None     # pose prompt every K frames
    depth_bernoulli: Optional[float] = None  # irregular: per-frame probability
    pose_bernoulli: Optional[float] = None
    first_frame_only: bool = False          # a single prompt at t=0, then image-only
    depth_enabled: bool = True
    pose_enabled: bool = True

    # -- depth corruption -------------------------------------------------- #
    num_depth_samples: int = 512            # sparsification: samples per prompt
    depth_missing: float = 0.0              # fraction of the samples dropped
    depth_noise_sigma: float = 0.0          # multiplicative log-normal sigma
    depth_outlier_rate: float = 0.0         # fraction replaced by a wild value
    depth_outlier_scale: float = 4.0        # multiplicative magnitude of outliers

    # -- pose corruption --------------------------------------------------- #
    translation_noise_m: float = 0.0
    rotation_noise_deg: float = 0.0

    # -- availability ------------------------------------------------------ #
    dropout: float = 0.0                    # probability a scheduled prompt vanishes

    seed: int = 0

    def key(self) -> str:
        return self.name

    def with_seed(self, seed: int) -> "PromptConfig":
        return replace(self, seed=seed)


@dataclass
class Prompt:
    """What the corrector is allowed to see at one frame."""

    frame: int
    has_depth: bool = False
    has_pose: bool = False
    # sparse depth samples on the cached lattice
    depth_rows: Optional[np.ndarray] = None      # (M,) int
    depth_cols: Optional[np.ndarray] = None      # (M,) int
    depth_values: Optional[np.ndarray] = None    # (M,) metric metres
    pose_c2w: Optional[np.ndarray] = None        # (3, 4) metric, possibly noisy
    # the confidence the sensor advertises (declared, not measured)
    depth_confidence: float = 0.0
    pose_confidence: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def num_depth_samples(self) -> int:
        return 0 if self.depth_values is None else int(self.depth_values.size)

    @property
    def is_empty(self) -> bool:
        return not (self.has_depth or self.has_pose)


# --------------------------------------------------------------------------- #
def _scheduled(cfg: PromptConfig, frame: int, n_frames: int, kind: str) -> bool:
    enabled = cfg.depth_enabled if kind == "depth" else cfg.pose_enabled
    if not enabled:
        return False
    if cfg.first_frame_only:
        return frame == 0
    interval = cfg.depth_interval if kind == "depth" else cfg.pose_interval
    bern = cfg.depth_bernoulli if kind == "depth" else cfg.pose_bernoulli
    if bern is not None:
        return None  # resolved by the caller, which owns the RNG
    if interval is None:
        return False
    return frame % interval == 0


def make_prompt(
    cfg: PromptConfig,
    sequence: str,
    frame: int,
    n_frames: int,
    gt_depth: Optional[np.ndarray] = None,
    gt_valid: Optional[np.ndarray] = None,
    gt_pose_c2w: Optional[np.ndarray] = None,
) -> Prompt:
    """Build the prompt for a single frame.

    Depends only on ``frame``'s own ground truth and on ``(cfg.seed, sequence,
    frame)``.  This is what makes the schedule causal and order-independent.
    """
    p = Prompt(frame=frame)

    want_depth = _scheduled(cfg, frame, n_frames, "depth")
    if want_depth is None:
        want_depth = _frame_rng(cfg.seed, sequence, frame, "sched_depth").random() < cfg.depth_bernoulli
    want_pose = _scheduled(cfg, frame, n_frames, "pose")
    if want_pose is None:
        want_pose = _frame_rng(cfg.seed, sequence, frame, "sched_pose").random() < cfg.pose_bernoulli

    if cfg.dropout > 0.0:
        drng = _frame_rng(cfg.seed, sequence, frame, "dropout")
        if want_depth and drng.random() < cfg.dropout:
            want_depth = False
        if want_pose and drng.random() < cfg.dropout:
            want_pose = False

    # -- depth ------------------------------------------------------------- #
    if want_depth and gt_depth is not None and gt_valid is not None and gt_valid.any():
        rng = _frame_rng(cfg.seed, sequence, frame, "depth")
        rows, cols = np.nonzero(gt_valid)
        n = rows.size
        take = min(cfg.num_depth_samples, n)
        pick = rng.choice(n, size=take, replace=False)
        rows, cols = rows[pick], cols[pick]
        vals = gt_depth[rows, cols].astype(np.float64)

        if cfg.depth_missing > 0.0:
            keep = rng.random(vals.size) >= cfg.depth_missing
            rows, cols, vals = rows[keep], cols[keep], vals[keep]
        if vals.size and cfg.depth_noise_sigma > 0.0:
            vals = vals * np.exp(rng.normal(0.0, cfg.depth_noise_sigma, size=vals.size))
        if vals.size and cfg.depth_outlier_rate > 0.0:
            hit = rng.random(vals.size) < cfg.depth_outlier_rate
            if hit.any():
                mult = np.where(rng.random(hit.sum()) < 0.5,
                                cfg.depth_outlier_scale, 1.0 / cfg.depth_outlier_scale)
                vals[hit] = vals[hit] * mult
        if vals.size:
            p.has_depth = True
            p.depth_rows, p.depth_cols, p.depth_values = rows, cols, vals
            # advertised confidence shrinks with declared noise, not with the
            # realised error -- a sensor does not know its own outliers
            p.depth_confidence = float(np.exp(-3.0 * cfg.depth_noise_sigma))

    # -- pose -------------------------------------------------------------- #
    if want_pose and gt_pose_c2w is not None:
        rng = _frame_rng(cfg.seed, sequence, frame, "pose")
        pose = np.array(gt_pose_c2w, dtype=np.float64).copy()
        if cfg.translation_noise_m > 0.0:
            pose[:3, 3] += rng.normal(0.0, cfg.translation_noise_m, size=3)
        if cfg.rotation_noise_deg > 0.0:
            axis = rng.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-12)
            ang = np.radians(rng.normal(0.0, cfg.rotation_noise_deg))
            pose[:3, :3] = axis_angle_to_matrix(axis * ang) @ pose[:3, :3]
        p.has_pose = True
        p.pose_c2w = pose
        p.pose_confidence = float(np.exp(-cfg.translation_noise_m))

    return p


def make_prompt_stream(
    cfg: PromptConfig,
    sequence: str,
    n_frames: int,
    gt_depth: Optional[np.ndarray] = None,
    gt_valid: Optional[np.ndarray] = None,
    gt_poses_c2w: Optional[np.ndarray] = None,
) -> List[Prompt]:
    """All prompts for a sequence, one per frame (possibly empty)."""
    out = []
    for t in range(n_frames):
        out.append(make_prompt(
            cfg, sequence, t, n_frames,
            gt_depth=None if gt_depth is None else gt_depth[t],
            gt_valid=None if gt_valid is None else gt_valid[t],
            gt_pose_c2w=None if gt_poses_c2w is None else gt_poses_c2w[t],
        ))
    return out


# --------------------------------------------------------------------------- #
# The schedules used in the evaluation
# --------------------------------------------------------------------------- #
CLEAN_NOISE = dict(depth_noise_sigma=0.0, depth_outlier_rate=0.0,
                   translation_noise_m=0.0, rotation_noise_deg=0.0,
                   depth_missing=0.0, dropout=0.0)

MODERATE_NOISE = dict(depth_noise_sigma=0.05, depth_outlier_rate=0.05,
                      translation_noise_m=0.10, rotation_noise_deg=0.5,
                      depth_missing=0.20, dropout=0.10)

HEAVY_NOISE = dict(depth_noise_sigma=0.15, depth_outlier_rate=0.15,
                   translation_noise_m=0.50, rotation_noise_deg=2.0,
                   depth_missing=0.40, dropout=0.25)

PROMPT_INTERVALS = (1, 5, 10, 30, 100)


def standard_configs() -> Dict[str, PromptConfig]:
    """The full evaluation grid required by the study."""
    cfgs: Dict[str, PromptConfig] = {}

    cfgs["rgb_only"] = PromptConfig(name="rgb_only", depth_enabled=False, pose_enabled=False)

    for noise_name, noise in (("clean", CLEAN_NOISE), ("moderate", MODERATE_NOISE),
                              ("heavy", HEAVY_NOISE)):
        for k in PROMPT_INTERVALS:
            cfgs[f"depth_k{k}_{noise_name}"] = PromptConfig(
                name=f"depth_k{k}_{noise_name}", depth_interval=k, pose_enabled=False, **noise)
            cfgs[f"pose_k{k}_{noise_name}"] = PromptConfig(
                name=f"pose_k{k}_{noise_name}", pose_interval=k, depth_enabled=False, **noise)
            cfgs[f"both_k{k}_{noise_name}"] = PromptConfig(
                name=f"both_k{k}_{noise_name}", depth_interval=k, pose_interval=k, **noise)

    for noise_name, noise in (("clean", CLEAN_NOISE), ("moderate", MODERATE_NOISE)):
        cfgs[f"depth_first_only_{noise_name}"] = PromptConfig(
            name=f"depth_first_only_{noise_name}", first_frame_only=True, pose_enabled=False, **noise)
        cfgs[f"both_first_only_{noise_name}"] = PromptConfig(
            name=f"both_first_only_{noise_name}", first_frame_only=True, **noise)
        for p in (0.02, 0.10):
            tag = f"{p:.2f}".replace(".", "")
            cfgs[f"depth_bern{tag}_{noise_name}"] = PromptConfig(
                name=f"depth_bern{tag}_{noise_name}", depth_bernoulli=p, pose_enabled=False, **noise)
            cfgs[f"both_bern{tag}_{noise_name}"] = PromptConfig(
                name=f"both_bern{tag}_{noise_name}", depth_bernoulli=p, pose_bernoulli=p, **noise)

    return cfgs


def training_config_sampler(seed: int) -> PromptConfig:
    """Randomised prompt regime for training, so one checkpoint covers many rigs."""
    rng = np.random.default_rng(seed)
    mode = rng.choice(["depth", "pose", "both"], p=[0.4, 0.2, 0.4])
    interval = int(rng.choice([1, 5, 10, 30, 100, 250]))
    use_bernoulli = rng.random() < 0.3
    return PromptConfig(
        name=f"train_{mode}_{interval}_{seed}",
        depth_interval=None if (use_bernoulli or mode == "pose") else interval,
        pose_interval=None if (use_bernoulli or mode == "depth") else interval,
        depth_bernoulli=(1.0 / interval) if (use_bernoulli and mode != "pose") else None,
        pose_bernoulli=(1.0 / interval) if (use_bernoulli and mode != "depth") else None,
        depth_enabled=mode != "pose",
        pose_enabled=mode != "depth",
        num_depth_samples=int(rng.choice([64, 256, 512, 2048])),
        depth_missing=float(rng.uniform(0.0, 0.4)),
        depth_noise_sigma=float(rng.uniform(0.0, 0.15)),
        depth_outlier_rate=float(rng.uniform(0.0, 0.15)),
        translation_noise_m=float(rng.uniform(0.0, 0.5)),
        rotation_noise_deg=float(rng.uniform(0.0, 2.0)),
        dropout=float(rng.uniform(0.0, 0.25)),
        seed=seed,
    )
