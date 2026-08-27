"""Causal evaluation loop: cached predictions + prompts -> corrections -> metrics."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from .anchors import (MetricAnchor, OfflineOracleSim3Anchor,
                      PerFrameOracleScaleAnchor, Prediction)
from .conventions import Sim3
from .prompts import Prompt, PromptConfig, make_prompt


@dataclass
class CachedSequence:
    """One cached sequence, loaded eagerly (a 500-frame chunk is ~200 MB in RAM)."""

    name: str
    meta: dict
    pred_pose_c2w: np.ndarray     # (S, 3, 4)
    pred_K: np.ndarray            # (S, 3, 3)
    pred_depth: np.ndarray        # (S, h, w) float16
    pred_depth_conf: np.ndarray   # (S, h, w) float16
    gt_pose_c2w: np.ndarray       # (S, 3, 4) rebased onto the first GT camera
    gt_world_from_local: np.ndarray  # (4, 4) the rebasing transform that was removed
    gt_K: np.ndarray              # (3, 3)
    gt_depth: Optional[np.ndarray]
    gt_depth_valid: Optional[np.ndarray]
    timestamps: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.pred_pose_c2w)

    @property
    def has_gt_depth(self) -> bool:
        return self.gt_depth is not None

    def prediction(self, t: int) -> Prediction:
        return Prediction(
            frame=t,
            pose_c2w=self.pred_pose_c2w[t].astype(np.float64),
            depth=self.pred_depth[t].astype(np.float32),
            depth_conf=self.pred_depth_conf[t].astype(np.float32),
            K=self.pred_K[t].astype(np.float64),
        )


def canonicalise_gt_to_first_camera(gt_poses_c2w: np.ndarray):
    """Express ground-truth poses in the frame of the sequence's first camera.

    LingbotMap defines its world frame as the first camera's frame, whereas a
    dataset's ground truth lives in some global frame.  Comparing the two
    directly measures the arbitrary offset between those frames, not any error
    of the model, so every raw (unaligned) number would be meaningless.

    Rebasing GT onto its own first camera fixes the gauge with a *rigid*
    transform: all metric distances, depths and drift are preserved exactly, and
    the only thing assumed known is the initial pose -- which is precisely what a
    single pose prompt at t=0 supplies, and what monocular SLAM evaluation
    conventionally assumes.  It grants no information about scale or drift.
    """
    T0 = np.eye(4)
    T0[:3, :4] = gt_poses_c2w[0]
    T0_inv = np.eye(4)
    T0_inv[:3, :3] = T0[:3, :3].T
    T0_inv[:3, 3] = -T0[:3, :3].T @ T0[:3, 3]
    R = np.einsum("ij,njk->nik", T0_inv[:3, :3], gt_poses_c2w[:, :3, :3])
    t = gt_poses_c2w[:, :3, 3] @ T0_inv[:3, :3].T + T0_inv[:3, 3]
    return np.concatenate([R, t[:, :, None]], axis=-1), T0


def load_cached(path: str, meta: Optional[dict] = None) -> CachedSequence:
    z = np.load(path)
    name = os.path.splitext(os.path.basename(path))[0]
    gt_poses, gt_world_from_local = canonicalise_gt_to_first_camera(
        z["gt_pose_c2w"].astype(np.float64))
    return CachedSequence(
        name=name,
        meta=meta or {},
        pred_pose_c2w=z["pred_pose_c2w"].astype(np.float64),
        pred_K=z["pred_K"].astype(np.float64),
        pred_depth=z["pred_depth"],
        pred_depth_conf=z["pred_depth_conf"],
        gt_pose_c2w=gt_poses,
        gt_world_from_local=gt_world_from_local,
        gt_K=z["gt_K"].astype(np.float64),
        gt_depth=z["gt_depth"] if "gt_depth" in z.files else None,
        gt_depth_valid=z["gt_depth_valid"] if "gt_depth_valid" in z.files else None,
        timestamps=z["timestamps"],
    )


def load_cache_dir(cache_dir: str, names: Optional[Sequence[str]] = None) -> List[CachedSequence]:
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {"sequences": {}}
    out = []
    for f in sorted(os.listdir(cache_dir)):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        if names is not None and name not in names:
            continue
        out.append(load_cached(os.path.join(cache_dir, f), manifest["sequences"].get(name, {})))
    return out


# --------------------------------------------------------------------------- #
def build_prompts(seq: CachedSequence, cfg: PromptConfig) -> List[Prompt]:
    """Prompt stream for a cached sequence.  Depth prompts need cached GT depth."""
    prompts = []
    for t in range(seq.n_frames):
        gd = gv = None
        if seq.has_gt_depth:
            gd = seq.gt_depth[t].astype(np.float32)
            gv = seq.gt_depth_valid[t]
        prompts.append(make_prompt(
            cfg, seq.name, t, seq.n_frames,
            gt_depth=gd, gt_valid=gv, gt_pose_c2w=seq.gt_pose_c2w[t],
        ))
    return prompts


@dataclass
class RunResult:
    corrections: List[Sim3]
    prompted: np.ndarray            # (S,) bool, any prompt at that frame
    depth_prompted: np.ndarray
    pose_prompted: np.ndarray
    update_ms_per_frame: float
    peak_extra_bytes: int


def run_anchor(seq: CachedSequence, anchor: MetricAnchor, prompts: Sequence[Prompt],
               depth_cache: "Optional[DepthEvalCache]" = None) -> RunResult:
    """Drive one anchor causally over a sequence."""
    anchor.reset()
    if isinstance(anchor, PerFrameOracleScaleAnchor):
        if not seq.has_gt_depth:
            raise ValueError("per_frame_oracle_scale needs cached ground-truth depth")
        if depth_cache is None:
            depth_cache = DepthEvalCache.build(seq)
        anchor.set_schedule({t: s for t, s in enumerate(depth_cache.oracle_scales())})
    if isinstance(anchor, OfflineOracleSim3Anchor):
        # Non-causal by design: fit once to everything, then hold it constant.
        anchor.fit(seq.pred_pose_c2w[:, :3, 3], seq.gt_pose_c2w[:, :3, 3])

    corrections: List[Sim3] = []
    t0 = time.perf_counter()
    for t in range(seq.n_frames):
        anchor.update(seq.prediction(t), prompts[t])
        corrections.append(anchor.correction)
    elapsed = time.perf_counter() - t0

    return RunResult(
        corrections=corrections,
        prompted=np.array([not p.is_empty for p in prompts]),
        depth_prompted=np.array([p.has_depth for p in prompts]),
        pose_prompted=np.array([p.has_pose for p in prompts]),
        update_ms_per_frame=1000.0 * elapsed / max(1, seq.n_frames),
        peak_extra_bytes=0,
    )


def apply_corrections(seq: CachedSequence, corrections: Sequence[Sim3]):
    """Corrected poses and depth for every frame."""
    poses = np.stack([corrections[t].apply_pose_c2w(seq.pred_pose_c2w[t])
                      for t in range(seq.n_frames)])
    scales = np.array([c.s for c in corrections])
    return poses, scales


@dataclass
class DepthEvalCache:
    """Sub-sampled (predicted, ground-truth) depth pairs, per frame.

    Every anchor in this study corrects depth by a single per-frame scalar, so
    the full depth metrics for ANY scale schedule are a cheap function of a fixed
    random pixel sample.  Precomputing it once per sequence turns the evaluation
    of one (config, anchor) pair from seconds into milliseconds, which is what
    makes the full 58-config grid affordable.
    """

    pred: np.ndarray        # (S, n) float64
    gt: np.ndarray          # (S, n) float64
    n_valid: np.ndarray     # (S,) valid pixels the sample was drawn from

    @staticmethod
    def build(seq: "CachedSequence", n_per_frame: int = 8000, seed: int = 0) -> "DepthEvalCache":
        rng = np.random.default_rng(seed)
        gt = seq.gt_depth.astype(np.float32)
        pd = seq.pred_depth.astype(np.float32)
        P = np.full((seq.n_frames, n_per_frame), np.nan)
        G = np.full((seq.n_frames, n_per_frame), np.nan)
        nv = np.zeros(seq.n_frames, dtype=np.int64)
        for t in range(seq.n_frames):
            m = seq.gt_depth_valid[t] & (gt[t] > 1e-3) & (pd[t] > 1e-6)
            idx = np.flatnonzero(m.ravel())
            nv[t] = idx.size
            if idx.size == 0:
                continue
            pick = idx if idx.size <= n_per_frame else rng.choice(idx, n_per_frame, replace=False)
            P[t, :pick.size] = pd[t].ravel()[pick]
            G[t, :pick.size] = gt[t].ravel()[pick]
        return DepthEvalCache(P, G, nv)

    def metrics(self, scales: np.ndarray) -> Dict[str, float]:
        p = self.pred * scales[:, None]
        g = self.gt
        m = np.isfinite(p) & np.isfinite(g)
        if not m.any():
            return {}
        pv, gv = p[m], g[m]
        diff = pv - gv
        ratio = np.maximum(pv / gv, gv / pv)
        return {
            "abs_rel": float(np.mean(np.abs(diff) / gv)),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "rmse_log": float(np.sqrt(np.mean((np.log(pv) - np.log(gv)) ** 2))),
            "sq_rel": float(np.mean(diff ** 2 / gv)),
            "mae": float(np.mean(np.abs(diff))),
            "delta_1_25": float(np.mean(ratio < 1.25)),
            "n": int(m.sum()),
        }

    def per_frame_log_scale_error(self, scales: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.log(self.gt) - np.log(self.pred * scales[:, None])
        out = np.full(lr.shape[0], np.nan)
        for t in range(lr.shape[0]):
            v = lr[t][np.isfinite(lr[t])]
            if v.size >= 16:
                out[t] = float(np.median(v))
        return out

    def oracle_scales(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.gt / self.pred
        out = np.ones(r.shape[0])
        for t in range(r.shape[0]):
            v = r[t][np.isfinite(r[t]) & (r[t] > 0)]
            if v.size >= 16:
                out[t] = float(np.median(v))
        return out


def gt_point_cloud(seq: "CachedSequence", seed: int = 0) -> np.ndarray:
    """The ground-truth cloud for a sequence.  Identical for every anchor, so it is
    built once per sequence rather than once per (config, anchor) pair."""
    max_d = float(seq.meta.get("max_valid_depth", 80.0)) if seq.meta else 80.0
    return M.build_point_cloud(seq.gt_depth.astype(np.float32), seq.gt_depth_valid,
                               seq.gt_K, seq.gt_pose_c2w, seed=seed, max_depth=max_d)


def evaluate_run(seq: CachedSequence, run: RunResult, point_cloud: bool = True,
                 seed: int = 0, depth_cache: Optional[DepthEvalCache] = None,
                 gt_points: Optional[np.ndarray] = None) -> Dict[str, float]:
    """All metrics for one (sequence, anchor, prompt-config) triple."""
    poses, scales = apply_corrections(seq, run.corrections)
    pred_c = poses[:, :3, 3]
    gt_c = seq.gt_pose_c2w[:, :3, 3]

    out: Dict[str, float] = {}
    out.update(M.ate(pred_c, gt_c))
    out.update(M.ate_aligned(pred_c, gt_c, with_scale=False))
    out.update(M.ate_aligned(pred_c, gt_c, with_scale=True))
    out.update(M.relative_pose_error(poses, seq.gt_pose_c2w, delta=1))
    out.update({f"{k}_d10": v for k, v in
                M.relative_pose_error(poses, seq.gt_pose_c2w, delta=10).items()})
    out.update(M.rotation_error_deg(poses, seq.gt_pose_c2w))
    out.update(M.error_vs_distance(pred_c, gt_c))
    out["trajectory_length_m"] = float(M.travelled_distance(gt_c)[-1])
    out["final_scale"] = float(scales[-1])
    out["update_ms_per_frame"] = run.update_ms_per_frame
    out["n_prompted_frames"] = int(run.prompted.sum())
    out["n_depth_prompts"] = int(run.depth_prompted.sum())
    out["n_pose_prompts"] = int(run.pose_prompted.sum())

    # position error vs frames since prompt
    since = M.frames_since_prompt(run.prompted)
    pos_err = np.linalg.norm(pred_c - gt_c, axis=1)
    out.update({f"ate_{k}": v for k, v in M.error_vs_frames_since_prompt(pos_err, since).items()})

    if seq.has_gt_depth:
        dc = depth_cache if depth_cache is not None else DepthEvalCache.build(seq, seed=seed)
        out.update(dc.metrics(scales))
        sc_err = dc.per_frame_log_scale_error(scales)
        out["depth_log_scale_err_rmse"] = float(np.sqrt(np.nanmean(sc_err ** 2)))
        out["depth_log_scale_err_final"] = float(np.nanmean(sc_err[-20:]))
        out["depth_log_scale_err_initial"] = float(np.nanmean(sc_err[:20]))
        d_since = M.frames_since_prompt(run.depth_prompted if run.depth_prompted.any() else run.prompted)
        out.update({f"depthscale_{k}": v for k, v in
                    M.error_vs_frames_since_prompt(np.abs(sc_err), d_since).items()})

        if point_cloud:
            max_d = float(seq.meta.get("max_valid_depth", 80.0)) if seq.meta else 80.0
            gt_v = seq.gt_depth_valid
            corr_d = seq.pred_depth.astype(np.float32) * scales[:, None, None].astype(np.float32)
            pred_pts = M.build_point_cloud(corr_d, gt_v, seq.pred_K[0], poses,
                                           seed=seed, max_depth=max_d)
            gt_pts = gt_points if gt_points is not None else gt_point_cloud(seq, seed)
            out.update(M.point_cloud_metrics(pred_pts, gt_pts))
    return out
