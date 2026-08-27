"""Metric anchors: exact recovery, robustness, causality, reset and serialisation."""

import numpy as np
import pytest

from prompted_lingbot.anchors import (
    BASELINES, CausalPoseSim3Anchor, DepthScalePoseSE3Anchor, FirstDepthScaleAnchor,
    OfflineOracleSim3Anchor, Prediction, RawAnchor, RunningDepthScaleAnchor,
    SlidingWindowSim3Anchor, robust_log_scale,
)
from prompted_lingbot.conventions import Sim3, axis_angle_to_matrix, umeyama_sim3
from prompted_lingbot.prompts import Prompt

H = W = 24
K = np.array([[30.0, 0.0, W / 2], [0.0, 30.0, H / 2], [0.0, 0.0, 1.0]])


def _world(rng, n=60, scale=1.0):
    """A synthetic sequence: true metric geometry, plus the model's scaled view."""
    centres = np.cumsum(rng.normal(scale=0.4, size=(n, 3)), axis=0)
    centres -= centres[0]
    poses = np.zeros((n, 3, 4))
    for i in range(n):
        poses[i, :3, :3] = axis_angle_to_matrix(rng.normal(scale=0.05, size=3))
        poses[i, :3, 3] = centres[i]
    depth = rng.uniform(2.0, 20.0, size=(n, H, W))
    return poses, depth


def _pred(poses_metric, depth_metric, i, scale):
    """What the frozen model would emit if its scale were 1/`scale` of metric."""
    p = poses_metric[i].copy()
    p[:3, 3] = p[:3, 3] / scale
    return Prediction(frame=i, pose_c2w=p, depth=depth_metric[i] / scale,
                      depth_conf=np.full((H, W), 3.0), K=K)


def _depth_prompt(frame, depth_metric, rng, n=200, noise=0.0, outlier=0.0):
    rows = rng.integers(0, H, n)
    cols = rng.integers(0, W, n)
    vals = depth_metric[frame][rows, cols].astype(float)
    if noise:
        vals = vals * np.exp(rng.normal(0, noise, n))
    if outlier:
        hit = rng.random(n) < outlier
        vals[hit] *= 5.0
    return Prompt(frame=frame, has_depth=True, depth_rows=rows, depth_cols=cols,
                  depth_values=vals, depth_confidence=1.0)


def _pose_prompt(frame, poses_metric, noise=0.0, rng=None):
    p = poses_metric[frame].copy()
    if noise:
        p[:3, 3] = p[:3, 3] + rng.normal(0, noise, 3)
    return Prompt(frame=frame, has_pose=True, pose_c2w=p, pose_confidence=1.0)


# --------------------------------------------------------------------------- #
# robust scale estimation
# --------------------------------------------------------------------------- #
def test_robust_log_scale_is_exact_without_noise():
    rng = np.random.default_rng(0)
    pred = rng.uniform(1, 10, 500)
    est = robust_log_scale(pred * 7.5, pred)
    assert est is not None and np.isclose(np.exp(est[0]), 7.5, rtol=1e-9)


def test_robust_log_scale_survives_noise_and_outliers():
    rng = np.random.default_rng(1)
    pred = rng.uniform(1, 10, 2000)
    metric = pred * 3.0 * np.exp(rng.normal(0, 0.05, 2000))
    hit = rng.random(2000) < 0.3
    metric[hit] *= 10.0 ** rng.choice([-1, 1], hit.sum())
    est = robust_log_scale(metric, pred)
    assert est is not None
    assert abs(np.exp(est[0]) - 3.0) / 3.0 < 0.05


def test_robust_log_scale_refuses_tiny_samples():
    assert robust_log_scale(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


# --------------------------------------------------------------------------- #
# identity behaviour
# --------------------------------------------------------------------------- #
def test_raw_anchor_is_the_identity():
    rng = np.random.default_rng(2)
    poses, depth = _world(rng)
    a = RawAnchor()
    for i in range(len(poses)):
        a.update(_pred(poses, depth, i, 5.0), Prompt(frame=i))
    out = a.correct(_pred(poses, depth, 0, 5.0))
    assert a.correction.s == 1.0
    assert np.allclose(out["pose_c2w"], _pred(poses, depth, 0, 5.0).pose_c2w)


@pytest.mark.parametrize("name", list(BASELINES))
def test_every_anchor_is_identity_without_prompts(name):
    if name in ("offline_oracle_sim3", "per_frame_oracle_scale"):
        pytest.skip("oracles are fitted from ground truth, not from prompts")
    rng = np.random.default_rng(3)
    poses, depth = _world(rng, n=30)
    a = BASELINES[name]()
    for i in range(len(poses)):
        a.update(_pred(poses, depth, i, 4.0), Prompt(frame=i))
    C = a.correction
    assert C.s == 1.0 and np.allclose(C.R, np.eye(3)) and np.allclose(C.t, 0.0)


# --------------------------------------------------------------------------- #
# exact recovery
# --------------------------------------------------------------------------- #
def test_first_depth_scale_recovers_the_exact_scale_and_then_freezes():
    rng = np.random.default_rng(4)
    poses, depth = _world(rng)
    a = FirstDepthScaleAnchor()
    for i in range(len(poses)):
        pr = _depth_prompt(i, depth, rng) if i % 5 == 0 else Prompt(frame=i)
        a.update(_pred(poses, depth, i, 6.0), pr)
        if i >= 1:
            assert np.isclose(a.correction.s, 6.0, rtol=1e-6)


def test_running_depth_scale_tracks_a_changing_scale():
    """A scale that drifts must be followed; a constant one must be held."""
    rng = np.random.default_rng(5)
    poses, depth = _world(rng, n=120)
    a = RunningDepthScaleAnchor()
    for i in range(120):
        s = 6.0 if i < 60 else 9.0
        pr = _depth_prompt(i, depth, rng, noise=0.02) if i % 5 == 0 else Prompt(frame=i)
        a.update(_pred(poses, depth, i, s), pr)
    assert abs(a.correction.s - 9.0) / 9.0 < 0.05


def test_causal_pose_sim3_recovers_a_known_sim3():
    rng = np.random.default_rng(6)
    poses, depth = _world(rng)
    a = CausalPoseSim3Anchor()
    for i in range(len(poses)):
        pr = _pose_prompt(i, poses) if i % 4 == 0 else Prompt(frame=i)
        a.update(_pred(poses, depth, i, 8.0), pr)
    assert abs(a.correction.s - 8.0) / 8.0 < 1e-4
    corrected = a.correct(_pred(poses, depth, 40, 8.0))
    assert np.allclose(corrected["centre"], poses[40][:3, 3], atol=1e-6)


def test_sliding_window_only_keeps_recent_correspondences():
    rng = np.random.default_rng(7)
    poses, depth = _world(rng, n=80)
    a = SlidingWindowSim3Anchor(window=6)
    for i in range(80):
        a.update(_pred(poses, depth, i, 3.0), _pose_prompt(i, poses))
    assert len(a._pred) == 6
    assert abs(a.correction.s - 3.0) / 3.0 < 1e-4


def test_depth_scale_pose_se3_uses_depth_for_scale():
    """With a 10% wrong pose scale, the combined anchor must keep the depth scale."""
    rng = np.random.default_rng(8)
    poses, depth = _world(rng)
    bad = poses.copy()
    bad[:, :3, 3] *= 1.10          # poses claim a 10%-too-large world
    a = DepthScalePoseSE3Anchor()
    for i in range(len(poses)):
        pr = Prompt(frame=i)
        if i % 5 == 0:
            dp = _depth_prompt(i, depth, rng)
            pp = _pose_prompt(i, bad)
            pr = Prompt(frame=i, has_depth=True, depth_rows=dp.depth_rows,
                        depth_cols=dp.depth_cols, depth_values=dp.depth_values,
                        depth_confidence=1.0, has_pose=True, pose_c2w=pp.pose_c2w,
                        pose_confidence=1.0)
        a.update(_pred(poses, depth, i, 5.0), pr)
    assert abs(a.correction.s - 5.0) / 5.0 < 0.02


def test_offline_oracle_is_flagged_non_causal_and_fits_globally():
    rng = np.random.default_rng(9)
    poses, depth = _world(rng)
    a = OfflineOracleSim3Anchor()
    assert a.is_causal is False
    a.fit(poses[:, :3, 3] / 4.0, poses[:, :3, 3])
    assert abs(a.correction.s - 4.0) / 4.0 < 1e-6


# --------------------------------------------------------------------------- #
# degenerate and missing input
# --------------------------------------------------------------------------- #
def test_pose_anchor_survives_collinear_correspondences():
    """A dead-straight run leaves the roll unobservable: scale must still be right
    and no NaN may escape."""
    n = 40
    poses = np.zeros((n, 3, 4))
    poses[:, :3, :3] = np.eye(3)
    poses[:, 2, 3] = np.linspace(0, 20, n)
    depth = np.full((n, H, W), 5.0)
    a = CausalPoseSim3Anchor()
    for i in range(n):
        a.update(_pred(poses, depth, i, 2.0), _pose_prompt(i, poses))
    C = a.correction
    assert np.isfinite(C.s) and np.isfinite(C.R).all() and np.isfinite(C.t).all()
    assert abs(C.s - 2.0) / 2.0 < 1e-3
    assert a._degenerate_skips > 0


def test_pose_anchor_holds_identity_below_the_minimum_correspondence_count():
    rng = np.random.default_rng(10)
    poses, depth = _world(rng)
    a = CausalPoseSim3Anchor()
    for i in range(3):
        a.update(_pred(poses, depth, i, 5.0), _pose_prompt(i, poses))
    assert a.correction.s == 1.0


def test_anchors_ignore_the_wrong_modality():
    rng = np.random.default_rng(11)
    poses, depth = _world(rng)
    d_only = FirstDepthScaleAnchor()
    p_only = CausalPoseSim3Anchor()
    for i in range(len(poses)):
        d_only.update(_pred(poses, depth, i, 5.0), _pose_prompt(i, poses))
        p_only.update(_pred(poses, depth, i, 5.0), _depth_prompt(i, depth, rng))
    assert d_only.correction.s == 1.0
    assert p_only.correction.s == 1.0


def test_empty_depth_prompt_is_ignored():
    rng = np.random.default_rng(12)
    poses, depth = _world(rng)
    a = FirstDepthScaleAnchor()
    empty = Prompt(frame=0, has_depth=True, depth_rows=np.zeros(0, int),
                   depth_cols=np.zeros(0, int), depth_values=np.zeros(0))
    a.update(_pred(poses, depth, 0, 5.0), empty)
    assert a.correction.s == 1.0


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["first_depth_scale", "running_depth_scale",
                                  "causal_pose_sim3", "sliding_pose_sim3",
                                  "depth_scale_pose_se3"])
def test_reset_clears_state_between_sequences(name):
    rng = np.random.default_rng(13)
    poses, depth = _world(rng)
    a = BASELINES[name]()
    for i in range(len(poses)):
        pr = Prompt(frame=i, has_depth=True, has_pose=True,
                    **{k: v for k, v in vars(_depth_prompt(i, depth, rng)).items()
                       if k in ("depth_rows", "depth_cols", "depth_values")},
                    pose_c2w=poses[i], depth_confidence=1.0, pose_confidence=1.0)
        a.update(_pred(poses, depth, i, 5.0), pr)
    assert a.correction.s != 1.0
    a.reset()
    assert a.correction.s == 1.0
    assert np.allclose(a.correction.R, np.eye(3))


@pytest.mark.parametrize("name", ["raw", "first_depth_scale", "running_depth_scale",
                                  "causal_pose_sim3", "sliding_pose_sim3",
                                  "depth_scale_pose_se3", "offline_oracle_sim3"])
def test_state_dict_roundtrip(name):
    import json
    rng = np.random.default_rng(14)
    poses, depth = _world(rng)
    a = BASELINES[name]()
    if name == "offline_oracle_sim3":
        a.fit(poses[:, :3, 3] / 3.0, poses[:, :3, 3])
    for i in range(len(poses)):
        dp = _depth_prompt(i, depth, rng)
        pr = Prompt(frame=i, has_depth=True, depth_rows=dp.depth_rows,
                    depth_cols=dp.depth_cols, depth_values=dp.depth_values,
                    depth_confidence=1.0, has_pose=True, pose_c2w=poses[i],
                    pose_confidence=1.0)
        a.update(_pred(poses, depth, i, 5.0), pr)
    state = json.loads(json.dumps(a.state_dict()))   # must be JSON-serialisable

    b = BASELINES[name]()
    b.load_state_dict(state)
    assert np.isclose(b.correction.s, a.correction.s)
    assert np.allclose(b.correction.R, a.correction.R)
    assert np.allclose(b.correction.t, a.correction.t)

    # and it must keep behaving identically afterwards
    nxt = _pred(poses, depth, 10, 5.0)
    dp = _depth_prompt(10, depth, rng)
    pr = Prompt(frame=10, has_depth=True, depth_rows=dp.depth_rows, depth_cols=dp.depth_cols,
                depth_values=dp.depth_values, depth_confidence=1.0,
                has_pose=True, pose_c2w=poses[10], pose_confidence=1.0)
    a.update(nxt, pr)
    b.update(nxt, pr)
    assert np.isclose(a.correction.s, b.correction.s)


def test_load_state_dict_rejects_a_foreign_state():
    a = FirstDepthScaleAnchor()
    with pytest.raises(ValueError):
        a.load_state_dict(RawAnchor().state_dict())
