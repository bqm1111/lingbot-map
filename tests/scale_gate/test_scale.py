"""Synthetic recovery tests for the scale estimators (plan tests 1-5)."""

import numpy as np
import pytest

from gates.scale_gate.scale import (
    ScaleEstimate, agreement_error, apply_scale_to_depth, apply_scale_to_poses,
    bootstrap_ci, depth_scale, joint_scale, pose_scale, trim_quantiles, weighted_median,
)


# ---------------- robust statistics ---------------- #
def test_weighted_median_matches_plain_median_without_weights():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert weighted_median(x) == pytest.approx(3.0)


def test_weighted_median_follows_the_weights():
    x = np.array([0.0, 10.0])
    assert weighted_median(x, np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert weighted_median(x, np.array([0.0, 1.0])) == pytest.approx(10.0)


def test_weighted_median_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_median(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        weighted_median(np.zeros(3), np.array([-1.0, 1.0, 1.0]))


def test_trim_drops_the_configured_tails():
    x = np.arange(100.0)
    xt, _, dropped = trim_quantiles(x, 0.05, 0.95)
    assert dropped > 0 and xt.min() >= 4.0 and xt.max() <= 95.0


# ---------------- 1. depth scalar recovery ---------------- #
def test_known_scalar_is_recovered_from_depth():
    rng = np.random.default_rng(0)
    d_pred = rng.uniform(1.0, 50.0, 5000)
    s_true = 21.7
    est = depth_scale(d_pred, s_true * d_pred)
    assert est.valid
    assert est.s == pytest.approx(s_true, rel=1e-9)
    assert est.mad_log == pytest.approx(0.0, abs=1e-9)


def test_depth_scalar_survives_heavy_outliers():
    rng = np.random.default_rng(1)
    d_pred = rng.uniform(1.0, 50.0, 5000)
    d_gt = 8.0 * d_pred
    d_gt[:1000] *= rng.uniform(20, 60, 1000)          # 20 % gross outliers
    est = depth_scale(d_pred, d_gt, trim=(0.01, 0.99))
    assert est.s == pytest.approx(8.0, rel=0.02)


def test_depth_scale_is_invalid_with_too_few_pixels():
    est = depth_scale(np.ones(10), np.ones(10) * 3.0, min_pixels=200)
    assert not est.valid and "valid pixels" in est.reason
    assert np.isnan(est.log_s)


def test_depth_scale_ignores_nonpositive_and_nonfinite():
    d_pred = np.array([1.0, 2.0, -1.0, np.nan, 4.0] * 100)
    d_gt = np.array([3.0, 6.0, 5.0, 5.0, 12.0] * 100)
    est = depth_scale(d_pred, d_gt, min_pixels=10)
    assert est.s == pytest.approx(3.0, rel=1e-9)
    assert est.n_raw == 300


def test_per_frame_dispersion_is_reported():
    d_pred = np.ones(600)
    frame = np.repeat([0, 1, 2], 200)
    d_gt = np.concatenate([np.full(200, 2.0), np.full(200, 4.0), np.full(200, 8.0)])
    est = depth_scale(d_pred, d_gt, frame_index=frame, min_pixels=10)
    d = est.to_dict()
    assert d["per_frame_s_min"] == pytest.approx(2.0)
    assert d["per_frame_s_max"] == pytest.approx(8.0)
    assert d["per_frame_log_s_std"] > 0


# ---------------- 2. pose scalar recovery ---------------- #
def test_known_scalar_is_recovered_from_translations():
    rng = np.random.default_rng(2)
    t_pred = rng.normal(size=(40, 3))
    s_true = 13.3
    est = pose_scale(t_pred, s_true * t_pred, min_translation_m=0.0)
    assert est.valid and est.s == pytest.approx(s_true, rel=1e-9)


def test_pose_scale_drops_pairs_below_the_motion_threshold():
    t_pred = np.array([[1e-6, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    t_gt = np.array([[1e-9, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    est = pose_scale(t_pred, t_gt, min_translation_m=0.5)
    assert est.n_raw == 2                      # the near-static pair is excluded
    assert est.s == pytest.approx(5.0, rel=1e-9)


def test_pose_scale_invalid_when_the_clip_barely_moves():
    t_pred = np.full((5, 3), 1e-6)
    est = pose_scale(t_pred, t_pred * 10, min_translation_m=0.5)
    assert not est.valid and "motion" in est.reason


# ---------------- 3. joint recovery ---------------- #
def test_joint_scale_averages_agreeing_estimators_in_log_space():
    d = ScaleEstimate(np.log(10.0), 1000, 1000, 0.01, "depth")
    p = ScaleEstimate(np.log(40.0), 20, 20, 0.02, "pose")
    assert joint_scale(d, p, w_depth=0.5).s == pytest.approx(20.0, rel=1e-9)
    assert joint_scale(d, p, w_depth=1.0).s == pytest.approx(10.0, rel=1e-9)
    assert joint_scale(d, p, w_depth=0.0).s == pytest.approx(40.0, rel=1e-9)


def test_joint_scale_falls_back_to_whichever_term_is_valid():
    d = ScaleEstimate(np.log(7.0), 500, 500, 0.01, "depth")
    bad = ScaleEstimate(float("nan"), 0, 0, float("nan"), "pose", valid=False, reason="x")
    assert joint_scale(d, bad).s == pytest.approx(7.0)
    assert joint_scale(bad, d).s == pytest.approx(7.0)
    assert not joint_scale(bad, bad).valid


def test_joint_scale_is_not_swamped_by_pixel_count():
    """A million depth pixels must not outvote 20 pose pairs at equal weight."""
    d = ScaleEstimate(np.log(10.0), 1_000_000, 1_000_000, 0.01, "depth")
    p = ScaleEstimate(np.log(40.0), 20, 20, 0.02, "pose")
    assert joint_scale(d, p, w_depth=0.5).s == pytest.approx(20.0, rel=1e-9)


def test_agreement_error_is_zero_for_identical_scales():
    a = ScaleEstimate(np.log(5.0), 10, 10, 0.0, "depth")
    b = ScaleEstimate(np.log(5.0), 10, 10, 0.0, "pose")
    assert agreement_error(a, b) == pytest.approx(0.0)
    c = ScaleEstimate(np.log(10.0), 10, 10, 0.0, "pose")
    assert agreement_error(a, c) == pytest.approx(np.log(2.0))


# ---------------- 4/5. applying a scale ---------------- #
def test_scaling_poses_moves_translations_but_never_rotations():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(3, 3))
    R = np.linalg.qr(q)[0]
    R *= np.sign(np.linalg.det(R))
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, :3] = R
    poses[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    out = apply_scale_to_poses(poses, 5.0, anchor=0)
    assert np.allclose(out[:, :3, :3], R)                       # rotations untouched
    assert np.allclose(out[:, :3, 3], [[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 0, 0]])


def test_scaling_poses_holds_the_anchor_fixed():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    out = apply_scale_to_poses(poses, 3.0, anchor=1)
    assert np.allclose(out[1, :3, 3], [1, 0, 0])
    assert np.allclose(out[0, :3, 3], [-2, 0, 0])
    assert np.allclose(out[2, :3, 3], [4, 0, 0])


def test_scaling_depth_is_a_plain_multiply():
    d = np.array([1.0, 2.0, 3.0])
    assert np.allclose(apply_scale_to_depth(d, 4.0), [4.0, 8.0, 12.0])


def test_applying_a_nonpositive_scale_is_refused():
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            apply_scale_to_depth(np.ones(3), bad)
        with pytest.raises(ValueError):
            apply_scale_to_poses(np.tile(np.eye(4), (2, 1, 1)), bad)


def test_depth_and_pose_scaling_stay_consistent_under_unprojection():
    """The invariant the whole gate rests on: one s scales the reconstruction uniformly."""
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    depth = np.full((4, 4), 10.0)
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, :3, 3] = [1.0, 0.0, 0.0]
    s = 7.0

    def unproject(d, K):
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        return np.stack([(u - K[0, 2]) * d / K[0, 0],
                         (v - K[1, 2]) * d / K[1, 1], d], -1).reshape(-1, 3)

    p0 = unproject(depth, K)
    p1 = unproject(apply_scale_to_depth(depth, s), K)
    assert np.allclose(p1, s * p0)                              # depth scales points
    sp = apply_scale_to_poses(poses, s)
    assert np.allclose(sp[1, :3, 3] - sp[0, :3, 3],
                       s * (poses[1, :3, 3] - poses[0, :3, 3]))  # and translations


# ---------------- bootstrap ---------------- #
def test_bootstrap_ci_excludes_zero_for_a_clear_effect():
    ci = bootstrap_ci(list(np.full(200, 0.05)), n_boot=2000, seed=0)
    assert ci["mean"] == pytest.approx(0.05) and ci["excludes_zero"]


def test_bootstrap_ci_includes_zero_for_noise():
    rng = np.random.default_rng(0)
    ci = bootstrap_ci(rng.normal(0, 1, 400).tolist(), n_boot=2000, seed=0)
    assert not ci["excludes_zero"]


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    v = np.random.default_rng(5).normal(0.01, 0.1, 100).tolist()
    assert bootstrap_ci(v, n_boot=1000, seed=7) == bootstrap_ci(v, n_boot=1000, seed=7)
