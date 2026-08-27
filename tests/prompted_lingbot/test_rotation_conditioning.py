"""Phase-1: the pose anchors must never accept an unobservable rotation.

The guard these tests cover replaced a singular-value-ratio check that never
fired on real driving data (measured ratios 0.033-0.085, always above its 5e-3
threshold) while the fitted rotation reached 92 deg of error.  Position error is
never sufficient evidence that a fit is good: every test here asserts on
orientation as well.
"""

import numpy as np
import pytest

from prompted_lingbot.anchors import (
    BASELINES, CausalPoseSim3Anchor, Prediction, SlidingWindowSim3Anchor,
)
from prompted_lingbot.conventions import (
    Sim3, axis_angle_to_matrix, rotation_conditioning, rotation_geodesic_deg,
)
from prompted_lingbot.prompts import Prompt

H = W = 16
K = np.array([[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]])
TRUE = Sim3(6.0, axis_angle_to_matrix(np.array([0.05, -0.3, 0.12])), np.array([3.0, -1.0, 0.5]))
SCENE_DEPTH_M = 20.0


def _stream(centres_metric, depth_m=SCENE_DEPTH_M):
    """Predictions in model units, plus the metric poses a sensor would report."""
    inv = TRUE.inverse()
    preds, poses = [], []
    for i, c in enumerate(centres_metric):
        p = np.zeros((3, 4))
        p[:3, :3] = np.eye(3)
        p[:3, 3] = inv.apply_points(c)
        preds.append(Prediction(frame=i, pose_c2w=p,
                                depth=np.full((H, W), depth_m / TRUE.s, np.float32),
                                depth_conf=np.full((H, W), 3.0, np.float32), K=K))
        g = np.zeros((3, 4))
        g[:3, :3] = TRUE.R
        g[:3, 3] = c
        poses.append(g)
    return preds, poses


def _drive(anchor, preds, poses, sigma_m=0.0, every=1, rng=None):
    anchor.reset()
    conf = float(np.exp(-sigma_m)) if sigma_m > 0 else 1.0
    for t, (pr, gp) in enumerate(zip(preds, poses)):
        prompt = Prompt(frame=t)
        if t % every == 0:
            g = gp.copy()
            if sigma_m > 0 and rng is not None:
                g[:3, 3] = g[:3, 3] + rng.normal(0, sigma_m, 3)
            prompt = Prompt(frame=t, has_pose=True, pose_c2w=g, pose_confidence=conf)
        anchor.update(pr, prompt)
    return anchor


# --------------------------------------------------------------------------- #
# the conditioning diagnostic itself
# --------------------------------------------------------------------------- #
def test_conditioning_reports_zero_extent_for_a_straight_line():
    line = np.linspace(0, 20, 12)[:, None] * np.array([[0.0, 0.0, 1.0]])
    c = rotation_conditioning(line, residual_rms=0.05, median_scene_depth=20.0)
    assert c["baseline"] > 1.0
    assert c["lateral"] < 1e-9 and c["vertical"] < 1e-9
    assert c["extent_ratio"] < 1e-9
    assert not np.isfinite(c["rotation_sigma_rad"])


def test_conditioning_reports_finite_uncertainty_for_a_3d_cloud():
    rng = np.random.default_rng(0)
    c = rotation_conditioning(rng.normal(scale=5.0, size=(20, 3)),
                              residual_rms=0.05, median_scene_depth=20.0)
    assert c["extent_ratio"] > 0.1
    assert 0.0 < c["rotation_sigma_rad"] < 0.1


def test_conditioning_uncertainty_scales_with_noise_over_extent():
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=4.0, size=(30, 3))
    a = rotation_conditioning(pts, residual_rms=0.05)["rotation_sigma_rad"]
    b = rotation_conditioning(pts, residual_rms=0.50)["rotation_sigma_rad"]
    assert b == pytest.approx(10.0 * a, rel=1e-6)
    c = rotation_conditioning(pts * 10.0, residual_rms=0.05)["rotation_sigma_rad"]
    assert c == pytest.approx(a / 10.0, rel=1e-6)


def test_conditioning_needs_three_points():
    assert rotation_conditioning(np.zeros((2, 3)))["n"] == 2
    assert not np.isfinite(rotation_conditioning(np.zeros((2, 3)))["rotation_sigma_rad"])


# --------------------------------------------------------------------------- #
# 1. straight synthetic trajectory -- rotation must be HELD
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [CausalPoseSim3Anchor, SlidingWindowSim3Anchor])
def test_straight_trajectory_holds_the_rotation(cls):
    centres = np.linspace(0, 60, 60)[:, None] * np.array([[0.0, 0.0, 1.0]])
    preds, poses = _stream(centres)
    a = _drive(cls(), preds, poses)

    assert a.rotation_held_fraction == 1.0, "a straight run must never accept a rotation"
    assert np.allclose(a.correction.R, np.eye(3)), "the held rotation must be the identity"
    # Scale is still observable along the direction of travel.
    assert abs(a.correction.s - TRUE.s) / TRUE.s < 0.05


def test_straight_trajectory_does_not_invent_orientation_despite_low_ate():
    """The exact failure that motivated this guard: low centre error, wild rotation."""
    centres = np.linspace(0, 60, 60)[:, None] * np.array([[0.0, 0.0, 1.0]])
    preds, poses = _stream(centres)
    a = _drive(SlidingWindowSim3Anchor(window=10), preds, poses)

    corrected_R = np.stack([a.correction.apply_rotations(p.pose_c2w[:3, :3]) for p in preds])
    gt_R = np.stack([g[:3, :3] for g in poses])
    # The correction holds the identity, so the orientation error is exactly the
    # frozen model's own -- never worse.
    assert float(rotation_geodesic_deg(corrected_R, gt_R).max()) < 90.0
    assert a.rotation_held_fraction == 1.0


# --------------------------------------------------------------------------- #
# 2. well-conditioned 3D trajectory -- rotation must be ACCEPTED and correct
# --------------------------------------------------------------------------- #
def test_well_conditioned_trajectory_accepts_and_recovers_the_rotation():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 60)
    centres = np.stack([20 * np.cos(t), 20 * np.sin(t), 6 * np.sin(0.7 * t)], 1)
    preds, poses = _stream(centres)
    a = _drive(CausalPoseSim3Anchor(), preds, poses)

    assert a.rotation_held_fraction < 0.5, "a 3D helix must permit a rotation fit"
    assert abs(a.correction.s - TRUE.s) / TRUE.s < 1e-3
    assert float(rotation_geodesic_deg(a.correction.R, TRUE.R)) < 1.0
    assert np.allclose(a.correction.t, TRUE.t, atol=1e-2)


def test_well_conditioned_trajectory_reduces_orientation_error():
    t = np.linspace(0, 4 * np.pi, 60)
    centres = np.stack([20 * np.cos(t), 20 * np.sin(t), 6 * np.sin(0.7 * t)], 1)
    preds, poses = _stream(centres)
    a = _drive(CausalPoseSim3Anchor(), preds, poses)

    gt_R = np.stack([g[:3, :3] for g in poses])
    raw_R = np.stack([p.pose_c2w[:3, :3] for p in preds])
    corr_R = np.stack([a.correction.apply_rotations(p.pose_c2w[:3, :3]) for p in preds])
    assert float(rotation_geodesic_deg(corr_R, gt_R).mean()) < \
           float(rotation_geodesic_deg(raw_R, gt_R).mean())


# --------------------------------------------------------------------------- #
# 3. noisy correspondences -- the extent must beat the declared noise
# --------------------------------------------------------------------------- #
def test_noise_makes_a_marginal_geometry_unobservable():
    """The same trajectory is accepted when clean and refused when the sensor
    declares noise comparable to its least extent."""
    # z uses a different frequency from y on purpose: z = a*sin(t) would make the
    # cloud planar and the third extent exactly zero, which tests nothing.
    # Vertical extent ~1.1 m RMS: above the absolute floor, but only ~20x a clean
    # prompt's 0.05 m noise, so 2 m of declared noise must kill it.
    t = np.linspace(0, 2 * np.pi, 40)
    centres = np.stack([20 * np.cos(t), 20 * np.sin(t), 1.5 * np.sin(3 * t)], 1)
    preds, poses = _stream(centres)
    rng = np.random.default_rng(3)

    clean = _drive(CausalPoseSim3Anchor(), preds, poses, sigma_m=0.0)
    noisy = _drive(CausalPoseSim3Anchor(), preds, poses, sigma_m=2.0, rng=rng)
    assert clean.rotation_held_fraction < noisy.rotation_held_fraction
    assert noisy.rotation_held_fraction == 1.0
    assert np.allclose(noisy.correction.R, np.eye(3))


def test_noisy_but_well_conditioned_geometry_is_still_accepted():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 4 * np.pi, 80)
    centres = np.stack([60 * np.cos(t), 60 * np.sin(t), 30 * np.sin(0.7 * t)], 1)
    preds, poses = _stream(centres)
    a = _drive(CausalPoseSim3Anchor(), preds, poses, sigma_m=0.3, rng=rng)
    assert a.rotation_held_fraction < 0.6
    assert float(rotation_geodesic_deg(a.correction.R, TRUE.R)) < 5.0


# --------------------------------------------------------------------------- #
# 4. insufficient correspondences
# --------------------------------------------------------------------------- #
def test_too_few_correspondences_leaves_the_correction_untouched():
    t = np.linspace(0, np.pi, 3)
    centres = np.stack([20 * np.cos(t), 20 * np.sin(t), 5 * t], 1)
    preds, poses = _stream(centres)
    a = _drive(CausalPoseSim3Anchor(), preds, poses)
    assert a.correction.s == 1.0
    assert np.allclose(a.correction.R, np.eye(3))
    assert np.allclose(a.correction.t, 0.0)


def test_sparse_prompts_on_a_short_window_hold_rotation():
    """A 10-slot window fed every 30th frame still needs spatial extent."""
    centres = np.linspace(0, 200, 300)[:, None] * np.array([[0.0, 0.0, 1.0]])
    preds, poses = _stream(centres)
    a = _drive(SlidingWindowSim3Anchor(window=10), preds, poses, every=30)
    assert a.rotation_held_fraction == 1.0
    assert np.allclose(a.correction.R, np.eye(3))


# --------------------------------------------------------------------------- #
# joint reporting contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["causal_pose_sim3", "sliding_pose_sim3",
                                  "depth_scale_pose_se3"])
def test_anchors_expose_rotation_held_fraction(name):
    """Every pose anchor must expose the diagnostic that stops ATE-only selection."""
    a = BASELINES[name]()
    inner = a if hasattr(a, "rotation_held_fraction") else a._pose
    assert hasattr(inner, "rotation_held_fraction")
    assert 0.0 <= inner.rotation_held_fraction <= 1.0


def test_conditioning_record_is_exposed_after_a_refit():
    t = np.linspace(0, 4 * np.pi, 40)
    centres = np.stack([20 * np.cos(t), 20 * np.sin(t), 6 * np.sin(0.7 * t)], 1)
    preds, poses = _stream(centres)
    a = _drive(CausalPoseSim3Anchor(), preds, poses)
    c = a._last_conditioning
    for key in ("n", "extent", "extent_ratio", "baseline_over_depth",
                "rotation_sigma_rad", "residual_rms", "prompt_sigma_m",
                "median_scene_depth_m", "accepted"):
        assert key in c, f"conditioning record is missing {key!r}"
    assert c["median_scene_depth_m"] == pytest.approx(SCENE_DEPTH_M, rel=0.05)
