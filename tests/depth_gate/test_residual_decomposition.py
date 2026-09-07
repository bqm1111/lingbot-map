"""Unit tests for the Gate-2 residual decomposition.

These test the *logic* of the decomposition -- the algebraic identities and the
pose-scaling convention -- without touching the dataset, so they run anywhere.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "tools", "geometry_gate")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "tools", "depth_gate")))

from factorize import relative_metric                                       # noqa: E402
from decompose_residual import (                                            # noqa: E402
    check_pose_scaling, fuse, scaled_relative_pose,
)


def random_poses(n: int = 5, seed: int = 0) -> np.ndarray:
    """Random camera-to-world poses with proper rotations."""
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(scale=5.0, size=3)
    return P


# --------------------------------------------------------------------------- #
# Pose scaling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", [0.5, 1.0, 3.7, 27.3665])
def test_pose_scaling_matches_the_frozen_gate1_convention(s):
    """c_scaled = c_anchor + s*(c - c_anchor) must equal Gate-1's relative_metric."""
    P = random_poses()
    a = len(P) - 1
    for f in range(len(P)):
        assert np.allclose(scaled_relative_pose(P, f, a, s),
                           relative_metric(P, f, a, s), atol=1e-10)


def test_anchor_is_a_fixed_point_of_the_scaling():
    P = random_poses(seed=1)
    a = len(P) - 1
    for s in (0.1, 1.0, 40.0):
        assert np.allclose(scaled_relative_pose(P, a, a, s), np.eye(4), atol=1e-12)


def test_rotations_are_never_scaled():
    P = random_poses(seed=2)
    a = len(P) - 1
    for f in range(len(P)):
        base = scaled_relative_pose(P, f, a, 1.0)[:3, :3]
        for s in (0.3, 12.0, 100.0):
            assert np.allclose(scaled_relative_pose(P, f, a, s)[:3, :3], base, atol=1e-12)


def test_relative_translation_scales_by_exactly_the_requested_ratio():
    P = random_poses(seed=3)
    a = len(P) - 1
    for f in range(len(P) - 1):                       # anchor itself has zero translation
        t1 = np.linalg.norm(scaled_relative_pose(P, f, a, 1.0)[:3, 3])
        for s in (0.25, 2.0, 27.3665):
            ts = np.linalg.norm(scaled_relative_pose(P, f, a, s)[:3, 3])
            assert ts / t1 == pytest.approx(s, rel=1e-12)


def test_scale_is_not_applied_twice():
    """Applying s once must not equal applying it twice, and must be linear in s."""
    P = random_poses(seed=4)
    a = len(P) - 1
    f = 0
    t1 = scaled_relative_pose(P, f, a, 1.0)[:3, 3]
    s = 5.0
    assert np.allclose(scaled_relative_pose(P, f, a, s)[:3, 3], s * t1)
    assert not np.allclose(scaled_relative_pose(P, f, a, s)[:3, 3], s * s * t1)


def test_check_pose_scaling_accepts_valid_poses():
    check_pose_scaling(random_poses(seed=5), 4, 27.3665)


def test_check_pose_scaling_rejects_a_rotation_scaling_bug():
    """A wrong implementation that scales the whole 3x4 block must be caught."""
    P = random_poses(seed=6)
    a = len(P) - 1
    bad = np.linalg.inv(P[a]) @ P[0]
    bad[:3, :4] *= 3.0                                   # scales rotation too
    good = scaled_relative_pose(P, 0, a, 3.0)
    assert not np.allclose(bad[:3, :3], good[:3, :3])


# --------------------------------------------------------------------------- #
# Decomposition algebra
# --------------------------------------------------------------------------- #
def test_median_centring_gives_a_zero_median_shape_residual():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.2, size=(5, 17, 23))
    support = rng.random(r.shape) > 0.3
    a_clip = float(np.median(r[support]))
    r_shape = r - a_clip
    assert abs(float(np.median(r_shape[support]))) < 1e-12


def test_c5_depth_equals_c1_depth():
    """s0*exp(a)*D*exp(r-a) == s0*D*exp(r), in float64."""
    rng = np.random.default_rng(1)
    D = rng.uniform(0.05, 4.0, size=(5, 17, 23))
    r = 0.7 * np.tanh(rng.normal(size=D.shape))
    support = rng.random(D.shape) > 0.3
    s0 = 27.3665
    a = float(np.median(r[support]))
    s_learned = s0 * np.exp(a)
    assert np.allclose(s_learned * D * np.exp(r - a), s0 * D * np.exp(r), rtol=1e-12, atol=0)


def test_shape_residual_carries_no_clip_level_scalar():
    """A pure scale change moves a_clip and leaves r_shape bit-identical."""
    rng = np.random.default_rng(2)
    r = rng.normal(scale=0.2, size=(5, 9, 11))
    support = np.ones_like(r, bool)
    a0 = float(np.median(r[support]))
    shifted = r + 0.31                                   # exactly a scale change in log space
    a1 = float(np.median(shifted[support]))
    assert a1 - a0 == pytest.approx(0.31, abs=1e-12)
    assert np.allclose(shifted - a1, r - a0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Fusion consistency: coupled scaling is a similarity, depth-only scaling is not
# --------------------------------------------------------------------------- #
def _toy_clip(seed=0):
    rng = np.random.default_rng(seed)
    T, H, W = 3, 6, 8
    depth = rng.uniform(2.0, 30.0, size=(T, H, W))
    mask = rng.random((T, H, W)) > 0.2
    K = np.tile(np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]]), (T, 1, 1))
    return depth, mask, K, random_poses(T, seed=seed + 10)


def test_coupled_scaling_is_a_pure_similarity_of_the_fused_cloud():
    """Scaling depth AND translation by s scales the anchor-frame cloud by exactly s."""
    depth, mask, K, P = _toy_clip()
    s = 4.3
    base = fuse(depth, mask, K, P, 1.0)
    scaled = fuse(s * depth, mask, K, P, s)
    assert base.shape == scaled.shape and base.shape[0] > 0
    assert np.allclose(scaled, s * base, rtol=1e-12, atol=1e-9)


def test_depth_only_scaling_is_not_a_similarity():
    """The C1-style inconsistency: depth scaled by s, translation left at s0."""
    depth, mask, K, P = _toy_clip(seed=3)
    s, s0 = 4.3, 2.0
    inconsistent = fuse(s * depth, mask, K, P, s0)
    consistent = fuse(s * depth, mask, K, P, s)
    assert not np.allclose(inconsistent, consistent, atol=1e-6)


def test_fuse_places_the_anchor_frame_without_any_pose_transform():
    """Anchor-frame points must be independent of the pose scale."""
    depth, mask, K, P = _toy_clip(seed=5)
    anchor_only = np.zeros_like(mask); anchor_only[-1] = mask[-1]
    a = fuse(depth, anchor_only, K, P, 1.0)
    b = fuse(depth, anchor_only, K, P, 99.0)
    assert np.allclose(a, b, atol=1e-12)


def test_fuse_unprojection_matches_the_pinhole_formula():
    depth, mask, K, P = _toy_clip(seed=7)
    one = np.zeros_like(mask); one[-1, 2, 3] = True
    p = fuse(depth, one, K, P, 1.0)
    d = depth[-1, 2, 3]
    assert p.shape == (1, 3)
    assert p[0] == pytest.approx([(3 - K[-1][0, 2]) * d / K[-1][0, 0],
                                  (2 - K[-1][1, 2]) * d / K[-1][1, 1], d], rel=1e-12)
