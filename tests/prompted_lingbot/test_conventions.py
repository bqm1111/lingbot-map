"""Convention tests: the geometry this prototype relies on must match the
existing LingbotMap pipeline exactly, and Sim(3) must act on depth, points and
poses in the documented way.

These run on CPU with synthetic data; no checkpoint or GPU is needed.
"""

import numpy as np
import pytest

from prompted_lingbot import conventions as cv


def _rand_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rand_c2w(rng):
    return np.concatenate([_rand_rotation(rng), rng.normal(scale=3.0, size=(3, 1))], axis=1)


def _K(H=48, W=64, f=55.0):
    return np.array([[f, 0.0, W / 2 - 0.5], [0.0, f, H / 2 - 0.5], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# 1. Back-projection agrees with the existing LingbotMap pipeline
# --------------------------------------------------------------------------- #
def test_backprojection_matches_upstream_pipeline():
    """``depth_to_world_points`` (c2w in) == ``depth_to_world_coords_points`` (w2c in).

    The upstream helper inverts its extrinsic argument internally, so the two
    agree only when it is handed the *inverse* of our camera-to-world matrix.
    This test is what pins that relationship down.
    """
    from lingbot_map.utils.geometry import depth_to_world_coords_points

    rng = np.random.default_rng(0)
    H, W = 48, 64
    K = _K(H, W)
    depth = rng.uniform(0.5, 20.0, size=(H, W))
    c2w = _rand_c2w(rng)

    mine = cv.depth_to_world_points(depth, K, c2w)

    w2c = np.zeros((3, 4))
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
    theirs, _, _ = depth_to_world_coords_points(depth.astype(np.float32), w2c, K)

    assert np.abs(mine - theirs).max() < 2e-4


def test_upstream_pipeline_disagrees_when_fed_c2w():
    """Guard against silently 'fixing' the inversion above.

    Feeding the upstream helper our camera-to-world matrix must produce a
    materially different cloud -- if this ever stops being true, the upstream
    convention changed and :mod:`prompted_lingbot.conventions` needs revisiting.
    """
    from lingbot_map.utils.geometry import depth_to_world_coords_points

    rng = np.random.default_rng(1)
    H, W = 32, 32
    K = _K(H, W)
    depth = rng.uniform(1.0, 10.0, size=(H, W))
    c2w = _rand_c2w(rng)

    mine = cv.depth_to_world_points(depth, K, c2w)
    wrong, _, _ = depth_to_world_coords_points(depth.astype(np.float32), c2w, K)
    assert np.abs(mine - wrong).max() > 1.0


# --------------------------------------------------------------------------- #
# 2. The reference camera
# --------------------------------------------------------------------------- #
def test_camera_centre_is_the_c2w_translation():
    rng = np.random.default_rng(2)
    c2w = _rand_c2w(rng)
    origin_cam = np.zeros(3)
    assert np.allclose(cv.camera_to_world(origin_cam, c2w), c2w[:3, 3])
    assert np.allclose(cv.camera_centre(c2w), c2w[:3, 3])


def test_identity_reference_camera_is_a_no_op():
    """A frame whose c2w is identity leaves camera points untouched."""
    rng = np.random.default_rng(3)
    H, W = 16, 16
    K = _K(H, W)
    depth = rng.uniform(1.0, 5.0, size=(H, W))
    eye = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    world = cv.depth_to_world_points(depth, K, eye)
    assert np.allclose(world, cv.depth_to_camera_points(depth, K))


def test_world_to_camera_inverts_camera_to_world():
    rng = np.random.default_rng(4)
    c2w = _rand_c2w(rng)
    pts = rng.normal(size=(20, 3))
    assert np.allclose(cv.world_to_camera(cv.camera_to_world(pts, c2w), c2w), pts, atol=1e-9)


def test_projection_roundtrip():
    rng = np.random.default_rng(5)
    H, W = 40, 60
    K = _K(H, W)
    depth = rng.uniform(1.0, 12.0, size=(H, W))
    cam = cv.depth_to_camera_points(depth, K)
    u, v, z = cv.project(cam, K)
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    assert np.allclose(u, uu, atol=1e-9)
    assert np.allclose(v, vv, atol=1e-9)
    assert np.allclose(z, depth)


# --------------------------------------------------------------------------- #
# 3. A known Sim(3) acts as specified
# --------------------------------------------------------------------------- #
def test_known_sim3_on_points_poses_and_depth():
    """Depth takes only the scale; points and camera centres take the full Sim(3)."""
    rng = np.random.default_rng(6)
    H, W = 24, 32
    K = _K(H, W)
    depth = rng.uniform(1.0, 9.0, size=(H, W))
    c2w = _rand_c2w(rng)
    S = cv.Sim3(2.5, _rand_rotation(rng), np.array([1.0, -2.0, 0.5]))

    world = cv.depth_to_world_points(depth, K, c2w)
    world_S = S.apply_points(world)

    # Transforming the pose and scaling the depth must reproduce the same cloud.
    c2w_S = S.apply_pose_c2w(c2w)
    depth_S = S.apply_depth(depth)
    world_via_pose = cv.depth_to_world_points(depth_S, K, c2w_S)

    assert np.abs(world_S - world_via_pose).max() < 1e-9
    assert np.allclose(cv.camera_centre(c2w_S), S.apply_centres(cv.camera_centre(c2w)))
    assert np.allclose(depth_S, 2.5 * depth)


def test_depth_is_invariant_to_the_rotation_and_translation_of_a_sim3():
    """A pure SE(3) correction must not change any depth value."""
    rng = np.random.default_rng(7)
    depth = rng.uniform(1.0, 9.0, size=(8, 8))
    S = cv.Sim3(1.0, _rand_rotation(rng), rng.normal(size=3))
    assert np.allclose(S.apply_depth(depth), depth)


def test_sim3_compose_and_inverse():
    rng = np.random.default_rng(8)
    A = cv.Sim3(1.7, _rand_rotation(rng), rng.normal(size=3))
    B = cv.Sim3(0.4, _rand_rotation(rng), rng.normal(size=3))
    x = rng.normal(size=(10, 3))
    assert np.allclose(A.compose(B).apply_points(x), A.apply_points(B.apply_points(x)))
    assert np.allclose(A.inverse().apply_points(A.apply_points(x)), x)
    ident = A.compose(A.inverse())
    assert np.isclose(ident.s, 1.0)
    assert np.allclose(ident.R, np.eye(3), atol=1e-12)
    assert np.allclose(ident.t, 0.0, atol=1e-12)


def test_identity_sim3_changes_nothing():
    rng = np.random.default_rng(9)
    I = cv.Sim3.identity()
    x = rng.normal(size=(7, 3))
    c2w = _rand_c2w(rng)
    d = rng.uniform(1, 5, size=(4, 4))
    assert np.allclose(I.apply_points(x), x)
    assert np.allclose(I.apply_pose_c2w(c2w), c2w)
    assert np.allclose(I.apply_depth(d), d)


# --------------------------------------------------------------------------- #
# 4. Umeyama recovers a synthetic Sim(3) exactly
# --------------------------------------------------------------------------- #
def test_umeyama_recovers_known_sim3_exactly():
    rng = np.random.default_rng(10)
    S = cv.Sim3(3.25, _rand_rotation(rng), np.array([-4.0, 7.5, 0.25]))
    src = rng.normal(scale=5.0, size=(64, 3))
    dst = S.apply_points(src)
    est = cv.umeyama_sim3(src, dst)
    assert np.isclose(est.s, S.s, rtol=1e-9)
    assert np.allclose(est.R, S.R, atol=1e-9)
    assert np.allclose(est.t, S.t, atol=1e-8)


def test_umeyama_without_scale_is_se3():
    rng = np.random.default_rng(11)
    S = cv.Sim3(1.0, _rand_rotation(rng), rng.normal(size=3))
    src = rng.normal(scale=2.0, size=(30, 3))
    est = cv.umeyama_sim3(src, S.apply_points(src), with_scale=False)
    assert np.isclose(est.s, 1.0)
    assert np.allclose(est.R, S.R, atol=1e-9)


def test_umeyama_needs_three_points():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError):
        cv.umeyama_sim3(rng.normal(size=(2, 3)), rng.normal(size=(2, 3)))


def test_degeneracy_detects_collinear_correspondences():
    t = np.linspace(0, 1, 25)[:, None]
    line = t * np.array([[1.0, 2.0, -0.5]])
    assert cv.sim3_degeneracy(line) < 1e-8
    rng = np.random.default_rng(13)
    assert cv.sim3_degeneracy(rng.normal(size=(25, 3))) > 0.1


# --------------------------------------------------------------------------- #
# 5. Rotation helpers
# --------------------------------------------------------------------------- #
def test_axis_angle_roundtrip():
    rng = np.random.default_rng(14)
    w = rng.normal(scale=0.4, size=(16, 3))
    assert np.allclose(cv.matrix_to_axis_angle(cv.axis_angle_to_matrix(w)), w, atol=1e-8)


def test_axis_angle_zero_is_identity():
    R = cv.axis_angle_to_matrix(np.zeros(3))
    assert np.allclose(R, np.eye(3))


def test_rotation_geodesic_is_zero_for_equal_rotations():
    rng = np.random.default_rng(15)
    R = _rand_rotation(rng)
    assert cv.rotation_geodesic_deg(R, R) < 1e-6
    flip = cv.axis_angle_to_matrix(np.array([np.pi, 0.0, 0.0]))
    assert abs(cv.rotation_geodesic_deg(R, R @ flip) - 180.0) < 1e-3
    quarter = cv.axis_angle_to_matrix(np.array([0.0, np.pi / 2, 0.0]))
    assert abs(cv.rotation_geodesic_deg(R, R @ quarter) - 90.0) < 1e-6
