"""Phase-2 sanity checks. Nothing downstream is trustworthy until these pass.

Each test pins one convention that, if wrong, would silently corrupt every
occupancy number: voxel indexing, axis order, boundary handling, the
camera-to-LiDAR transform, Z-depth back-projection, and mask semantics.
"""

import os

import numpy as np
import pytest

from prompted_lingbot.conventions import Sim3, axis_angle_to_matrix
from prompted_lingbot.occ_datasets import (
    apply_transform, camera_points_from_depth, load_occ3d_target,
    load_semantickitti_target, read_kitti_calib, relative_c2w,
)
from prompted_lingbot.occupancy import (
    OCC3D_NUSCENES_GRID, OCC3D_SINGLE_CAMERA_X_CUT, SEMANTICKITTI_GRID,
    accumulate_scores, binary_occupancy_scores, occupancy_from_points, voxelize_points,
)

KITTI_ROOT = "data/kitti/dataset"
OCC3D_ROOT = ("/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/"
              "Occupancy3D-nuScenes-trainval/gts")


# --------------------------------------------------------------------------- #
# grid definitions
# --------------------------------------------------------------------------- #
def test_semantickitti_grid_matches_the_official_definition():
    g = SEMANTICKITTI_GRID
    assert g.dims == (256, 256, 32)
    assert g.voxel_size == 0.2
    assert g.origin == (0.0, -25.6, -2.0)
    assert g.scene_size == (51.2, 51.2, 6.4)
    assert np.allclose(g.upper, [51.2, 25.6, 4.4])
    assert g.empty_class == 0 and g.ignore_label == 255


def test_occ3d_grid_matches_the_official_definition():
    g = OCC3D_NUSCENES_GRID
    assert g.dims == (200, 200, 16)
    assert g.voxel_size == 0.4
    assert g.origin == (-40.0, -40.0, -1.0)
    # 16 * 0.4 = 6.4 so z spans -1.0 .. 5.4. OccAny's nuscenes.py pc_range lists a
    # z-max of 3.0, which is inconsistent with 16 voxels and is unused dead
    # metadata -- only pc_range[:3] is read, as the grid origin.
    assert np.allclose(g.upper, [40.0, 40.0, 5.4])
    assert g.empty_class == 17 and g.ignore_label == 255


# --------------------------------------------------------------------------- #
# voxel indexing
# --------------------------------------------------------------------------- #
def test_single_point_lands_in_the_expected_voxel():
    g = SEMANTICKITTI_GRID
    # origin (0, -25.6, -2), voxel 0.2 -> this point is inside voxel (5, 3, 10)
    p = np.array([[0.0 + 5 * 0.2 + 0.1, -25.6 + 3 * 0.2 + 0.1, -2.0 + 10 * 0.2 + 0.1]])
    idx, keep = voxelize_points(p, g)
    assert keep.all()
    assert idx.tolist() == [[5, 3, 10]]


def test_voxel_centres_round_trip_to_their_own_index():
    g = OCC3D_NUSCENES_GRID
    want = np.array([[0, 0, 0], [199, 199, 15], [100, 50, 8]])
    idx, keep = voxelize_points(g.voxel_centres(want), g)
    assert keep.all()
    assert idx.tolist() == want.tolist()


def test_the_grid_origin_lands_in_voxel_zero():
    g = SEMANTICKITTI_GRID
    idx, keep = voxelize_points(np.array([[0.0, -25.6, -2.0]]), g)
    assert keep.all() and idx.tolist() == [[0, 0, 0]]


def test_voxel_boundaries_split_the_two_neighbouring_voxels():
    """A point just below a boundary goes low, just above goes high.

    The boundary value *itself* is not asserted: ``floor((p - origin)/size)`` is
    float arithmetic, and e.g. ``-25.6 + 0.2`` is not exactly representable, so an
    exact boundary can land either side by one ULP. OccAny uses the identical
    expression and inherits the identical ambiguity, so pinning it would be
    testing numpy, not the convention.
    """
    g = SEMANTICKITTI_GRID
    eps = 1e-9
    for axis in range(3):
        for k in (1, 7, 30):
            p = np.array(g.origin, float).copy()
            p[axis] = g.origin[axis] + k * g.voxel_size
            below = p.copy(); below[axis] -= eps
            above = p.copy(); above[axis] += eps
            i_lo, _ = voxelize_points(below[None], g)
            i_hi, _ = voxelize_points(above[None], g)
            assert i_lo[0, axis] == k - 1, f"axis {axis} k {k}"
            assert i_hi[0, axis] == k, f"axis {axis} k {k}"


def test_voxelization_is_deterministic():
    g = SEMANTICKITTI_GRID
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, -25, -2], [51, 25, 4], size=(5000, 3))
    a = occupancy_from_points(pts, g)
    b = occupancy_from_points(pts, g)
    c = occupancy_from_points(pts[rng.permutation(len(pts))], g)
    assert np.array_equal(a, b), "same input must give the same volume"
    assert np.array_equal(a, c), "point order must not change the volume"


def test_points_outside_the_evaluation_range_are_dropped():
    g = SEMANTICKITTI_GRID
    pts = np.array([
        [-0.001, 0.0, 0.0],          # below x
        [51.2, 0.0, 0.0],            # at the upper x face -> index 256, out
        [10.0, -25.61, 0.0],         # below y
        [10.0, 25.6, 0.0],           # at the upper y face
        [10.0, 0.0, -2.001],         # below z
        [10.0, 0.0, 4.4],            # at the upper z face
        [10.0, 0.0, 0.0],            # the only interior point
    ])
    idx, keep = voxelize_points(pts, g)
    assert keep.tolist() == [False] * 6 + [True]
    assert idx.shape == (1, 3)


def test_axis_order_is_x_y_z_not_transposed():
    """A point far along +x must produce a large FIRST index, not a large third."""
    g = SEMANTICKITTI_GRID
    idx, _ = voxelize_points(np.array([[50.0, 0.0, 0.0]]), g)
    assert idx[0, 0] > 200 and idx[0, 1] < 200 and idx[0, 2] < 32
    idx, _ = voxelize_points(np.array([[1.0, 24.0, 0.0]]), g)
    assert idx[0, 1] > 200


def test_duplicate_points_aggregate_rather_than_double_count():
    g = SEMANTICKITTI_GRID
    p = g.voxel_centres(np.array([[7, 8, 9]]))
    vol = occupancy_from_points(np.repeat(p, 100, axis=0), g)
    assert int(vol.sum()) == 1
    assert bool(vol[7, 8, 9])


def test_min_points_per_voxel_threshold_counts_correctly():
    g = SEMANTICKITTI_GRID
    a = g.voxel_centres(np.array([[1, 1, 1]]))
    b = g.voxel_centres(np.array([[2, 2, 2]]))
    pts = np.concatenate([np.repeat(a, 5, 0), np.repeat(b, 2, 0)])
    assert int(occupancy_from_points(pts, g, min_points_per_voxel=1).sum()) == 2
    assert int(occupancy_from_points(pts, g, min_points_per_voxel=3).sum()) == 1
    assert bool(occupancy_from_points(pts, g, min_points_per_voxel=3)[1, 1, 1])


def test_empty_input_gives_an_empty_volume():
    vol = occupancy_from_points(np.zeros((0, 3)), SEMANTICKITTI_GRID)
    assert vol.shape == SEMANTICKITTI_GRID.dims and not vol.any()


# --------------------------------------------------------------------------- #
# Z-depth back-projection and frame conversions
# --------------------------------------------------------------------------- #
def test_zdepth_backprojection_puts_the_principal_ray_on_the_optical_axis():
    K = np.array([[100.0, 0.0, 15.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]])
    depth = np.zeros((16, 32)) - 1.0
    depth[7, 15] = 5.0            # not the principal pixel
    depth[8, 16] = 5.0            # (u, v) = (16, 8)
    pts = camera_points_from_depth(depth, K, None, 0.0, 0.1, 100.0)
    assert pts.shape == (2, 3)
    assert np.allclose(pts[:, 2], 5.0)
    # x = (u - cx) * d / fx
    assert np.allclose(sorted(pts[:, 0]), sorted([(15 - 15.5) * 5 / 100, (16 - 15.5) * 5 / 100]))


def test_zdepth_not_ray_distance():
    """An off-axis pixel keeps z == depth; its norm is strictly larger."""
    K = np.array([[50.0, 0.0, 10.0], [0.0, 50.0, 10.0], [0.0, 0.0, 1.0]])
    depth = np.full((21, 21), -1.0)
    depth[0, 0] = 8.0
    p = camera_points_from_depth(depth, K, None, 0.0, 0.1, 100.0)[0]
    assert p[2] == pytest.approx(8.0)
    assert np.linalg.norm(p) > 8.0


def test_depth_filtering_by_confidence_and_range():
    K = np.eye(3)
    depth = np.array([[1.0, 2.0], [3.0, 200.0]])
    conf = np.array([[3.0, 1.0], [3.0, 3.0]])
    pts = camera_points_from_depth(depth, K, conf, 2.0, 0.5, 100.0)
    assert pts.shape[0] == 2                      # 2.0 fails conf, 200 fails range
    assert sorted(pts[:, 2]) == [1.0, 3.0]


def test_relative_c2w_composes_camera_to_world_correctly():
    rng = np.random.default_rng(0)
    def c2w():
        m = np.eye(4)
        m[:3, :3] = axis_angle_to_matrix(rng.normal(scale=0.4, size=3))
        m[:3, 3] = rng.normal(scale=2.0, size=3)
        return m
    A, B = c2w(), c2w()
    p_a = rng.normal(size=(20, 3))
    world = apply_transform(A, p_a)               # camera A -> world
    p_b = apply_transform(np.linalg.inv(B), world)  # world -> camera B
    assert np.allclose(apply_transform(relative_c2w(A, B), p_a), p_b)


def test_relative_c2w_of_a_frame_with_itself_is_the_identity():
    m = np.eye(4)
    m[:3, :3] = axis_angle_to_matrix(np.array([0.1, -0.2, 0.3]))
    m[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(relative_c2w(m, m), np.eye(4), atol=1e-12)


def test_a_global_sim3_cancels_in_the_anchor_frame_except_for_scale():
    """Key structural fact: rendering in the anchor camera's own frame makes the
    rigid part of a global Sim(3) pure gauge. Only the scale survives."""
    rng = np.random.default_rng(1)
    C = Sim3(3.7, axis_angle_to_matrix(rng.normal(scale=0.5, size=3)), rng.normal(size=3))
    pose_f, pose_t = np.eye(4), np.eye(4)
    pose_f[:3, :3] = axis_angle_to_matrix(rng.normal(scale=0.3, size=3))
    pose_f[:3, 3] = rng.normal(size=3)
    pose_t[:3, :3] = axis_angle_to_matrix(rng.normal(scale=0.3, size=3))
    pose_t[:3, 3] = rng.normal(size=3)
    p_cam_f = rng.normal(size=(30, 3)) + np.array([0, 0, 5.0])

    plain = apply_transform(relative_c2w(pose_f, pose_t), p_cam_f) * C.s

    corr_f = np.eye(4); corr_f[:3, :4] = C.apply_pose_c2w(pose_f[:3, :4])
    corr_t = np.eye(4); corr_t[:3, :4] = C.apply_pose_c2w(pose_t[:3, :4])
    viaC = apply_transform(relative_c2w(corr_f, corr_t), C.s * p_cam_f)
    assert np.allclose(plain, viaC, atol=1e-9)


# --------------------------------------------------------------------------- #
# camera <-> LiDAR
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(KITTI_ROOT), reason="KITTI not available")
def test_camera_to_lidar_is_the_inverse_of_Tr():
    calib = read_kitti_calib(os.path.join(KITTI_ROOT, "sequences", "08", "calib.txt"))
    Tr = calib["Tr"]
    cam2velo = np.linalg.inv(Tr)
    assert np.allclose(Tr @ cam2velo, np.eye(4), atol=1e-9)
    # A point 10 m in front of the camera (+z) must be ~10 m ahead of the
    # velodyne (+x), and roughly at the camera's height offset.
    ahead = apply_transform(cam2velo, np.array([[0.0, 0.0, 10.0]]))[0]
    assert ahead[0] == pytest.approx(10.0, abs=0.5)
    assert abs(ahead[1]) < 0.5


@pytest.mark.skipif(not os.path.isdir(KITTI_ROOT), reason="KITTI not available")
def test_lidar_points_land_inside_the_ssc_grid():
    """Sanity on real data: a real velodyne scan must populate the SSC volume."""
    p = os.path.join(KITTI_ROOT, "sequences", "08", "velodyne", "000000.bin")
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    vol = occupancy_from_points(pts, SEMANTICKITTI_GRID)
    frac = vol.sum() / np.prod(SEMANTICKITTI_GRID.dims)
    assert 0.005 < frac < 0.15, f"implausible occupied fraction {frac}"


# --------------------------------------------------------------------------- #
# targets and masks
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(KITTI_ROOT), reason="KITTI not available")
def test_semantickitti_target_marks_invalid_as_ignore():
    target, valid = load_semantickitti_target(KITTI_ROOT, "08", 0)
    assert target.shape == SEMANTICKITTI_GRID.dims
    assert np.all(target[~valid] == 255)
    assert np.all(target[valid] != 255)
    assert 0.3 < valid.mean() < 1.0
    assert 0.005 < ((target != 0) & valid).mean() < 0.3


@pytest.mark.skipif(not os.path.isdir(OCC3D_ROOT), reason="Occ3D not available")
def test_occ3d_camera_mask_and_single_camera_crop():
    scene = sorted(os.listdir(OCC3D_ROOT))[0]
    sample = sorted(os.listdir(os.path.join(OCC3D_ROOT, scene)))[0]
    gt_dir = os.path.join(OCC3D_ROOT, scene, sample)

    full, valid_full = load_occ3d_target(gt_dir, single_camera=False)
    front, valid_front = load_occ3d_target(gt_dir, single_camera=True)
    assert full.shape == OCC3D_NUSCENES_GRID.dims
    # The rear half is entirely excluded in the single-camera setting.
    assert np.all(front[:OCC3D_SINGLE_CAMERA_X_CUT] == 255)
    assert not valid_front[:OCC3D_SINGLE_CAMERA_X_CUT].any()
    assert valid_front.sum() < valid_full.sum()
    # The front half is untouched by the crop.
    assert np.array_equal(front[OCC3D_SINGLE_CAMERA_X_CUT:],
                          full[OCC3D_SINGLE_CAMERA_X_CUT:])
    # 17 (free) must survive as a real label, not be confused with ignore.
    assert (front == 17).any()


@pytest.mark.skipif(not os.path.isdir(OCC3D_ROOT), reason="Occ3D not available")
def test_occ3d_lidar_mask_is_off_by_default():
    scene = sorted(os.listdir(OCC3D_ROOT))[0]
    sample = sorted(os.listdir(os.path.join(OCC3D_ROOT, scene)))[0]
    gt_dir = os.path.join(OCC3D_ROOT, scene, sample)
    a, _ = load_occ3d_target(gt_dir, apply_lidar_mask=False, single_camera=False)
    b, _ = load_occ3d_target(gt_dir, apply_lidar_mask=True, single_camera=False)
    assert (b == 255).sum() >= (a == 255).sum()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_binary_scores_are_perfect_for_a_perfect_prediction():
    g = SEMANTICKITTI_GRID
    target = np.zeros(g.dims, np.int32)
    target[10:20, 10:20, 5:8] = 11
    s = binary_occupancy_scores(target != 0, target, g)
    assert s["iou"] == 1.0 and s["precision"] == 1.0 and s["recall"] == 1.0
    assert s["n_gt_occupied"] == 10 * 10 * 3


def test_binary_scores_are_zero_for_a_disjoint_prediction():
    g = SEMANTICKITTI_GRID
    target = np.zeros(g.dims, np.int32)
    target[10:20, 10:20, 5:8] = 11
    pred = np.zeros(g.dims, bool)
    pred[100:110, 100:110, 5:8] = True
    s = binary_occupancy_scores(pred, target, g)
    assert s["iou"] == 0.0 and s["tp"] == 0


def test_ignore_voxels_are_excluded_from_every_term():
    g = SEMANTICKITTI_GRID
    target = np.zeros(g.dims, np.int32)
    target[0:10, 0:10, 0:4] = 255      # ignore
    pred = np.zeros(g.dims, bool)
    pred[0:10, 0:10, 0:4] = True       # predicting inside ignore must be free
    s = binary_occupancy_scores(pred, target, g)
    assert s["fp"] == 0 and s["tp"] == 0 and s["fn"] == 0
    assert s["n_valid_voxels"] == np.prod(g.dims) - 10 * 10 * 4


def test_occ3d_empty_class_17_is_not_treated_as_occupied():
    g = OCC3D_NUSCENES_GRID
    target = np.full(g.dims, 17, np.int32)     # entirely free
    target[5:9, 5:9, 2:4] = 3                  # a small real object
    s = binary_occupancy_scores(target != 17, target, g)
    assert s["iou"] == 1.0
    assert s["n_gt_occupied"] == 4 * 4 * 2
    # Predicting "occupied" everywhere must NOT score well.
    s2 = binary_occupancy_scores(np.ones(g.dims, bool), target, g)
    assert s2["recall"] == 1.0 and s2["precision"] < 0.001


def test_extra_valid_mask_further_restricts_evaluation():
    g = SEMANTICKITTI_GRID
    target = np.zeros(g.dims, np.int32)
    target[10:20, 10:20, 5:8] = 11
    valid = np.zeros(g.dims, bool)
    valid[10:15, 10:20, 5:8] = True
    s = binary_occupancy_scores(target != 0, target, g, valid=valid)
    assert s["n_gt_occupied"] == 5 * 10 * 3
    assert s["n_valid_voxels"] == 5 * 10 * 3


def test_accumulate_pools_counts_not_averages_of_ratios():
    rows = [{"tp": 1, "fp": 0, "fn": 99, "n_pred_occupied": 1, "n_gt_occupied": 100,
             "n_valid_voxels": 1000},
            {"tp": 99, "fp": 1, "fn": 0, "n_pred_occupied": 100, "n_gt_occupied": 99,
             "n_valid_voxels": 1000}]
    a = accumulate_scores(rows)
    assert a["tp"] == 100 and a["fp"] == 1 and a["fn"] == 99
    assert a["iou"] == pytest.approx(100 / 200)
    assert a["n_frames"] == 2


def test_shape_mismatch_is_rejected():
    g = SEMANTICKITTI_GRID
    with pytest.raises(ValueError):
        binary_occupancy_scores(np.zeros((4, 4, 4), bool), np.zeros(g.dims, np.int32), g)


def test_voxelize_rejects_bad_shapes():
    with pytest.raises(ValueError):
        voxelize_points(np.zeros((5, 2)), SEMANTICKITTI_GRID)
