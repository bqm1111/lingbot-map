"""Calibration, projection, preprocessing and manifest tests (plan tests 6-13)."""

import json
import os

import numpy as np
import pytest

from gates.scale_gate.kitti import (
    KittiCalibration, Preprocess, build_clips, load_poses_cam2_c2w, parse_calibration,
    project_lidar_to_depth, read_manifest, validate_sequence, write_manifest,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KITTI = os.path.join(REPO, "data/kitti/dataset")
HAS_KITTI = os.path.isdir(os.path.join(KITTI, "sequences", "08"))
kitti_only = pytest.mark.skipif(not HAS_KITTI, reason="SemanticKITTI unavailable")


def _calib(tmp_path, p2_col3=(0.0, 0.0, 0.0)) -> str:
    """A minimal calib.txt with identity Tr and a controllable P2 4th column."""
    p = tmp_path / "calib.txt"
    P2 = np.zeros((3, 4))
    P2[:3, :3] = [[700.0, 0, 600.0], [0, 700.0, 180.0], [0, 0, 1.0]]
    P2[:, 3] = p2_col3
    Tr = np.eye(4)[:3, :4]
    p.write_text("P2: " + " ".join(f"{x:.12e}" for x in P2.ravel()) + "\n"
                 + "Tr: " + " ".join(f"{x:.12e}" for x in Tr.ravel()) + "\n")
    return str(p)


# ---------------- 6. calibration ---------------- #
def test_calibration_shapes_and_finiteness(tmp_path):
    c = parse_calibration(_calib(tmp_path))
    assert c.K.shape == (3, 3) and c.Tr.shape == (4, 4) and c.t_cam2.shape == (3,)
    assert np.isfinite(c.K).all() and np.isfinite(c.Tr).all()
    assert c.K[0, 0] == pytest.approx(700.0) and c.K[0, 2] == pytest.approx(600.0)


def test_p2_baseline_becomes_a_translation_not_an_intrinsic(tmp_path):
    """P2 is a projection matrix: its 4th column is a baseline, not part of K."""
    c = parse_calibration(_calib(tmp_path, p2_col3=(46.6, 0.35, 0.006)))
    assert np.allclose(c.K, [[700, 0, 600], [0, 700, 180], [0, 0, 1]])
    assert c.t_cam2[0] == pytest.approx(46.6 / 700.0)
    assert c.t_cam2[1] == pytest.approx(0.35 / 700.0)
    assert c.t_cam2[2] == pytest.approx(0.006)


def test_calibration_rejects_malformed_files(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("Tr: " + " ".join("0" for _ in range(12)) + "\n")
    with pytest.raises(ValueError):
        parse_calibration(str(p))


def test_poses_are_offset_from_cam0_to_cam2(tmp_path):
    c = parse_calibration(_calib(tmp_path, p2_col3=(700.0, 0.0, 0.0)))  # t_cam2 = (1,0,0)
    pp = tmp_path / "poses.txt"
    pp.write_text("1 0 0 5  0 1 0 0  0 0 1 0\n")
    out = load_poses_cam2_c2w(str(pp), c)
    assert out.shape == (1, 4, 4)
    assert np.allclose(out[0, :3, :3], np.eye(3))            # rotation untouched
    assert np.allclose(out[0, :3, 3], [4.0, 0.0, 0.0])       # 5 - t_cam2


# ---------------- 7. LiDAR -> image projection ---------------- #
def test_known_point_projects_to_the_expected_pixel():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    calib = KittiCalibration(K=K, t_cam2=np.zeros(3), Tr=np.eye(4), P2=np.zeros((3, 4)))
    pts = np.array([[1.0, 2.0, 10.0]])                       # -> u=60, v=60
    d, v = project_lidar_to_depth(pts, calib, (80, 100), K)
    assert v.sum() == 1 and v[60, 60]
    assert d[60, 60] == pytest.approx(10.0)


def test_points_behind_or_outside_the_camera_are_dropped():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    calib = KittiCalibration(K=K, t_cam2=np.zeros(3), Tr=np.eye(4), P2=np.zeros((3, 4)))
    pts = np.array([[0, 0, -5.0], [0, 0, np.nan], [1000.0, 0, 10.0], [0, 0, 10.0]])
    d, v = project_lidar_to_depth(pts, calib, (80, 100), K)
    assert v.sum() == 1 and v[40, 50]                        # only the valid centre point


# ---------------- 9. z-buffering ---------------- #
def test_zbuffer_keeps_the_nearest_point_in_a_shared_pixel():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    calib = KittiCalibration(K=K, t_cam2=np.zeros(3), Tr=np.eye(4), P2=np.zeros((3, 4)))
    pts = np.array([[0, 0, 30.0], [0, 0, 5.0], [0, 0, 12.0]])  # same pixel, three depths
    d, v = project_lidar_to_depth(pts, calib, (80, 100), K)
    assert v.sum() == 1
    assert d[40, 50] == pytest.approx(5.0), "the far point bled through the near one"


def test_depth_range_filter_is_applied():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    calib = KittiCalibration(K=K, t_cam2=np.zeros(3), Tr=np.eye(4), P2=np.zeros((3, 4)))
    pts = np.array([[0, 0, 0.5], [0, 0, 200.0], [0, 0, 20.0]])
    d, v = project_lidar_to_depth(pts, calib, (80, 100), K, min_depth=1.0, max_depth=80.0)
    assert v.sum() == 1 and d[40, 50] == pytest.approx(20.0)


# ---------------- 8. original -> processed coordinates ---------------- #
def test_preprocess_matches_the_demo_resize_for_kitti():
    p = Preprocess.build((370, 1226))
    assert p.proc_hw == (154, 518)                     # round(370*518/1226/14)*14
    assert p.sx == pytest.approx(518 / 1226) and p.sy == pytest.approx(154 / 370)


def test_preprocess_scales_intrinsics_and_pixels_consistently():
    p = Preprocess.build((370, 1226))
    K = np.array([[707.09, 0, 601.89], [0, 707.09, 183.11], [0, 0, 1.0]])
    Ks = p.scale_intrinsics(K)
    # A point projected with K then mapped must equal the same point projected with Ks.
    X = np.array([[2.0, 1.0, 15.0]])
    uv = np.array([[X[0, 0] * K[0, 0] / X[0, 2] + K[0, 2],
                    X[0, 1] * K[1, 1] / X[0, 2] + K[1, 2]]])
    direct = np.array([[X[0, 0] * Ks[0, 0] / X[0, 2] + Ks[0, 2],
                        X[0, 1] * Ks[1, 1] / X[0, 2] + Ks[1, 2]]])
    assert np.allclose(p.map_pixels(uv), direct)


def test_preprocess_refuses_a_resolution_that_would_be_cropped():
    with pytest.raises(RuntimeError):
        Preprocess.build((1200, 600))          # portrait: resized height would exceed 518


# ---------------- 10/11. manifests ---------------- #
@kitti_only
def test_manifest_generation_is_deterministic():
    a = build_clips(KITTI, ["08"], 5, 5, max_clips_per_sequence=6)
    b = build_clips(KITTI, ["08"], 5, 5, max_clips_per_sequence=6)
    assert a == b and len(a) == 6
    r = a[0]
    assert r["frame_ids"] == [0, 5, 10, 15, 20]
    assert r["clip_id"] == "08_000000_000020_s5"
    assert all(not os.path.isabs(p) for p in r["image_paths"])


@kitti_only
def test_train_and_validation_sequences_do_not_leak():
    train = build_clips(KITTI, ["09", "10"], 5, 5, max_clips_per_sequence=4)
    val = build_clips(KITTI, ["08"], 5, 5, max_clips_per_sequence=4)
    assert {c["sequence"] for c in train}.isdisjoint({c["sequence"] for c in val})
    assert {c["clip_id"] for c in train}.isdisjoint({c["clip_id"] for c in val})


def test_clip_length_below_two_is_refused():
    with pytest.raises(ValueError):
        build_clips(KITTI, ["08"], 1, 5)


def test_manifest_round_trips_and_hashes_stably(tmp_path):
    recs = [{"clip_id": "a", "frame_ids": [0, 1]}, {"clip_id": "b", "frame_ids": [2, 3]}]
    p1, p2 = str(tmp_path / "m1.jsonl"), str(tmp_path / "m2.jsonl")
    h1, h2 = write_manifest(p1, recs), write_manifest(p2, recs)
    assert h1 == h2 and read_manifest(p1) == recs
    assert write_manifest(p2, recs[::-1]) != h1        # order is part of the identity


# ---------------- real-data sanity ---------------- #
@kitti_only
def test_real_sequence_validates_cleanly():
    inv = validate_sequence(KITTI, "08")
    assert inv.ok, inv.problems
    assert inv.n_images == inv.n_lidar == inv.n_poses == 4071
    assert inv.image_hw == (370, 1226)


@kitti_only
def test_real_lidar_projects_into_the_image_plausibly():
    from gates.scale_gate.kitti import read_velodyne
    calib = parse_calibration(os.path.join(KITTI, "sequences/08/calib.txt"))
    pts = read_velodyne(os.path.join(KITTI, "sequences/08/velodyne/000100.bin"))[:, :3]
    pre = Preprocess.build((370, 1226))
    d, v = project_lidar_to_depth(pts, calib, pre.proc_hw, pre.scale_intrinsics(calib.K))
    assert v.sum() > 2000, "implausibly few LiDAR pixels landed in the image"
    assert 1.0 < float(d[v].min()) and float(d[v].max()) < 80.0
    assert 5.0 < float(np.median(d[v])) < 40.0
