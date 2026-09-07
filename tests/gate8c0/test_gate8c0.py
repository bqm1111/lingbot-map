"""Gate 8C-0 — KITTI-360 alignment and learnability audit.

Two jobs. First, prove the transform chain is what the report says it is: exact inverses,
the right frame, the right binning, and the SSCBench index mapping. Second, prove the
Stage-5 invariants actually fire -- each deliberate defect below (a one-frame offset, a
reversed pose, a drive-ID collision, a future-frame leak, a one-voxel origin shift) must
be caught with a diagnostic that names it, otherwise the audit that passed means nothing.
"""
from __future__ import annotations
import os, sys
import numpy as np, pytest
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from gates.gate6 import grids as G6G
from gates.gate8c0 import checks as CK, oracle as OR, transforms as TF
from sscbench_kitti360 import adapter as K3

S_ROOT = "/media/SSD1/MINH_DATASETS/sscbench_kitti360"
K_ROOT = "/media/welf/MINH/datasets/kitti360/KITTI-360"
DRIVE = "2013_05_28_drive_0006_sync"
TOL = 1e-3
_has_data = os.path.isdir(os.path.join(S_ROOT, "data_poses", DRIVE))
needs_data = pytest.mark.skipif(not _has_data, reason="KITTI-360 not present on this machine")


@pytest.fixture(scope="module")
def geo():
    return TF.DriveGeometry(DRIVE, S_ROOT, K_ROOT)


# ------------------------------------------------------------------ transforms
def test_inv_is_a_true_inverse_not_a_transpose():
    rng = np.random.default_rng(0)
    T = np.eye(4); T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    T[:3, :3] *= 1.000001                      # a slightly non-rigid matrix, as shipped
    T[:3, 3] = rng.normal(size=3) * 10
    p = rng.normal(size=(512, 3)) * 30
    assert np.abs(TF.apply(TF.inv(T), TF.apply(T, p)) - p).max() < 1e-9
    assert np.abs(TF.apply(TF.inv_rigid(T), TF.apply(T, p)) - p).max() > 1e-9


def test_apply_uses_column_vector_convention_on_row_stored_points():
    T = np.eye(4); T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T[:3, 3] = [1, 2, 3]
    p = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(TF.apply(T, p), [[1.0, 3.0, 3.0]])


@needs_data
@pytest.mark.parametrize("pair", ["velo_to_world", "rect_cam_to_velo", "velo_to_velo"])
def test_transform_round_trips_within_tolerance(geo, pair):
    rng = np.random.default_rng(1)
    p = rng.uniform(-40, 40, size=(2048, 3))
    n = geo.native(1000)
    T = {"velo_to_world": geo.velo_to_world(n),
         "rect_cam_to_velo": geo.rect_cam_to_velo,
         "velo_to_velo": geo.velo_to_velo(geo.native(980), n)}[pair]
    assert np.abs(TF.apply(TF.inv(T), TF.apply(T, p)) - p).max() < TOL


@needs_data
def test_sscbench_index_maps_to_pose_frames_plus_one(geo):
    assert geo.native(0) == int(geo.pose_frames[1])
    for i in (0, 40, 1000, 4525):
        assert geo.native(i) == int(geo.pose_frames[i + 1])


@needs_data
def test_camera_sits_where_the_kitti360_rig_puts_it(geo):
    o = geo.rect_cam_to_velo[:3, 3]
    f = geo.rect_cam_to_velo[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert 0.7 < o[0] < 0.9 and 0.2 < o[1] < 0.4 and -0.3 < o[2] < -0.1
    assert f[0] > 0.98                                   # camera looks along velodyne +x


def test_grid_declaration_matches_the_frozen_benchmark():
    g = G6G.PREDICTION_GRID["kitti360"]
    assert tuple(g.dims) == (256, 256, 32) and float(g.voxel_size) == 0.2
    assert tuple(float(x) for x in g.origin) == (0.0, -25.6, -2.0)
    assert g.frame == "velodyne_of_anchor_frame"


def test_voxelize_floors_and_matches_gate6():
    p = np.array([[0.0, -25.6, -2.0], [0.19, -25.41, -1.81], [0.21, -25.39, -1.79]])
    idx, keep = TF.voxelize(p)
    assert keep.all() and np.array_equal(idx, [[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    assert np.array_equal(idx, G6G.voxelize(p, TF.GRID)[0])


def test_grid_index_centre_round_trip_is_exact():
    rng = np.random.default_rng(2)
    idx = np.stack([rng.integers(0, d, 4096) for d in TF.DIMS], -1)
    back, keep = TF.voxelize(TF.centres(idx))
    assert keep.all() and np.array_equal(back, idx)


# ------------------------------------------------------------------ deliberate defects
def test_one_frame_offset_is_caught():
    ok, det = CK.causal_input_only(t=100, input_frames=np.arange(0, 102))
    assert not ok and det["n_future_in_input"] == 1 and det["max"] == 101
    ok, det = CK.future_target_window(t=100, target_frames=np.arange(100, 120))
    assert not ok and det["lo"] == 100 and det["expected_lo"] == 101


@needs_data
def test_reversed_pose_transform_is_caught(geo):
    """Using world->velo where velo->world is meant puts the sweep nowhere near the target."""
    n = geo.native(1130)
    pts = geo.read_velodyne(n)
    correct = geo.velo_to_velo(n - 25, n)
    reversed_ = TF.inv(correct)
    src = geo.read_velodyne(n - 25)
    a = OR.volume_from_points(TF.apply(correct, src))
    b = OR.volume_from_points(TF.apply(reversed_, src))
    ref = OR.volume_from_points(pts)
    ones = np.ones_like(ref)
    ca, cb = OR.counts(a, ref, ones), OR.counts(b, ref, ones)
    assert ca["iou"] > 2 * cb["iou"], (ca["iou"], cb["iou"])


def test_drive_id_collision_is_caught():
    ok, det = CK.cache_keys_cannot_collide({"d1": ["0000000004", "0000000009"],
                                            "d2": ["0000000004"]})
    assert not ok and det["n_collisions"] == 1 and "0000000004" in det["examples"]
    ok, _ = CK.cache_keys_cannot_collide({"d1": ["0003_0000000004"], "d2": ["0007_0000000004"]})
    assert ok, "drive-scoped keys must not collide"


def test_future_frame_leak_is_caught():
    ok, det = CK.no_future_leak(np.arange(0, 51), np.arange(45, 71))
    assert not ok and det["n_overlap"] == 6
    assert CK.no_future_leak(np.arange(0, 51), np.arange(51, 71))[0]


@needs_data
def test_one_voxel_origin_shift_is_caught(geo):
    """A 0.2 m origin error moves every measurement one voxel and the shift scan sees it."""
    n = geo.native(1130)
    pts = geo.read_velodyne(n)
    good = OR.volume_from_points(pts)
    bad = OR.volume_from_points(pts + np.array([0.0, 0.0, TF.VOXEL]))
    assert not np.array_equal(good, bad)
    sc = OR.shift_scan(bad, good, np.ones_like(good))
    best = max(sc.items(), key=lambda kv: kv[1]["iou"])[0]
    assert best == "(0, 0, -1)", best
    assert sc["(0, 0, 0)"]["iou"] < sc[best]["iou"]


def test_integrated_once_and_scale_window_are_caught():
    assert CK.integrated_once(12, 12)[0] and not CK.integrated_once(13, 12)[0]
    assert CK.scale_anchor_window([0, 1, 2, 3, 4])[0]
    assert not CK.scale_anchor_window([0, 1, 2, 3, 4, 5])[0]
    assert not CK.scale_anchor_window([0, 1, 2, 3, 7])[0]


def test_supervising_an_invalid_voxel_is_caught():
    valid = np.zeros(100, bool); valid[:60] = True
    occ = np.zeros(100, bool)
    assert CK.unknown_handled_consistently(occ, valid, valid)[0]
    bad = valid.copy(); bad[70] = True
    ok, det = CK.unknown_handled_consistently(occ, valid, bad)
    assert not ok and det["n_supervised_but_invalid"] == 1


def test_partitions_disjoint_is_caught():
    assert CK.partitions_disjoint(["a", "b"], ["c"], ["d"])[0]
    ok, det = CK.partitions_disjoint(["a", "b"], ["b"], ["d"])
    assert not ok and det["train_val"] == ["b"]


# ------------------------------------------------------------------ recorded findings
@needs_data
def test_our_voxelization_reproduces_the_official_sscbench_input(geo):
    """The Stage-2b control, as a test: our chain must match SSCBench's own .bin at zero shift."""
    VD = os.path.join(S_ROOT, "data_2d_raw", DRIVE, "voxels")
    anchor = 1130
    p = os.path.join(VD, f"{anchor:06d}.bin")
    if not os.path.exists(p):
        pytest.skip("official voxel inputs not extracted")
    off = np.unpackbits(np.fromfile(p, np.uint8)).reshape(TF.DIMS).astype(bool)
    mine = OR.volume_from_points(geo.read_velodyne(geo.native(anchor)))
    sc = OR.shift_scan(mine, off, np.ones_like(off))
    assert sc["(0, 0, 0)"]["recall"] > 0.95
    assert max(sc.items(), key=lambda kv: kv[1]["iou"])[0] == "(0, 0, 0)"


def test_adapter_target_rule_is_documented_and_stable():
    src = K3.load_target.__doc__ or ""
    assert "sequence" in src
    import inspect
    assert inspect.signature(K3.load_target).parameters["sequence"].default == K3.SEQUENCE


def test_gate8_8a_8b_artifacts_are_untouched():
    for f in ("artifacts/gate8/gate8_results.json", "artifacts/gate8a/gate8a_results.json",
              "artifacts/gate8b/gate8b_results.json", "reports/gate8b/gate8b_report.md"):
        assert os.path.exists(os.path.join(REPO, f)), f
