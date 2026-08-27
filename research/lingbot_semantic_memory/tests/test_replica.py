"""Replica backend: pose convention, depth scale and stride, checked on real data.

The pose-convention test is the important one. Replica's ``traj.txt`` is consumed by
the popular Nice-SLAM loader with an axis flip (``c2w[:3, 1] *= -1; c2w[:3, 2] *= -1``)
that is correct for its OpenGL raycasting and **wrong** for OpenCV projection. Rather
than trust either convention, this warps ground-truth depth between two real frames and
asserts the identity convention wins by a wide margin.
"""

import os

import numpy as np
import pytest
import torch

from research.lingbot_semantic_memory.config import ChunkSpec, DataConfig
from research.lingbot_semantic_memory.dataset_adapter import (
    ReplicaBackend, build_splits, get_backend, invert_se3, scale_intrinsics,
)
from research.lingbot_semantic_memory.reprojection import _transfer, patch_center_pixels

ROOT = "data/Replica"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(REPO, ROOT, "office0")), reason="Replica unavailable")


def _cfg(**kw) -> DataConfig:
    base = dict(dataset="replica", root=ROOT, image_dirname="results",
                train_sequences=["room0", "room1"], val_sequences=["office4"],
                chunk_length=8, stride=2, val_chunks=1)
    base.update(kw)
    return DataConfig(**base)


def test_backend_reports_expected_geometry():
    b = get_backend(_cfg())
    assert isinstance(b, ReplicaBackend)
    assert b.has_labels is False
    assert b.sequence_length("office0") == 2000
    assert b.original_hw("office0") == (680, 1200)
    K = b.intrinsics("office0")
    assert K[0, 0] == pytest.approx(600.0) and K[1, 1] == pytest.approx(600.0)
    assert K[0, 2] == pytest.approx(599.5) and K[1, 2] == pytest.approx(339.5)
    assert b.depth_scale() == pytest.approx(6553.5)


def test_replica_has_no_labels():
    b = get_backend(_cfg())
    ch = ChunkSpec("office0", 0, 4, "val", 1)
    assert b.patch_labels(ch, (294, 518), (21, 37)) is None


def test_stride_selects_the_right_frames():
    ch = ChunkSpec("office0", 100, 8, "val", 3)
    assert list(ch.frame_indices) == [100, 103, 106, 109, 112, 115, 118, 121]
    paths = get_backend(_cfg()).image_paths(ch)
    assert os.path.basename(paths[0]) == "frame000100.jpg"
    assert os.path.basename(paths[-1]) == "frame000121.jpg"
    assert all(os.path.exists(p) for p in paths)
    assert "s3" in ch.name


def test_depth_is_metric_and_plausible_for_a_room():
    b = get_backend(_cfg())
    d = b.oracle_depth(ChunkSpec("office0", 0, 3, "val", 1), (294, 518))
    assert d.shape == (3, 294, 518)
    finite = d[d > 0]
    assert 0.1 < float(np.median(finite)) < 10.0, "indoor depth should be a few metres"
    assert float(finite.max()) < 30.0


def test_poses_are_valid_rigid_transforms():
    b = get_backend(_cfg())
    E = b.oracle_poses_w2c(ChunkSpec("office0", 0, 5, "val", 4))
    assert E.shape == (5, 3, 4)
    for e in E:
        R = e[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_invert_se3_round_trips():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((3, 3))
    R = np.linalg.qr(q)[0]
    R *= np.sign(np.linalg.det(R))
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = rng.standard_normal(3)
    w2c = invert_se3(c2w)
    back = np.eye(4)
    back[:3, :3] = w2c[:3, :3].T
    back[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
    assert np.allclose(back, c2w, atol=1e-9)


def _warp_error(flip: bool, i: int = 100, j: int = 116) -> float:
    """Median relative depth error when warping frame i into frame j."""
    b = get_backend(_cfg())
    traj = np.loadtxt(os.path.join(REPO, ROOT, "office0", "traj.txt")).reshape(-1, 4, 4)
    F = np.diag([1.0, -1.0, -1.0, 1.0])

    def w2c(k):
        c2w = traj[k] @ F if flip else traj[k]
        return torch.from_numpy(invert_se3(c2w)).float()

    d = b.oracle_depth(ChunkSpec("office0", i, j - i + 1, "val", 1), (680, 1200))
    di, dj = torch.from_numpy(d[0]), torch.from_numpy(d[-1])
    K = torch.from_numpy(b.intrinsics("office0")).float()
    uv = patch_center_pixels((34, 60), (680, 1200))
    z = di[uv[:, 1].long(), uv[:, 0].long()]
    ok = z > 0
    uvd, zd = _transfer(uv[ok], z[ok], K, w2c(i), w2c(j))
    inb = ((uvd[:, 0] >= 0) & (uvd[:, 0] < 1200) & (uvd[:, 1] >= 0) & (uvd[:, 1] < 680) & (zd > 0))
    zobs = dj[uvd[inb, 1].long().clamp(0, 679), uvd[inb, 0].long().clamp(0, 1199)]
    v = zobs > 0
    return float(((zd[inb][v] - zobs[v]).abs() / zobs[v]).median())


def test_traj_is_camera_to_world_in_opencv_axes_not_the_niceslam_flip():
    """The convention is established by measurement; regressing it must fail loudly."""
    identity_err = _warp_error(flip=False)
    flipped_err = _warp_error(flip=True)
    assert identity_err < 0.01, f"as-is c2w should warp near-perfectly, got {identity_err:.4f}"
    assert flipped_err > 10 * identity_err, (
        f"the Nice-SLAM y/z flip should be clearly worse here: "
        f"{flipped_err:.4f} vs {identity_err:.4f}")


def test_preprocessing_is_a_pure_resize_to_the_expected_grid():
    from research.lingbot_semantic_memory.dataset_adapter import assert_pure_resize
    cfg = _cfg()
    assert_pure_resize(cfg, (680, 1200), (294, 518))       # must not raise
    assert 294 // cfg.patch_size == 21 and 518 // cfg.patch_size == 37
    K = scale_intrinsics(get_backend(cfg).intrinsics("office0"), (680, 1200), (294, 518))
    assert K[0, 0] == pytest.approx(600.0 * 518 / 1200)
    assert K[1, 1] == pytest.approx(600.0 * 294 / 680)


def test_splits_are_scene_disjoint_and_deterministic():
    cfg = _cfg(train_sequences=["room0", "room1", "room2"], val_sequences=["office4"], val_chunks=3)
    tr1, va1 = build_splits(cfg, seed=0)
    tr2, va2 = build_splits(cfg, seed=0)
    assert [c.name for c in tr1] == [c.name for c in tr2]
    assert [c.name for c in va1] == [c.name for c in va2]
    assert set(c.sequence for c in tr1).isdisjoint(set(c.sequence for c in va1))
    assert len(va1) == 3
    for c in tr1 + va1:
        assert c.stride == cfg.stride
        assert c.start + c.length * c.stride <= 2000
