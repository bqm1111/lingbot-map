"""Synthetic tests for 3D observation tracks.

Two cameras look at a fronto-parallel plane with exactly known geometry, so the
correct voxel assignment, the correct rejection of an inconsistent depth, and the
correct scale-relative voxel size are all checkable in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_sidecar.config import TrackConfig
from semantic_sidecar.lingbot_features import camera_center
from semantic_sidecar.tracks import (
    FrameObservations,
    _empty_trackset,
    build_tracks_from_observations,
    frame_observations,
    reprojection_residual,
    voxel_size_for_scene,
)



def _plane_frame(rig, cam_idx: int, conf: float = 2.0):
    H, W, K = rig["H"], rig["W"], rig["K"]
    E = rig["extrinsics"][cam_idx]
    depth = torch.full((H, W), rig["plane_z"] - float(camera_center(E)[2]))
    conf_map = torch.full((H, W), conf)
    return depth, conf_map, K, E


def test_voxel_size_is_scale_relative():
    cfg = TrackConfig(voxel_token_scale=None, rel_voxel_size=0.02)
    assert voxel_size_for_scene(cfg, 4.0) == pytest.approx(0.08)
    assert voxel_size_for_scene(cfg, 40.0) == pytest.approx(0.8)
    # A configured metric size wins outright.
    assert voxel_size_for_scene(TrackConfig(metric_voxel_size=0.25), 40.0) == pytest.approx(0.25)


def test_voxel_size_in_token_footprints():
    """One token footprint at depth d is patch_size * d / focal; the voxel scales with it."""
    cfg = TrackConfig(voxel_token_scale=1.5)
    # focal 40, depth 4, patch 14 -> footprint 1.4 -> voxel 2.1
    assert voxel_size_for_scene(cfg, 4.0, focal=40.0, patch_size=14) == pytest.approx(2.1)
    # Doubling the focal halves the footprint and therefore the voxel.
    assert voxel_size_for_scene(cfg, 4.0, focal=80.0, patch_size=14) == pytest.approx(1.05)
    # Without a focal it falls back to the relative rule rather than guessing.
    assert voxel_size_for_scene(cfg, 4.0) == pytest.approx(0.08)


def test_frame_observations_filters_low_confidence(rig):
    cfg = TrackConfig(conf_threshold=1.3, min_observations=1)
    depth, conf, K, E = _plane_frame(rig, 0, conf=1.0)
    obs = frame_observations(depth, conf, K, E, 14, cfg, median_depth=4.0)
    assert obs.points.shape[0] == 0

    depth, conf, K, E = _plane_frame(rig, 0, conf=2.0)
    obs = frame_observations(depth, conf, K, E, 14, cfg, median_depth=4.0)
    h, w = rig["H"] // 14, rig["W"] // 14
    assert obs.points.shape[0] == h * w
    assert torch.allclose(obs.points[:, 2], torch.full((h * w,), rig["plane_z"]), atol=1e-4)
    # View direction points from the surface back at the camera, i.e. towards -z.
    assert bool((obs.view_dir[:, 2] < 0).all())


def test_two_views_of_the_same_surface_share_tracks(rig):
    cfg = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=0.1, max_observations=32)
    per_frame, depth_maps = [], {}
    for i in (0, 1):
        depth, conf, K, E = _plane_frame(rig, i)
        per_frame.append((i, frame_observations(depth, conf, K, E, 14, cfg, 4.0)))
        depth_maps[i] = (depth, K, E)

    tracks = build_tracks_from_observations(per_frame, voxel_size=0.08, cfg=cfg, depth_maps=depth_maps)
    assert tracks.num_tracks > 0
    counts = tracks.counts()
    # min_observations = 2 means every surviving track is genuinely multi-view.
    assert int(counts.min()) >= 2
    assert set(tracks.obs_frame.tolist()) == {0, 1}
    # CSR invariants.
    assert int(tracks.obs_ptr[-1]) == tracks.num_observations
    assert tracks.obs_ptr.numel() == tracks.num_tracks + 1


def test_inconsistent_depth_is_rejected(rig):
    """An observation whose frame's depth map contradicts the track centre is dropped.

    The observations themselves are identical in both runs; only the depth map used
    for the cross-view check differs, so this isolates the reprojection filter from
    the voxel assignment.
    """
    cfg = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=0.05, max_observations=32)
    per_frame, depth_ok, depth_broken = [], {}, {}
    for i in (0, 1):
        depth, conf, K, E = _plane_frame(rig, i)
        per_frame.append((i, frame_observations(depth, conf, K, E, 14, cfg, 4.0)))
        depth_ok[i] = (depth, K, E)
        # Frame 1 claims the surface is 40 % further away than the track centre.
        depth_broken[i] = (depth * (1.4 if i == 1 else 1.0), K, E)

    t_good = build_tracks_from_observations(per_frame, 0.08, cfg, depth_ok)
    t_bad = build_tracks_from_observations(per_frame, 0.08, cfg, depth_broken)
    assert t_good.num_tracks > 0
    assert set(t_good.obs_frame.tolist()) == {0, 1}
    # Frame 1's observations fail the consistency check, so no track keeps two views.
    assert t_bad.num_tracks == 0

    # Relaxing the tolerance past the disagreement lets them back in, proving the
    # rejection came from the tolerance and not from some other filter.
    loose = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=1.0, max_observations=32)
    t_loose = build_tracks_from_observations(per_frame, 0.08, loose, depth_broken)
    assert t_loose.num_tracks == t_good.num_tracks


def test_multiple_observations_are_preserved_and_capped(rig):
    cfg = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=1e9, max_observations=3)
    per_frame = []
    for i in range(6):
        depth, conf, K, E = _plane_frame(rig, 0)
        per_frame.append((i, frame_observations(depth, conf, K, E, 14, cfg, 4.0)))
    tracks = build_tracks_from_observations(per_frame, 0.08, cfg, depth_maps=None)
    counts = tracks.counts()
    assert int(counts.max()) <= 3
    assert int(counts.min()) >= 2


def test_cap_is_deterministic(rig):
    cfg = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=1e9, max_observations=3, seed=7)
    per_frame = []
    for i in range(6):
        depth, conf, K, E = _plane_frame(rig, 0)
        per_frame.append((i, frame_observations(depth, conf, K, E, 14, cfg, 4.0)))
    a = build_tracks_from_observations(per_frame, 0.08, cfg, depth_maps=None)
    b = build_tracks_from_observations(per_frame, 0.08, cfg, depth_maps=None)
    assert torch.equal(a.obs_frame, b.obs_frame)
    assert torch.equal(a.obs_token, b.obs_token)


def test_empty_and_single_observation_inputs():
    cfg = TrackConfig(min_observations=2)
    empty = build_tracks_from_observations([], 0.1, cfg)
    assert empty.num_tracks == 0 and empty.num_observations == 0
    assert int(empty.obs_ptr[-1]) == 0

    one = FrameObservations(
        token_index=torch.tensor([0]),
        points=torch.zeros(1, 3),
        depth=torch.tensor([1.0]),
        conf=torch.tensor([2.0]),
        view_dir=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    tracks = build_tracks_from_observations([(0, one)], 0.1, cfg)
    assert tracks.num_tracks == 0  # a single view cannot satisfy min_observations = 2

    cfg1 = TrackConfig(min_observations=1)
    tracks1 = build_tracks_from_observations([(0, one)], 0.1, cfg1)
    assert tracks1.num_tracks == 1 and tracks1.num_observations == 1


def test_reprojection_residual_detects_wrong_depth(rig):
    H, W, K = rig["H"], rig["W"], rig["K"]
    E = rig["extrinsics"][0]
    depth = torch.full((H, W), 4.0)
    centres = torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 8.0]])
    res = reprojection_residual(centres, K, E, depth)
    assert float(res[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(res[1]) == pytest.approx(0.5, abs=1e-3)
    # A point behind the camera is unusable, not silently accepted.
    behind = reprojection_residual(torch.tensor([[0.0, 0.0, -1.0]]), K, E, depth)
    assert np.isinf(float(behind[0]))


def test_trackset_roundtrip(tmp_path, rig):
    cfg = TrackConfig(conf_threshold=1.3, min_observations=2, reproj_tol=1e9, max_observations=8)
    per_frame = []
    for i in (0, 1):
        depth, conf, K, E = _plane_frame(rig, i)
        per_frame.append((i, frame_observations(depth, conf, K, E, 14, cfg, 4.0)))
    tracks = build_tracks_from_observations(per_frame, 0.08, cfg)
    tracks.save(str(tmp_path), "scene")

    from semantic_sidecar.tracks import TrackSet, tracks_are_cached

    assert tracks_are_cached(str(tmp_path), "scene")
    loaded = TrackSet.load(str(tmp_path), "scene")
    assert loaded.num_tracks == tracks.num_tracks
    assert torch.equal(loaded.obs_ptr, tracks.obs_ptr)
    assert torch.equal(loaded.obs_frame, tracks.obs_frame)
    assert torch.allclose(loaded.centres, tracks.centres)
