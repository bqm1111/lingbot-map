"""Coordinate conventions: unprojection, projection and camera centres must agree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_sidecar.lingbot_features import (
    camera_center,
    patch_centre_pixels,
    patch_grid,
    pool_to_patch_grid,
    project_to_camera,
    unproject_depth,
)


def test_unproject_then_project_roundtrip(rig):
    H, W, K = rig["H"], rig["W"], rig["K"]
    E = rig["extrinsics"][1]
    depth = torch.full((H, W), rig["plane_z"])
    world = unproject_depth(depth, K, E)
    assert world.shape == (H, W, 3)

    uv, z = project_to_camera(world.reshape(-1, 3), K, E)
    v, u = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    assert torch.allclose(uv[:, 0], u.reshape(-1), atol=1e-3)
    assert torch.allclose(uv[:, 1], v.reshape(-1), atol=1e-3)
    assert torch.allclose(z, depth.reshape(-1), atol=1e-4)


def test_camera_centre_matches_construction(rig):
    for E, centre in zip(rig["extrinsics"], rig["centres"]):
        assert torch.allclose(camera_center(E), centre, atol=1e-5)


def test_two_views_of_a_plane_agree_in_world_space(rig):
    """A plane seen from two cameras must unproject to the same world points."""
    H, W, K = rig["H"], rig["W"], rig["K"]
    z = rig["plane_z"]
    worlds = []
    for E in rig["extrinsics"]:
        centre = camera_center(E)
        # Depth to a plane at world z = plane_z is plane_z - centre_z (cameras face +z).
        depth = torch.full((H, W), z - float(centre[2]))
        worlds.append(unproject_depth(depth, K, E))
    for wpts in worlds:
        assert torch.allclose(wpts[..., 2], torch.full((H, W), z), atol=1e-4)
    # The baseline (0.5) is an exact multiple of the pixel footprint at this depth
    # (4 / 40 = 0.1), so most samples of the second view must coincide with the first.
    a = worlds[0].reshape(-1, 3)
    b = worlds[1].reshape(-1, 3)
    dist = torch.cdist(b, a).min(dim=1).values
    assert float(dist.median()) < 1e-4
    # Sanity: the second camera really is displaced, so the clouds are not identical.
    assert float((b - a).abs().max()) > 0.4


def test_patch_grid_and_centres():
    assert patch_grid(294, 518, 14) == (21, 37)
    centres = patch_centre_pixels(28, 42, 14)
    assert centres.shape == (2 * 3, 2)
    assert torch.allclose(centres[0], torch.tensor([6.5, 6.5]))
    assert torch.allclose(centres[-1], torch.tensor([20.5, 34.5]))


def test_pool_to_patch_grid_averages():
    dense = torch.arange(2 * 28 * 42, dtype=torch.float32).reshape(2, 28, 42)
    pooled = pool_to_patch_grid(dense, 2, 3)
    assert pooled.shape == (6, 2)
    # Top-left token is the mean of the top-left 14x14 block of channel 0.
    assert pooled[0, 0] == pytest.approx(float(dense[0, :14, :14].mean()), rel=1e-5)


def test_unproject_rejects_bad_shape(rig):
    with pytest.raises(ValueError):
        unproject_depth(torch.zeros(3, 4, 5), rig["K"], rig["extrinsics"][0])
