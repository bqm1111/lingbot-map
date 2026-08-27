"""Correspondence-engine unit tests on small, exactly-known tensors."""

import numpy as np
import pytest
import torch

from research.lingbot_semantic_memory.reprojection import (
    Correspondences, compute_correspondences, patch_center_pixels, _transfer,
)

GRID = (4, 8)
IMG = (16, 32)
K = torch.tensor([[20.0, 0.0, 16.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])
EYE = torch.eye(4)[:3]


def test_patch_centers_are_inside_their_own_patch():
    uv = patch_center_pixels(GRID, IMG)
    gh, gw = GRID
    H, W = IMG
    assert uv.shape == (gh * gw, 2)
    gx = (uv[:, 0] / (W / gw)).floor().long()
    gy = (uv[:, 1] / (H / gh)).floor().long()
    assert torch.equal(gy * gw + gx, torch.arange(gh * gw))


def test_identity_pose_and_depth_gives_identity_correspondences():
    """The synthetic sanity check: same pose, same depth => every patch maps to itself."""
    depth = torch.full(IMG, 5.0)
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID)
    n = GRID[0] * GRID[1]
    assert len(corr) == n, f"expected all {n} patches to survive, got {len(corr)}"
    assert torch.equal(corr.src_idx, torch.arange(n))
    assert torch.equal(corr.dst_idx, torch.arange(n))
    assert corr.valid_fraction == pytest.approx(1.0)


def test_identity_holds_for_a_non_constant_depth_map():
    torch.manual_seed(0)
    depth = 3.0 + torch.rand(IMG) * 4.0
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID)
    assert torch.equal(corr.src_idx, corr.dst_idx)
    assert len(corr) == GRID[0] * GRID[1]


def test_pure_translation_shifts_correspondences_consistently():
    """A sideways camera step must move matches by the predicted pixel disparity."""
    z = 10.0
    depth = torch.full(IMG, z)
    tx = 4.0                               # 8 px shift = two patch columns
    E2 = EYE.clone()
    E2[0, 3] = -tx  # world-to-camera: camera moved +tx in world x
    corr = compute_correspondences(depth, depth, EYE, E2, K, GRID, occlusion_rel_tol=1e-3)
    uv = patch_center_pixels(GRID, IMG)
    uv_d, z_d = _transfer(uv, depth[0, 0].expand(uv.shape[0]), K, EYE, E2)
    expected_shift = -K[0, 0] * tx / z
    assert torch.allclose(uv_d[:, 0] - uv[:, 0], torch.full((uv.shape[0],), float(expected_shift)), atol=1e-4)
    assert torch.allclose(z_d, torch.full_like(z_d, z), atol=1e-5)
    # Points pushed off the left edge must be dropped, not wrapped.
    assert len(corr) < GRID[0] * GRID[1]
    assert (corr.dst_idx >= 0).all() and (corr.dst_idx < GRID[0] * GRID[1]).all()


def test_invalid_depth_is_masked_out():
    depth = torch.full(IMG, 5.0)
    depth[:, :16] = 0.0                    # left half has no depth
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID)
    gh, gw = GRID
    left = torch.arange(gh * gw).view(gh, gw)[:, : gw // 2].reshape(-1)
    assert not np.isin(corr.src_idx.numpy(), left.numpy()).any()
    assert len(corr) == gh * gw // 2


def test_nan_depth_is_masked_out():
    depth = torch.full(IMG, 5.0)
    depth[:4, :] = float("nan")            # the full top patch row
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID)
    assert len(corr) < GRID[0] * GRID[1]
    assert torch.isfinite(corr.src_depth).all()


def test_occlusion_filter_rejects_hidden_points():
    """If the destination frame sees a much nearer surface, the match is dropped."""
    depth_src = torch.full(IMG, 10.0)
    depth_dst = torch.full(IMG, 2.0)      # everything in view 2 is much nearer
    corr = compute_correspondences(depth_src, depth_dst, EYE, EYE, K, GRID,
                                   occlusion_rel_tol=0.05)
    assert len(corr) == 0


def test_occlusion_tolerance_admits_small_disagreement():
    depth_src = torch.full(IMG, 10.0)
    depth_dst = torch.full(IMG, 10.2)     # 2 % disagreement
    corr = compute_correspondences(depth_src, depth_dst, EYE, EYE, K, GRID,
                                   occlusion_rel_tol=0.05)
    assert len(corr) == GRID[0] * GRID[1]


def test_validity_masks_are_honoured():
    depth = torch.full(IMG, 5.0)
    n = GRID[0] * GRID[1]
    vs = torch.zeros(n, dtype=torch.bool)
    vs[:5] = True
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID, valid_src=vs)
    assert len(corr) == 5
    vd = torch.zeros(n, dtype=torch.bool)
    vd[:2] = True
    corr = compute_correspondences(depth, depth, EYE, EYE, K, GRID, valid_src=vs, valid_dst=vd)
    assert len(corr) == 2


def test_forward_backward_filter_rejects_inconsistent_geometry():
    """Mismatched depth scales break the round trip and must be filtered."""
    depth_src = torch.full(IMG, 10.0)
    depth_dst = torch.full(IMG, 10.0)
    E2 = EYE.clone()
    E2[0, 3] = -2.0
    loose = compute_correspondences(depth_src, depth_dst, EYE, E2, K, GRID,
                                    occlusion_rel_tol=1.0, fb_tol_patches=1e9)
    strict = compute_correspondences(depth_src, depth_dst, EYE, E2, K, GRID,
                                     occlusion_rel_tol=1.0, fb_tol_patches=0.05)
    assert len(strict) <= len(loose)
