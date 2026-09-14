"""The causal clip scale enters exactly once, through the mapper, and reaches depth and translation together."""
from __future__ import annotations
import os, sys
import numpy as np, pytest, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.occupancy import VoxelGrid
from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry, network_input

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="mapper runs on CUDA")


def _clip(depth_canonical: float, scale: float, shift_canonical: float = 0.0) -> LocalClip:
    H, W = 8, 8; f = 8.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    poses = np.tile(np.eye(4), (5, 1, 1)); poses[:, 0, 3] = shift_canonical     # every frame shares one canonical x offset
    return LocalClip(anchor_index=4, stream_indices=np.arange(5), keys=np.array([str(i) for i in range(5)]),
                     depth_canonical=np.full((5, H, W), depth_canonical, np.float32), conf=np.full((5, H, W), 10.0, np.float32),
                     pose_c2w_canonical=poses, K=np.tile(K, (5, 1, 1)), scale_m_per_unit=scale, log_s_frames=np.full(5, np.log(scale)))


@needs_cuda
def test_surface_lands_at_scaled_metric_depth():
    dev = torch.device("cuda:0")
    grid = VoxelGrid(dims=(8, 8, 60), voxel_size=0.2, origin=(-0.8, -0.8, 0.0), frame="cam", empty_class=0, ignore_label=255, name="t")
    S, F, stats = build_local_geometry(_clip(depth_canonical=20.0, scale=0.25), grid, np.eye(4), dev)   # metric depth 5 m
    occ_z = torch.nonzero(S.reshape(grid.dims))[:, 2].float() * 0.2
    assert stats["n_rays"] == 5 * 64 and occ_z.numel() > 0
    assert 4.5 <= occ_z.median().item() <= 5.5, "surface must sit at s * z_canonical = 5 m, not at 20 m or 1.25 m"
    free_z = torch.nonzero(F.reshape(grid.dims))[:, 2].float() * 0.2
    assert free_z.max().item() < 5.0 and free_z.min().item() >= 1.0        # carved before the surface, from 1 m


@needs_cuda
def test_translation_is_scaled_with_depth():
    dev = torch.device("cuda:0")
    grid = VoxelGrid(dims=(40, 8, 60), voxel_size=0.2, origin=(-4.0, -0.8, 0.0), frame="cam", empty_class=0, ignore_label=255, name="t")
    # anchor = identity by construction; a common canonical shift of all frames cancels in inv(P_4) P_q
    S_a, _, _ = build_local_geometry(_clip(20.0, 0.25, shift_canonical=0.0), grid, np.eye(4), dev)
    S_b, _, _ = build_local_geometry(_clip(20.0, 0.25, shift_canonical=8.0), grid, np.eye(4), dev)
    assert torch.equal(S_a, S_b)


def test_network_input_layout():
    S = torch.zeros(2 * 2 * 2, dtype=torch.bool); F = torch.ones_like(S); S[0] = True
    x = network_input(S, F, (2, 2, 2))
    assert tuple(x.shape) == (1, 3, 2, 2, 2) and x[0, 1].sum() == 0 and x[0, 0].sum() == 1 and x[0, 2].sum() == 8
