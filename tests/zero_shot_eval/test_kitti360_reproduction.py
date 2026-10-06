"""The grid-generic prediction code reproduces the frozen KITTI-360 pipeline (evidence, provenance, B0, B3 labels) exactly."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path")


@needs_cuda
def test_grid_generic_code_reproduces_gate16_on_a_kitti360_anchor():
    from core.datasets.kitti360 import SSCBENCH_KITTI360_GRID
    from tools.baseline_reconstruction.current.geometry import LocalClip
    from tools.delayed_semantic import config as PC
    from tools.delayed_semantic.gate16 import config as G16, evaluate as G16EV
    from tools.delayed_semantic.model import voxel_centres as frozen_centres
    from tools.delayed_semantic.stream import DriveStream
    from tools.semantic_completion.evaluation import baseline_nearest as B0
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.evidence.build import camera_to_grid
    from tools.semantic_completion.evidence.lift import grid_world_keys, load_evidence
    from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk
    from tools.zero_shot_eval.prediction.evidence import lift_evidence
    from tools.zero_shot_eval.prediction.grid import GridSpec, voxel_centres, world_keys
    from tools.zero_shot_eval.prediction.semantics import b3_features, nearest_evidence
    P = PC.Paths(); drive, t = PC.DEV_DRIVE, 4
    record = f"{G16.PREDICTIONS}/{drive}/{t:05d}.npz"
    if not os.path.exists(record):
        pytest.skip("Gate 16 prediction records not present")
    dev = torch.device("cuda:0"); grid = GridSpec.of(SSCBENCH_KITTI360_GRID); T = camera_to_grid()
    assert torch.equal(world_keys(grid, np.linalg.inv(T), dev), grid_world_keys(np.linalg.inv(T), dev))
    flat = torch.arange(0, grid.n_voxels, 997, device=dev); assert torch.equal(voxel_centres(grid, flat), frozen_centres(flat))
    clip = LocalClip.load(P.g15.clip_path(drive, t)); S, F_ = load_hard_map(drive, t, dev)
    ev = lift_evidence(clip, [f.token_path for f in DriveStream(drive).history(t).frames()], S, T, grid, dev)
    ref = load_evidence(P.g15.evidence_path(drive, t), dev)
    assert torch.equal(ev.flat, ref.flat_index) and torch.equal(ev.count, ref.count) and float((ev.feature - ref.feature).abs().max()) < 1e-3
    with np.load(P.provenance_path(drive, t)) as z:
        assert np.array_equal(ev.frame_bits.cpu().numpy(), z["frame_bits"])
    E = ev.dense(grid); n1, d1 = nearest_evidence(E, grid); n0, d0 = B0.nearest_evidence(E)
    assert np.array_equal(n1, n0) and np.array_equal(d1, d0)
    net = load_frozen_geometry(0, dev); occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]; tr = trunk[0].reshape(24, -1)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    with np.load(P.g15.eval_mask_path(t)) as z:
        gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
    query = ((O | ((gt >= 1) & (gt <= 18))) & ~E).nonzero(as_tuple=True)[0]
    evd = ev.scorer_inputs(tr, voxel_centres(grid, ev.flat)); evd["feature"] = ref.feature
    f = b3_features(G16EV.load_head(G16.head_path("B3", 0), dev), grid, query, evd, tr, torch.stack([F_, S, O]).reshape(3, *grid.dims))
    lab = torch.zeros(grid.n_voxels, dtype=torch.long, device=dev); lab[query] = TR.classify(f, text); lab[ref.flat_index] = TR.classify(ref.feature, text)
    with np.load(record) as z:
        oi = O.nonzero(as_tuple=True)[0]
        assert np.array_equal(np.unpackbits(z["O"])[:grid.n_voxels].astype(bool), O.cpu().numpy())
        assert np.array_equal(lab[oi].to(torch.uint8).cpu().numpy(), z["label_B3_s0"])
