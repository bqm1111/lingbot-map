"""The OccAny comparison reuses the frozen pipeline exactly and scores both methods with OccAny's own rules."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.occany_comparison import config as C
from tools.occany_comparison import scoring as SC
from tools.occany_comparison.analysis import bootstrap_delta, pooled_rates

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the mapper runs on CUDA")
FROZEN_SK_CLIP = os.path.join(C.DATA_ROOT, "..", "lingbot_baseline_reconstruction", "target_clips", "semantickitti", "08", "00004.npz")


@needs_cuda
@pytest.mark.skipif(not os.path.exists(FROZEN_SK_CLIP), reason="frozen target clip not on this machine")
def test_last_slot_geometry_is_bit_identical_to_the_frozen_function():
    from core.datasets import grids as G
    from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry
    from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels
    from core.datasets import sources as S
    dev = torch.device("cuda:0")
    local = LocalClip.load(FROZEN_SK_CLIP)
    (seg,) = S.segments("semantickitti", C.REPO_ROOT)
    T = seg.frames[4].T_cam_to_grid
    grid = G.PREDICTION_GRID["semantickitti"]
    S0, F0, _ = build_local_geometry(local, grid, T, dev)
    S1, F1 = hard_channels(build_mapper(WindowClip.from_local(local), dev), grid, T)
    assert torch.equal(S0, S1) and torch.equal(F0, F1) and int(S0.sum()) > 0


def test_pooling_is_a_one_voxel_max_dilation_of_the_unmasked_volume():
    occ = torch.zeros(9, 9, 9, dtype=torch.bool); occ[4, 4, 4] = True; occ[0, 0, 0] = True
    p = SC.pool(occ)
    assert int(p.sum()) == 27 + 8 and bool(p[3:6, 3:6, 3:6].all()) and bool(p[0:2, 0:2, 0:2].all())


@pytest.mark.parametrize("dataset", C.DATASETS)
def test_fast_counts_equal_occany_ssc_metrics_with_ignore(dataset):
    rng = np.random.default_rng(0)
    run = C.OCCANY[dataset]
    label = rng.integers(0, run.n_classes, size=(20, 20, 8)).astype(np.uint8)
    label[rng.random(label.shape) < 0.3] = C.IGNORE
    occ = rng.random(label.shape) < 0.4
    keep = torch.from_numpy(label != C.IGNORE); gt = keep & torch.from_numpy(label != run.empty_class)
    assert SC.fast_counts(torch.from_numpy(occ), gt, keep) == SC.official_counts(dataset, occ, label)


def test_protocol_windows():
    fwd, cau = C.PROTOCOLS["forward"], C.PROTOCOLS["causal"]
    assert fwd.offsets[fwd.anchor_slot] == 0 and cau.offsets[cau.anchor_slot] == 0
    assert fwd.offsets == (0, 2, 4, 6, 8) and cau.offsets == (-4, -3, -2, -1, 0)


def test_bootstrap_with_unit_weights_returns_the_point_estimate():
    rng = np.random.default_rng(1)
    ours = rng.integers(10, 100, size=(30, 3, 3)).astype(np.int64)      # [samples, seeds, 3]
    ref = rng.integers(10, 100, size=(30, 3)).astype(np.int64)
    blocks = np.repeat(np.arange(10), 3)
    point = np.mean([pooled_rates(ours[:, k])["iou"] for k in range(3)]) - pooled_rates(ref)["iou"]
    r = bootstrap_delta(ours, ref, blocks, n_boot=200, seed=0)
    assert abs(r["point"] - point) < 1e-12 and r["lo"] <= point <= r["hi"]
