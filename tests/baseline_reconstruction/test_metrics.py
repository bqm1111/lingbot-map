"""Occupancy metric arithmetic and range binning."""
from __future__ import annotations
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.baseline_reconstruction.metrics import occupancy as M


def test_counts_and_ratios():
    c = M.OccupancyCounts(tp=6, fp=2, fn=4)
    assert abs(c.iou - 0.5) < 1e-12 and abs(c.precision - 0.75) < 1e-12 and abs(c.recall - 0.6) < 1e-12 and c.n_gt == 10
    assert (c + M.OccupancyCounts(1, 1, 1)).tp == 7
    assert M.OccupancyCounts(0, 0, 0).iou == 0.0


def test_counts_respect_mask():
    pred = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool); gt = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
    mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)          # the last voxel is outside the mask
    c = M.counts(pred, gt, mask)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)


def test_counts_by_range_totals_match():
    n = 1000; g = torch.Generator().manual_seed(0)
    pred, gt = torch.rand(n, generator=g) < 0.3, torch.rand(n, generator=g) < 0.3
    mask = torch.rand(n, generator=g) < 0.8; rb = torch.randint(0, 5, (n,), generator=g)
    per = M.counts_by_range(pred, gt, mask, rb, 5)
    tot = M.counts(pred, gt, mask)
    assert per[-1] == tot and sum((p.tp for p in per[:-1])) == tot.tp


def test_range_bins_by_camera_z():
    dims, vs, origin = (4, 1, 1), 10.0, (0.0, 0.0, 0.0)
    # grid x maps to camera z (velodyne-like frame: camera z = grid x)
    T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float64)
    bins = M.range_bins_for_grid(dims, vs, origin, T, edges_m=(5.0, 10.0, 20.0, 30.0))
    assert list(bins) == [1, 2, 3, 4]        # centres at 5, 15, 25, 35 m


def test_seed_statistics():
    s = M.seed_statistics([0.1, 0.2, 0.3])
    assert abs(s["mean"] - 0.2) < 1e-12 and abs(s["median"] - 0.2) < 1e-12 and s["n"] == 3 and s["sd"] > 0
