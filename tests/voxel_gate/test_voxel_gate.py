"""Gate-3 correctness tests: frozen protocol, no leakage, declared region semantics."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from gates.scale_gate.config import load_config                                    # noqa: E402
from prompted_lingbot.occupancy import (                                     # noqa: E402
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)
from prompted_lingbot.occ_datasets import SemanticKittiOccSpec               # noqa: E402
from gates.voxel_gate import data as vdata                                         # noqa: E402
from gates.voxel_gate.c3 import c3_points, scaled_relative_pose                    # noqa: E402
from gates.voxel_gate.controls import control, enumerate_controls                  # noqa: E402
from gates.voxel_gate.losses import compute_loss                                   # noqa: E402
from gates.voxel_gate.models import VoxelCorrector3D, apply_region, count_params   # noqa: E402
from gates.voxel_gate.voxels import (                                              # noqa: E402
    FEATURE_NAMES, FORBIDDEN_INPUTS, N_FEATURES, build_input, correction_region,
    dense_from_sparse, dilate, distance_bins, pack, unpack,
)
from gates.scale_gate.scale import bootstrap_ci                                    # noqa: E402

CFG = os.path.join(_ROOT, "configs/voxel_gate/visible_correction.yaml")
ART = os.path.join(_ROOT, "artifacts/voxel_gate")
DEV = torch.device("cpu")

cfg = load_config(CFG)
has_cache = os.path.isdir(os.path.join(ART, "cache_c3"))
has_index = os.path.exists(os.path.join(ART, "c3_val.json"))
needs_cache = pytest.mark.skipif(not (has_cache and has_index), reason="C3 cache not built")


# --------------------------------------------------------------------------- #
# Frozen protocol
# --------------------------------------------------------------------------- #
def test_grid_dimensions_and_voxel_size_are_unchanged():
    assert tuple(G.dims) == (256, 256, 32)
    assert G.voxel_size == 0.2
    assert tuple(G.origin) == (0.0, -25.6, -2.0)
    assert (G.empty_class, G.ignore_label) == (0, 255)


@needs_cache
def test_c3_reproduces_the_frozen_gate2_result():
    clips = vdata.clip_index(cfg, "val")
    iou = float(np.mean([c["iou"] for c in clips]))
    assert len(clips) == 163
    assert abs(iou - 0.0778) <= 5e-4, f"C3 IoU {iou} outside tolerance of 0.0778"


@needs_cache
def test_cached_scores_match_the_frozen_evaluator_exactly():
    """The fast sparse scorer must agree with binary_occupancy_scores voxel for voxel."""
    c = vdata.clip_index(cfg, "val")[0]
    s = vdata.sample(cfg, c["clip_id"], 3, None, DEV)
    fast = vdata.scores(s["occupied"], s["gt"], s["keep"])
    spec = SemanticKittiOccSpec.build(os.path.join(_ROOT, "data/kitti/dataset"), "08")
    target, valid = spec.target(c["anchor_frame"])
    ref = binary_occupancy_scores(s["occupied"].numpy(), target, G, valid=valid)
    for k in ("tp", "fp", "fn", "iou", "precision", "recall", "n_pred_occupied"):
        assert fast[k] == pytest.approx(ref[k]), k


# --------------------------------------------------------------------------- #
# C3 geometry: coupled scale, no r_shape
# --------------------------------------------------------------------------- #
def _poses(n=5, seed=0):
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(scale=4.0, size=3)
    return P


def test_c3_scales_depth_and_pose_translation_together():
    """Scaling by s must be a pure similarity of the fused cloud."""
    rng = np.random.default_rng(0)
    T, H, W = 5, 8, 12
    dep = rng.uniform(0.1, 2.0, (T, H, W)).astype(np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    K = np.tile(np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]]), (T, 1, 1))
    P = _poses(T)
    p1, *_ = c3_points(dep, conf, K, P, 1.0, 1.5, 0.0, 1e9)
    p2, *_ = c3_points(dep, conf, K, P, 3.5, 1.5, 0.0, 1e9)
    assert p1.shape == p2.shape and len(p1) > 0
    assert np.allclose(p2, 3.5 * p1, rtol=1e-9, atol=1e-9)


def test_rotations_are_never_scaled():
    P = _poses(seed=2)
    base = scaled_relative_pose(P, 0, 4, 1.0)[:3, :3]
    for s in (0.4, 9.0, 27.3665):
        assert np.allclose(scaled_relative_pose(P, 0, 4, s)[:3, :3], base, atol=1e-12)


def test_r_shape_is_absent_from_the_gate3_code_path():
    """The deployed path must contain no reference to the discarded spatial residual."""
    from gates import voxel_gate
    import gates.voxel_gate.c3, gates.voxel_gate.data, gates.voxel_gate.models, gates.voxel_gate.voxels
    for mod in (voxel_gate.c3, voxel_gate.data, voxel_gate.models, voxel_gate.voxels):
        src = open(mod.__file__).read()
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        code = code.split('"""')[0] + "".join(code.split('"""')[2::2])   # drop docstrings
        assert "r_shape" not in code, f"{mod.__name__} references r_shape in code"
    assert cfg.c3.use_r_shape is False


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
@needs_cache
def test_model_input_is_bit_identical_when_the_targets_are_mutated():
    c = vdata.clip_index(cfg, "val")[0]
    d = dict(np.load(os.path.join(ART, "cache_c3", f"{c['clip_id']}.npz")))
    norm = {f"{k}_mean": 0.0 for k in ("log1p_count", "n_frames", "mean_confidence",
                                       "mean_point_depth")}
    norm.update({f"{k}_std": 1.0 for k in ("log1p_count", "n_frames", "mean_confidence",
                                           "mean_point_depth")})
    def make(dd):
        inp = vdata.inputs(dd, DEV)
        R = correction_region(inp["occupied"], 3)
        return build_input(inp["feat5"], R, norm)
    a = make(d)
    d2 = dict(d)
    d2["gt_flat"] = np.zeros(0, np.int32)
    d2["vc_flat"] = np.zeros(0, np.int32)
    d2["valid_bits"] = pack(np.zeros(G.dims, bool))
    b = make(d2)
    assert torch.equal(a, b), "model input changed when LiDAR targets were mutated"


def test_input_channel_names_declare_no_forbidden_quantity():
    assert len(FEATURE_NAMES) == N_FEATURES == 6
    for name in FEATURE_NAMES:
        for bad in FORBIDDEN_INPUTS:
            assert bad not in name, f"input channel {name!r} names a forbidden quantity"


def test_inputs_reads_no_target_key():
    """vdata.inputs must raise if it ever touches a target key."""
    class Guard(dict):
        def __getitem__(self, k):
            assert k not in vdata.TARGET_KEYS, f"inputs() read target key {k!r}"
            return super().__getitem__(k)
    n = 7
    d = Guard({"c3_flat": np.arange(n, dtype=np.int32),
               "c3_count": np.ones(n, np.float32),
               "c3_n_frames": np.ones(n, np.float32),
               "c3_sum_conf": np.ones(n, np.float32),
               "c3_sum_depth": np.ones(n, np.float32),
               "gt_flat": np.zeros(1, np.int32), "vc_flat": np.zeros(1, np.int32),
               "valid_bits": pack(np.zeros(G.dims, bool))})
    out = vdata.inputs(d, DEV)
    assert int(out["occupied"].sum()) == n


@needs_cache
def test_correction_region_is_computable_without_any_target():
    """The region depends only on C3 occupancy, so it survives target deletion."""
    c = vdata.clip_index(cfg, "val")[0]
    d = dict(np.load(os.path.join(ART, "cache_c3", f"{c['clip_id']}.npz")))
    occ = vdata.inputs(d, DEV)["occupied"]
    d2 = dict(d); d2["gt_flat"] = np.zeros(0, np.int32); d2["vc_flat"] = np.zeros(0, np.int32)
    occ2 = vdata.inputs(d2, DEV)["occupied"]
    assert torch.equal(correction_region(occ, 3), correction_region(occ2, 3))


# --------------------------------------------------------------------------- #
# Correction region semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_region_contains_every_c3_occupied_voxel(radius):
    """Dilation is extensive, so "preserve C3 outside R" == "force empty outside R"."""
    rng = np.random.default_rng(0)
    occ = torch.from_numpy(rng.random((16, 16, 8)) > 0.97)
    R = correction_region(occ, radius)
    assert bool((occ & ~R).sum() == 0)


def test_apply_region_leaves_voxels_outside_the_region_untouched():
    rng = np.random.default_rng(1)
    occ = torch.from_numpy(rng.random((10, 10, 6)) > 0.9)
    R = correction_region(occ, 1)
    wild = torch.from_numpy(rng.random((10, 10, 6)) > 0.5)      # model wants everything
    out = apply_region(wild, occ, R)
    assert torch.equal(out & ~R, occ & ~R), "occupancy outside the region changed"
    assert torch.equal(out & R, wild & R)


def test_controls_respect_the_region_too():
    rng = np.random.default_rng(2)
    occ = torch.from_numpy(rng.random((12, 12, 8)) > 0.95)
    R = correction_region(occ, 1)
    for kw in enumerate_controls(cfg).values():
        out = control(occ, R, **kw)
        assert torch.equal(out & ~R, occ & ~R), f"{kw} changed voxels outside the region"


# --------------------------------------------------------------------------- #
# Model behaviour
# --------------------------------------------------------------------------- #
def test_zero_initialised_model_reproduces_v0():
    m = VoxelCorrector3D(6, 16, 3, 3).eval()
    rng = np.random.default_rng(3)
    occ = torch.from_numpy(rng.random((8, 8, 4)) > 0.8)
    x = torch.zeros(1, 6, 8, 8, 4)
    x[0, 0] = occ.float()
    x[0, 1:5] = torch.from_numpy(rng.normal(size=(4, 8, 8, 4)).astype(np.float32))
    x[0, 5] = 1.0
    with torch.no_grad():
        p = torch.sigmoid(m(x))[0, 0] >= 0.5
    assert torch.equal(p, occ), "zero-initialised corrector does not reproduce C3"


def test_model_can_both_add_and_remove_voxels():
    """The residual is unbounded in sign: forcing the head reproduces either behaviour."""
    rng = np.random.default_rng(4)
    occ = torch.from_numpy(rng.random((8, 8, 4)) > 0.8)
    x = torch.zeros(1, 6, 8, 8, 4); x[0, 0] = occ.float(); x[0, 5] = 1.0
    m = VoxelCorrector3D(6, 8, 3, 3).eval()
    with torch.no_grad():
        torch.nn.init.constant_(m.head.bias, +12.0)             # add everything
        assert bool((torch.sigmoid(m(x))[0, 0] >= 0.5).all())
        torch.nn.init.constant_(m.head.bias, -12.0)             # remove everything
        assert bool((torch.sigmoid(m(x))[0, 0] >= 0.5).sum() == 0)


def test_model_is_small():
    assert count_params(VoxelCorrector3D(6, 16, 3, 3)) < 50_000


def test_loss_is_restricted_to_the_supervision_mask():
    logit = torch.zeros(1, 1, 6, 6, 4, requires_grad=True)
    y = torch.zeros(1, 1, 6, 6, 4); y[0, 0, 0, 0, 0] = 1.0
    mask = torch.zeros(1, 1, 6, 6, 4); mask[0, 0, 3:, 3:, :] = 1.0   # excludes the positive
    a = compute_loss(logit, y, mask, 3.0, 0.5)["loss"]
    y2 = y.clone(); y2[0, 0, 0, 0, 0] = 0.0
    b = compute_loss(logit, y2, mask, 3.0, 0.5)["loss"]
    assert torch.allclose(a, b), "loss reacted to a target outside the supervision mask"


# --------------------------------------------------------------------------- #
# Split hygiene and selection provenance
# --------------------------------------------------------------------------- #
@needs_cache
def test_source_and_sequence08_clip_ids_are_disjoint():
    src = {c["clip_id"] for c in vdata.clip_index(cfg, "train")}
    val = {c["clip_id"] for c in vdata.clip_index(cfg, "val")}
    assert src and val and not (src & val)
    assert {c["sequence"] for c in vdata.clip_index(cfg, "val")} == {"08"}
    assert "08" not in set(cfg.data.source_train_sequences)
    assert "08" not in set(cfg.data.source_select_sequences)


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "region_selection.json")),
                    reason="selection not run")
def test_region_and_control_were_selected_on_source_only():
    sel = json.load(open(os.path.join(ART, "region_selection.json")))
    assert "08" not in sel["source_train_sequences"]
    assert "08" not in sel["source_select_sequences"]
    assert sel["selected_radius"] in list(cfg.region.candidate_radii)
    assert sel["selected_control"] in enumerate_controls(cfg)


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "runs/voxel_cnn3d/train.json")),
                    reason="model not trained")
def test_threshold_and_hyperparameters_were_not_selected_on_sequence_08():
    t = json.load(open(os.path.join(ART, "runs/voxel_cnn3d/train.json")))
    assert t["sequence_08_used_in_training"] is False
    assert "08" not in t["source_train_sequences"] + t["source_select_sequences"]
    assert t["selected_threshold"] in [float(x) for x in cfg.train.threshold_grid]
    assert t["norm"]["n_clips"] == t["n_train_clips"]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_bootstrap_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    a = rng.random(50); b = a + 0.02                      # a constant paired improvement
    d = list(b - a)
    c1 = bootstrap_ci(d, 10000, 0)
    c2 = bootstrap_ci(d, 10000, 0)
    assert c1 == c2, "bootstrap is not deterministic at a fixed seed"
    assert c1["lo"] == pytest.approx(0.02) and c1["hi"] == pytest.approx(0.02)
    assert bootstrap_ci(list(b) , 10000, 0)["mean"] != pytest.approx(c1["mean"])


def test_bootstrap_seed_changes_the_interval_but_not_the_mean():
    rng = np.random.default_rng(1)
    d = list(rng.normal(0.01, 0.05, 200))
    c0, c1 = bootstrap_ci(d, 10000, 0), bootstrap_ci(d, 10000, 1)
    assert c0["mean"] == pytest.approx(c1["mean"])
    assert (c0["lo"], c0["hi"]) != (c1["lo"], c1["hi"])


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(5)
    v = rng.random(G.dims) > 0.99
    assert np.array_equal(unpack(pack(v)), v)


def test_distance_bins_tile_the_grid_without_overlap():
    bins = distance_bins(((0, 10), (10, 20), (20, 40), (40, 80)))
    stack = np.stack(list(bins.values()))
    assert stack.sum(0).max() == 1
    assert int(stack.sum()) == int(np.prod(G.dims))


def test_dense_from_sparse_matches_the_frozen_voxeliser():
    rng = np.random.default_rng(6)
    pts = rng.uniform([1, -10, -1], [40, 10, 3], size=(5000, 3))
    from gates.voxel_gate.voxels import sparse_voxel_features
    sp = sparse_voxel_features(pts, np.zeros(5000, np.int8),
                               np.ones(5000, np.float32), np.ones(5000, np.float32))
    a = dense_from_sparse(sp, DEV)[0] > 0
    b = torch.from_numpy(occupancy_from_points(pts, G, 1))
    assert torch.equal(a, b)


def test_dilate_is_extensive_and_monotone():
    rng = np.random.default_rng(7)
    v = torch.from_numpy(rng.random((10, 10, 6)) > 0.95)
    d1, d2 = dilate(v, 1), dilate(v, 2)
    assert bool((v & ~d1).sum() == 0) and bool((d1 & ~d2).sum() == 0)
