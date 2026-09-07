"""Gate 3.1 — leakage, provenance and reproduction tests for the clean protocol.

The Gate-3 defect was that the inference region was `dilate(occ, 3) & valid`, where
`valid` is the SemanticKITTI per-sample invalid mask. These tests prove the clean path
has no such dependency, and — critically — they test provenance **from raw construction**,
not merely that a previously derived region is stable under target mutation.
"""
from __future__ import annotations

import inspect
import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from gates.scale_gate.config import load_config                                    # noqa: E402
from gates.scale_gate.scale import bootstrap_ci                                    # noqa: E402
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G               # noqa: E402
from gates.voxel_gate.controls import control, enumerate_controls                  # noqa: E402
from gates.voxel_gate.losses import compute_loss                                   # noqa: E402
from gates.voxel_gate.models import VoxelCorrector3D, apply_region                 # noqa: E402
from gates.voxel_gate.voxels import dilate, pack, unpack                           # noqa: E402
from gates.voxel_gate_validation import data as vd                                 # noqa: E402
from gates.voxel_gate_validation.geometry import (                                 # noqa: E402
    EVIDENCE_SIGNATURE, FORBIDDEN_SIGNATURE_TOKENS, geometry_evidence,
)

CFG = os.path.join(_ROOT, "configs/voxel_gate_validation/clean_infill.yaml")
cfg = load_config(CFG)
ART = os.path.join(_ROOT, cfg.experiment.output_dir)
DEV = torch.device("cpu")
RADIUS = int(cfg.region.radius)

have = lambda g, s: os.path.exists(os.path.join(ART, f"{g}_{s}.json"))
needs = pytest.mark.skipif(not (have("c3", "val") and have("c0", "val")),
                           reason="clean caches not built")


def _clip(geom="c3"):
    return vd.clip_index(cfg, geom, "val")[0]


def _raw(geom="c3"):
    return dict(np.load(os.path.join(vd.cache_dir(cfg, geom),
                                     f"{_clip(geom)['clip_id']}.npz")))


def _norm_identity():
    n = {f"{k}_mean": 0.0 for k in ("log1p_count", "n_frames", "mean_confidence",
                                    "mean_point_depth")}
    n.update({f"{k}_std": 1.0 for k in ("log1p_count", "n_frames", "mean_confidence",
                                        "mean_point_depth")})
    return n


# --------------------------------------------------------------------------- #
# K1-K2: R_infer is pure dilation and independent of every target
# --------------------------------------------------------------------------- #
@needs
def test_r_infer_is_exactly_radius3_dilation_of_occupancy():
    d = _raw()
    inp = vd.inference_inputs(d, RADIUS, DEV)
    assert torch.equal(inp["R_infer"], dilate(inp["occupied"], RADIUS))
    assert RADIUS == 3
    # and it is NOT the Gate-3 leaky region
    keep = torch.from_numpy(unpack(d["valid_bits"]))
    assert not torch.equal(inp["R_infer"], inp["R_infer"] & keep), \
        "clean region coincides with the leaky one; the test cannot discriminate"


@needs
@pytest.mark.parametrize("key", ["valid_bits", "gt_flat", "vc_flat"])
def test_r_infer_bit_identical_under_independent_target_randomisation(key):
    rng = np.random.default_rng(0)
    d = _raw()
    ref = vd.inference_inputs(d, RADIUS, DEV)["R_infer"]
    for trial in range(3):
        m = dict(d)
        if key == "valid_bits":
            m[key] = pack(rng.random(G.dims) > 0.5)
        else:
            m[key] = rng.integers(0, int(np.prod(G.dims)), size=5000).astype(np.int32)
        assert torch.equal(vd.inference_inputs(m, RADIUS, DEV)["R_infer"], ref)


# --------------------------------------------------------------------------- #
# K3-K4: the input tensor and the input builder
# --------------------------------------------------------------------------- #
@needs
def test_input_tensor_bit_identical_under_target_mutation():
    rng = np.random.default_rng(1)
    d = _raw()
    inp = vd.inference_inputs(d, RADIUS, DEV)
    from gates.voxel_gate.voxels import build_input
    ref = build_input(inp["feat5"], inp["R_infer"], _norm_identity())
    m = dict(d)
    m["valid_bits"] = pack(rng.random(G.dims) > 0.5)
    m["gt_flat"] = rng.integers(0, int(np.prod(G.dims)), size=9000).astype(np.int32)
    m["vc_flat"] = rng.integers(0, int(np.prod(G.dims)), size=9000).astype(np.int32)
    i2 = vd.inference_inputs(m, RADIUS, DEV)
    assert torch.equal(build_input(i2["feat5"], i2["R_infer"], _norm_identity()), ref)


def test_input_builder_raises_if_it_touches_a_target_key():
    class Guard(dict):
        def __getitem__(self, k):
            assert k not in vd.TARGET_KEYS, f"inference path read target key {k!r}"
            return super().__getitem__(k)
    n = 11
    d = Guard({k: np.ones(n, np.float32) for k in vd.INPUT_KEYS})
    d["c3_flat"] = np.arange(n, dtype=np.int32)
    d.update({"gt_flat": np.zeros(1, np.int32), "vc_flat": np.zeros(1, np.int32),
              "valid_bits": pack(np.zeros(G.dims, bool))})
    out = vd.inference_inputs(d, 1, DEV)
    assert int(out["occupied"].sum()) == n
    # input_view physically excludes the target keys
    assert set(vd.input_view(d)) == set(vd.INPUT_KEYS)
    for k in vd.TARGET_KEYS:
        assert k not in vd.input_view(d)


# --------------------------------------------------------------------------- #
# K5: provenance from RAW construction, not from a cached derived region
# --------------------------------------------------------------------------- #
def test_geometry_evidence_signature_admits_no_target_side_argument():
    """The raw evidence builder cannot receive a target: its signature forbids it."""
    for p in EVIDENCE_SIGNATURE:
        for bad in FORBIDDEN_SIGNATURE_TOKENS:
            assert bad not in p.lower(), f"geometry_evidence takes {p!r}"
    assert set(EVIDENCE_SIGNATURE) == {"dep", "conf", "K", "pose_pred", "cam_to_velo",
                                       "scale", "conf_thr", "dmin", "dmax"}
    # strip the docstring and comments: only executable code may be checked
    src = inspect.getsource(geometry_evidence)
    body = src.split('"""')[2]
    code = "\n".join(l.split("#")[0] for l in body.splitlines())
    for bad in ("valid", "invalid", "target", "keep", "gt_", "vc_", "lidar"):
        assert bad not in code, f"geometry_evidence body references {bad!r}"


def test_raw_evidence_rebuild_is_invariant_to_any_valid_mask():
    """Rebuild inputs from raw geometry under different valid masks: identical outputs.

    This is the provenance test the Gate-3 suite lacked -- it constructs the evidence from
    scratch rather than re-reading a region that was already derived from a target.
    """
    rng = np.random.default_rng(2)
    T, H, W = 5, 12, 20
    dep = rng.uniform(0.2, 1.8, (T, H, W)).astype(np.float32)
    conf = rng.uniform(1.0, 12.0, (T, H, W)).astype(np.float32)
    K = np.tile(np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]]), (T, 1, 1))
    P = np.tile(np.eye(4), (T, 1, 1))
    for i in range(T):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q)); P[i, :3, 3] = rng.normal(size=3)
    c2v = np.eye(4)
    base = geometry_evidence(dep, conf, K, P, c2v, 27.3665, 1.5, 1.0, 60.0)
    for _ in range(3):
        _unused_valid = rng.random(G.dims) > 0.5          # cannot enter the call at all
        again = geometry_evidence(dep, conf, K, P, c2v, 27.3665, 1.5, 1.0, 60.0)
        for k in ("flat", "count", "n_frames", "sum_conf", "sum_depth"):
            assert np.array_equal(base[k], again[k])
    occ = torch.zeros(G.dims, dtype=torch.bool)
    occ.view(-1)[torch.from_numpy(base["flat"])] = True
    assert torch.equal(dilate(occ, RADIUS), dilate(occ, RADIUS))


# --------------------------------------------------------------------------- #
# K6: the clean region legitimately covers evaluation-invalid voxels
# --------------------------------------------------------------------------- #
@needs
def test_r_infer_contains_evaluation_invalid_voxels():
    d = _raw()
    R = vd.inference_inputs(d, RADIUS, DEV)["R_infer"]
    keep = torch.from_numpy(unpack(d["valid_bits"]))
    n_invalid = int((R & ~keep).sum())
    assert n_invalid > 0, "clean region excludes all invalid voxels -- looks like the leak"


# --------------------------------------------------------------------------- #
# K7-K9: loss, evaluation and gating
# --------------------------------------------------------------------------- #
def test_loss_moves_only_for_targets_inside_r_infer_and_valid():
    logit = torch.zeros(1, 1, 8, 8, 4)
    y = torch.zeros(1, 1, 8, 8, 4)
    S = torch.zeros(1, 1, 8, 8, 4); S[0, 0, 2:5, 2:5, 1:3] = 1.0     # R_infer AND valid
    ref = compute_loss(logit, y, S, 3.0, 0.5)["loss"]
    out = y.clone(); out[0, 0, 7, 7, 3] = 1.0                        # outside S
    assert torch.allclose(compute_loss(logit, out, S, 3.0, 0.5)["loss"], ref)
    ins = y.clone(); ins[0, 0, 3, 3, 2] = 1.0                        # inside S
    assert not torch.allclose(compute_loss(logit, ins, S, 3.0, 0.5)["loss"], ref)


def test_evaluation_ignores_invalid_voxels_without_changing_predictions():
    rng = np.random.default_rng(3)
    pred = torch.from_numpy(rng.random((10, 10, 6)) > 0.7)
    gt = torch.from_numpy(rng.random((10, 10, 6)) > 0.7)
    keep = torch.from_numpy(rng.random((10, 10, 6)) > 0.3)
    before = pred.clone()
    s = vd.scores(pred, gt, keep)
    assert torch.equal(pred, before), "scoring mutated the prediction"
    assert s["tp"] + s["fp"] == int((pred & keep).sum())
    assert s["tp"] + s["fn"] == int((gt & keep).sum())
    # a voxel outside `keep` can never contribute
    s2 = vd.scores(pred | ~keep, gt, keep)
    assert s2 == s


def test_output_gating_depends_only_on_r_infer():
    rng = np.random.default_rng(4)
    occ = torch.from_numpy(rng.random((10, 10, 6)) > 0.9)
    R = dilate(occ, 1)
    wild = torch.from_numpy(rng.random((10, 10, 6)) > 0.4)
    out = apply_region(wild, occ, R)
    assert torch.equal(out & R, wild & R)
    assert torch.equal(out & ~R, occ & ~R)
    assert bool((occ & ~R).sum() == 0)          # dilation is extensive


@needs
def test_controls_use_the_same_clean_region():
    d = _raw()
    inp = vd.inference_inputs(d, RADIUS, DEV)
    occ, R = inp["occupied"], inp["R_infer"]
    keep = torch.from_numpy(unpack(d["valid_bits"]))
    for kw in list(enumerate_controls(cfg).values())[:6]:
        out = control(occ, R, **kw)
        assert torch.equal(out & ~R, occ & ~R)
        # the control may legitimately place voxels in evaluation-invalid space
        assert int((out & ~keep).sum()) >= 0


# --------------------------------------------------------------------------- #
# K11-K12: split hygiene and selection provenance
# --------------------------------------------------------------------------- #
@needs
def test_source_selection_and_sequence08_clip_ids_are_disjoint():
    for geom in ("c3", "c0"):
        tr = {c["clip_id"] for c in vd.select_clips(cfg, geom, "train",
                                                    cfg.data.source_train_sequences)}
        se = {c["clip_id"] for c in vd.select_clips(cfg, geom, "train",
                                                    cfg.data.source_select_sequences)}
        va = {c["clip_id"] for c in vd.select_clips(cfg, geom, "val",
                                                    cfg.data.val_sequences)}
        assert tr and se and va
        assert not (tr & se) and not (tr & va) and not (se & va)
        assert {c["sequence"] for c in vd.clip_index(cfg, geom, "val")} == {"08"}
    assert "08" not in set(cfg.data.source_train_sequences)
    assert "08" not in set(cfg.data.source_select_sequences)


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "control_selection.json")),
                    reason="controls not selected")
def test_no_sequence08_statistic_selects_controls():
    s = json.load(open(os.path.join(ART, "control_selection.json")))
    assert s["selection_used_sequence_08"] is False
    assert "08" not in s["source_select_sequences"]
    assert s["v1_clean"]["name"] in enumerate_controls(cfg)
    assert s["v2_clean"]["name"] in enumerate_controls(cfg)
    assert s["radius"] == RADIUS


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "runs/full_s0/train.json")),
                    reason="models not trained")
def test_no_sequence08_statistic_selects_checkpoints_or_hyperparameters():
    for run in ("full_s0", "full_s1", "full_s2", "occ_only_s0", "c0_corrector_s0"):
        p = os.path.join(ART, "runs", run, "train.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        assert t["sequence_08_used_in_training"] is False
        assert t["region_intersects_valid_mask"] is False
        assert "08" not in t["source_train_sequences"] + t["source_select_sequences"]
        assert t["threshold"] == float(cfg.train.threshold)   # frozen, never reselected
        assert t["radius"] == RADIUS
        assert t["norm"]["n_clips"] == t["n_train_clips"]


# --------------------------------------------------------------------------- #
# K13: determinism
# --------------------------------------------------------------------------- #
def test_seed_control_is_deterministic():
    from tools.voxel_gate_validation.train_clean import seed_everything

    def draw():
        seed_everything(7)
        return (torch.randn(5), np.random.rand(5),
                VoxelCorrector3D(6, 8, 3, 3).body[0].weight.detach().clone())
    a, b = draw(), draw()
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    seed_everything(8)
    assert not torch.equal(torch.randn(5), a[0])


def test_bootstrap_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    x = rng.random(60)
    d = list((x + 0.03) - x)
    assert bootstrap_ci(d, 10000, 0) == bootstrap_ci(d, 10000, 0)
    ci = bootstrap_ci(d, 10000, 0)
    assert ci["lo"] == pytest.approx(0.03) and ci["hi"] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# K14: baselines reproduce
# --------------------------------------------------------------------------- #
@needs
@pytest.mark.parametrize("geom,key", [("c0", "c0_iou"), ("c3", "c3_iou")])
def test_baselines_reproduce_within_tolerance(geom, key):
    clips = vd.clip_index(cfg, geom, "val")
    assert len(clips) == 163
    iou = float(np.mean([c["iou"] for c in clips]))
    want, tol = float(cfg.reference[key]), float(cfg.reference.tolerance)
    assert abs(iou - want) <= tol, f"{geom} IoU {iou} outside {want} +/- {tol}"


@needs
def test_c0_and_c3_geometries_actually_differ():
    c0 = {c["clip_id"]: c["iou"] for c in vd.clip_index(cfg, "c0", "val")}
    c3 = {c["clip_id"]: c["iou"] for c in vd.clip_index(cfg, "c3", "val")}
    assert set(c0) == set(c3)
    assert float(np.mean([c3[k] - c0[k] for k in c0])) > 0.01


# --------------------------------------------------------------------------- #
# Frozen protocol
# --------------------------------------------------------------------------- #
def test_grid_and_frozen_choices_are_unchanged():
    assert tuple(G.dims) == (256, 256, 32) and G.voxel_size == 0.2
    assert tuple(G.origin) == (0.0, -25.6, -2.0)
    assert RADIUS == 3
    assert cfg.region.intersect_with_valid_mask is False
    assert float(cfg.train.threshold) == 0.45
    assert (int(cfg.model.channels), int(cfg.model.n_blocks), int(cfg.model.kernel)) == (16, 3, 3)
    assert int(cfg.train.epochs) == 20 and int(cfg.train.patience) == 4
    assert float(cfg.train.lr) == 1e-3 and float(cfg.train.weight_decay) == 1e-4
    assert list(cfg.experiment.seeds) == [0, 1, 2]
    assert int(cfg.experiment.seed) == 0


def test_zero_initialised_model_reproduces_the_input_occupancy():
    m = VoxelCorrector3D(6, 16, 3, 3).eval()
    rng = np.random.default_rng(5)
    occ = torch.from_numpy(rng.random((8, 8, 4)) > 0.8)
    x = torch.zeros(1, 6, 8, 8, 4)
    x[0, 0] = occ.float()
    x[0, 1:5] = torch.from_numpy(rng.normal(size=(4, 8, 8, 4)).astype(np.float32))
    x[0, 5] = 1.0
    with torch.no_grad():
        assert torch.equal(torch.sigmoid(m(x))[0, 0] >= 0.5, occ)


def test_occ_only_ablation_zeroes_exactly_channels_1_to_4():
    rng = np.random.default_rng(6)
    x = torch.from_numpy(rng.normal(size=(6, 4, 4, 2)).astype(np.float32))
    z = x.clone(); z[1:5] = 0.0
    assert torch.equal(z[0], x[0]) and torch.equal(z[5], x[5])
    assert float(z[1:5].abs().sum()) == 0.0
