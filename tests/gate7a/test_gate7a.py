"""Gate 7A — completion reachability and oracle-envelope diagnosis.

The tests are grouped by the claim they defend, not by the module they exercise:

* **metric distance** -- indices are never metres, and the two grid resolutions are
  handled separately and correctly;
* **construction identities** -- ``r = 0`` is the base, the oracle adds only true
  positives and never a false positive, morphology is a superset, recall and IoU are
  monotone;
* **semantic transport** -- existing voxels keep their semantics, added voxels take the
  declared nearest source, exact ties are averaged;
* **provenance** -- the Gate-6 predictions are byte-for-byte what they were, no released
  dataset directory is written, and nothing here trains anything.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gates.gate6 import grids as G6G, vocab                                     # noqa: E402
from gates.gate7a import config as C, distance as D, envelopes as E, frustum as FR, \
    pipeline as P7, stats as S, transport as TR                           # noqa: E402

ART = os.path.join(REPO, "artifacts", "gate7a")
G6ART = os.path.join(REPO, "artifacts", "gate6")


def _json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Metric distance, and the two grid resolutions
# --------------------------------------------------------------------------- #
def test_distance_is_metres_not_voxel_indices():
    """Two voxels three apart are 0.6 m apart at 0.2 m and 1.2 m apart at 0.4 m."""
    occ = np.zeros((8, 8, 8), bool)
    occ[1, 1, 1] = True
    d, d2 = D.edt(occ)
    assert d2[4, 1, 1] == 9 and d[4, 1, 1] == pytest.approx(3.0)
    assert D.metres(d, 0.2)[4, 1, 1] == pytest.approx(0.6)
    assert D.metres(d, 0.4)[4, 1, 1] == pytest.approx(1.2)
    # the same lattice distance is inside 0.8 m only on the fine grid
    assert D.within_radius(np.array([9]), 0.8, 0.2)[0]
    assert not D.within_radius(np.array([9]), 0.8, 0.4)[0]


def test_exact_euclidean_not_chebyshev_or_manhattan():
    occ = np.zeros((8, 8, 8), bool)
    occ[1, 1, 1] = True
    d, d2 = D.edt(occ)
    assert d2[2, 2, 2] == 3                     # sqrt(3), not 1 and not 3
    assert d[2, 2, 2] == pytest.approx(np.sqrt(3.0))
    assert d2[3, 1, 1] == 4 and d2[2, 3, 1] == 5


@pytest.mark.parametrize("vs,r,d2_in,d2_out", [(0.2, 0.4, 4, 5), (0.2, 1.2, 36, 37),
                                               (0.4, 0.4, 1, 2), (0.4, 2.0, 25, 26),
                                               (0.2, 4.0, 400, 401)])
def test_radius_boundary_is_exact_despite_binary_floating_point(vs, r, d2_in, d2_out):
    """``1.2 / 0.2`` is ``5.999999999999999``; a naive test would drop the boundary shell."""
    assert D.within_radius(np.array([d2_in]), r, vs)[0]
    assert not D.within_radius(np.array([d2_out]), r, vs)[0]


def test_empty_base_yields_infinite_distance_and_no_membership():
    occ = np.zeros((4, 4, 4), bool)
    d, d2 = D.edt(occ)
    assert np.isinf(d).all() and (d2 == -1).all()
    assert not D.within_radius(d2.reshape(-1), 4.0, 0.2).any()


def test_shells_partition_the_ball_and_carry_exact_distances():
    sh = D.shells(25)
    seen = set()
    for d2, offs in sh.items():
        assert ((offs ** 2).sum(axis=1) == d2).all()
        for o in map(tuple, offs):
            assert o not in seen
            seen.add(o)
    assert (0, 0, 0) not in seen
    assert len(seen) == sum(len(o) for o in sh.values())


def test_quantiles_from_histogram_are_monotone_and_bin_bounded():
    h = D.histogram(np.array([0.01, 0.3, 0.3, 5.0]), 0.05, 80.0)
    q = D.quantiles_from_histogram(h, (0.25, 0.5, 0.75, 0.99), 0.05)
    vals = [q[k] for k in ("p25", "p50", "p75", "p99")]
    assert vals == sorted(vals)
    assert q["p25"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Nearest-neighbour indices and tie handling on synthetic volumes
# --------------------------------------------------------------------------- #
def test_nearest_source_indices_on_a_synthetic_volume():
    occ = np.zeros((9, 9, 9), bool)
    occ[1, 4, 4] = True
    occ[7, 4, 4] = True
    d, d2, src = D.edt_with_source(occ)
    a = np.ravel_multi_index((1, 4, 4), occ.shape)
    b = np.ravel_multi_index((7, 4, 4), occ.shape)
    assert src[2, 4, 4] == a and src[6, 4, 4] == b
    assert d2[2, 4, 4] == 1 and d2[6, 4, 4] == 1


def test_exact_distance_ties_are_averaged_not_arbitrarily_broken():
    dims = (9, 5, 5)
    occ = np.zeros(dims, bool)
    occ[1, 2, 2] = True
    occ[7, 2, 2] = True
    d, d2 = D.edt(occ)
    src = np.flatnonzero(occ.reshape(-1))
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mid = np.ravel_multi_index((4, 2, 2), dims)
    dst = np.array([mid])
    got = TR.propagate_probs(dims, src, probs, dst, d2.reshape(-1)[dst], "cpu")
    assert got[0].tolist() == pytest.approx([0.5, 0.5])
    st = TR.propagate_stats(dims, src, probs, dst, d2.reshape(-1)[dst], "cpu")
    assert st["n_tied"][0] == 2 and st["assigned"][0]


def test_propagate_stats_agrees_with_the_full_vector_path():
    rng = np.random.default_rng(0)
    dims = (12, 12, 12)
    occ = np.zeros(dims, bool)
    occ.reshape(-1)[rng.choice(12 ** 3, 40, replace=False)] = True
    d, d2 = D.edt(occ)
    src = np.flatnonzero(occ.reshape(-1))
    probs = torch.tensor(rng.random((len(src), 5)), dtype=torch.float32)
    probs = probs / probs.sum(1, keepdim=True)
    d2f = d2.reshape(-1)
    dst = np.flatnonzero((d2f > 0) & (d2f <= 9))
    full = TR.propagate_probs(dims, src, probs, dst, d2f[dst], "cpu")
    st = TR.propagate_stats(dims, src, probs, dst, d2f[dst], "cpu")
    assert st["assigned"].all()
    assert np.array_equal(full.argmax(1).numpy(), st["channel"])
    assert np.allclose(full.max(1).values.numpy(), st["max_prob"], atol=1e-6)


def test_every_voxel_inside_the_maximum_radius_finds_a_source():
    rng = np.random.default_rng(1)
    dims = (16, 16, 16)
    occ = np.zeros(dims, bool)
    occ.reshape(-1)[rng.choice(16 ** 3, 12, replace=False)] = True
    d, d2 = D.edt(occ)
    d2f = d2.reshape(-1)
    dst = np.flatnonzero((d2f > 0) & (d2f <= D.d2_threshold(4.0, 0.2)))
    probs = torch.ones(int(occ.sum()), 3) / 3
    st = TR.propagate_stats(dims, np.flatnonzero(occ.reshape(-1)), probs, dst, d2f[dst],
                            "cpu")
    assert st["assigned"].all()


def test_propagation_is_deterministic():
    rng = np.random.default_rng(2)
    dims = (10, 10, 10)
    occ = np.zeros(dims, bool)
    occ.reshape(-1)[rng.choice(1000, 25, replace=False)] = True
    d, d2 = D.edt(occ)
    src = np.flatnonzero(occ.reshape(-1))
    probs = torch.tensor(rng.random((len(src), 4)), dtype=torch.float32)
    d2f = d2.reshape(-1)
    dst = np.flatnonzero((d2f > 0) & (d2f <= 16))
    a = TR.propagate_stats(dims, src, probs, dst, d2f[dst], "cpu")
    b = TR.propagate_stats(dims, src, probs, dst, d2f[dst], "cpu")
    for k in ("channel", "max_prob", "entropy", "n_tied"):
        assert np.array_equal(a[k], b[k])


# --------------------------------------------------------------------------- #
# The two constructions
# --------------------------------------------------------------------------- #
def _toy(seed=0, n=800):
    rng = np.random.default_rng(seed)
    base_occ = rng.random(n) < 0.10
    gt = rng.random(n) < 0.30
    t_ch = np.where(gt, rng.integers(0, 4, n), -1).astype(np.int32)
    base_ch = np.where(base_occ, rng.integers(0, 4, n), -1).astype(np.int32)
    prop_ch = rng.integers(0, 4, n).astype(np.int32)
    d2 = np.where(base_occ, 0, rng.integers(1, 500, n)).astype(np.int64)
    return base_occ, base_ch, prop_ch, d2, t_ch


def test_radius_zero_reproduces_the_base_volume_and_its_metrics():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy()
    base = E.count(t_ch, np.where(base_occ, base_ch, -1).astype(np.int32), 4)
    blocks = E.evaluate(base_occ, base_ch, prop_ch, d2, t_ch, 0.2, C.RADII_M, 4)
    for constr in C.CONSTRUCTIONS:
        b = blocks[constr][0.0]
        assert (b.btp, b.bfp, b.bfn) == (base.btp, base.bfp, base.bfn)
        assert np.array_equal(b.tp, base.tp) and np.array_equal(b.fp, base.fp)
        assert b.n_added == 0


def test_oracle_adds_only_ground_truth_occupied_voxels():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(3)
    gt = t_ch >= 0
    for r in C.RADII_M:
        _p, added = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, "oracle")
        assert not (added & ~gt).any()
        assert not (added & base_occ).any()


def test_oracle_false_positive_count_never_moves():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(4)
    base = E.count(t_ch, np.where(base_occ, base_ch, -1).astype(np.int32), 4)
    blocks = E.evaluate(base_occ, base_ch, prop_ch, d2, t_ch, 0.2, C.RADII_M, 4)
    for r in C.RADII_M:
        assert blocks["oracle"][float(r)].bfp == base.bfp
        assert blocks["oracle"][float(r)].n_added_fp == 0


def test_oracle_recall_and_iou_are_monotone_non_decreasing():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(5)
    blocks = E.evaluate(base_occ, base_ch, prop_ch, d2, t_ch, 0.2, C.RADII_M, 4)
    prev_r = prev_i = -1.0
    for r in sorted(C.RADII_M):
        b = blocks["oracle"][float(r)]
        rec = b.btp / max(b.btp + b.bfn, 1)
        iou = b.btp / max(b.btp + b.bfp + b.bfn, 1)
        assert rec >= prev_r - 1e-12 and iou >= prev_i - 1e-12
        prev_r, prev_i = rec, iou


def test_morphology_is_a_superset_of_the_base_and_of_the_oracle():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(6)
    gt = t_ch >= 0
    for r in C.RADII_M:
        pm, am = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, "morph")
        po, ao = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, "oracle")
        assert ((pm >= 0) | ~base_occ).all()
        assert not (base_occ & (pm < 0)).any()
        assert not (ao & ~am).any()
        assert ((po >= 0) <= (pm >= 0)).all()


def test_morphology_recall_is_monotone_and_never_below_the_base():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(7)
    base = E.count(t_ch, np.where(base_occ, base_ch, -1).astype(np.int32), 4)
    blocks = E.evaluate(base_occ, base_ch, prop_ch, d2, t_ch, 0.2, C.RADII_M, 4)
    E.assert_invariants(blocks, base, C.RADII_M)


def test_added_true_positives_are_identical_for_both_constructions():
    """The oracle's additions are exactly morphology's additions that are occupied."""
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(8)
    gt = t_ch >= 0
    for r in C.RADII_M:
        _pm, am = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, "morph")
        _po, ao = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, "oracle")
        assert np.array_equal(am & gt, ao)


def test_existing_voxel_semantics_are_never_changed():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(9)
    gt = t_ch >= 0
    for constr in C.CONSTRUCTIONS:
        for r in C.RADII_M:
            p, _a = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, constr)
            assert np.array_equal(p[base_occ], base_ch[base_occ])


def test_added_voxels_take_the_declared_propagated_class():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(10)
    gt = t_ch >= 0
    for constr in C.CONSTRUCTIONS:
        p, a = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, 2.0, constr)
        assert np.array_equal(p[a], prop_ch[a])
        assert (p[~a & ~base_occ] == -1).all()


def test_count_partition_identities_hold():
    base_occ, base_ch, prop_ch, d2, t_ch = _toy(11)
    gt = t_ch >= 0
    for constr in C.CONSTRUCTIONS:
        for r in C.RADII_M:
            p, _a = E.build(base_occ, base_ch, prop_ch, d2, gt, 0.2, r, constr)
            b = E.count(t_ch, p, 4)
            assert b.btp + b.bfn == int(gt.sum())
            assert b.btp + b.bfp == int((p >= 0).sum())
            assert int(b.tp.sum()) == b.btp - int(((t_ch >= 0) & (p >= 0)
                                                   & (t_ch != p)).sum())
            assert int((b.tp + b.fn).sum()) == int(gt.sum())


def test_empty_base_clip_adds_nothing_under_either_construction():
    n = 200
    base_occ = np.zeros(n, bool)
    base_ch = np.full(n, -1, np.int32)
    prop_ch = np.full(n, -1, np.int32)
    d2 = np.full(n, -1, np.int64)
    t_ch = np.where(np.arange(n) % 3 == 0, 1, -1).astype(np.int32)
    blocks = E.evaluate(base_occ, base_ch, prop_ch, d2, t_ch, 0.2, C.RADII_M, 4)
    for constr in C.CONSTRUCTIONS:
        for r in C.RADII_M:
            b = blocks[constr][float(r)]
            assert b.n_added == 0 and b.btp == 0 and b.bfp == 0


# --------------------------------------------------------------------------- #
# Camera geometry
# --------------------------------------------------------------------------- #
def test_toy_camera_projection_and_frustum_membership():
    K = np.array([[[100.0, 0, 50.0], [0, 100.0, 25.0], [0, 0, 1.0]]] * 1)
    pts = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 0.0, -5.0]])
    u, v, z = FR.project(pts, K[0])
    assert u[0] == pytest.approx(50.0) and v[0] == pytest.approx(25.0)
    assert u[1] == pytest.approx(60.0)
    assert z[2] < 0


def test_frustum_rejects_behind_out_of_bounds_and_out_of_range():
    T = 1
    H, W = 50, 100
    K = np.array([[[100.0, 0, 50.0], [0, 100.0, 25.0], [0, 0, 1.0]]])
    dep = np.full((T, H, W), 10.0, np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    pose = np.eye(4)[None]
    centres = np.array([
        [0.0, 0.0, 10.0],          # dead centre, in range
        [0.0, 0.0, -10.0],         # behind the camera
        [0.0, 0.0, 0.5],           # nearer than the frozen minimum
        [0.0, 0.0, 100.0],         # beyond the frozen maximum
        [100.0, 0.0, 10.0],        # projects far outside the image
    ])
    out = FR.analyse(centres, np.eye(4), pose, K, dep, conf, 1.0, 1.5, 1.0, 60.0,
                     np.sqrt(3) * 0.2)
    assert out["in_any"].tolist() == [True, False, False, False, False]
    assert out["klass"][0] == FR.NEAR_SURFACE      # depth 10 == predicted 10
    assert out["klass"][1] == FR.OUTSIDE


def test_frustum_residual_classes_use_the_declared_tolerance():
    K = np.array([[[100.0, 0, 50.0], [0, 100.0, 25.0], [0, 0, 1.0]]])
    dep = np.full((1, 50, 100), 10.0, np.float32)
    conf = np.full((1, 50, 100), 5.0, np.float32)
    pose = np.eye(4)[None]
    tol = np.sqrt(3) * 0.2
    centres = np.array([[0.0, 0.0, 10.0 + tol * 0.5],
                        [0.0, 0.0, 10.0 + tol * 3.0],
                        [0.0, 0.0, 10.0 - tol * 3.0]])
    out = FR.analyse(centres, np.eye(4), pose, K, dep, conf, 1.0, 1.5, 1.0, 60.0, tol)
    assert out["klass"].tolist() == [FR.NEAR_SURFACE, FR.BEHIND, FR.IN_FRONT]


def test_low_confidence_pixels_are_no_valid_predicted_depth():
    K = np.array([[[100.0, 0, 50.0], [0, 100.0, 25.0], [0, 0, 1.0]]])
    dep = np.full((1, 50, 100), 10.0, np.float32)
    conf = np.zeros((1, 50, 100), np.float32)          # below the frozen threshold
    out = FR.analyse(np.array([[0.0, 0.0, 10.0]]), np.eye(4), np.eye(4)[None], K, dep,
                     conf, 1.0, 1.5, 1.0, 60.0, np.sqrt(3) * 0.2)
    assert out["klass"][0] == FR.NO_DEPTH and out["in_any"][0]


def test_voxel_centres_land_at_the_grid_centre_offset():
    G = G6G.EVAL_GRID["semantickitti"]
    c = FR.voxel_centres(np.array([0]), G)
    assert c[0].tolist() == pytest.approx([G.origin[0] + G.voxel_size / 2,
                                           G.origin[1] + G.voxel_size / 2,
                                           G.origin[2] + G.voxel_size / 2])


def test_occ3d_native_grid_is_the_coarse_one_and_is_unambiguous():
    G = G6G.EVAL_GRID["occ3d"]
    assert tuple(G.dims) == (200, 200, 16) and G.voxel_size == 0.4
    assert G6G.NEEDS_REDUCTION["occ3d"] and not G6G.NEEDS_REDUCTION["semantickitti"]
    assert G6G.EVAL_GRID["semantickitti"].voxel_size == 0.2
    assert G6G.EVAL_GRID["kitti360"].voxel_size == 0.2


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_multiplicity_bootstrap_matches_the_gate6_draws():
    from gates.gate6 import stats as G6S
    U, B, seed = 9, 50, 0
    mult = S.multiplicities(U, B, seed)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, U, size=(B, U))
    block = np.arange(U * 4, dtype=np.float64).reshape(U, 4)
    for b in range(B):
        assert np.allclose(S.resample(block, mult)[b], block[draws[b]].sum(axis=0))
    assert mult.sum(axis=1).tolist() == [U] * B
    assert hasattr(G6S, "paired_bootstrap")


def test_bootstrap_metrics_match_the_gate6_definitions():
    pc = np.array([[[4.0, 1, 1], [0, 0, 0], [2, 2, 2]]])
    assert S.miou(pc)[0] == pytest.approx((4 / 6 + 2 / 6) / 2)
    assert S.binary_iou(np.array([[3.0, 1, 1]]))[0] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# Provenance: Gate 6 is untouched, nothing is trained, nothing is written
# --------------------------------------------------------------------------- #
def test_gate6_prediction_rollup_hashes_are_unchanged():
    a = _json(os.path.join(ART, "stage0_audit.json"))
    assert a is not None, "run tools/gate7a/stage0.py"
    for ds, rec in a["gate6"]["predictions"].items():
        man = _json(os.path.join(G6ART, f"prediction_manifest_{ds}.json"))
        assert man["rollup_sha256"] == rec["rollup_sha256"]
        assert _sha256(os.path.join(G6ART,
                                    f"prediction_manifest_{ds}.json")) == rec["manifest_sha256"]


def test_gate6_precommit_still_matches_its_pin():
    pin = _json(os.path.join(G6ART, "precommit_pin.json"))
    assert _sha256(os.path.join(REPO, pin["path"])) == pin["sha256"]


def test_gate7a_precommit_still_matches_its_pin():
    pin = _json(os.path.join(ART, "precommit_pin.json"))
    assert pin is not None, "run tools/gate7a/precommit.py"
    assert _sha256(os.path.join(REPO, pin["path"])) == pin["sha256"]


def test_pre_existing_dirty_files_are_untouched():
    a = _json(os.path.join(ART, "stage0_audit.json"))
    for rel, rec in a["pre_existing_dirty"].items():
        assert _sha256(os.path.join(REPO, rel)) == rec["sha256"], \
            f"{rel} was modified; Gate 7A must preserve it"


def test_no_gate7a_output_path_is_inside_a_released_dataset_directory():
    forbidden = ("data/kitti/dataset", "/media/SSD1/MINH_DATASETS/sscbench_kitti360",
                 "/media/SSD1/MINH_DATASETS/nuscenes",
                 "/media/welf/MINH/datasets/kitti360",
                 "/media/SSD1/MINH_DATASETS/lingbot_gate6/predictions",
                 "/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics")
    for p in (glob.glob(os.path.join(REPO, "tools", "gate7a", "*.py"))
              + glob.glob(os.path.join(REPO, "gates", "gate7a", "*.py"))):
        t = ast.parse(open(p).read())
        for node in ast.walk(t):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "open" and len(node.args) > 1:
                mode = node.args[1]
                if isinstance(mode, ast.Constant) and "w" in str(mode.value):
                    seg = ast.get_source_segment(open(p).read(), node) or ""
                    assert not any(f in seg for f in forbidden), f"{p}: {seg}"
    # and nothing on disk under those roots carries a Gate-7A name
    for root in forbidden:
        if os.path.isdir(root):
            assert not glob.glob(os.path.join(root, "*gate7a*"))


def _code_only(path):
    """Source with docstrings and string literals removed, so prose cannot trip a scan."""
    t = ast.parse(open(path).read())
    for node in ast.walk(t):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    for node in ast.walk(t):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(ast.fix_missing_locations(t))


def test_no_optimizer_no_backward_no_parameter_update():
    banned = ("optim.", "Optimizer", ".backward(", "requires_grad_", "loss.backward",
              "nn.Module", "torch.nn", "state_dict", "train()", "fit(", "GradScaler")
    for p in (glob.glob(os.path.join(REPO, "gates", "gate7a", "*.py"))
              + glob.glob(os.path.join(REPO, "tools", "gate7a", "*.py"))):
        code = _code_only(p)
        for b in banned:
            assert b not in code, f"{os.path.basename(p)} contains {b!r}"


def test_gate7a_never_writes_a_prediction_artifact():
    """Gate 7A is oracle analysis; it must not emit anything shaped like a prediction."""
    for p in glob.glob(os.path.join(ART, "*")):
        base = os.path.basename(p).lower()
        assert "prediction" not in base and "manifest" not in base, base


def test_the_precommit_declares_gate7a_as_target_dependent_oracle_analysis():
    pin = _json(os.path.join(ART, "precommit_pin.json"))
    text = open(os.path.join(REPO, pin["path"])).read()
    assert "TARGET-DEPENDENT ORACLE ANALYSIS" in text
    assert "not target-free inference" in text.lower() or \
           "not a deployable prediction" in text.lower()
    for r in C.RADII_M:
        assert f"{r}" in text or f"{r:g}" in text


# --------------------------------------------------------------------------- #
# Results, when they exist
# --------------------------------------------------------------------------- #
def _summary(ds):
    return _json(os.path.join(ART, f"summary_{ds}.json"))


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_base_metrics_reproduce_gate6_exactly(ds):
    s = _summary(ds)
    if s is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    g6 = _json(os.path.join(G6ART, f"summary_{ds}.json"))
    for cond in C.CONDITIONS:
        assert s["base"][cond]["binary_iou"] == pytest.approx(
            g6["conditions"][cond]["binary_iou_pooled"], abs=1e-9)
        assert s["base"][cond]["ssc_miou"] == pytest.approx(
            g6["conditions"][cond]["ssc_miou"], abs=1e-9)
        assert s["base"][cond]["binary_iou_mean_per_clip"] == pytest.approx(
            g6["conditions"][cond]["binary_iou_mean_per_clip"], abs=1e-9)
    assert s["n_clips"] == g6["n_clips"]


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_every_clip_reproduced_the_frozen_geometry(ds):
    s = _summary(ds)
    if s is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    assert s["n_verified_against_pinned"] == s["n_clips"]
    assert s["max_projection_roundtrip_pixel_error"] < 1e-6


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_reported_envelopes_obey_the_structural_invariants(ds):
    s = _summary(ds)
    if s is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    for cond in C.CONDITIONS:
        base = s["base"][cond]
        o = s["envelopes"][cond]["oracle"]
        m = s["envelopes"][cond]["morph"]
        assert o["recall_monotonic_non_decreasing"] and o["iou_monotonic_non_decreasing"]
        assert o["false_positives_constant"]
        assert m["recall_monotonic_non_decreasing"]
        for row in o["by_radius"]:
            assert row["fp"] == base["fp"]
            assert row["n_added_false_positive"] == 0
        for row in m["by_radius"]:
            assert row["n_added"] >= s["envelopes"][cond]["oracle"]["by_radius"][
                s["radii_m"].index(row["radius_m"])]["n_added"]
        assert o["by_radius"][0]["binary_iou"] == pytest.approx(base["binary_iou"])
        assert m["by_radius"][0]["binary_iou"] == pytest.approx(base["binary_iou"])


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_miss_fractions_are_bounded_and_monotone(ds):
    s = _summary(ds)
    if s is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    for cond in C.CONDITIONS:
        f = s["miss_distance"][cond]["all"]["miss_fraction_within_radius"]
        vals = [f[f"{r:g}"] for r in s["radii_m"]]
        assert vals == sorted(vals) and 0.0 <= vals[-1] <= 1.0
        assert vals[0] == 0.0


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_frustum_counts_partition_the_coverage_misses(ds):
    s = _summary(ds)
    if s is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    for cond in C.CONDITIONS:
        f = s["frustum"][cond]["all"]
        assert f["in_any_input_frustum"] + f["outside_all_input_frusta"] == \
            f["n_coverage_miss"]
        assert sum(f["residual_classes"].values()) == f["n_coverage_miss"]
        assert f["residual_classes"]["outside_all_frusta"] == \
            f["outside_all_input_frusta"]


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_morphology_at_the_dilation_radius_sits_inside_b_d(ds):
    """B-D is a Chebyshev ball of two voxels; the 0.4 m Euclidean ball is a subset of it.

    A free end-to-end check that the distance transform, the radius test and the frozen
    ``dilate_r2`` agree: morphology at 0.4 m on B-R cannot have higher recall than B-D.
    """
    d = _summary(ds)
    if d is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    m = d["envelopes"]["B-R"]["morph"]["by_radius"][1]
    assert m["radius_m"] == 0.4
    assert m["binary_recall"] <= d["base"]["B-D"]["binary_recall"] + 1e-12
    assert m["tp"] <= d["base"]["B-D"]["tp"]


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_propagated_semantics_stay_above_chance_at_every_distance(ds):
    d = _summary(ds)
    if d is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    chance = 1.0 / len(d["class_names"])
    for cond in C.CONDITIONS:
        for row in d["semantic_transport"][cond]["by_interval"]:
            if row["n_voxels"] == 0:
                continue
            assert row["top1_accuracy"] > 2 * chance, (cond, row["interval_m"])


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_added_volume_and_precision_bookkeeping_is_consistent(ds):
    d = _summary(ds)
    if d is None:
        pytest.skip(f"gate7a summary for {ds} not produced yet")
    for cond in C.CONDITIONS:
        base = d["base"][cond]
        for constr in C.CONSTRUCTIONS:
            for row in d["envelopes"][cond][constr]["by_radius"]:
                assert row["n_added"] == row["n_added_true_positive"] + \
                    row["n_added_false_positive"]
                assert row["tp_recovered"] == row["n_added_true_positive"]
                assert row["fn_recovered"] == row["n_added_true_positive"]
                assert row["fp"] == base["fp"] + row["n_added_false_positive"]


def test_report_is_fully_generated_and_has_no_placeholders():
    p = os.path.join(REPO, "reports", "gate7a", "completion_reachability.md")
    if not os.path.exists(p):
        pytest.skip("gate7a report not written yet")
    text = open(p).read()
    assert "{{" not in text and "}}" not in text
    assert "failed:" not in text
    assert "TARGET-DEPENDENT" in text or "oracle analysis" in text


def test_gate6_report_corrections_are_present_and_recorded():
    rec = _json(os.path.join(ART, "gate6_report_correction.json"))
    if rec is None:
        pytest.skip("correction record not written yet")
    assert rec["gate6_numerical_artifacts_untouched"]
    assert rec["sha256_after"] != rec["sha256_before"]
    p = os.path.join(REPO, REL := "reports/gate6/frozen_trident_semantic_lifting.md")
    assert _sha256(p) == rec["sha256_after"]
    text = open(p).read()
    assert "All reported differences except" in text
    assert "binary IoU (pooled)" in text
    assert "DINO v1" in text
    assert "0.2643" in text and "0.2563" in text          # the 0-10 m reversal
    assert "no paired confidence interval" in text
