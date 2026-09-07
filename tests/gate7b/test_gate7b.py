"""Gate 7B — native causal streaming replay, metric gauge, complementary depth.

Grouped by the claim each defends: **causality** (no future frame ever reaches a causal
map), **the gauge** (one scalar, applied to depth and translation and to nothing else),
**the map** (free / occupied / unknown, and never free behind a surface), **the
complementary evidence** (MoGe proposes only where it is allowed to), and **provenance**
(Gates 6 and 7A are byte-identical afterwards, and no dataset directory is written).
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gates.gate6 import grids as G6G                                            # noqa: E402
from gates.gate7b import config as C, depth as D7, evidence as EV, fuse as FZ, \
    rays as RY, replay as RP, scale as SC, streams as ST, voxmap as VM     # noqa: E402
from gates.gate7b.voxmap import EvidenceVolume                                   # noqa: E402

ART = os.path.join(REPO, "artifacts", "gate7b")
G6ART = os.path.join(REPO, "artifacts", "gate6")
G7A = os.path.join(REPO, "artifacts", "gate7a")


def _json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Streams and boundaries
# --------------------------------------------------------------------------- #
def test_streams_are_deduplicated_and_chronological():
    from gates.scale_gate.config import REPO_ROOT
    for ds in C.DATASETS:
        segs = ST.build(ds, REPO_ROOT)
        seen = set()
        for s in segs:
            orders = [f.order for f in s.frames]
            assert orders == sorted(orders), f"{ds}/{s.name} is not chronological"
            assert len(set(orders)) == len(orders), f"{ds}/{s.name} has a repeated frame"
            for f in s.frames:
                assert f.key not in seen, f"{ds}: frame {f.key} appears in two segments"
                seen.add(f.key)
            assert [f.index for f in s.frames] == list(range(len(s.frames)))


def test_state_resets_only_at_a_true_sequence_boundary():
    from gates.scale_gate.config import REPO_ROOT
    segs = {ds: ST.build(ds, REPO_ROOT) for ds in C.DATASETS}
    assert len(segs["semantickitti"]) == 1          # one sequence, one stream
    assert len(segs["kitti360"]) == 1               # one drive, one stream
    assert len(segs["occ3d"]) == 150                # 150 official scenes
    for ds, ss in segs.items():
        for s in ss:
            assert len(s) >= 5, f"{ds}/{s.name} is shorter than the anchor block"
    # the replay resets exactly once per segment, at its start
    src = open(os.path.join(REPO, "gates", "gate7b", "replay.py")).read()
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("``")]
    assert sum(ln.count("model.clean_kv_cache()") for ln in code) == 1


def test_every_anchor_maps_into_its_own_segment():
    from gates.scale_gate.config import REPO_ROOT
    for ds in C.DATASETS:
        for s in ST.build(ds, REPO_ROOT):
            for cid, t in s.anchors.items():
                assert 0 <= t < len(s)


def test_keyframe_rule_keeps_every_stream_inside_the_frozen_rope_table():
    for n in (5, 40, 815, 1777, 4071, 10000):
        k = RP.keyframe_interval_for(n)
        assert k >= 1
        assert RP.rope_slots(n, k) <= RP.MAX_FRAME_NUM - RP.ROPE_MARGIN
        if k > 1:                       # minimality: one step smaller would not fit
            assert RP.rope_slots(n, k - 1) > RP.MAX_FRAME_NUM - RP.ROPE_MARGIN
    assert RP.keyframe_interval_for(815) == 1        # depends only on length,
    assert RP.keyframe_interval_for(1777) == 2       # never on a benchmark score


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon,t,expect", [(1, 10, 10), (5, 10, 6), (20, 10, 0),
                                              ("all", 10, 0), (50, 3, 0)])
def test_causal_window_starts_where_declared(horizon, t, expect):
    assert FZ.horizon_start(t, horizon) == expect


def test_no_future_frame_can_enter_a_causal_window():
    for t in range(0, 40):
        for h in C.HORIZONS:
            lo = FZ.horizon_start(t, h)
            assert lo <= t
            assert max(range(lo, t + 1)) == t          # the window ends AT t
    src = open(os.path.join(REPO, "tools", "gate7b", "run_stream_eval.py")).read()
    assert "range(lo, t + 1)" in src
    assert "a future frame entered the window" in src


def test_reach_prefilter_can_only_remove_frames_that_cannot_contribute():
    """A camera further away than max_depth + box radius cannot write into the box."""
    n = 12
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n) * 30.0                # 30 m apart along z
    G = G6G.PREDICTION_GRID["semantickitti"]
    m = FZ.reach_mask(poses, n - 1, 1.0, np.eye(4), G, 60.0)
    assert m[n - 1]                                     # the anchor always reaches
    far = np.linalg.norm(poses[:, :3, 3] - poses[n - 1, :3, 3], axis=1)
    box_r = 0.5 * float(np.linalg.norm(np.asarray(G.dims) * G.voxel_size))
    assert not m[far > 60.0 + 2 * box_r].any()


def test_recoverability_is_isolated_from_any_deployable_output():
    src = open(os.path.join(REPO, "tools", "gate7b", "recoverability.py")).read()
    assert "diagnostic_only" in src
    assert "recoverability_" in src                     # its own artifact namespace
    ev = open(os.path.join(REPO, "tools", "gate7b", "run_stream_eval.py")).read()
    assert "recoverability" not in ev                   # the evaluator cannot import it
    assert "range(t + 1" not in ev


# --------------------------------------------------------------------------- #
# The metric gauge
# --------------------------------------------------------------------------- #
def _cands(vals):
    return [{"log_s": float(v)} for v in vals]


def test_fixed_anchor_scale_uses_only_the_anchor_frames():
    vals = np.log([10.0, 10.0, 10.0, 10.0, 10.0] + [100.0] * 20)
    st = SC.ScaleState(policy="G-A")
    out = [st.observe(i, c) for i, c in enumerate(_cands(vals))]
    assert out[-1] == pytest.approx(10.0)
    assert st.series[SC.N_ANCHOR_FRAMES:].std() == pytest.approx(0.0)


def test_running_median_uses_only_frames_up_to_t():
    vals = np.log([1.0, 2.0, 3.0, 100.0, 200.0])
    st = SC.ScaleState(policy="G-B")
    out = [st.observe(i, c) for i, c in enumerate(_cands(vals))]
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(np.sqrt(2.0))
    assert out[2] == pytest.approx(2.0)
    # truncating the stream cannot change any earlier value
    for cut in range(1, len(vals) + 1):
        st2 = SC.ScaleState(policy="G-B")
        o2 = [st2.observe(i, c) for i, c in enumerate(_cands(vals[:cut]))]
        assert np.allclose(o2, out[:cut])


def test_per_frame_scale_follows_every_frame():
    vals = np.log([1.0, 4.0, 9.0])
    st = SC.ScaleState(policy="G-C")
    out = [st.observe(i, c) for i, c in enumerate(_cands(vals))]
    assert np.allclose(out, [1.0, 4.0, 9.0])


def test_per_frame_scaling_exposes_surface_duplication_rather_than_hiding_it():
    """Two views of one plane, gauged differently, must land in different voxels.

    G-C is an ablation precisely because this happens; the test asserts the mechanism is
    visible rather than silently averaged away.
    """
    dims, vs = (8, 8, 40), 0.5            # long axis along camera z
    origin = torch.tensor([-2.0, -2.0, 0.0], dtype=torch.float64)
    K = np.array([[8.0, 0, 4.0], [0, 8.0, 4.0], [0, 0, 1.0]])
    dirs = RY.pixel_rays(K, (9, 9), "cpu")
    acc = torch.zeros((9, 9), dtype=torch.bool)
    acc[4, 4] = True
    seen = []
    for s in (1.0, 1.25):
        vol = EvidenceVolume(dims, "cpu", 2)
        d = torch.full((9, 9), 8.0 * s, dtype=torch.float64)
        RY.cast_frame(vol, dirs, d, acc, torch.eye(4, dtype=torch.float64),
                      torch.eye(4, dtype=torch.float64), origin, vs, dims, 0, 0, 1.0,
                      carve=False)
        seen.append(set(vol.occupied().nonzero(as_tuple=True)[0].tolist()))
    assert seen[0] and seen[1] and seen[0] != seen[1]


def test_the_same_scalar_scales_depth_and_translation_and_leaves_rotation_alone():
    sys.path.insert(0, os.path.join(REPO, "tools", "depth_gate"))
    from decompose_residual import scaled_relative_pose
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (4, 1, 1))
    for i in range(4):
        q = rng.normal(size=(3, 3))
        u, _s, vt = np.linalg.svd(q)
        poses[i, :3, :3] = u @ vt
        poses[i, :3, 3] = rng.normal(size=3) * 5
    for s in (1.0, 2.5, 0.3):
        for f in range(3):
            T1 = scaled_relative_pose(poses, f, 3, 1.0)
            Ts = scaled_relative_pose(poses, f, 3, s)
            assert np.allclose(Ts[:3, :3], T1[:3, :3])          # rotation untouched
            assert np.allclose(Ts[:3, 3], s * T1[:3, 3])        # translation scaled
    # and the depth scaling uses the very same scalar
    d = torch.tensor([[2.0]], dtype=torch.float32)
    conf = torch.tensor([[5.0]], dtype=torch.float32)
    _ok, dm = EV.lingbot_gate(d, conf, 2.5)
    assert float(dm[0, 0]) == pytest.approx(5.0)


def test_a_gauge_change_transforms_old_and_new_observations_together():
    """The whole window is rematerialised with s(t), so no stale surface can survive."""
    src = open(os.path.join(REPO, "gates", "gate7b", "fuse.py")).read()
    assert "scale_at(anchor)" in src
    dims, vs = (8, 8, 40), 0.5            # long axis along camera z
    origin = torch.tensor([-2.0, -2.0, 0.0], dtype=torch.float64)
    K = np.array([[8.0, 0, 4.0], [0, 8.0, 4.0], [0, 0, 1.0]])
    dirs = RY.pixel_rays(K, (9, 9), "cpu")
    acc = torch.zeros((9, 9), dtype=torch.bool)
    acc[4, 4] = True
    # two frames of the same surface, both rescaled by the same new gauge
    for s in (1.0, 1.5):
        vol = EvidenceVolume(dims, "cpu", 2)
        for _f in range(2):
            d = torch.full((9, 9), 6.0 * s, dtype=torch.float64)
            RY.cast_frame(vol, dirs, d, acc, torch.eye(4, dtype=torch.float64),
                          torch.eye(4, dtype=torch.float64), origin, vs, dims, 0, 0, 1.0,
                          carve=False)
        assert int(vol.occupied().sum()) == 1, "one surface, one voxel, at any gauge"


def test_scale_candidate_abstains_on_too_little_overlap():
    z = np.zeros((4, 4))
    c = SC.frame_candidate(z + 5.0, np.ones((4, 4), bool), z + 1.0, z + 5.0)
    assert not np.isfinite(c["log_s"])
    n = 64
    md = np.full((n, n), 20.0)
    ld = np.full((n, n), 2.0)
    c = SC.frame_candidate(md, np.ones((n, n), bool), ld, np.full((n, n), 5.0))
    assert np.exp(c["log_s"]) == pytest.approx(10.0)


def test_scale_candidate_rejects_gross_outliers():
    n = 64
    md = np.full((n, n), 20.0)
    ld = np.full((n, n), 2.0)
    md[0, :] = 20000.0                          # a row of gross outliers
    c = SC.frame_candidate(md, np.ones((n, n), bool), ld, np.full((n, n), 5.0))
    assert np.exp(c["log_s"]) == pytest.approx(10.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Depth conventions
# --------------------------------------------------------------------------- #
def test_depth_convention_conversion_is_z_not_ray_distance():
    K = np.array([[10.0, 0, 5.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    dirs = RY.pixel_rays(K, (11, 11), "cpu")
    assert torch.allclose(dirs[..., 2], torch.ones_like(dirs[..., 2]))
    off = dirs[0, 0]                                     # a corner ray
    assert float(off.norm()) > 1.0                       # ray length exceeds its z
    p = off * 10.0                                       # z-depth 10 m
    assert float(p[2]) == pytest.approx(10.0)
    assert float(p.norm()) > 10.0


def test_lingbot_gate_is_the_frozen_rule():
    d = torch.tensor([[0.5, 2.0, 100.0]], dtype=torch.float32)
    c = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
    ok, dm = EV.lingbot_gate(d, c, 1.0)
    assert ok.tolist() == [[False, True, False]]
    ok2, _ = EV.lingbot_gate(d, torch.tensor([[1.0, 1.0, 1.0]]), 1.0)
    assert not ok2.any()                                 # conf 1.0 < 1.5
    assert D7.CONF_THRESHOLD == 1.5 and D7.MIN_DEPTH_M == 1.0 and D7.MAX_DEPTH_M == 60.0


def test_moge_cache_is_the_calibrated_variant_and_frame_consistent():
    from gates.scale_gate.config import REPO_ROOT
    ix = D7.moge_frame_index("kitti360", REPO_ROOT)
    if not ix:
        pytest.skip("kitti360 MoGe cache unavailable")
    k = sorted(ix)[0]
    a = D7.load_moge_frame("kitti360", k, ix)
    b = D7.load_moge_frame("kitti360", k, ix)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert "semantickitti" not in D7.MOGE_CLIP_CACHE      # the A-variant is not reused
    assert "occ3d" not in D7.MOGE_CLIP_CACHE


# --------------------------------------------------------------------------- #
# The evidence map
# --------------------------------------------------------------------------- #
def _toy_cast(depth=6.0, dims=(20, 20, 20), vs=1.0, stride=1, band=0.2):
    origin = torch.tensor([-10.0, -10.0, -10.0], dtype=torch.float64)
    vol = EvidenceVolume(dims, "cpu", 3)
    K = np.array([[10.0, 0, 5.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    dirs = RY.pixel_rays(K, (11, 11), "cpu")
    d = torch.full((11, 11), float(depth), dtype=torch.float64)
    acc = torch.zeros((11, 11), dtype=torch.bool)
    acc[5, 5] = True
    st = RY.cast_frame(vol, dirs, d, acc, torch.eye(4, dtype=torch.float64),
                       torch.eye(4, dtype=torch.float64), origin, vs, dims, 0, 0, 1.0,
                       carve_stride=stride, band_half_m=band)
    return vol, st, dims


def test_ray_marks_free_before_occupied_at_and_unknown_behind():
    vol, _st, dims = _toy_cast()
    occ = vol.occupied().reshape(dims)
    free = vol.free().reshape(dims)
    unk = vol.unknown().reshape(dims)
    assert occ[10, 10, 16] and not occ[10, 10, 15]
    assert free[10, 10, 11:16].all()
    assert unk[10, 10, 17:].all(), "space behind the surface must stay unknown"
    assert not free[10, 10, 17:].any()


def test_the_three_states_are_disjoint_and_exhaustive():
    vol, _st, dims = _toy_cast()
    o, f, u = vol.occupied(), vol.free(), vol.unknown()
    p = vol.provisional()
    n = o.numel()
    assert int((o & f).sum()) == 0 and int((o & u).sum()) == 0
    assert int((f & u).sum()) == 0
    assert int(o.sum()) + int(f.sum()) + int(u.sum()) + int(p.sum()) <= n


def test_repeated_observations_are_fused_not_unioned():
    dims, vs = (20, 20, 20), 1.0
    origin = torch.tensor([-10.0, -10.0, -10.0], dtype=torch.float64)
    vol = EvidenceVolume(dims, "cpu", 3)
    f = torch.tensor([7])
    vol.add_occupied(f, 0)
    l1 = float(vol.logodds[7])
    vol.add_occupied(f, 1)
    assert float(vol.logodds[7]) == pytest.approx(2 * l1)
    assert int(vol.n_obs[7]) == 2
    for _ in range(20):
        vol.add_occupied(f, 2)
        vol.clamp()
    assert float(vol.logodds[7]) == pytest.approx(VM.L_CLAMP)


def test_free_evidence_can_overturn_a_single_occupied_observation():
    vol = EvidenceVolume((4, 4, 4), "cpu", 2)
    f = torch.tensor([1])
    vol.add_occupied(f, 0)
    assert bool(vol.occupied()[1])
    for _ in range(3):
        vol.add_free(f)
    vol.clamp()
    assert not bool(vol.occupied()[1]) and bool(vol.free()[1])


def test_semantic_fusion_does_not_alter_geometry():
    vol = EvidenceVolume((4, 4, 4), "cpu", 3)
    f = torch.tensor([2])
    vol.add_occupied(f, 0)
    before = (vol.logodds.clone(), vol.w_occ.clone(), vol.w_free.clone(),
              vol.n_obs.clone(), vol.touched.clone())
    vol.add_semantics(f, torch.tensor([[0.1, 0.7, 0.2]]), torch.tensor([2.0]))
    assert torch.equal(vol.logodds, before[0]) and torch.equal(vol.w_occ, before[1])
    assert torch.equal(vol.w_free, before[2]) and torch.equal(vol.n_obs, before[3])
    assert torch.equal(vol.touched, before[4])
    assert int(vol.semantic_channel()[2]) == 1


def test_semantic_weight_never_uses_the_teacher_confidence():
    src = open(os.path.join(REPO, "gates", "gate7b", "evidence.py")).read()
    fn = src[src.index("def semantic_weight"):]
    assert "probs" not in fn and "max" not in fn.split("return")[0].split("\n")[-3]
    conf = torch.tensor([[3.0, 0.1]])
    acc = EV.Acceptance(lingbot=torch.tensor([[True, False]]),
                        moge=torch.tensor([[False, True]]),
                        moge_weight=torch.tensor([[0.0, 0.4]]),
                        depth_lingbot_m=torch.zeros(1, 2, dtype=torch.float64),
                        depth_moge_m=torch.zeros(1, 2, dtype=torch.float64))
    w = EV.semantic_weight("S3", acc, conf)
    assert float(w[0, 0]) > 0 and float(w[0, 1]) == pytest.approx(0.4)


def test_carve_stride_only_decimates_free_space_not_occupancy():
    v1, s1, _ = _toy_cast(stride=1)
    v4, s4, _ = _toy_cast(stride=4)
    assert s1["n_occ_updates"] == s4["n_occ_updates"] == 1
    assert int(v1.occupied().sum()) == int(v4.occupied().sum())


# --------------------------------------------------------------------------- #
# Complementary depth
# --------------------------------------------------------------------------- #
def _acc_inputs():
    d = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32)
    conf = torch.tensor([[5.0, 0.1, 0.1]], dtype=torch.float32)
    mg = torch.tensor([[20.0, 21.0, 0.1]], dtype=torch.float32)
    mm = torch.tensor([[True, True, True]])
    return d, conf, mg, mm


def test_moge_rescue_only_on_rays_lingbot_rejects_and_only_when_eligible():
    d, conf, mg, mm = _acc_inputs()
    a = EV.accept("S3", d, conf, 10.0, mg, mm)
    assert a.lingbot.tolist() == [[True, False, False]]
    # pixel 0 accepted by LingBot -> no rescue; pixel 1 eligible; pixel 2 out of range
    assert a.moge.tolist() == [[False, True, False]]
    assert float(a.moge_weight[0, 1]) == pytest.approx(VM.W_MOGE)


def test_s1_and_s2_never_produce_moge_evidence():
    d, conf, mg, mm = _acc_inputs()
    for var in ("S1", "S2"):
        a = EV.accept(var, d, conf, 10.0, mg, mm)
        assert not a.moge.any() and float(a.moge_weight.sum()) == 0.0


def test_s2_relaxes_confidence_but_not_range():
    d = torch.tensor([[2.0, 8.0]], dtype=torch.float32)
    conf = torch.tensor([[0.6, 0.6]], dtype=torch.float32)
    strict = EV.accept("S1", d, conf, 10.0, None, None)
    relaxed = EV.accept("S2", d, conf, 10.0, None, None, conf_threshold=0.5)
    assert not strict.lingbot.any()
    assert relaxed.lingbot.tolist() == [[True, False]]      # 80 m still rejected
    assert 1.5 in EV.S2_CONF_SWEEP and min(EV.S2_CONF_SWEEP) == 0.0


def test_s4_weight_falls_with_disagreement_and_drops_hopeless_candidates():
    d = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    conf = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    mg = torch.tensor([[20.5, 55.0]], dtype=torch.float32)   # agrees / disagrees badly
    mm = torch.tensor([[True, True]])
    a = EV.accept("S4", d, conf, 10.0, mg, mm)
    assert float(a.moge_weight[0, 0]) > float(a.moge_weight[0, 1])
    assert float(a.moge_weight[0, 0]) <= VM.W_MOGE


def test_established_free_space_rejects_a_contradictory_rescue_proposal():
    dims, vs = (9, 9, 30), 1.0            # long axis along camera z
    origin = torch.tensor([-4.0, -4.0, -2.0], dtype=torch.float64)
    K = np.array([[9.0, 0, 4.0], [0, 9.0, 4.0], [0, 0, 1.0]])
    dirs = RY.pixel_rays(K, (9, 9), "cpu")
    keep = torch.zeros((9, 9), dtype=torch.bool)
    keep[4, 4] = True
    depth = torch.full((9, 9), 6.0, dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float64)
    grid = type("G", (), {"origin": (-4.0, -4.0, -2.0), "voxel_size": vs, "dims": dims})
    vol = EvidenceVolume(dims, "cpu", 2)
    assert bool(FZ._veto(vol, dirs, depth, keep, eye, origin, grid, "cpu")[4, 4])
    for _ in range(4):                                   # carve that voxel free
        f, _ok = RY.to_grid((dirs[4, 4] * 6.0).unsqueeze(0), eye, origin, vs, dims)
        vol.add_free(f)
    vol.clamp()
    assert not bool(FZ._veto(vol, dirs, depth, keep, eye, origin, grid, "cpu")[4, 4])


def test_moge_only_occupancy_stays_provisional_until_confirmed():
    vol = EvidenceVolume((4, 4, 4), "cpu", 2)
    f = torch.tensor([3])
    vol.add_occupied(f, 0, source=VM.SRC_MOGE, weight=VM.W_MOGE)
    assert bool(vol.provisional()[3]) and not bool(vol.occupied()[3])
    assert bool(vol.occupied(allow_provisional=True)[3])
    vol.add_occupied(f, 1, source=VM.SRC_MOGE, weight=VM.W_MOGE)
    assert bool(vol.occupied()[3]) and not bool(vol.provisional()[3])


def test_moge_cannot_outweigh_lingbot_per_observation():
    assert VM.W_MOGE < VM.W_LINGBOT


# --------------------------------------------------------------------------- #
# Determinism and provenance
# --------------------------------------------------------------------------- #
def test_replay_of_the_map_is_deterministic():
    outs = []
    for _ in range(2):
        vol, _s, dims = _toy_cast()
        outs.append((vol.logodds.clone(), vol.occupied().clone()))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_scale_state_replay_is_deterministic():
    vals = np.log(np.linspace(5.0, 30.0, 40))
    runs = []
    for _ in range(2):
        st = SC.ScaleState(policy="G-B")
        runs.append([st.observe(i, {"log_s": float(v)}) for i, v in enumerate(vals)])
    assert runs[0] == runs[1]


def test_prediction_never_opens_a_target_file():
    """The Gate-6 auditor is installed around a fusion; it raises on any target path."""
    from gates.gate6.audit import Gate6Audit
    with Gate6Audit() as audit:
        vol, _s, _d = _toy_cast()
        st = SC.ScaleState(policy="G-A")
        st.observe(0, {"log_s": 0.0})
    assert not audit.violations, audit.violations


def test_no_module_on_the_prediction_path_imports_gate6_targets():
    for name in ("streams", "replay", "depth", "scale", "voxmap", "rays", "evidence",
                 "fuse", "config"):
        src = open(os.path.join(REPO, "gates", "gate7b", f"{name}.py")).read()
        assert "gate6.targets" not in src and "semantic_target" not in src, name


def test_no_optimizer_no_backward_no_training():
    banned = ("optim.", "Optimizer", ".backward(", "requires_grad_(True)", "loss",
              "GradScaler", ".train()", "fit(")
    for p in (glob.glob(os.path.join(REPO, "gates", "gate7b", "*.py"))
              + glob.glob(os.path.join(REPO, "tools", "gate7b", "*.py"))):
        t = ast.parse(open(p).read())
        for node in ast.walk(t):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        code = ast.unparse(ast.fix_missing_locations(t))
        for b in banned:
            assert b not in code, f"{os.path.basename(p)} contains {b!r}"


def test_lingbot_is_loaded_frozen():
    src = open(os.path.join(REPO, "gates", "gate7b", "replay.py")).read()
    assert "requires_grad_(False)" in src
    assert "LingBot parameters are trainable" in src


def test_gate6_and_gate7a_artifacts_are_unchanged():
    a = _json(os.path.join(ART, "stage0_audit.json"))
    assert a is not None, "run tools/gate7b/stage0.py"
    for gate, root in (("gate6", G6ART), ("gate7a", G7A)):
        for rel, want in a[gate]["artifacts"].items():
            p = os.path.join(root, rel)
            assert os.path.exists(p), f"{gate}/{rel} disappeared"
            assert _sha(p) == want, f"{gate}/{rel} was modified"
        pin = a[gate]["precommit"]
        assert _sha(os.path.join(REPO, pin["path"])) == pin["pinned"]
    for rel, want in a["reports"].items():
        assert _sha(os.path.join(REPO, rel)) == want, f"{rel} was modified"
    for ds, d in a["gate6"]["predictions"].items():
        man = _json(os.path.join(G6ART, f"prediction_manifest_{ds}.json"))
        assert man["rollup_sha256"] == d["rollup_sha256"]


def test_pre_existing_dirty_files_are_preserved():
    a = _json(os.path.join(ART, "stage0_audit.json"))
    for rel, rec in a["pre_existing_dirty"].items():
        assert _sha(os.path.join(REPO, rel)) == rec["sha256"], rel


def test_nothing_is_written_into_a_dataset_directory():
    forbidden = ("data/kitti/dataset", "/media/SSD1/MINH_DATASETS/sscbench_kitti360",
                 "/media/SSD1/MINH_DATASETS/nuscenes",
                 "/media/welf/MINH/datasets/kitti360",
                 "/media/SSD1/MINH_DATASETS/lingbot_gate6",
                 "/media/SSD1/MINH_DATASETS/lingbot_gate5_moge",
                 "/media/SSD1/MINH_DATASETS/lingbot_gate5_2")
    for p in (glob.glob(os.path.join(REPO, "gates", "gate7b", "*.py"))
              + glob.glob(os.path.join(REPO, "tools", "gate7b", "*.py"))):
        src = open(p).read()
        t = ast.parse(src)
        for node in ast.walk(t):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("savez_compressed", "savez", "save",
                                           "makedirs"):
                seg = ast.get_source_segment(src, node) or ""
                assert not any(f in seg for f in forbidden), f"{p}: {seg[:90]}"
    for root in forbidden:
        if os.path.isdir(root):
            assert not glob.glob(os.path.join(root, "**", "*gate7b*"), recursive=True)


def test_gate7b_precommit_matches_its_pin():
    pin = _json(os.path.join(ART, "precommit_pin.json"))
    assert pin is not None, "run tools/gate7b/precommit.py"
    assert _sha(os.path.join(REPO, pin["path"])) == pin["sha256"]
    text = open(os.path.join(REPO, pin["path"])).read()
    assert "DIAGNOSTIC ONLY" in text
    assert "ABLATION ONLY" in text


# --------------------------------------------------------------------------- #
# Results, when they exist
# --------------------------------------------------------------------------- #
def _eval(ds, tag):
    return _json(os.path.join(ART, f"eval_{ds}_{tag}.json"))


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_streaming_results_are_causal_and_complete(ds):
    d = _eval(ds, "S1_G-A_hall")
    if d is None:
        pytest.skip(f"{ds} S1 not produced yet")
    g6 = _json(os.path.join(G6ART, f"summary_{ds}.json"))
    assert d["n_anchors"] == g6["n_clips"], "every official timestamp must be evaluated"
    for r in d["per_clip"]:
        assert r["n_window"] >= 1 and r["n_window"] <= r["t"] + 1
        assert r["n_occ_lingbot"] + r["n_occ_moge_only"] == r["n_occ_eval"]


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_horizon_recall_is_non_decreasing_in_history(ds):
    got = [(h, _eval(ds, f"S1_G-A_h{h}")) for h in (1, 5, 20, 50, "all")]
    got = [(h, d) for h, d in got if d is not None]
    if len(got) < 2:
        pytest.skip(f"{ds} horizon sweep incomplete")
    rec = [d["summary"]["binary_recall"] for _h, d in got]
    assert rec == sorted(rec), f"{ds} recall fell with more causal history: {rec}"


@pytest.mark.parametrize("ds", list(C.DATASETS))
def test_moge_variants_only_add_evidence(ds):
    s1 = _eval(ds, "S1_G-A_hall")
    s3 = _eval(ds, "S3_G-A_hall")
    if s1 is None or s3 is None:
        pytest.skip(f"{ds} S1/S3 incomplete")
    assert s3["summary"]["binary_recall"] >= s1["summary"]["binary_recall"] - 1e-9
    assert sum(r["n_occ_moge_only"] for r in s1["per_clip"]) == 0
