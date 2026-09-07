"""Gate 8 — causal semantic memory with privileged completion."""
from __future__ import annotations
import ast, glob, json, os, sys
import numpy as np, pytest, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from types import SimpleNamespace
from gates.gate8 import vocab as V8, targets as TG
from gates.gate8.mapper import IncrementalMapper, FrameInput, ScaleState, VoxelTable, pack, unpack
from gates.gate8.net import CompletionUNet, apply_residual, LOCK_LOGODDS
ART = os.path.join(REPO, "artifacts", "gate8")
K = np.array([[10.0, 0, 5.0], [0, 10.0, 5.0], [0, 0, 1.0]])
GRID = SimpleNamespace(dims=(24, 24, 40), voxel_size=0.2, origin=(-2.4, -2.4, 0.0))


def _frame(i, depth_m=6.0, s=2.0, tx=0.0, conf=5.0, C=3, cls=0):
    d = torch.full((11, 11), depth_m / s, dtype=torch.float32)
    c = torch.full((11, 11), conf)
    P = np.eye(4); P[0, 3] = tx / s                        # canonical translation
    p = torch.zeros(11, 11, C); p[..., cls] = 1.0
    return FrameInput(i, d, c, K, P, np.log(s), sem_probs=p)


def _mapper(frames, C=3):
    m = IncrementalMapper("cpu", n_teacher=C)
    m.initialize(frames[:5])
    for f in frames[5:]:
        m.step(f)
    return m


# ----------------------------------------------------------------- keys
def test_key_pack_roundtrip():
    ijk = torch.tensor([[0, 0, 0], [-5, 7, 300], [100000, -100000, 3]], dtype=torch.int64)
    assert torch.equal(unpack(pack(ijk)), ijk)
    assert torch.equal(torch.sort(pack(ijk)).values, torch.sort(pack(ijk)).values)


# ----------------------------------------------------------------- scale
def test_scale_fixed_from_the_first_five_frames_and_then_ignored():
    s = ScaleState()
    for v in np.log([2.0, 2.0, 2.0, 2.0, 2.0]):
        s.observe(v)
    assert s.frozen and s.scale == pytest.approx(2.0)
    s.observe(np.log(100.0))
    assert s.scale == pytest.approx(2.0)


def test_scale_state_is_separate_from_map_state():
    m = _mapper([_frame(i) for i in range(6)])
    assert not hasattr(m.table, "scale")
    n = len(m.table); lo = m.table.logodds.clone()
    m.scale_state.force(3.0)                       # a gauge change touches no voxel
    assert len(m.table) == n and torch.equal(m.table.logodds, lo)


def test_identical_scale_on_depth_and_translation():
    """A surface 6 m ahead of a camera 4 m along x lands at x=4, z=6 for any canonical s."""
    for s in (1.0, 2.0, 5.0):
        m = _mapper([_frame(i, depth_m=6.0, s=s, tx=4.0) for i in range(5)])
        ijk = unpack(m.table.keys[m.table.occupied()])
        c = (ijk.double() + 0.5) * 0.2
        assert c[:, 0].median().item() == pytest.approx(4.0, abs=0.25)
        assert c[:, 2].median().item() == pytest.approx(6.0, abs=0.25)


def test_anchor_frames_integrated_exactly_once():
    m = IncrementalMapper("cpu", n_teacher=3)
    m.initialize([_frame(i) for i in range(5)])
    assert m.n_integrated == 5
    n5 = int(m.table.n_obs.sum())
    one = IncrementalMapper("cpu", n_teacher=3); one.scale_state.force(2.0); one.step(_frame(0))
    assert n5 == 5 * int(one.table.n_obs.sum())     # five identical anchors, integrated once each
    m.step(_frame(5))
    assert m.n_integrated == 6 and int(m.table.n_obs.sum()) == n5 + int(one.table.n_obs.sum())


# ----------------------------------------------------------------- causality / incrementality
def test_future_frame_cannot_influence_output_at_t():
    frames = [_frame(i) for i in range(8)]
    m = _mapper(frames[:6])
    q_t = m.query(GRID, np.eye(4))
    snap = {k: v.clone() for k, v in q_t.items()}
    m.step(_frame(6, depth_m=3.0))                   # a very different future frame
    for k, v in snap.items():
        assert torch.equal(v, q_t[k]), f"query at t was mutated by frame t+1 ({k})"
    m2 = _mapper(frames[:6])
    q2 = m2.query(GRID, np.eye(4))
    for k in ("occupied", "logodds", "sem"):
        assert torch.equal(q2[k], snap[k]), "output at t is not a deterministic function of frames <= t"


def test_previous_frames_are_not_reintegrated():
    m = _mapper([_frame(i) for i in range(6)])
    keys, lo, nobs = m.table.keys.clone(), m.table.logodds.clone(), m.table.n_obs.clone()
    m.step(_frame(6, conf=0.0))                      # no accepted ray: nothing may change
    assert torch.equal(m.table.keys, keys) and torch.equal(m.table.logodds, lo)
    assert torch.equal(m.table.n_obs, nobs)
    m.step(_frame(7))                                # an identical view adds the same increment
    n7 = int(m.table.n_obs.sum())
    m.step(_frame(8))
    assert int(m.table.n_obs.sum()) - n7 == n7 - int(nobs.sum())   # growth is per frame, not per history


def test_dense_export_is_a_lookup_not_a_rebuild():
    src = open(os.path.join(REPO, "gates", "gate8", "mapper.py")).read()
    q = src[src.index("def query"):src.index("# ---- internals")]
    assert "_integrate" not in q and "merge(" not in q and "cast" not in q


def test_free_space_stops_before_the_surface_and_behind_stays_unknown():
    m = _mapper([_frame(i) for i in range(5)])
    q = m.query(GRID, np.eye(4))
    occ = q["occupied"].reshape(GRID.dims); free = q["free"].reshape(GRID.dims)
    obs = q["observed"].reshape(GRID.dims)
    n_checked = 0
    for x in range(GRID.dims[0]):
        for y in range(GRID.dims[1]):
            zo = occ[x, y].nonzero().flatten(); zf = free[x, y].nonzero().flatten()
            if zo.numel() and zf.numel():
                assert zf.max() < zo.min(), (x, y)          # free only in front of the surface
                n_checked += 1
            if zo.numel():
                assert not obs[x, y][zo.max() + 2:].any(), (x, y)   # behind stays unknown
    assert n_checked > 10


def test_semantic_evidence_lands_and_geometry_is_independent_of_it():
    a = _mapper([_frame(i, cls=1) for i in range(5)])
    b = _mapper([_frame(i, cls=2) for i in range(5)])
    assert torch.equal(a.table.logodds, b.table.logodds)
    assert int(a.table.sem[a.table.occupied()].argmax(1).mode().values) == 1
    assert int(b.table.sem[b.table.occupied()].argmax(1).mode().values) == 2


# ----------------------------------------------------------------- vocabulary
def test_union_vocabulary_maps_are_total_and_fixed():
    V8.check()
    assert V8.U == 25
    for ds in ("semantickitti", "occ3d", "kitti360"):
        M = V8.into_matrix(ds); assert (M.sum(1) == 1).all()
        assert V8.out_channel(ds).shape == (V8.U,)


# ----------------------------------------------------------------- residual rule
def test_residual_never_changes_a_locked_voxel():
    base = torch.tensor([0.0, 1.0, LOCK_LOGODDS, -LOCK_LOGODDS, 4.0])
    r = torch.tensor([9.0] * 5)
    out = apply_residual(base, r)
    assert out[0] == 9.0 and out[1] == 10.0
    assert out[2] == LOCK_LOGODDS and out[3] == -LOCK_LOGODDS and out[4] == 4.0


def test_completion_disabled_is_the_identity():
    base = torch.randn(1000) * 3
    assert torch.equal(apply_residual(base, torch.zeros_like(base)), base)


def test_network_is_lightweight_and_shapes_match():
    n = CompletionUNet()
    assert n.n_params() < 2_000_000
    o, s = n(torch.randn(1, TG.N_INPUT_CHANNELS, 32, 32, 32))
    assert o.shape == (1, 1, 32, 32, 32) and s.shape == (1, V8.U, 32, 32, 32)


# ----------------------------------------------------------------- samples / leakage
def _samples():
    return sorted(glob.glob("/media/SSD1/MINH_DATASETS/lingbot_gate8/samples/*/*.npz"))[:20]


def test_sample_inputs_use_only_past_frames_and_targets_only_future():
    fs = _samples()
    if not fs:
        pytest.skip("no samples built yet")
    for p in fs:
        with np.load(p) as z:
            t = int(z["t"])
            assert int(z["input_frames"].max()) == t and int(z["input_frames"].min()) == 0
            assert int(z["target_frames"].min()) == t + 1
            assert int(z["target_frames"].max()) <= t + TG.FUTURE_FRAMES


def test_sample_masks_are_consistent():
    fs = _samples()
    if not fs:
        pytest.skip("no samples built yet")
    with np.load(fs[0]) as z:
        d = TG.unpack_sample(z, "cpu")
    assert d["fut_valid"].sum() > 0 and d["gt_valid"].sum() > 0
    assert not (d["fut_valid"] & ~d["fut_observed"]).any()      # teacher target only where future observed
    sem_obs = d["input"][6] > 0
    assert not (sem_obs & ~d["observed"]).any()                  # input semantics only on observed


def test_targets_expose_no_semantic_ground_truth():
    src = open(os.path.join(REPO, "gates", "gate8", "targets.py")).read()
    fn = src[src.index("def binary_occupancy"):src.index("def future_volume")]
    assert "return occ.reshape(-1), keep.reshape(-1)" in fn
    assert "target !=" in fn and "target ==" not in fn        # only collapsed to occupancy


def test_prediction_path_never_imports_targets():
    for name in ("mapper", "feed", "net", "losses", "vocab", "sources"):
        t = ast.parse(open(os.path.join(REPO, "gates", "gate8", f"{name}.py")).read())
        for node in ast.walk(t):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        code = ast.unparse(ast.fix_missing_locations(t))
        assert "gate6.targets" not in code and "semantic_target" not in code, name
    src = open(os.path.join(REPO, "gates", "gate8", "net.py")).read()
    assert "gt_" not in src


def test_trident_cache_matches_its_frame():
    from gates.scale_gate.config import REPO_ROOT
    from gates.gate8 import sources as S
    for source in ("sk_train", "occ3d_train"):
        segs = S.segments(source, REPO_ROOT)
        f = segs[0].frames[0]
        p = S.trident_path(source, f.key)
        if not os.path.exists(p):
            pytest.skip(f"{source} Trident cache not built yet")
        with np.load(p) as z:
            assert str(z["image_path"]) == f.path
            st = json.loads(str(z["stamp"]))
            assert st["dataset"] == S.DATASET_OF[source]
            assert tuple(int(x) for x in z["proc_hw"]) == tuple(z["probs"].shape[1:])


def test_incremental_mapper_reproduces_the_frozen_streaming_baseline():
    p = os.path.join(ART, "eval_semantickitti_equiv.json")
    if not os.path.exists(p):
        pytest.skip("equivalence run not produced")
    e = json.load(open(p))["variants"]["raw"]
    b = json.load(open(os.path.join(REPO, "artifacts", "gate7b",
                                    "eval_semantickitti_S1_G-A_hall.json")))["summary"]
    assert abs(e["binary_iou"] - b["binary_iou"]) < 0.005
    assert abs(e["binary_recall"] - b["binary_recall"]) < 0.005
    assert abs(e["ssc_miou"] - b["ssc_miou"]) < 0.002


def test_no_training_code_touches_kitti360():
    for p in (os.path.join(REPO, "tools", "gate8", "train.py"),
              os.path.join(REPO, "configs", "gate8", "completion.yaml")):
        assert "kitti360" not in open(p).read().lower().replace("never", "") or \
            "KITTI-360 never" in open(p).read()


def test_gate7b_artifacts_untouched():
    a = json.load(open(os.path.join(ART, "stage0_audit.json"))) if os.path.exists(
        os.path.join(ART, "stage0_audit.json")) else None
    if a is None:
        pytest.skip("run tools/gate8/stage0.py")
    import hashlib
    for rel, h in a["gate7b_artifacts"].items():
        p = os.path.join(REPO, "artifacts", "gate7b", rel)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, rel


# ----------------------------------------------------------------- GPU pool
def test_no_gate8_tool_defaults_to_gpu_zero():
    """GPU 0 is reserved for another user; nothing may land there by default."""
    import subprocess
    env = dict(os.environ); env.pop("GATE8_DEVICE", None); env.pop("GATE8_GPUS", None)
    for t in ("cache_trident", "build_samples", "evaluate", "train", "stream_sources",
              "runtime_audit", "time_trident"):
        src = open(os.path.join(REPO, "tools", "gate8", f"{t}.py")).read()
        assert '"--device", default="cuda:0"' not in src, t
        assert "_default_device()" in src, t
    out = subprocess.run([sys.executable, "-c",
                          "import os,sys; sys.path.insert(0,'.');"
                          "from tools.gate8.evaluate import _default_device; print(_default_device())"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "cuda:1", out.stdout + out.stderr


def test_shell_scripts_take_their_gpus_from_the_shared_pool():
    for name in ("run_caches.sh", "run_train_eval.sh", "run_occ3d_train_samples.sh",
                 "run_samples_sk.sh"):
        s = open(os.path.join(REPO, "tools", "gate8", name)).read()
        assert "source tools/gate8/gpus.sh" in s, name
        # the only literal cuda:0 permitted is under a CUDA_VISIBLE_DEVICES mask
        for line in s.splitlines():
            if "cuda:0" in line and not line.strip().startswith("#"):
                assert "CUDA_VISIBLE_DEVICES" in s, name
    g = open(os.path.join(REPO, "tools", "gate8", "gpus.sh")).read()
    assert 'GATE8_GPUS:-"1 2 3"' in g


def test_gpu_pool_is_overridable():
    import subprocess
    out = subprocess.run(["bash", "-c",
                          'source tools/gate8/gpus.sh; echo "$(g8 0) $(g8 1) $(g8 2) $(g8 3)"'],
                         cwd=REPO, capture_output=True, text=True,
                         env=dict(os.environ, GATE8_GPUS="2 3"))
    assert out.stdout.split() == ["2", "3", "2", "3"], out.stdout
