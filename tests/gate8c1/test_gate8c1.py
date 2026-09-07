"""Gate 8C-1 — KITTI-360-only training with untouched SemanticKITTI and Occ3D evaluation.

Two jobs. First, prove the dataset firewall: no SemanticKITTI or Occ3D path is reachable
from any training, validation, selection or calibration code, and the runtime file audit
that guards the training run actually fires. Second, prove the rebuilt supervision is what
the report says: raw-LiDAR endpoints agree with it by construction, ground-truth poses and
future sweeps never reach the inference path, scale is fixed once, and every frame is
integrated exactly once.
"""
from __future__ import annotations
import ast, glob, inspect, json, os, sys
import numpy as np, pytest, torch, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from gates.gate8c0 import oracle as OR, transforms as TF
from gates.gate8c1 import rawtarget as RT, sources as SRC
from sscbench_kitti360.audit import FileAudit, ForbiddenAccess

TOOLS = os.path.join(REPO, "tools", "gate8c1")
#: every module that runs before the firewall lifts
PRE_FIREWALL = ["build_targets.py", "build_samples.py", "train.py", "selection.py",
                "validate_targets.py"]
TARGET_TOKENS = ("semantickitti", "occ3d", "nuscenes", "semantic_kitti", "kitti/dataset")
_has_data = os.path.isdir(os.path.join(SRC.SSCBENCH_ROOT, "data_poses", SRC.VAL_DRIVE))
needs_data = pytest.mark.skipif(not _has_data, reason="KITTI-360 not present")
_has_samples = bool(glob.glob(os.path.join(
    os.path.dirname(SRC.sample_path(SRC.TRAIN_DRIVES[0], 0)), "*.npz")))
needs_samples = pytest.mark.skipif(not _has_samples, reason="Gate 8C-1 samples not built")


# ----------------------------------------------------------------- dataset firewall
def _code_strings_and_names(path):
    """Every identifier and every non-docstring string constant of a module.

    Prose that *describes* the firewall is not a violation; a path or an identifier that
    reaches a target dataset is. Docstrings are excluded for exactly that reason.
    """
    tree = ast.parse(open(path).read())
    docstrings = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(n, clean=False)
            if d is not None:
                docstrings.add(d)
    names, strings = set(), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value not in docstrings:
                strings.append(n.value)
    return names, strings


@pytest.mark.parametrize("mod", PRE_FIREWALL)
def test_no_target_dataset_path_or_identifier_before_the_firewall(mod):
    """A pre-firewall module may *mention* the target datasets in prose but must never
    name one in an identifier or in anything path-shaped."""
    names, strings = _code_strings_and_names(os.path.join(TOOLS, mod))
    bad_names = {n for n in names if any(t in n.lower() for t in TARGET_TOKENS)}
    assert not bad_names, f"{mod}: identifiers {bad_names}"
    import re
    # path-shaped == absolute, or a real filename, or three-plus slash-separated segments.
    # "SemanticKITTI / Occ3D" in a log message is prose, not a path.
    def path_shaped(v):
        return (v.startswith("/") or v.startswith("~")
                or re.search(r"\.(npy|npz|bin|png|pt|json|txt)\b", v) is not None
                or re.search(r"[\w.-]+(/[\w.-]+){2,}", v) is not None)
    bad_paths = [v for v in strings
                 if path_shaped(v) and any(t in v.lower() for t in TARGET_TOKENS)
                 and v not in list(SRC.FORBIDDEN_PATH_TOKENS)]
    assert not bad_paths, f"{mod}: path-shaped strings {bad_paths[:4]}"


@pytest.mark.parametrize("mod", PRE_FIREWALL)
def test_pre_firewall_modules_do_not_import_target_loaders(mod):
    tree = ast.parse(open(os.path.join(TOOLS, mod)).read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {x.name for x in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module)
    # gate6.targets resolves benchmark labels for all three datasets; it must not appear
    assert "gate6.targets" not in imported, f"{mod} imports the benchmark target loader"


def test_firewall_predicate_rejects_target_paths():
    for p in ("/media/SSD1/MINH_DATASETS/nuscenes/x.npz",
              "data/kitti/dataset/sequences/08/velodyne/000000.bin",
              "/media/SSD1/MINH_DATASETS/sscbench_kitti360/preprocess/labels/d/000000_1_1.npy"):
        with pytest.raises(SRC.TargetAccessViolation):
            SRC.assert_no_target_access([p])
    SRC.assert_no_target_access(["/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/x/0.npz",
                                 SRC.TRAIN_DRIVES[0], SRC.VAL_DRIVE])


def test_runtime_file_audit_actually_fires():
    """The guard the training run executes inside must raise, not merely record."""
    with pytest.raises(ForbiddenAccess):
        with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True):
            open("/media/SSD1/MINH_DATASETS/nuscenes/does_not_exist.json")


def test_sscbench_completion_labels_are_never_read():
    """Gate 8C-0 disqualified `_1_1.npy`; no Gate 8C-1 code may open one."""
    assert "preprocess/labels" in SRC.FORBIDDEN_PATH_TOKENS
    for mod in PRE_FIREWALL + ["eval_target.py"]:
        _, strings = _code_strings_and_names(os.path.join(TOOLS, mod))
        assert not [v for v in strings if "_1_1" in v], mod


def test_target_evaluator_refuses_without_a_frozen_manifest():
    src = open(os.path.join(TOOLS, "eval_target.py")).read()
    assert "refusing to open" in src and "frozen_manifest.json" in src
    tree = ast.parse(src)
    args = [c.value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "add_argument"
            for c in n.args if isinstance(c, ast.Constant)]
    assert "--checkpoint" not in args and "--threshold" not in args
    assert "--dataset" in args and "--mode" in args and "--seed" in args


def test_all_three_seeds_share_one_configuration_except_seed():
    cfg = {s: yaml.safe_load(open(os.path.join(REPO, "configs", "gate8c1", f"seed{s}.yaml")))
           for s in (0, 1, 2)}
    keys = set().union(*[set(v) for v in cfg.values()])
    diff = {k for k in keys if len({repr(v.get(k)) for v in cfg.values()}) > 1}
    assert diff == {"seed"}, diff
    for c in cfg.values():
        assert c["train_drives"] == list(SRC.TRAIN_DRIVES) and c["val_drive"] == SRC.VAL_DRIVE
        for d in c["train_drives"] + [c["val_drive"]]:
            assert "drive" in d and d.startswith("2013_05_28_")


def test_training_drives_and_validation_drive_are_disjoint():
    assert SRC.VAL_DRIVE not in SRC.TRAIN_DRIVES
    from sscbench_kitti360 import adapter as K3
    assert all(d in K3.OFFICIAL_SPLIT["train"] for d in SRC.TRAIN_DRIVES)
    assert SRC.VAL_DRIVE in K3.OFFICIAL_SPLIT["val"]


# ----------------------------------------------------------------- rebuilt supervision
def test_conflicting_evidence_becomes_unknown_not_a_vote():
    src = inspect.getsource(RT.build)
    assert "state[fre & ~occ] = FREE" in src and "state[occ & ~fre] = OCCUPIED" in src
    assert "conflict stays UNKNOWN" in src


def test_a_sweep_never_carves_the_voxel_it_measured():
    src = inspect.getsource(RT.build)
    assert "sweep_free[hit] = 0" in src


@needs_data
def test_rebuilt_target_agrees_with_its_own_sweep_by_construction():
    geo = TF.DriveGeometry(SRC.TRAIN_DRIVES[1], SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    ancs = [a for a in SRC.anchors(SRC.TRAIN_DRIVES[1], REPO) if a.n_future >= 10][:2]
    if not ancs:
        pytest.skip("no eligible anchors")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for anc in ancs:
        t = RT.build(geo, anc.native_frame,
                     [anc.native_frame] + list(anc.future_natives), dev)
        idx, _ = TF.voxelize(geo.read_velodyne(anc.native_frame))
        flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), TF.DIMS)
        sup = t.valid[flat]
        assert sup.any()
        assert t.occupied[flat][sup].all(), "a supervised endpoint is not OCCUPIED"
        assert not (t.free[flat]).any(), "an endpoint was marked FREE"


@needs_data
def test_rebuilt_target_has_no_systematic_one_voxel_offset():
    """The test that failed for SSCBench's label must pass for the rebuilt one."""
    geo = TF.DriveGeometry(SRC.TRAIN_DRIVES[1], SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    anc = [a for a in SRC.anchors(SRC.TRAIN_DRIVES[1], REPO) if a.n_future >= 20][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = RT.build(geo, anc.native_frame, [anc.native_frame] + list(anc.future_natives), dev)
    vol = OR.volume_from_points(geo.read_velodyne(anc.native_frame))
    sc = OR.shift_scan(vol, t.occupied.reshape(TF.DIMS), t.valid.reshape(TF.DIMS))
    assert max(sc.items(), key=lambda kv: kv[1]["iou"])[0] == "(0, 0, 0)"


@needs_data
def test_occupancy_recall_grows_with_future_sweeps():
    geo = TF.DriveGeometry(SRC.TRAIN_DRIVES[0], SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    anc = [a for a in SRC.anchors(SRC.TRAIN_DRIVES[0], REPO) if a.n_future >= 20][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full = RT.build(geo, anc.native_frame, [anc.native_frame] + list(anc.future_natives), dev)
    rec = []
    for k in (1, 5, 20):
        sub = RT.build(geo, anc.native_frame,
                       [anc.native_frame] + list(anc.future_natives[:k]), dev)
        rec.append((sub.occupied & full.occupied).sum() / max(full.occupied.sum(), 1))
    assert all(b >= a - 1e-9 for a, b in zip(rec, rec[1:])), rec
    assert rec[-1] > rec[0]


# ----------------------------------------------------------------- causality
@needs_samples
def test_cached_samples_are_causal_and_use_only_future_frames_for_targets():
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(SRC.sample_path(SRC.TRAIN_DRIVES[1], 0)), "*.npz")))[::97][:8]
    for p in files:
        with np.load(p) as z:
            t = int(z["t"]); inp = z["input_frames"]; fut = z["target_frames"]
            assert inp.min() == 0 and inp.max() == t
            assert fut.min() == t + 1 and fut.max() <= t + SRC.FUTURE_FRAMES
            assert not (set(inp.tolist()) & set(fut.tolist()))
            assert set(np.unique(np.unpackbits(z["gt_occ"])).tolist()) <= {0, 1}


def test_ground_truth_poses_and_future_lidar_are_absent_from_the_inference_path():
    """`rawtarget` is the only module allowed to touch them, and nothing on the inference
    path imports it."""
    for mod in ("gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8/feed.py",
                "tools/gate8c1/eval_target.py"):
        src = open(os.path.join(REPO, mod)).read()
        assert "rawtarget" not in src, mod
        assert "read_velodyne" not in src, mod
        assert "cam0_to_world" not in src, mod


def test_scale_is_fixed_once_and_applied_identically_to_depth_and_translation():
    from gates.gate8.mapper import IncrementalMapper, ScaleState
    s = ScaleState()
    for v in np.log([2.0] * 5):
        s.observe(v)
    assert s.frozen and abs(s.scale - 2.0) < 1e-9
    s.observe(np.log(100.0))
    assert abs(s.scale - 2.0) < 1e-9, "scale moved after freezing"
    src = inspect.getsource(IncrementalMapper._integrate)
    assert "d_m = s * f.depth_canonical" in src
    assert "T[:3, 3] *= s" in src and "identical scalar on translation" in src


def test_each_frame_is_integrated_exactly_once():
    from tools.gate8c1.eval_target import window_map
    src = inspect.getsource(window_map)
    assert "m.step(feed.frame(j))" in src
    assert src.count("m.step(") == 1
    tree = ast.parse(inspect.getsource(window_map))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    assert len(loops) == 1, "more than one loop could integrate a frame twice"


def test_protocol_offsets_are_causal_or_explicitly_labelled_non_causal():
    from tools.gate8c1 import eval_target as ET
    assert max(ET.PAST5_OFFSETS) == 0 and len(ET.PAST5_OFFSETS) == 5
    assert min(ET.OCCANY_FWD_OFFSETS) == 0 and max(ET.OCCANY_FWD_OFFSETS) > 0
    src = open(os.path.join(TOOLS, "eval_target.py")).read()
    assert "non-causal" in src
    assert '"causal": mode in ("stream", "past5")' in src


# ----------------------------------------------------------------- prior artifacts
def test_gate8c0_artifacts_remain_unchanged():
    p = os.path.join(REPO, "artifacts", "gate8c0", "hashes.json")
    if not os.path.exists(p):
        pytest.skip("Gate 8C-0 hashes not present")
    import hashlib
    h = json.load(open(p))
    bad = []
    for rel, rec in h["frozen"].items():
        # recorded before the gate packages moved into gates/; resolve either layout so
        # the check cannot pass vacuously just because the old path stopped resolving
        f = os.path.join(REPO, rel)
        if not os.path.exists(f):
            f = os.path.join(REPO, "gates", rel)
        assert os.path.exists(f), f"Gate 8C-0 frozen file is missing entirely: {rel}"
        if rec:
            d = hashlib.sha256(open(f, "rb").read()).hexdigest()
            if d != rec["sha256"]:
                bad.append(rel)
    assert not bad, f"Gate 8C-0 frozen files changed: {bad}"


# ----------------------------------------------------------------- OccAny comparison
def test_occany_official_metric_agrees_with_our_counting():
    """The SC numbers are computed by OccAny's own evaluator; ours must match it exactly."""
    from gates.gate8c1 import occany_eval as OE
    if not OE.available():
        pytest.skip("OccAny checkout not present")
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 20, size=(1, 16, 16, 8)).astype(np.uint8)
    tgt = rng.integers(0, 20, size=(1, 16, 16, 8)).astype(np.uint8)
    tgt[rng.random(tgt.shape) < 0.2] = 255
    tp, fp, fn = OE.official_sc_counts(pred, tgt, n_classes=20, empty_class=0)
    keep = tgt != 255
    p = (pred != 0) & keep; g = (tgt != 0) & keep
    assert (tp, fp, fn) == (int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()))


def test_occany_pooling_default_is_a_max_pool_not_a_vote():
    from gates.gate8b import pooling as PL
    occ = torch.zeros(5, 5, 5, dtype=torch.bool); occ[2, 2, 2] = True
    assert int(PL.geometry_dilation(occ).sum()) == 27
    assert int(PL.geometry_majority(occ).sum()) == 1


def test_published_occany_numbers_are_quoted_not_recomputed():
    from gates.gate8c1 import occany_eval as OE
    assert OE.PUBLISHED_5FRAME["semantickitti"]["sc_iou"] == 0.2591
    assert OE.PUBLISHED_5FRAME["occ3d"]["sc_iou"] == 0.2355
    doc = open(os.path.join(REPO, "artifacts", "gate8c1", "occany_reproduction.md")).read()
    assert "could not be run" in doc and "not run" in doc
