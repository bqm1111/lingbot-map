"""Gate 8B — leave-one-dataset-out transfer evaluation.

What has to hold: the two new folds cannot see their target anywhere; KITTI-360's training
and validation partitions are disjoint and official; every cached training sample's target
frames come strictly after its input frames; the matched five-frame setting keeps nothing
between clips; the OccAny post-processing is the official function; and the target
evaluator reads its checkpoint and threshold from the frozen manifest and nowhere else.
"""
from __future__ import annotations
import ast, glob, inspect, os, random, subprocess, sys
import numpy as np, pytest, torch, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from gates.gate8 import sources as S
from gates.gate8b import pooling as PL, sources as B
from sscbench_kitti360 import adapter as K3

OCCANY_PY = "/home/minh/workspace/third_party/occany_env/bin/python"
G8B_SAMPLES = f"{B.G8B_ROOT}/samples"


# ----------------------------------------------------------------- folds and partitions
def test_every_fold_excludes_its_target_from_train_and_val():
    for name, f in B.FOLDS.items():
        for src in f["train"] + f["val"]:
            assert S.DATASET_OF[src] != f["target"], (name, src)
        assert f["target"] == name


def test_fold_configs_match_gate8a_cell_b_except_sources():
    a = yaml.safe_load(open(os.path.join(REPO, "configs", "gate8a", "cellB_uniform_focal.yaml")))
    for tgt in ("semantickitti", "occ3d"):
        c = yaml.safe_load(open(os.path.join(REPO, "configs", "gate8b", f"fold_{tgt}.yaml")))
        for k in a:
            if k in ("train_sources", "val_sources"):
                continue
            assert c[k] == a[k], (tgt, k)
        assert c["sampler"] == "uniform" and c["occ_loss"] == "focal_dice"
        assert tgt not in c["train_sources"] + c["val_sources"]
        assert all(S.DATASET_OF[s] != tgt for s in c["train_sources"] + c["val_sources"])
        assert c["train_sources"] == B.FOLDS[tgt]["train"] and c["val_sources"] == B.FOLDS[tgt]["val"]


def test_kitti360_train_and_val_partitions_are_disjoint_and_official():
    assert B.K360_VAL_DRIVE not in B.K360_TRAIN_DRIVES
    assert B.K360_VAL_DRIVE in K3.OFFICIAL_SPLIT["val"]
    assert all(d in K3.OFFICIAL_SPLIT["train"] for d in B.K360_TRAIN_DRIVES)
    assert not any(d in K3.OFFICIAL_SPLIT["test"] for d in B.K360_TRAIN_DRIVES)


def test_registry_does_not_change_existing_source_paths():
    assert S.stream_path("sk_train", "00").startswith(S.G8_ROOT)
    assert S.stream_path("kitti360", "x").startswith("/media/SSD1/MINH_DATASETS/lingbot_gate7b")
    assert S.stream_path("k360_train", "x").startswith(B.G8B_ROOT)
    assert S.DATASET_OF["k360_train"] == "kitti360" and S.DATASET_OF["kitti360"] == "kitti360"


def test_target_loader_default_is_the_validation_drive():
    sig = inspect.signature(K3.load_target)
    assert sig.parameters["sequence"].default == K3.SEQUENCE == B.K360_VAL_DRIVE
    src = inspect.getsource(K3.load_target)
    assert 'f"{anchor:06d}_1_1.npy"' in src and "sequence" in src


@pytest.mark.skipif(not os.path.isdir(os.path.join(K3.__file__.rsplit("/", 2)[0])), reason="repo")
def test_k360_train_frames_never_come_from_the_val_drive():
    try:
        segs = S.segments("k360_train", REPO)
    except FileNotFoundError:
        pytest.skip("k360_train targets not fetched on this machine")
    for seg in segs:
        assert seg.name in B.K360_TRAIN_DRIVES and seg.name != B.K360_VAL_DRIVE
        for f in seg.frames:
            assert B.K360_VAL_DRIVE not in f.path
            if f.gt_ref is not None:
                assert f.gt_ref["sequence"] == seg.name
        o = [f.order for f in seg.frames]
        assert o == sorted(o) and len(set(o)) == len(o)


# ----------------------------------------------------------------- causal separation
def _sample_files(source, n=12):
    fs = sorted(glob.glob(f"{G8B_SAMPLES}/{source}/*.npz"))
    if not fs:
        pytest.skip(f"no cached {source} samples on this machine")
    random.Random(0).shuffle(fs)
    return fs[:n]


@pytest.mark.parametrize("source", ["k360_train", "kitti360"])
def test_cached_samples_record_disjoint_input_and_target_frames(source):
    for f in _sample_files(source):
        with np.load(f) as z:
            t = int(z["t"]); inp = z["input_frames"]; tgt = z["target_frames"]
            assert inp.max() == t and inp.min() == 0
            assert tgt.min() == t + 1 and tgt.max() <= t + 20
            assert not set(inp.tolist()) & set(tgt.tolist())
            assert len(tgt) >= 5


def test_k360_train_sample_targets_are_binary_only():
    for f in _sample_files("k360_train", 4):
        with np.load(f) as z:
            occ = np.unpackbits(z["gt_occ"])
            assert set(np.unique(occ).tolist()) <= {0, 1}
            assert "gt_label" not in z.files and "target" not in z.files


# ----------------------------------------------------------------- balanced draws
def test_fold_sampler_draws_each_source_with_probability_one_half():
    sys.argv = ["x"]
    from tools.gate8b.train import FoldSamples
    s = FoldSamples.__new__(FoldSamples)
    s.sources = ["a", "b"]; s.by_source = {"a": list(range(1000)), "b": list(range(10))}
    s._offset = {"a": 0, "b": 1000}; s.rng = random.Random(0)
    n = 20000
    hits = sum(s.draw() >= 1000 for _ in range(n)) / n
    assert abs(hits - 0.5) < 0.02, hits


# ----------------------------------------------------------------- matched five-frame setting
def test_clip_feed_builds_a_fresh_map_per_clip_with_no_carry_over():
    from gates.gate8b.clips import ClipFeed, build_map
    from gates.gate8.mapper import IncrementalMapper
    src = inspect.getsource(build_map)
    assert "IncrementalMapper(" in src and "force(clip.scale)" in src
    assert "n_integrated == len(clip.frames)" in src
    # the feed holds no mapper and no map state at all
    assert not any(isinstance(v, IncrementalMapper) for v in vars(ClipFeed).values())
    fsrc = inspect.getsource(ClipFeed)
    assert "IncrementalMapper" not in fsrc and "ScaleState" not in fsrc


def test_clip_scale_is_the_pinned_g51b_scalar():
    from gates.gate8b.clips import ClipFeed
    src = inspect.getsource(ClipFeed)
    assert "load_scales" in src and "moge" not in src.lower()


# ----------------------------------------------------------------- OccAny post-processing
@pytest.mark.skipif(not os.path.exists(OCCANY_PY), reason="OccAny environment not present")
def test_pooling_matches_the_official_occany_function(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.integers(0, 20, size=(18, 22, 14)).astype(np.uint8)
    x[rng.random(x.shape) < 0.55] = 0
    src = tmp_path / "in.npy"; dst = tmp_path / "out.npz"
    np.save(src, x)
    r = subprocess.run([OCCANY_PY, os.path.join(REPO, "tools", "gate8b",
                                                "occany_pooling_reference.py"), str(src), str(dst)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    z = np.load(dst)
    t = torch.from_numpy(x); occ = t != 0
    assert (PL.geometry_dilation(occ).numpy() == (z["geometry_dilation"] != 0)).all()
    assert (PL.geometry_majority(occ).numpy() == (z["geometry_majority"] != 0)).all()
    assert (PL.semantic_separate(t, 20, 19, 0).numpy() == z["semantic_separate"]).all()
    assert ((z["semantic_separate"] != 0) == (x != 0)).all(), "semantic pooling moved occupancy"


def test_occany_geometry_default_is_a_max_pool_not_a_vote():
    occ = torch.zeros(5, 5, 5, dtype=torch.bool); occ[2, 2, 2] = True
    assert int(PL.geometry_dilation(occ).sum()) == 27
    assert int(PL.geometry_majority(occ).sum()) == 1          # a single voxel never wins a vote


# ----------------------------------------------------------------- target exclusion in code
def test_target_evaluator_takes_no_checkpoint_or_threshold_argument():
    src = open(os.path.join(REPO, "tools", "gate8b", "eval_target.py")).read()
    tree = ast.parse(src)
    args = [c.value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "add_argument"
            for c in n.args if isinstance(c, ast.Constant)]
    assert "--fold" in args and "--setting" in args
    assert "--checkpoint" not in args and "--threshold" not in args
    assert "refusing to evaluate" in src and "frozen_fold_" in src


def test_no_target_data_in_the_selection_path():
    for f in ("select_fold.py", "train.py"):
        src = open(os.path.join(REPO, "tools", "gate8b", f)).read()
        assert "semantic_target" not in src and "G6T" not in src
    src = open(os.path.join(REPO, "tools", "gate8b", "select_fold.py")).read()
    assert "target not in SOURCES" in src


def test_gate8a_artifacts_are_untouched():
    import json
    a = json.load(open(os.path.join(REPO, "artifacts", "gate8a", "stage0_audit.json")))
    assert a.get("drift_since_first_run") == []
    for f in ("gate8a_results.json", "frozen_manifest.json", "checkpoints/cellB_uniform_focal_last.pt"):
        assert os.path.exists(os.path.join(REPO, "artifacts", "gate8a", f))
