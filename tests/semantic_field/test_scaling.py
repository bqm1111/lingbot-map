"""Scaling study infrastructure: bounded loading, deterministic resume, source membership, budgets.

CPU tests run by default. The GPU equivalence test against the original trainers needs the caches and a GPU:
    B4_SCALING_GPU_TESTS=1 PYTHONPATH=$PWD python -m pytest tests/semantic_field/test_scaling.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from tools.semantic_field.scaling.anchor_store import AnchorLoadError, DiskAnchorSequence, tensor_bytes      # noqa: E402
from tools.semantic_field.scaling.b4_step import StepOutput                                                    # noqa: E402
from tools.semantic_field.scaling.config import OriginalBudgets, stretched_steps                               # noqa: E402
from tools.semantic_field.scaling.resumable import OptimSpec, draw_digest, train_resumable                     # noqa: E402
from tools.semantic_field.scaling.source_pool import SourcePoolError, check_membership                          # noqa: E402

KEYS = [("2013_05_28_drive_0003_sync", t) for t in range(10)]


def _fake_loader(key):
    torch.manual_seed(key[1])
    return {"drive": key[0], "t": key[1], "feature": torch.randn(100, 16).half(), "flat": torch.arange(100)}


def test_disk_sequence_returns_loader_values_and_respects_the_host_cache_bound():
    one = tensor_bytes(_fake_loader(KEYS[0]))
    seq = DiskAnchorSequence(KEYS, _fake_loader, lambda k: [f"/x/{k[1]}.npz"], "cpu", host_cache_bytes=3 * one)
    for i in [0, 1, 2, 3, 0, 4, 5, 1, 9, 9]:
        a, ref = seq[i], _fake_loader(KEYS[i])
        assert a["t"] == ref["t"] and torch.equal(a["feature"], ref["feature"]) and a["feature"].dtype == torch.float16
        assert seq.stats.bytes_now <= 3 * one
    assert seq.stats.bytes_peak <= 3 * one and seq.stats.evictions > 0 and seq.stats.hits >= 1


def test_an_anchor_larger_than_the_cache_is_loaded_but_not_cached():
    seq = DiskAnchorSequence(KEYS, _fake_loader, lambda k: [], "cpu", host_cache_bytes=10)
    assert torch.equal(seq[2]["feature"], _fake_loader(KEYS[2])["feature"])
    assert seq.stats.bytes_now == 0 and seq.stats.misses == 1


def test_load_failures_name_the_anchor_and_files_and_are_not_skipped():
    def broken(key):
        raise KeyError("feature")
    seq = DiskAnchorSequence(KEYS, broken, lambda k: [f"/data/evidence/{k[0]}/{k[1]:05d}.npz"], "cpu", 0)
    with pytest.raises(AnchorLoadError, match=r"drive=2013_05_28_drive_0003_sync t=4.*00004\.npz.*KeyError"):
        seq[4]


def test_update_budgets_follow_the_declared_formula():
    budgets = OriginalBudgets()
    assert (budgets.b4_steps, budgets.b3_steps) == (6000, 3000)
    assert stretched_steps(6000, 9066) == 40056 and stretched_steps(3000, 9066) == 20028
    assert stretched_steps(6000, 1358) == 6000
    with pytest.raises(ValueError):
        stretched_steps(6000, 1000)


@pytest.mark.parametrize("drive", ["2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync", "08"])
def test_development_reserved_and_foreign_drives_cannot_enter_a_pool(drive):
    with pytest.raises(SourcePoolError):
        check_membership((("2013_05_28_drive_0000_sync", 10), (drive, 11)))


def test_duplicate_anchors_are_rejected_and_training_drives_accepted():
    check_membership((("2013_05_28_drive_0000_sync", 10), ("2013_05_28_drive_0010_sync", 10)))
    with pytest.raises(SourcePoolError):
        check_membership((("2013_05_28_drive_0000_sync", 10), ("2013_05_28_drive_0000_sync", 10)))


def _toy_run(tmp_path, name, total, stop=None, every=5):
    torch.manual_seed(0); model = torch.nn.Linear(4, 1)
    data = torch.randn(64, 4, generator=torch.Generator().manual_seed(1))

    def step(gen, gen_d):
        i = torch.randint(64, (8,), generator=gen); j = torch.randperm(8, generator=gen_d)[:4]
        loss = model(data[i][j]).pow(2).mean()
        return StepOutput(loss, {"loss": float(loss)}, [f"{i.tolist()}:{j.tolist()};"])

    return model, train_resumable(model, step, total, 3, OptimSpec(1e-2, 0.01, 1.0), str(tmp_path / name), every, 1000, stop, name)


def test_resume_reproduces_the_uninterrupted_sequence(tmp_path):
    m1, s1 = _toy_run(tmp_path, "straight.pt", 20)
    _toy_run(tmp_path, "split.pt", 20, stop=7)
    m2, s2 = _toy_run(tmp_path, "split.pt", 20)                     # resumes at 7 from the checkpoint written at the stop
    assert s1.draws == s2.draws and draw_digest(s1.draws) == draw_digest(s2.draws)
    assert [r["loss"] for r in s1.log] == [r["loss"] for r in s2.log]
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))


def test_resume_refuses_a_checkpoint_of_another_run(tmp_path):
    _toy_run(tmp_path, "run.pt", 10, stop=5)
    with pytest.raises(RuntimeError, match="total_steps"):
        _toy_run(tmp_path, "run.pt", 12)


@pytest.mark.skipif(not os.environ.get("B4_SCALING_GPU_TESTS") or not torch.cuda.is_available(), reason="needs caches + GPU")
def test_disk_backed_b4_reproduces_the_original_trainer(tmp_path):
    from tools.semantic_field.scaling import verify_loader as V
    keys = V.selection()[:6]
    rec = V.verify_b4(keys, "cuda:0", 4, str(tmp_path))
    assert rec["draw_digest_equal"] and rec["original_vs_new"]["log"]["target_counts_equal"]
    first_step = {k: v for k, v in rec["original_vs_new"]["tensors"].items() if k.startswith("step1_")}
    assert all(v for k, v in first_step.items() if k.endswith("_exact")) and all(v == 0.0 for k, v in first_step.items() if k.endswith("_diff"))
    assert rec["params_finite"] and rec["cache"]["bytes_peak"] <= rec["cache_bound_bytes"]


def test_registration_refuses_reserved_and_foreign_drives_and_is_idempotent():
    from core.datasets import kitti360_folds as KF
    from tools.semantic_field.scaling import registration as R
    for bad in ("2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync", "08"):
        with pytest.raises(RuntimeError):
            R.check_added_drives((bad,))
    before = tuple(KF.K360_TRAIN_DRIVES)
    try:
        once = R.register_added_drives(); twice = R.register_added_drives()
        assert once == twice and len(set(once)) == len(once) and set(R.ADDED_DRIVES) <= set(once)
    finally:
        KF.K360_TRAIN_DRIVES = before


def test_storage_never_relocates_existing_drives_or_replaces_real_directories(tmp_path, monkeypatch):
    from tools.semantic_field.scaling import storage as ST
    with pytest.raises(RuntimeError, match="not an added drive"):
        ST.place("evidence", "2013_05_28_drive_0003_sync")
    fixed = tmp_path / "fixed"; (fixed / "2013_05_28_drive_0005_sync").mkdir(parents=True)     # a real directory already there
    monkeypatch.setitem(ST.CACHE_KINDS, "evidence", (lambda: str(fixed), "hot"))
    monkeypatch.setattr(ST, "HOT_ROOTS", (str(tmp_path / "hot"),)); monkeypatch.setattr(ST, "LAYOUT_PATH", str(tmp_path / "layout.json"))
    monkeypatch.setattr(ST, "MIN_FREE_GB", 0.0)                                  # the temp dir's disk is not the declared hot tier
    with pytest.raises(RuntimeError, match="real directory"):
        ST.place("evidence", "2013_05_28_drive_0005_sync", str(tmp_path / "hot"))
    placed = ST.place("evidence", "2013_05_28_drive_0000_sync", str(tmp_path / "hot"))
    assert os.path.islink(placed.link) and os.path.realpath(placed.link) == os.path.realpath(placed.target)
    assert ST.place("evidence", "2013_05_28_drive_0000_sync", str(tmp_path / "hot")).target == placed.target   # idempotent


def test_gpu_time_counts_overlapping_jobs_on_one_gpu_once():
    from tools.semantic_field.scaling.resources import union_seconds
    assert union_seconds([(0, 10), (5, 15), (20, 30)]) == 25          # two overlapping jobs + one later job
    assert union_seconds([(0, 10), (0, 10), (0, 10)]) == 10           # three runs packed on one GPU
    assert union_seconds([]) == 0
