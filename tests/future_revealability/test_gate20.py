"""Checks for the properties Gate 20's conclusion depends on.

Three groups: the pre-registration cannot have been edited after the fact; the population is the frozen one
and holds nothing the causal system observed; and the future record cannot leak into a prediction.

The GPU-dependent checks reuse the gate's own validators and skip where there is no device.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

from tools.delayed_semantic import config as C
from tools.delayed_semantic.stream import CausalHistory, DriveStream, FutureAccess
from tools.future_revealability import config as G, eval_masks as EM
from tools.semantic_completion import config as G15
from tools.semantic_completion.evidence import build as G15B
from tools.temporal_semantics import config as G18

DRIVE = "2013_05_28_drive_0007_sync"
#: frozen pilot record, artifacts/delayed_semantic_pilot/analysis.json (pooled/B0)
FROZEN_0006 = {"completed_never_observed": 2_743_212, "completed_later_revealed": 559_021}


def _records(drive: str, n: int = 12):
    d = os.path.join(G.DATA_ROOT, "population", drive)
    if not os.path.isdir(d):
        pytest.skip("Gate 20 population records not built")
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    return [os.path.join(d, f) for f in fs[:: max(1, len(fs) // n)][:n]]


# ---- the pre-registration -------------------------------------------------------------------------

def test_decision_rule_hash_is_the_one_registered_before_the_run():
    """If the thresholds moved after the results were seen, this fails."""
    md = os.path.join(G.ARTIFACT_DIR, "decision_rule.md")
    rec = os.path.join(G.ARTIFACT_DIR, "decision_rule.sha256")
    if not os.path.exists(rec):
        pytest.skip("no pre-registration on disk")
    want = open(rec).read().split()[0]
    assert hashlib.sha256(open(md, "rb").read()).hexdigest() == want


def test_thresholds_in_code_match_the_registered_document():
    T = G.Thresholds()
    text = open(os.path.join(G.ARTIFACT_DIR, "decision_rule.md")).read()
    for v in (T.a_p0_eventual, T.a_p1_eventual, T.a_far_eventual, T.c_p0_eventual, T.c_p1_eventual,
              T.c_far_eventual):
        assert f"{v:.0f} %" in text or f"{v:.0f}%" in text or f"**{v:.0f} %**" in text, v
    assert f"{T.a_teacher_miou:.1f}" in text and f"{T.c_teacher_miou:.1f}" in text


# ---- the frozen conventions ------------------------------------------------------------------------

def test_grid_and_bins_are_the_frozen_ones():
    assert tuple(C.GRID_DIMS) == (256, 256, 32) and C.VOXEL_SIZE_M == 0.2
    assert tuple(C.GRID_ORIGIN_M) == (0.0, -25.6, -2.0)
    assert G.DISTANCE_EDGES_M == G15.DISTANCE_EDGES_M == G18.DISTANCE_EDGES_M
    assert G.CLASS_NAMES == G18.CLASS_NAMES and G.THING_LABELS == G18.THING_LABELS
    assert G.TRAIN_DRIVES == G15.TRAIN_DRIVES and G.DEV_DRIVE == G15.VAL_DRIVE


def test_horizons_are_monotone_and_start_at_zero():
    assert G.HORIZONS_S[0] == 0.0
    assert list(G.HORIZONS_S) == sorted(G.HORIZONS_S)
    assert G.HORIZONS_S[-1] == G.EVENTUAL_S == C.TargetConfig().horizon_s
    assert G.DECISION_LEVEL in G.LEVELS and G.DECISION_LEVEL in G.DIAG_LEVELS


def test_eval_masks_reproduce_the_frozen_0006_masks_bit_exactly():
    r = EM.verify(n_anchors=4)
    assert r["bit_exact"], r["rows"]


# ---- the population --------------------------------------------------------------------------------

def test_nothing_is_revealed_at_horizon_zero():
    for p in _records(DRIVE):
        with np.load(p) as z:
            for k in ("geo_first_dt", "img_first_dt", "vis_first_dt", "dep_first_dt"):
                assert not (z[k] <= 0.0).any(), (p, k)


def test_reveal_levels_are_nested():
    """A camera-visible return is inside the image, and is a return. Violations mean a wiring error."""
    for p in _records(DRIVE):
        with np.load(p) as z:
            assert np.all(z["geo_first_dt"] <= z["img_first_dt"] + 1e-6)
            assert np.all(z["img_first_dt"] <= z["vis_first_dt"] + 1e-6)
            assert np.all(z["vis_first_dt"] <= z["dep_first_dt"] + 1e-2)


def test_P0_holds_no_voxel_the_causal_system_observed():
    paths = C.Paths()
    for p in _records(DRIVE, 6):
        t = int(os.path.basename(p)[:-4])
        with np.load(p) as z:
            p0 = z["flat_index"].astype(np.int64)
        with np.load(paths.g15.evidence_path(DRIVE, t)) as z:
            ev = z["flat_index"].astype(np.int64)
        assert len(np.intersect1d(p0, ev)) == 0


def test_P0_sem_gt_is_always_a_real_class():
    for p in _records(DRIVE):
        with np.load(p) as z:
            s = z["sem_gt"].astype(np.int64)
            assert s.min() >= 1 and s.max() <= 18


def test_population_sizes_reproduce_the_frozen_0006_regions():
    """P0 is the pilot's `completed_never_observed`; the deployable level is its `completed_later_revealed`."""
    d = os.path.join(G.DATA_ROOT, "population", G.DEV_DRIVE)
    if not os.path.isdir(d) or len(os.listdir(d)) < 590:
        pytest.skip("0006 records incomplete")
    n_p0 = n_dep = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(d, f)) as z:
            n_p0 += len(z["flat_index"])
            n_dep += int((z["dep_first_dt"] <= G.EVENTUAL_S).sum())
    assert n_p0 == FROZEN_0006["completed_never_observed"]
    assert n_dep == FROZEN_0006["completed_later_revealed"]


# ---- leakage ---------------------------------------------------------------------------------------

def test_causal_history_refuses_a_future_frame():
    stream = DriveStream(DRIVE)
    h = stream.history(100)
    assert h.frame(100).index == 100
    with pytest.raises(FutureAccess):
        h.frame(101)


def test_future_record_module_is_not_imported_by_the_prediction_path():
    """`causal_state` must not be able to reach the future record, even accidentally."""
    import inspect
    from tools.future_revealability import build_targets as BT
    src = inspect.getsource(BT.causal_state)
    for bad in ("reveal_record", "target_path", "load_mask", "sem_gt", "future"):
        assert bad not in src, bad


def test_prediction_path_opens_no_forbidden_file():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from core.datasets.audit import FileAudit
    from tools.future_revealability import build_targets as BT
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    dev = torch.device("cuda:0")
    paths = C.Paths()
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    t = int(G15B.anchor_ids(DRIVE, paths.g15)[20])
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True) as audit:
        BT.causal_state(DRIVE, t, net, text.to(dev), dev, paths)
    assert audit.violations == []


def test_opening_the_future_does_not_move_the_causal_state():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    from tools.delayed_semantic.targets import ViewCache
    from tools.future_revealability import build_targets as BT
    from tools.future_revealability.future_visibility import reveal_record
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    dev = torch.device("cuda:0")
    paths = C.Paths()
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    stream = DriveStream(DRIVE)
    geo = TF.DriveGeometry(DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    t = int(G15B.anchor_ids(DRIVE, paths.g15)[20])
    st = BT.causal_state(DRIVE, t, net, text, dev, paths)
    before = BT._state_digest(st)
    reveal_record(stream, geo, t, ViewCache(geo, text, C.TargetConfig(), dev), None, C.TargetConfig(), dev)
    assert BT._state_digest(st) == before
    assert BT._state_digest(BT.causal_state(DRIVE, t, net, text, dev, paths)) == before


def test_future_record_reproduces_the_frozen_delayed_targets():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from tools.future_revealability.future_visibility import validate_against_frozen_targets
    ids = [int(t) for t in G15B.anchor_ids(DRIVE, C.Paths().g15)[:: 150][:3]]
    r = validate_against_frozen_targets(DRIVE, ids, "cuda:0")
    assert r["reproduces_frozen_targets"], r["rows"]
