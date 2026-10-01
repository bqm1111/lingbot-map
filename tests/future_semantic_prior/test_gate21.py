"""Gate 21 unit tests: the pre-registration cannot move, the model cannot see the future, and the frozen
definitions are the frozen ones.

    PYTHONPATH=$PWD python -m pytest tests/future_semantic_prior -q
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
import torch

from tools.delayed_semantic import config as C
from tools.future_revealability import config as G20
from tools.future_semantic_prior import config as G, features as FE, selection_bias as SB, targets as TG
from tools.future_semantic_prior.model import (ClassPriorHead, build_head, channel_spec, soft_distillation,
                                               uses_evidence)

CUDA = torch.cuda.is_available()
DEV = "cuda:0" if CUDA else "cpu"


# ---- pre-registration -------------------------------------------------------------------------------
def test_decision_rule_hash_is_unchanged():
    p = os.path.join(G.ARTIFACT_DIR, "decision_rule.md")
    want = open(os.path.join(G.ARTIFACT_DIR, "decision_rule.sha256")).read().strip()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    assert h.hexdigest() == want, "decision_rule.md was edited after it was hashed"


def test_thresholds_match_the_document():
    text = open(os.path.join(G.ARTIFACT_DIR, "decision_rule.md")).read()
    th = G.Thresholds()
    assert th.a_delta_e2_min == 1.0 and "+1.00" in text
    assert th.c_delta_e2_max == 0.25 and "+0.25" in text
    assert th.a_ci_lower_above == 0.0
    assert f"{th.b0_e2_miou:.3f}" in text, "the B0 reference value is not the one registered"
    assert G.PRIMARY_CONTROL == "C3_full"


def test_primary_metric_is_E2():
    """A typo that made E1 the decision population would invert the experiment's meaning."""
    assert "mIoU on `E2`" in open(os.path.join(G.ARTIFACT_DIR, "decision_rule.md")).read()


# ---- channels: nothing from the future, and a clean factorial ---------------------------------------
def test_channel_groups_are_disjoint():
    geo = {n for n, _ in G.GEOMETRY_CHANNELS}
    sem = {n for n, _ in G.SEMANTIC_CHANNELS}
    assert not (geo & sem), "C2/C3/C4 is only a factorial if the two groups do not overlap"
    assert set(FE.ALL_CHANNEL_NAMES) == geo | sem


def test_no_channel_is_derived_from_the_future():
    import re
    for control in G.LEARNED_CONTROLS:
        for name, _ in channel_spec(control):
            assert not re.search(r"future|reveal|delayed|target|sem_gt", name, re.I), name


def test_control_channel_sets():
    assert channel_spec("C1_prior") == ()
    assert channel_spec("C2_geometry") == G.GEOMETRY_CHANNELS
    assert channel_spec("C3_full") == G.GEOMETRY_CHANNELS + G.SEMANTIC_CHANNELS
    c4 = {n for n, _ in channel_spec("C4_semantics")}
    assert "trunk" not in c4 and "hard_S" not in c4, "C4 must carry no frozen latent features"
    assert "occupancy" in c4, "C4 still needs the occupancy mask to know which voxels it answers for"
    assert {"evidence_proj", "evidence_mask", "evidence_dist"} <= c4
    assert not uses_evidence("C2_geometry") and uses_evidence("C3_full") and uses_evidence("C4_semantics")


def test_parameter_counts_are_small_and_documented():
    want = {"C1_prior": 18, "C2_geometry": 223_506, "C3_full": 257_394, "C4_semantics": 256_498}
    for control, n in want.items():
        h = build_head(control)
        assert h.n_params() == n, f"{control} has {h.n_params():,} parameters, expected {n:,}"
        assert h.n_params() < 1_000_000, "section 7 caps the head at about 1M trainable parameters"
    assert build_head("C3_full").describe()["receptive_field_radius_m"] == 6.0


def test_unknown_control_is_rejected():
    with pytest.raises(KeyError):
        channel_spec("C9_whatever")


# ---- the loss ----------------------------------------------------------------------------------------
def test_soft_distillation_is_minimised_at_the_target():
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(64, 18), -1)
    exact = soft_distillation(torch.log(q), q)
    entropy = -(q * torch.log(q)).sum(-1).mean()
    assert torch.allclose(exact, entropy, atol=1e-5), "at p == q the loss must equal the target entropy"
    assert soft_distillation(torch.log(q) + 0.5, q).item() == pytest.approx(exact.item(), abs=1e-5), \
        "the loss must be invariant to a constant logit shift"
    worse = soft_distillation(torch.randn(64, 18), q)
    assert worse > exact


def test_soft_distillation_equals_kl_up_to_target_entropy():
    torch.manual_seed(1)
    q = torch.softmax(torch.randn(32, 18), -1)
    lg = torch.randn(32, 18)
    kl = torch.nn.functional.kl_div(torch.log_softmax(lg, -1), q, reduction="none").sum(-1).mean()
    ent = -(q * torch.log(q)).sum(-1).mean()
    assert soft_distillation(lg, q).item() == pytest.approx((kl + ent).item(), abs=1e-5)


def test_soft_distillation_is_empty_safe():
    assert soft_distillation(torch.zeros(0, 18), torch.zeros(0, 18)).item() == 0.0


def test_targets_are_never_hardened():
    """A hard target would make the gate a different experiment; check a real cached target is soft."""
    d = G.TRAIN_DRIVES[0]
    t = sorted(int(f[:-4]) for f in os.listdir(os.path.join(G.DATA_ROOT, "targets", d)))[0]
    q = TG.load_audit(d, t)["q_future"].astype(np.float64)
    assert q.shape[1] == 18 and np.abs(q.sum(1) - 1).max() < 2e-3
    assert q.max(1).mean() < 0.999, "the cached target is one-hot; it should be a probability vector"


# ---- the prior control -------------------------------------------------------------------------------
def test_class_prior_head_ignores_its_input():
    h = ClassPriorHead()
    idx = torch.arange(10)
    a = h.logits_at(torch.randn(1, 0, 4, 4, 4), idx)
    b = h.logits_at(torch.randn(1, 0, 4, 4, 4), idx)
    assert torch.equal(a, b) and a.shape == (10, 18)
    assert h.n_params() == 18


# ---- frozen conventions ------------------------------------------------------------------------------
def test_grid_and_bins_are_the_frozen_ones():
    assert tuple(int(d) for d in G.GRID_DIMS) == (256, 256, 32)
    assert G.N_VOXELS == 256 * 256 * 32 and G.N_CLASSES == 18
    assert G.DISTANCE_EDGES_M == (1.0, 2.0, 4.0, 8.0)
    assert G.REVEAL_LEVEL == "visible" and G.REVEAL_READOUT == "mean_probability"
    assert G.EVENTUAL_H == len(G20.HORIZONS_S) - 1 and G20.HORIZONS_S[G.EVENTUAL_H] == 10.0
    assert G.GEOMETRY_SEED == 0, "B0 must not move between controls or seeds"
    assert len(G.CLASS_NAMES) == 18 and G.THING_LABELS == (1, 2, 3, 4, 5, 6)


def test_training_firewall_forbids_the_dev_drive_and_all_ground_truth():
    f = " ".join(G.TRAIN_FORBIDDEN)
    for pat in ("drive_0006", "preprocess/labels", "eval_masks", "velodyne", "nuscenes", "occ3d_gt"):
        assert pat in f, f"{pat} is not on the training firewall's list"


def test_input_firewall_forbids_both_reveal_caches():
    f = " ".join(G.INPUT_FORBIDDEN)
    assert G.DATA_ROOT in f and "lingbot_gate20_future_reveal" in f


def test_train_anchors_never_include_the_dev_drive():
    from tools.future_semantic_prior.train import train_anchors
    items = train_anchors()
    assert items and all(d in G.TRAIN_DRIVES for d, _ in items)
    assert G.DEV_DRIVE not in {d for d, _ in items}


# ---- the mutation detector ---------------------------------------------------------------------------
def test_fingerprint_detects_a_single_element_change():
    x = torch.randn(3, 5, 7)
    f = FE.fingerprint(x)
    assert FE.fingerprint(x.clone()) == f
    y = x.clone()
    y[1, 2, 3] += 1e-3
    assert FE.fingerprint(y) != f


# ---- populations and targets -------------------------------------------------------------------------
def test_E1_and_E2_partition_E0_on_a_real_anchor():
    from tools.future_semantic_prior import evaluate as EV
    t = EV.dev_anchor_ids(stride=1)[0]
    rec = EV.dev_record(t)
    pops = EV.populations(rec, torch.device("cpu"))
    assert bool(pops["E0"].all())
    assert torch.equal(pops["E1"] | pops["E2"], pops["E0"])
    assert not bool((pops["E1"] & pops["E2"]).any())
    assert bool((pops["E1_P1"] <= pops["E1"]).all()) and bool((pops["E2_P1"] <= pops["E2"]).all())


def test_frozen_dev_population_sizes():
    from tools.future_semantic_prior.validate import FROZEN_DEV, FROZEN_TRAIN_UR, check_9_population_sizes
    r = check_9_population_sizes()
    assert r["dev"] == {k: FROZEN_DEV[k] for k in ("E0", "E1", "E2")}, r["dev"]
    assert r["UR_train"] == FROZEN_TRAIN_UR, r["UR_train"]


def test_training_targets_are_the_revealed_rows_of_P0():
    d = G.TRAIN_DRIVES[0]
    ts = sorted(int(f[:-4]) for f in os.listdir(os.path.join(G.DATA_ROOT, "targets", d)))[:5]
    for t in ts:
        a = TG.load_audit(d, t)
        with np.load(G20.population_path(d, t)) as z:
            p0, rev, mt = z["flat_index"], np.isfinite(z["vis_first_dt"]), z["mean_top1"][:, G.EVENTUAL_H]
        assert np.array_equal(a["flat_index"], p0[rev])
        dis = TG.argmax_disagreement(a["q_future"], mt[rev])
        assert dis["n_non_tie"] == 0, \
            "the float16 target changed an argmax that was not an exact tie, so it lost real information"


def test_load_targets_cannot_reach_the_ground_truth_column():
    d = G.TRAIN_DRIVES[0]
    t = sorted(int(f[:-4]) for f in os.listdir(os.path.join(G.DATA_ROOT, "targets", d)))[0]
    tb = TG.load_targets(d, t, torch.device("cpu"))
    assert set(vars(tb)) == {"flat_index", "q_future"}, "the training view of the cache gained a field"


# ---- selection bias ----------------------------------------------------------------------------------
def test_tv_distance_is_zero_against_itself():
    p = np.array([3, 1, 0, 6], np.int64)
    assert SB.tv_distance(p, p) == 0.0
    assert SB.tv_distance(np.array([1, 0]), np.array([0, 1])) == pytest.approx(100.0)


def test_geometry_of_flat_index_round_trips():
    X, Y, Z = (int(d) for d in C.GRID_DIMS)
    ix, iy, iz = 10, 200, 5
    f = np.array([(ix * Y + iy) * Z + iz])
    g = SB.geometry_of(f)
    assert g["height_m"][0] == pytest.approx((iz + 0.5) * 0.2 - 2.0)
    cx, cy = (ix + 0.5) * 0.2, (iy + 0.5) * 0.2 - 25.6
    assert g["range_m"][0] == pytest.approx(float(np.hypot(cx, cy)))


# ---- the additive change to Gate 20 is additive ------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_gate20_reveal_record_is_unchanged_by_the_soft_target_option():
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    from tools.delayed_semantic.stream import DriveStream
    from tools.delayed_semantic.targets import ViewCache
    from tools.future_revealability.future_visibility import reveal_record
    from tools.semantic_completion.evidence import trident as TR
    dev = torch.device(DEV)
    drive = G.TRAIN_DRIVES[0]
    t = sorted(int(f[:-4]) for f in os.listdir(os.path.join(G20.DATA_ROOT, "population", drive)))[0]
    with np.load(G20.population_path(drive, t)) as z:
        rows = torch.from_numpy(z["flat_index"].astype(np.int64)).to(dev)
        mt = z["mean_top1"][:, G.EVENTUAL_H]
    m = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)
    m[rows] = True
    cfg = C.TargetConfig()
    text, _, _ = TR.load_text_embeddings(C.Paths().g15.text_embeddings)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    views = ViewCache(geo, text.to(dev), cfg, dev)
    stream = DriveStream(drive)
    a = reveal_record(stream, geo, t, views, m, cfg, dev, want_mean_prob=False)
    b = reveal_record(stream, geo, t, views, m, cfg, dev, want_mean_prob=True)
    assert a.mean_prob is None and b.mean_prob is not None
    for f in ("flat_index", "geo_first_dt", "img_first_dt", "vis_first_dt", "vis_n_views", "mean_top1",
              "last_top1", "first_top1"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f"the option changed {f}"
    seen = np.isfinite(b.vis_first_dt)
    assert np.array_equal(b.mean_prob[seen].argmax(1).astype(np.uint8), mt[seen])


# ---- the head -----------------------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_logits_at_matches_a_full_forward():
    torch.manual_seed(0)
    h = build_head("C2_geometry").to(DEV).eval()
    x = torch.randn(1, h.cin, *[int(d) for d in G.GRID_DIMS], device=DEV)
    idx = torch.randint(0, G.N_VOXELS, (256,), device=DEV)
    with torch.no_grad():
        full = h(x)[0].reshape(G.N_CLASSES, -1).T[idx]
        got = h.logits_at(x, idx)
    assert torch.equal(full, got)


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_build_input_channel_order_matches_the_spec():
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    dev = torch.device(DEV)
    paths = C.Paths()
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    st = FE.frozen_causal_state(G.TRAIN_DRIVES[0], 4, net, text.to(dev), dev, paths)
    h = build_head("C3_full").to(dev)
    x = FE.build_input(st, h, dev)
    ch = FE.causal_channels(st, h, dev)
    o = 0
    for name, w in channel_spec("C3_full"):
        assert torch.equal(x[0, o:o + w], ch[name]), f"channel {name} is at the wrong offset"
        o += w
    assert o == x.shape[1] == h.cin


# ---- the trained controls (skipped until the checkpoints exist) ---------------------------------------
def _ck(control: str, seed: int) -> str:
    return G.checkpoint_path(control, seed, G.TrainConfig().steps)


@pytest.mark.skipif(not os.path.exists(_ck("C1_prior", 0)), reason="C1 is not trained yet")
def test_c1_predicts_one_constant_class():
    """If C1 ever predicted more than one class it would not be a class-prior control."""
    from tools.future_semantic_prior import evaluate as EV
    h = build_head("C1_prior")
    h.load_state_dict(torch.load(_ck("C1_prior", 0), map_location="cpu", weights_only=False)["state_dict"])
    h.eval()
    t = EV.dev_anchor_ids(stride=1)[0]
    rec = EV.dev_record(t)
    idx = torch.from_numpy(rec["flat_index"].astype(np.int64))
    with torch.no_grad():
        pred = h.logits_at(None, idx).argmax(1) + 1
    assert len(torch.unique(pred)) == 1
    assert 1 <= int(pred[0]) <= 18


@pytest.mark.skipif(not os.path.exists(_ck("C3_full", 0)), reason="C3 is not trained yet")
def test_the_reported_checkpoint_is_the_final_step():
    """No dev-metric selection: the step the analysis reads must be the last one trained."""
    steps = G.TrainConfig().steps
    rec = json.load(open(os.path.join(G.seed_dir(0), "train_C3_full.json")))
    assert rec["steps"] == steps
    assert max(ck["step"] for ck in rec["checkpoints"]) == steps
    assert os.path.exists(_ck("C3_full", 0))


@pytest.mark.skipif(not os.path.exists(_ck("C3_full", 0)), reason="C3 is not trained yet")
def test_every_run_kept_the_geometry_frozen():
    for control in G.LEARNED_CONTROLS:
        for seed in G.SEEDS:
            p = os.path.join(G.seed_dir(seed), f"train_{control}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            assert r["geometry_unchanged"] and r["geometry_grads_none"]
            assert r["firewall"]["violations"] == []
            assert r["head"]["params"] < 1_000_000
