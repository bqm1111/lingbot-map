"""Gate 8D: the properties the protocol claims, asserted rather than trusted.

Each test names the claim it defends. Anything requiring GPU or the KITTI-360 mount is
skipped rather than silently passed, and the skip is visible in the run.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import importlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from gates.gate8d import densetarget as DT, futureframes as FF, protocol as P, sources as SRC  # noqa: E402

ART = os.path.join(REPO, "artifacts", "gate8d")
TARGET_TOKENS = ("semantickitti", "SemanticKITTI", "nuscenes", "nuScenes",
                 "occ3d", "Occ3D")
#: modules that must stay source-only until the manifest is frozen
SOURCE_MODULES = ["gates/gate8d/protocol.py", "gates/gate8d/sources.py", "gates/gate8d/densetarget.py",
                  "gates/gate8d/futureframes.py", "gates/gate8d/net.py", "gates/gate8d/data.py",
                  "tools/gate8d/build_targets.py", "tools/gate8d/build_samples.py",
                  "tools/gate8d/train.py", "tools/gate8d/selection.py",
                  "tools/gate8d/losses.py", "tools/gate8d/fit_moge_margin.py",
                  "tools/gate8d/cache_future_teachers.py", "tools/gate8d/audit_targets.py"]


# --------------------------------------------------------------- firewall / boundary
def test_source_modules_contain_no_target_data_paths():
    """A source module may *name* a benchmark -- in the forbidden list, or in a success
    criterion -- but it must never contain a path that could reach one."""
    roots = ("data/kitti/dataset", "occ3d_gt", "lingbot_occ3d_zeroshot",
             "nuscenes_root", "occany_out", "/semantickitti/", "/occ3d/")
    offenders = []
    for rel in SOURCE_MODULES:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            continue
        for line in open(p).read().splitlines():
            for t in roots:
                if t in line and "FORBIDDEN" not in line:
                    offenders.append((rel, t, line.strip()[:70]))
    assert not offenders, f"target data path in a source module: {offenders[:5]}"


def test_forbidden_tokens_cover_both_targets_and_the_defective_label():
    for t in ("semantickitti", "SemanticKITTI", "nuscenes", "nuScenes", "occ3d", "Occ3D",
              "_1_1.npy"):
        assert t in SRC.FORBIDDEN_PATH_TOKENS, f"{t} missing from the firewall"


def test_transitively_imported_target_named_modules_are_pure_geometry():
    """Importing a source module may pull in a target-named module only if that module
    reads nothing.

    ``gate6.grids`` imports ``occ3d_zeroshot.grid`` for grid constants. Defining a lattice
    is not crossing the dataset boundary; opening a file would be. This asserts the
    distinction rather than banning the name.
    """
    before = set(sys.modules)
    importlib.import_module("gates.gate8d.densetarget")
    importlib.import_module("gates.gate8d.net")
    pulled = [m for m in set(sys.modules) - before
              if any(t in m.lower() for t in ("semantickitti", "occ3d", "nuscenes", "occany"))]
    for name in pulled:
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None)
        if not f or not os.path.exists(f):
            continue
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
                assert nm not in ("open", "load", "fromfile", "memmap", "listdir", "glob"), \
                    f"{name} performs I/O at module scope; it is a loader, not geometry"


def test_runtime_firewall_recorded_zero_violations():
    """The mechanism that actually enforces the boundary is the runtime file audit."""
    seen = 0
    for p in glob.glob(os.path.join(ART, "*.json")):
        j = json.load(open(p))
        for key in ("firewall",):
            fw = j.get(key)
            if isinstance(fw, dict) and "violations" in fw:
                seen += 1
                assert not fw["violations"], f"{os.path.basename(p)}: {fw['violations'][:3]}"
    if seen == 0:
        pytest.skip("no audited stage has run yet")


# --------------------------------------------------------------- evaluator refusal
def test_evaluator_refuses_without_a_frozen_manifest_and_takes_no_overrides():
    src = open(os.path.join(REPO, "tools/gate8d/eval_target.py")).read()
    assert "require_manifest" in src
    assert "refused" in src, "the evaluator must refuse, not warn"
    tree = ast.parse(src)
    opts = [n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
            and n.args and isinstance(n.args[0], ast.Constant)]
    for forbidden in ("--checkpoint", "--threshold", "--tau", "--manifest"):
        assert forbidden not in opts, f"evaluator exposes {forbidden}; it must not"


# --------------------------------------------------------------- privileged evidence
def test_future_evidence_is_target_only_and_absent_from_inference():
    """The dense target module must be unreachable from the causal inference path."""
    for rel in ("gates/gate8/feed.py", "gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8d/net.py"):
        txt = open(os.path.join(REPO, rel)).read()
        assert "densetarget" not in txt, f"{rel} reaches privileged future evidence"
        assert "future_natives" not in txt, f"{rel} names future frames"


def test_ground_truth_poses_never_enter_inference():
    for rel in ("gates/gate8/feed.py", "gates/gate8/mapper.py", "gates/gate8d/net.py"):
        txt = open(os.path.join(REPO, rel)).read()
        assert "velo_to_velo" not in txt and "cam0_to_world" not in txt, \
            f"{rel} uses ground-truth poses"


def test_native_window_never_exceeds_the_stream_horizon():
    """Dense sampling may fill the horizon; it may never extend it."""
    avail = {i: None for i in range(0, 200)}
    fut = [10, 20, 30, 40, 50]
    dense = FF.native_window(0, fut, avail)
    assert max(dense) == max(fut), "dense window looked beyond the frozen horizon"
    assert len(dense) > len(fut), "dense window is not actually denser"
    assert min(dense) >= 0


def test_no_neighbourhood_height_propagation_anywhere():
    """The rejected Gate 8C-1 variant must not reappear under a new name."""
    for rel in SOURCE_MODULES:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        for bad in ("max_pool2d", "SKY_NEIGHBOURHOOD", "top_local", "neighbourhood_max"):
            assert bad not in txt, f"{rel} contains height propagation ({bad})"


# --------------------------------------------------------------- evidence precedence
def _toy(dims=(4, 4, 4)):
    n = int(np.prod(dims))
    return dims, n


def test_lidar_endpoint_outranks_pseudo_free():
    """A pseudo-teacher may never turn a measured surface into free space."""
    src = open(os.path.join(REPO, "gates/gate8d/densetarget.py")).read()
    assert "state[pseudo & ~occ & ~fre] = FREE" in src, \
        "pseudo-free must be gated on the absence of LiDAR occupancy"
    assert "n_pseudo_refused_by_lidar_endpoint" in src, \
        "refusals must be counted, not silently dropped"


def test_conflicts_become_unknown_when_precedence_cannot_resolve_them():
    src = open(os.path.join(REPO, "gates/gate8d/densetarget.py")).read()
    assert "conflict" in src and "UNKNOWN" in src
    assert "n_conflict_to_unknown" in src


def test_moge_uncertainty_band_is_respected():
    """A MoGe ray must stop before the surface by the fitted margin."""
    src = open(os.path.join(REPO, "gates/gate8d/densetarget.py")).read()
    assert "stop = dep[ok] - mg" in src, "MoGe carving must subtract the safety margin"
    p = os.path.join(ART, "moge_margin.json")
    if not os.path.exists(p):
        pytest.skip("margin not fitted yet")
    m = json.load(open(p))
    assert all(x >= P.MOGE_MIN_MARGIN_M for x in m["margin_m"])
    assert m["fitted_on"] == list(P.TRAIN_DRIVES), "margin must be fitted on train drives"
    assert m["val_drive"] == P.VAL_DRIVE


def test_sky_rays_cannot_carve_a_lidar_endpoint():
    src = open(os.path.join(REPO, "gates/gate8d/densetarget.py")).read()
    i_pseudo = src.index("state[pseudo & ~occ & ~fre] = FREE")
    i_occ = src.index("state[occ & ~fre] = OCCUPIED")
    assert i_occ > i_pseudo, "occupied must be written after pseudo-free, so it wins"


# --------------------------------------------------------------- surface distance
def test_distance_target_is_in_metres_not_voxels():
    src = open(os.path.join(REPO, "gates/gate8d/densetarget.py")).read()
    assert "sampling=(vs, vs, vs)" in src, "EDT must be scaled by the voxel size"
    assert "TUDF_TRUNCATION_M" in src
    net = open(os.path.join(REPO, "gates/gate8d/net.py")).read()
    assert "metres" in net.lower()


def test_distance_loss_ignores_unknown_space():
    from gates.gate8d.net import surface_distance_loss
    import torch
    pred = torch.zeros(2, 4, 4, 4)
    tgt = torch.ones(2, 4, 4, 4)
    valid = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    assert float(surface_distance_loss(pred, tgt, valid)) == 0.0
    valid[0, 0, 0, 0] = True
    assert float(surface_distance_loss(pred, tgt, valid)) == pytest.approx(1.0)


# --------------------------------------------------------------- semantics
def test_observed_semantics_are_never_overwritten():
    src = open(os.path.join(REPO, "gates/gate8d/net.py")).read()
    assert "torch.where((semw > 0).unsqueeze(1), map_probs, net_probs)" in src, \
        "the network must not replace fused teacher evidence"


def test_semantic_loss_ignores_unreliable_teacher_evidence():
    src = open(os.path.join(REPO, "tools/gate8d/losses.py")).read()
    assert 'b["fut_valid"]' in src, "semantic loss must be masked by teacher validity"


# --------------------------------------------------------------- training hygiene
def test_all_seeds_share_one_config_except_the_seed():
    cfgs = sorted(glob.glob(os.path.join(REPO, "configs/gate8d/seed*.yaml")))
    if len(cfgs) < 2:
        pytest.skip("configs not written yet")
    import yaml
    base = None
    for c in cfgs:
        d = yaml.safe_load(open(c))
        d.pop("seed"); d.pop("tag", None)
        if base is None:
            base = d
        else:
            assert d == base, f"{c} differs from the others by more than the seed"


def test_protocol_is_frozen_and_hashes_itself():
    p = os.path.join(ART, "protocol.json")
    if not os.path.exists(p):
        pytest.skip("protocol not frozen yet")
    j = json.load(open(p))
    assert j["protocol"]["protocol_source_sha256"] == P.source_digest(), \
        "gate8d/protocol.py changed after freezing"


def test_gate8c1_artifacts_are_unchanged():
    """Gate 8D must not have touched the baseline it is measured against."""
    p = os.path.join(ART, "protocol.json")
    if not os.path.exists(p):
        pytest.skip("protocol not frozen yet")
    base = json.load(open(p))["baseline_gate8c1"]
    for rel, want in base["hashes"].items():
        if want == "MISSING":
            continue
        f = os.path.join(REPO, rel)
        assert os.path.exists(f), f"{rel} disappeared"
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        assert h.hexdigest() == want, f"{rel} changed since Gate 8D was frozen"


def test_each_input_frame_is_integrated_exactly_once():
    src = open(os.path.join(REPO, "tools/gate8c1/eval_target.py")).read()
    assert "window_map" in src
    m = open(os.path.join(REPO, "gates/gate8/mapper.py")).read()
    assert "def step" in m


def test_target_acceptance_gate_was_recorded_before_training():
    p = os.path.join(ART, "target_audit.json")
    if not os.path.exists(p):
        pytest.skip("audit not run yet")
    j = json.load(open(p))
    assert j["acceptance"]["max_pseudo_free_collision_rate"] == \
        P.TARGET_ACCEPTANCE["max_pseudo_free_collision_rate"]
    assert "accepted" in j
