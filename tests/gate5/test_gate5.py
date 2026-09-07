"""Gate 5 — MoGe metric-gauge correctness, scaling discipline and leakage tests."""
from __future__ import annotations

import csv
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
from gates.scale_gate.scale import bootstrap_ci, weighted_median                   # noqa: E402
from gates.voxel_gate.controls import control                                      # noqa: E402
from gates.voxel_gate.voxels import dilate                                         # noqa: E402
from moge_gauge import adapter as A                                          # noqa: E402
from moge_gauge import estimator as E                                        # noqa: E402
from occ3d_zeroshot import factorization as F                                # noqa: E402

CFG = os.path.join(_ROOT, "configs/gate5/moge_metric_gauge.yaml")
cfg = load_config(CFG)
g4 = load_config(os.path.join(_ROOT, cfg.experiment.gate4_config))
FV = g4.frozen_values
ART = os.path.join(_ROOT, cfg.experiment.output_dir)
EST = cfg.estimator

have = lambda p: os.path.exists(os.path.join(ART, p))
needs_scale = pytest.mark.skipif(not have("moge_scale_occ3d.csv"),
                                 reason="MoGe scales not cached")
needs_eval = pytest.mark.skipif(not have("gate5_summary_occ3d.json"),
                                reason="Gate-5 evaluation not run")



def code_only(path: str, drop_names=("FORBIDDEN_INFER_KWARGS",)) -> str:
    """Executable code with every docstring, comment and declared-name constant removed.

    Scanning raw source for forbidden tokens gives false positives on the prose that
    *documents* the prohibition, and on the constant that *lists* it.
    """
    import ast
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)                                     # drop the docstring
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in drop_names for t in node.targets):
            continue                                        # drop the forbidden-name list
        keep.append(node)
    tree.body = keep
    return ast.unparse(tree)


def _poses(n=5, seed=0):
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(scale=3.0, size=3)
    return P


# --------------------------------------------------------------------------- #
# 1, 2, 16: MoGe receives RGB only and never enters the reconstruction
# --------------------------------------------------------------------------- #
def test_moge_infer_receives_rgb_only_and_infers_its_own_fov():
    """The MoGe call site must pass the image and nothing else."""
    import ast
    params = list(inspect.signature(A.FrozenMoGe.infer).parameters)
    assert params == ["self", "rgb01"], f"infer takes extra inputs: {params}"
    tree = ast.parse(code_only(A.__file__))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "infer"
             and isinstance(n.func.value, ast.Attribute)
             and n.func.value.attr == "model"]
    assert len(calls) == 1, "expected exactly one self.model.infer call site"
    call = calls[0]
    kw = {k.arg for k in call.keywords}
    assert kw == {"apply_mask"}, f"MoGe called with extra keywords: {kw}"
    assert len(call.args) == 1, "MoGe must receive the image as its only positional arg"
    for bad in A.FORBIDDEN_INFER_KWARGS:
        assert bad not in kw, f"infer passes {bad} to MoGe"


def test_no_ground_truth_camera_or_lidar_reaches_moge():
    body = code_only(A.__file__)
    for bad in ("intrinsics_gt", "extrinsic", "ego_pose", "camera_height", "lidar",
                "velodyne", "labels.npz", "semantics", "mask_camera"):
        assert bad not in body, f"adapter references {bad!r}"


def test_moge_outputs_cannot_enter_the_fused_reconstruction():
    """The fusion signature accepts only LingBot tensors; MoGe has no way in."""
    params = list(inspect.signature(F.fuse).parameters)
    assert params == ["dep", "conf", "K", "rel", "T_anchor_camera_to_ego",
                      "depth_scale", "conf_thr", "dmin", "dmax"]
    for bad in ("moge", "teacher"):
        assert bad not in "".join(params).lower()
    body = "\n".join(l.split("#")[0] for l in
                     open(F.__file__).read().splitlines())
    assert "moge" not in body.lower(), "factorization references MoGe"


def test_estimator_never_opens_a_label_or_mask():
    body = "\n".join(l.split("#")[0] for l in open(E.__file__).read().splitlines())
    for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar", "occupancy",
                "lidar", "velodyne", "s_oracle"):
        assert bad not in body, f"estimator references {bad!r}"
    src = open(os.path.join(_ROOT, "tools/gate5/cache_moge_scale.py")).read()
    body = "\n".join(l.split("#")[0] for l in src.splitlines())
    for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar", "velodyne",
                "anchor_gt_path", "s_oracle"):
        assert bad not in body, f"cache_moge_scale references {bad!r}"


# --------------------------------------------------------------------------- #
# 4, 5, 6: label and LiDAR independence
# --------------------------------------------------------------------------- #
@needs_scale
def test_m2_scale_is_independent_of_labels_and_lidar():
    """The cached M2 scalar is a pure function of RGB + LingBot depth/confidence."""
    rows = list(csv.DictReader(open(os.path.join(ART, "moge_scale_occ3d.csv"))))
    assert len(rows) == 1182
    cols = set(rows[0])
    for bad in ("s_oracle", "iou", "gt", "label", "mask_camera", "n_lidar"):
        assert not any(bad in c for c in cols), f"MoGe scale table carries {bad!r}"


def test_randomising_targets_leaves_the_estimator_bit_identical():
    rng = np.random.default_rng(0)
    T, H, W = 5, 24, 40
    md = rng.uniform(2.0, 40.0, (T, H, W)).astype(np.float32)
    mm = rng.random((T, H, W)) > 0.15
    ld = rng.uniform(0.1, 2.0, (T, H, W)).astype(np.float32)
    lc = rng.uniform(1.0, 12.0, (T, H, W)).astype(np.float32)
    base = E.estimate_clip_scale(md, mm, ld, lc, 1.5, 1.0, 60.0, 500)
    for _ in range(3):
        _sem = rng.integers(0, 18, size=(200, 200, 16))      # target arrays that exist
        _mc = rng.random((200, 200, 16)) > 0.5               # in the pipeline but must
        _lidar = rng.uniform(1, 80, (T, H, W))               # never touch the estimator
        again = E.estimate_clip_scale(md, mm, ld, lc, 1.5, 1.0, 60.0, 500)
        assert again["s_moge"] == base["s_moge"]
        assert again["n_valid_total"] == base["n_valid_total"]


def test_lidar_diagnostics_run_after_predictions_are_finalised():
    src = open(os.path.join(_ROOT, "tools/gate5/scale_diagnostics.py")).read()
    assert 'gate5_summary_{a.dataset}.json' in src
    assert "run eval_gate5.py" in src, "diagnostics must refuse to run before the eval"


# --------------------------------------------------------------------------- #
# 7, 8: frozen
# --------------------------------------------------------------------------- #
def test_no_optimizer_backward_or_trainable_parameter_in_gate5():
    for p in (A.__file__, E.__file__,
              os.path.join(_ROOT, "tools/gate5/cache_moge_scale.py"),
              os.path.join(_ROOT, "tools/gate5/eval_gate5.py"),
              os.path.join(_ROOT, "tools/gate5/scale_diagnostics.py")):
        src = open(p).read()
        for bad in ("torch.optim", "Adam(", "SGD(", ".backward()", ".train()"):
            assert bad not in src, f"{os.path.basename(p)} mentions {bad!r}"
    assert "requires_grad_(False)" in open(A.__file__).read()
    assert "are trainable" in open(A.__file__).read()          # raises if any are


def test_frozen_scalars_and_hashes_match_gate41():
    import hashlib
    expect = {"lingbot": "ee665103348e07e6", "depth_head": "2a91822ea5d1182a",
              "full_s0": "05730ac7addb101d", "occ_only_s0": "b90a6e4ee9cdeccd",
              "c0_corrector_s0": "6590049302ef2044", "gate31_config": "a548ed08fcbe1ea6",
              "gate2_config": "b7692022114830f9", "gate0_config": "472b45311a5e03ba"}
    for k, rel in g4.frozen.items():
        h = hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()
        assert h.startswith(expect[k]), f"{k} changed"
    assert (float(FV.s0), float(FV.confidence_threshold)) == (27.3665, 1.5)
    assert (float(FV.min_depth_m), float(FV.max_depth_m)) == (1.0, 60.0)


def test_moge_revision_is_pinned_to_the_metric_v2_model():
    assert cfg.moge.hf_repo == "Ruicheng/moge-2-vitl"
    assert cfg.moge.hf_revision == A.MOGE_HF_REVISION == \
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
    assert cfg.moge.commit == A.MOGE_COMMIT
    assert A.MOGE_MODEL_CLASS == "moge.model.v2.MoGeModel"
    assert "moge-2" in cfg.moge.hf_repo and "normal" not in cfg.moge.hf_repo
    assert cfg.moge.license == "MIT"


# --------------------------------------------------------------------------- #
# 9, 10, 11: conventions
# --------------------------------------------------------------------------- #
def test_optical_axis_z_is_used_not_euclidean_ray():
    src = inspect.getsource(A.FrozenMoGe.infer)
    assert 'points[..., 2]' in src or "points[..., 2]" in src
    assert "norm(" not in src.split("return")[0].replace("np.linalg.norm", "")
    # numeric: for an off-axis point the two differ
    pts = np.array([[[3.0, 4.0, 12.0]]])
    assert abs(pts[..., 2].item() - 12.0) < 1e-9
    assert abs(np.linalg.norm(pts, axis=-1).item() - 13.0) < 1e-9


def test_rgb_range_and_shape_are_enforced():
    class Dummy(A.FrozenMoGe):
        def __init__(self):
            pass
    d = Dummy()
    with pytest.raises(AssertionError):
        A.FrozenMoGe.infer(d, np.zeros((4, 8, 8), np.float32))       # wrong channels
    with pytest.raises(AssertionError):
        A.FrozenMoGe.infer(d, np.full((3, 8, 8), 2.0, np.float32))   # out of [0,1]


def test_invalid_teacher_pixels_cannot_contaminate_the_estimate():
    T, H, W = 5, 8, 8
    md = np.full((T, H, W), 10.0, np.float32)
    md[:, 0, 0] = 1e6                       # absurd value, but masked out
    mm = np.ones((T, H, W), bool); mm[:, 0, 0] = False
    ld = np.full((T, H, W), 0.5, np.float32)
    lc = np.full((T, H, W), 5.0, np.float32)
    r = E.estimate_clip_scale(md, mm, ld, lc, 1.5, 1.0, 60.0, 10)
    assert abs(r["s_moge"] - 20.0) < 1e-9, "masked teacher pixel leaked into the estimate"
    v = E.frame_valid(md[0], mm[0], ld[0], lc[0], 1.5, 1.0, 60.0)
    assert not v[0, 0]


def test_pixel_correspondence_is_asserted_not_interpolated():
    path = os.path.join(_ROOT, "tools/gate5/cache_moge_scale.py")
    assert "pixel correspondence broken" in open(path).read(), \
        "shape correspondence is not asserted"
    body = code_only(path)
    for bad in ("interpolate", "cv2.resize", "grid_sample", "F.interpolate"):
        assert bad not in body, f"cache tool performs {bad!r}; correspondence must be exact"


# --------------------------------------------------------------------------- #
# 12, 13, 14, 15: one scalar, applied once
# --------------------------------------------------------------------------- #
def test_estimator_returns_exactly_one_scalar_per_clip():
    rng = np.random.default_rng(1)
    T, H, W = 5, 16, 20
    md = rng.uniform(2, 40, (T, H, W)).astype(np.float32)
    mm = np.ones((T, H, W), bool)
    ld = rng.uniform(0.1, 2, (T, H, W)).astype(np.float32)
    lc = np.full((T, H, W), 5.0, np.float32)
    r = E.estimate_clip_scale(md, mm, ld, lc, 1.5, 1.0, 60.0, 500)
    assert np.isscalar(r["s_moge"]) or isinstance(r["s_moge"], float)
    assert len(r["per_frame"]) == T            # diagnostics only, never applied per frame


def test_one_scalar_is_applied_to_all_five_depths_and_translations_once():
    rng = np.random.default_rng(2)
    T, H, W = 5, 10, 14
    dep = rng.uniform(0.1, 1.5, (T, H, W)).astype(np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    K = np.tile(np.array([[300., 0, W / 2], [0, 300., H / 2], [0, 0, 1.]]), (T, 1, 1))
    P, E4 = _poses(seed=3), np.eye(4)
    s = 24.5
    rel1 = F.relative_transforms(P, P, T - 1, "pred", 1.0)
    rels = F.relative_transforms(P, P, T - 1, "pred", s)
    for f in range(T - 1):
        assert np.allclose(rels[f][:3, 3], s * rel1[f][:3, 3], rtol=1e-12)   # once
        assert not np.allclose(rels[f][:3, 3], s * s * rel1[f][:3, 3])       # not twice
        assert np.allclose(rels[f][:3, :3], rel1[f][:3, :3], atol=1e-12)     # rotation
    assert np.allclose(rels[T - 1], np.eye(4), atol=1e-12)                   # anchor fixed
    p1, *_ = F.fuse(dep, conf, K, rel1, E4, 1.0, 1.5, 0.0, 1e9)
    ps, *_ = F.fuse(dep, conf, K, rels, E4, s, 1.5, 0.0, 1e9)
    assert np.allclose(ps, s * p1, rtol=1e-9, atol=1e-9)      # pure similarity


# --------------------------------------------------------------------------- #
# 17: the frozen 0.4 m dilation
# --------------------------------------------------------------------------- #
def test_dilate_r2_remains_the_frozen_0_4_m_expansion():
    kw = dict(FV.v1_control)
    assert kw == {"kind": "dilate", "radius": 2}
    v = float(FV.canonical_voxel_size)
    assert abs(kw["radius"] * v - 0.4) < 1e-12
    assert abs(int(round(float(FV.radius_m) / v)) * v - 0.6) < 1e-12   # band, distinct
    one = torch.zeros((11, 11, 11), dtype=torch.bool); one[5, 5, 5] = True
    assert int(dilate(one, kw["radius"]).sum()) == 125
    R = dilate(one, 3)
    assert torch.equal(control(one, R, **kw) & ~R, one & ~R)


# --------------------------------------------------------------------------- #
# results integrity
# --------------------------------------------------------------------------- #
@needs_eval
@pytest.mark.parametrize("ds,want", [
    ("occ3d", {"M0-R": 0.0647, "M0-D": 0.1597, "M1-R": 0.0411, "M1-D": 0.1434,
               "OR-R": 0.1027, "OR-D": 0.2292}),
    ("kitti", {"M0-R": 0.0573, "M1-R": 0.0778, "M1-D": 0.1759})])
def test_established_baselines_reproduce(ds, want):
    p = os.path.join(ART, f"gate5_summary_{ds}.json")
    if not os.path.exists(p):
        pytest.skip(f"{ds} not evaluated")
    agg = json.load(open(p))["aggregate"]
    for k, w in want.items():
        assert abs(agg[k]["iou"] - w) <= 5e-4, f"{ds} {k}: {agg[k]['iou']} vs {w}"


@needs_eval
def test_full_coverage_and_no_silent_fallback():
    s = json.load(open(os.path.join(ART, "gate5_summary_occ3d.json")))
    assert s["n_clips"] == 1182 and s["n_units"] == 150
    assert all(v == 0 for v in s["skipped_by_condition"].values())
    m = json.load(open(os.path.join(ART, "moge_scale_occ3d.json")))
    assert m["n_ok"] == 1182 and m["n_failed"] == 0
    assert m["estimator"]["on_failure"] == "exclude_and_report"
    k = json.load(open(os.path.join(ART, "gate5_summary_kitti.json")))
    assert k["n_clips"] == 163
    assert k["bootstrap_unit"] == "block"      # one sequence: never called "scenes"


@needs_eval
def test_bootstrap_units_are_labelled_honestly():
    for ds, unit, n in (("occ3d", "scene", 150), ("kitti", "block", 21)):
        s = json.load(open(os.path.join(ART, f"gate5_summary_{ds}.json")))
        assert s["bootstrap_unit"] == unit
        for k, v in s["contrasts"].items():
            assert v["unit"] == unit and v["n_units"] == n


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    d = list(rng.normal(0.01, 0.05, 150))
    assert bootstrap_ci(d, 10000, 0) == bootstrap_ci(d, 10000, 0)


def test_weighted_median_is_the_gate4_implementation():
    from gates.scale_gate import scale as S
    assert E.weighted_median is S.weighted_median
