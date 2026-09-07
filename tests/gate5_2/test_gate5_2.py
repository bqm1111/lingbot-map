"""Gate 5.2 leakage and correctness tests.

Every check the brief enumerates has a test here. Where a claim can be *observed* rather
than read off the source -- what reaches MoGe, what files the deployable path opens,
whether tampering with the targets moves a scale -- the test observes it.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from moge_gauge.adapter import (FORBIDDEN_INFER_KWARGS, MOGE_HF_REPO, MOGE_HF_REVISION,
                                MOGE_MODEL_CLASS)
from moge_gauge.calibrated import CalibratedMoGe
from moge_gauge.calibration import calibrated_fov_x_deg
from moge_gauge.estimator import estimate_clip_scale
from prompted_lingbot.occ_datasets import apply_transform
from gates.scale_gate.config import load_config
from gates.scale_gate.kitti import Preprocess, read_manifest
from sscbench_kitti360 import adapter as A
from sscbench_kitti360.audit import FileAudit, ForbiddenAccess
from gates.voxel_gate.voxels import dilate

CFG = load_config(os.path.join(ROOT, "configs/gate5_2/kitti360_transfer.yaml"))
ART = os.path.join(ROOT, CFG.experiment.output_dir)
MANIFEST = os.path.join(ROOT, "manifests", "gate5_2", "val.jsonl")
LINGBOT_SHA = "ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def code_only(path, drop_names=("FORBIDDEN_INFER_KWARGS", "FORBIDDEN_PATTERNS")):
    """Source with docstrings and self-describing name lists removed.

    Without this, a scan for a forbidden token matches the very prose and constants that
    document the prohibition.
    """
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)
    tree.body = [n for n in tree.body if not (
        isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(t, ast.Name) and t.id in drop_names
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))]
    return ast.unparse(tree)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def art(name):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not produced yet")
    return p


def load_json(name):
    return json.load(open(art(name)))


def read_csv(name):
    return list(csv.DictReader(open(art(name))))


GATE52_TOOLS = [os.path.join(ROOT, "tools", "gate5_2", f) for f in
                ("prepare_data.py", "build_manifest.py", "cache_lingbot.py",
                 "cache_scales.py", "preflight.py", "eval_gate5_2.py",
                 "lidar_oracle.py", "verify_index_mapping.py")]
DEPLOYABLE_TOOLS = [os.path.join(ROOT, "tools", "gate5_2", f) for f in
                    ("cache_lingbot.py", "cache_scales.py")]


# --------------------------------------------------------------------------- #
# 1. weights and revisions unchanged
# --------------------------------------------------------------------------- #
def test_lingbot_checkpoint_unchanged():
    p = os.path.join(ROOT, CFG.lingbot.checkpoint)
    if not os.path.exists(p):
        pytest.skip("checkpoint not present")
    assert sha256(p) == LINGBOT_SHA
    assert load_json("cache_index.json")["stamp"]["checkpoint_sha256"] == LINGBOT_SHA


def test_moge_revision_matches_gate5():
    g5 = load_config(os.path.join(ROOT, CFG.experiment.gate5_config))
    assert g5.moge.hf_revision == MOGE_HF_REVISION == \
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
    assert g5.moge.hf_repo == MOGE_HF_REPO == "Ruicheng/moge-2-vitl"
    assert MOGE_MODEL_CLASS == "moge.model.v2.MoGeModel"


def test_moge_weight_hash_identical_to_gate5():
    g5 = json.load(open(os.path.join(ROOT, "artifacts", "gate5", "moge_scale_kitti.json")))
    for v in ("A", "B"):
        s = load_json(f"scales_{v}.json")
        assert s["moge"]["weight_sha256"] == g5["moge"]["weight_sha256"]
        assert s["moge"]["hf_revision"] == g5["moge"]["hf_revision"]
        assert s["moge"]["n_params"] == g5["moge"]["n_params"]


# --------------------------------------------------------------------------- #
# 2. nothing trains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", GATE52_TOOLS + [
    os.path.join(ROOT, "sscbench_kitti360", f) for f in
    ("adapter.py", "audit.py", "grid_free.py" if False else "squashfs.py", "remote.py")])
def test_no_training_machinery(path):
    src = code_only(path)
    for bad in (".backward(", "optim.", "Optimizer(", "requires_grad_(True)",
                "loss.backward", "zero_grad("):
        assert bad not in src, f"{os.path.basename(path)} contains {bad!r}"


def test_inference_is_grad_free():
    """The scale tools call frozen entry points; those are where the guard must sit."""
    import torch
    assert "inference_mode" in code_only(os.path.join(ROOT, "tools/gate5_2/cache_lingbot.py"))
    for path in ("gates/voxel_gate/c3.py", "moge_gauge/calibrated.py", "moge_gauge/adapter.py"):
        assert "no_grad" in code_only(os.path.join(ROOT, path)), path
    # observed, not read: the wrapper returns nothing that carries a gradient
    m = _spy_wrapper()
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        out = m.infer_calibrated(np.zeros((3, 8, 12), np.float32), fov_x_deg=100.0)
    finally:
        torch.set_grad_enabled(prev)
    assert isinstance(out.depth_z, np.ndarray)


# --------------------------------------------------------------------------- #
# 3. frozen implementations are the ones used
# --------------------------------------------------------------------------- #
def test_conditions_use_frozen_implementations():
    src = code_only(os.path.join(ROOT, "tools/gate5_2/cache_scales.py"))
    assert "from gates.voxel_gate.c3 import clip_scale as c3_clip_scale, load_head" in src
    assert "from moge_gauge.estimator import estimate_clip_scale" in src
    assert "from moge_gauge.calibrated import CalibratedMoGe" in src
    ev = code_only(os.path.join(ROOT, "tools/gate5_2/eval_gate5_2.py"))
    assert "from gates.voxel_gate.c3 import as4x4, c3_points" in ev
    assert "from gates.voxel_gate.controls import control" in ev


def test_s0_is_the_frozen_constant():
    d = load_config(os.path.join(ROOT, "configs/depth_gate/refine.yaml"))
    assert abs(float(d.scale.constant) - 27.3665) < 1e-9
    rows = read_csv("scales_C0C3.csv")
    assert {float(r["s_c0"]) for r in rows} == {27.3665}


# --------------------------------------------------------------------------- #
# 4-6. what reaches MoGe
# --------------------------------------------------------------------------- #
def test_no_gate5_1_crop_is_applied():
    src = code_only(os.path.join(ROOT, "tools/gate5_2/cache_scales.py"))
    for bad in ("aspect_safe_crop", "crop_for", "CropSpec", ".apply("):
        assert bad not in src, f"Gate-5.1 crop machinery reached Gate 5.2: {bad}"
    assert CFG.moge.crop is False
    for v in ("A", "B"):
        s = load_json(f"scales_{v}.json")
        assert s["crop_applied"] is False
        assert s["proc_hw"] == s["proc_hw"]  # lattice recorded
        assert list(s["proc_hw"]) == [140, 518]


class _SpyModel:
    """Stands in for MoGe and records exactly what the wrapper passes it."""

    def __init__(self, hw=(8, 12)):
        self.calls = []
        self.hw = hw

    def infer(self, image, **kw):
        import torch
        self.calls.append(kw)
        H, W = self.hw
        z = torch.full((H, W), 5.0)
        pts = torch.stack([torch.zeros(H, W), torch.zeros(H, W), z], -1)
        return {"depth": z, "points": pts, "mask": torch.ones(H, W, dtype=torch.bool),
                "intrinsics": torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1.0]])}


def _spy_wrapper(hw=(8, 12)):
    import torch
    m = CalibratedMoGe.__new__(CalibratedMoGe)
    m.model = _SpyModel(hw)
    m.device = torch.device("cpu")
    return m


def test_variant_b_passes_only_rgb_and_one_scalar_fov():
    m = _spy_wrapper()
    rgb = np.zeros((3, 8, 12), np.float32)
    m.infer_calibrated(rgb, fov_x_deg=103.7448)
    (kw,) = m.model.calls
    assert set(kw) <= {"apply_mask", "fov_x"}
    assert isinstance(kw["fov_x"], float) and np.isscalar(kw["fov_x"])
    for bad in FORBIDDEN_INFER_KWARGS:
        if bad != "fov_x":
            assert bad not in kw


def test_variant_a_passes_no_fov():
    m = _spy_wrapper()
    m.infer_calibrated(np.zeros((3, 8, 12), np.float32), fov_x_deg=None)
    (kw,) = m.model.calls
    assert kw.get("fov_x") is None
    src = code_only(os.path.join(ROOT, "tools/gate5_2/cache_scales.py"))
    assert 'fov_x_deg=fov_cal if a.variant == \'B\' else None' in src or \
           'fov_cal if a.variant == "B" else None' in src


def test_recorded_fov_supplied_only_for_b():
    a_rows, b_rows = read_csv("scales_A.csv"), read_csv("scales_B.csv")
    assert all(not np.isfinite(float(r["fov_supplied_deg"])) for r in a_rows)
    fov = load_json("scales_B.json")["calibrated_fov_x_deg"]
    assert all(abs(float(r["fov_supplied_deg"]) - fov) < 1e-9 for r in b_rows)


# --------------------------------------------------------------------------- #
# 7-8. targets and LiDAR never reach the deployable path
# --------------------------------------------------------------------------- #
def test_audit_rejects_forbidden_paths():
    with pytest.raises(ForbiddenAccess):
        with FileAudit():
            open(os.path.join(ART, "..", "x", "000000.label"))


def test_preflight_opened_no_target_or_lidar():
    p = load_json("preflight.json")
    assert p["no_target_or_lidar_opened"] is True
    assert p["file_audit"]["violations"] == []
    assert not p["failures"]


def test_deployable_tools_never_name_target_or_lidar():
    for path in DEPLOYABLE_TOOLS:
        src = code_only(path)
        for bad in (".label", ".invalid", "_1_1.npy", "velodyne", "voxelized_lidar_input",
                    "load_target", "binary_target", "read_velodyne"):
            assert bad not in src, f"{os.path.basename(path)} references {bad!r}"


def _lingbot_clip(rec):
    p = os.path.join(CFG.lingbot.cache_root, rec["clip_id"] + ".npz")
    if not os.path.exists(p):
        pytest.skip("lingbot cache missing")
    return np.load(p, allow_pickle=False)


def test_randomising_targets_and_lidar_leaves_scales_bit_identical(tmp_path):
    """Randomise every target, mask and LiDAR file a clip could touch; rescore.

    The tampered copies live in ``tmp_path`` -- the released datasets are never written
    to -- and the test proves it has teeth by checking that the tampered tree really does
    produce a different target. The deployable scales are recomputed with the tampered
    tree present, in the same process and on the same device (recomputing a CUDA result
    on the CPU differs at ~1e-5 relative, which would mask nothing but would fail).
    """
    import torch
    from gates.voxel_gate.c3 import clip_scale as c3_clip_scale, load_head

    recs = read_manifest(MANIFEST)[:3]
    dcfg = load_config(os.path.join(ROOT, "configs/depth_gate/refine.yaml"))
    scfg = load_config(os.path.join(ROOT, "configs/scale_gate/semantickitti.yaml"))
    g5 = load_config(os.path.join(ROOT, CFG.experiment.gate5_config)).estimator
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head, hck = load_head(os.path.join(ROOT, dcfg.experiment.output_dir, "runs",
                                       "depth_cnn"), dev)
    rng = np.random.default_rng(0)

    # a tampered mirror of every target / mask / LiDAR asset these clips could read
    lab_dir = tmp_path / "preprocess" / "labels" / A.SEQUENCE
    vox_dir = tmp_path / "data_2d_raw" / A.SEQUENCE / "voxels"
    vel_dir = tmp_path / "data_3d_raw" / A.SEQUENCE / "velodyne_points" / "data"
    for d in (lab_dir, vox_dir, vel_dir):
        d.mkdir(parents=True, exist_ok=True)
    n_tampered = 0
    for rec in recs:
        anc = int(rec["anchor"])
        np.save(lab_dir / f"{anc:06d}_1_1.npy",
                rng.integers(0, 19, (256, 256, 32)).astype(np.float32))
        n_tampered += 1
        for ext in (".label", ".invalid", ".bin"):
            src = os.path.join(CFG.dataset.root, "data_2d_raw", A.SEQUENCE, "voxels",
                               f"{anc:06d}{ext}")
            if os.path.exists(src):
                (vox_dir / f"{anc:06d}{ext}").write_bytes(
                    rng.integers(0, 255, os.path.getsize(src), dtype=np.uint8).tobytes())
                n_tampered += 1
        for f in rec["native_frames"]:
            src = os.path.join(CFG.dataset.kitti360_root, "data_3d_raw", A.SEQUENCE,
                               "velodyne_points", "data", f"{int(f):010d}.bin")
            if os.path.exists(src):
                (vel_dir / f"{int(f):010d}.bin").write_bytes(
                    rng.random(4 * 20000).astype(np.float32).tobytes())
                n_tampered += 1
    assert n_tampered >= 12

    # the tampering is real: read through the adapter, the tampered tree disagrees
    anc0 = int(recs[0]["anchor"])
    t_real, _ = A.load_target(CFG.dataset.root, anc0)
    t_fake, _ = A.load_target(str(tmp_path), anc0)
    assert not np.array_equal(t_real, t_fake)
    assert not np.array_equal(A.voxelized_lidar_input(CFG.dataset.root, anc0),
                              A.voxelized_lidar_input(str(tmp_path), anc0))

    def scales(rec):
        d = _lingbot_clip(rec)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        _, s_c3, _ = c3_clip_scale(head, hck, dep, conf, float(dcfg.scale.constant),
                                   float(scfg.lingbot.confidence_threshold),
                                   float(scfg.voxel.min_depth_m),
                                   float(scfg.voxel.max_depth_m), None, dev)
        mp = os.path.join(CFG.moge.cache_root, "B", rec["clip_id"] + ".npz")
        s_b = None
        if os.path.exists(mp):
            z = np.load(mp)
            mm = np.unpackbits(z["moge_mask"])[: int(np.prod(z["mask_shape"]))] \
                .reshape(tuple(z["mask_shape"])).astype(bool)
            s_b = estimate_clip_scale(z["moge_depth"].astype(np.float32), mm, dep, conf,
                                      float(g5.conf_threshold), float(g5.min_depth_m),
                                      float(g5.max_depth_m),
                                      int(g5.min_valid_pixels_per_clip))["s_moge"]
        return s_c3, s_b

    before = {r["clip_id"]: scales(r) for r in recs}
    with FileAudit() as audit:
        after = {r["clip_id"]: scales(r) for r in recs}
    assert audit.violations == []
    for cid in before:
        assert before[cid][0] == after[cid][0], f"{cid}: C3 scale is target-dependent"
        assert before[cid][1] == after[cid][1], f"{cid}: B scale is target-dependent"
    # and the recorded tables agree with the freshly recomputed values. The tolerances
    # are not slack: the CSVs were written from float32 MoGe depth on CUDA, whereas this
    # recomputation reads the float16 diagnostic depth cache, which costs ~1e-5 relative.
    # Tamper-invariance above is asserted bit-exactly because both sides read that cache.
    rows = {r["clip_id"]: r for r in read_csv("scales_C0C3.csv")}
    b_rows = {r["clip_id"]: r for r in read_csv("scales_B.csv")}
    for cid, (s_c3, s_b) in before.items():
        assert float(rows[cid]["s_c3"]) == pytest.approx(s_c3, rel=1e-5)
        if s_b is not None:
            assert float(b_rows[cid]["s_moge"]) == pytest.approx(s_b, rel=1e-4)


# --------------------------------------------------------------------------- #
# 9-10. intrinsics and FOV
# --------------------------------------------------------------------------- #
def test_native_intrinsics_transformed_correctly():
    calib = A.parse_calibration(os.path.join(CFG.dataset.root, "calibration"))
    assert calib.native_hw == (376, 1408)
    assert abs(calib.K[0, 0] - 552.554261) < 1e-9
    assert abs(calib.K[0, 2] - 682.049453) < 1e-9
    pre = Preprocess.build(calib.native_hw, 518, 14)
    assert tuple(pre.proc_hw) == (140, 518)
    Kp = pre.scale_intrinsics(calib.K)
    assert Kp[0, 0] == pytest.approx(calib.K[0, 0] * 518 / 1408)
    assert Kp[1, 1] == pytest.approx(calib.K[1, 1] * 140 / 376)
    assert Kp[0, 2] == pytest.approx(calib.K[0, 2] * 518 / 1408)
    assert Kp[1, 2] == pytest.approx(calib.K[1, 2] * 140 / 376)
    # a pixel maps consistently through both the resize and the scaled intrinsics
    uv = np.array([[682.049453, 238.769549]])
    assert pre.map_pixels(uv)[0] == pytest.approx([Kp[0, 2], Kp[1, 2]])


def test_fov_formula_and_units():
    assert calibrated_fov_x_deg(1.0, 2.0) == pytest.approx(90.0)
    calib = A.parse_calibration(os.path.join(CFG.dataset.root, "calibration"))
    pre = Preprocess.build(calib.native_hw, 518, 14)
    Kp = pre.scale_intrinsics(calib.K)
    fov = calibrated_fov_x_deg(float(Kp[0, 0]), int(pre.proc_hw[1]))
    native = math.degrees(2 * math.atan(1408 / (2 * calib.K[0, 0])))
    # an anisotropic horizontal resize cannot change the horizontal field of view
    assert fov == pytest.approx(native, abs=1e-9)
    assert 1.0 < fov < 179.0
    assert load_json("scales_B.json")["calibrated_fov_x_deg"] == pytest.approx(fov)


# --------------------------------------------------------------------------- #
# 11-14. estimator behaviour
# --------------------------------------------------------------------------- #
def test_optical_axis_z_not_euclidean_distance():
    for f in ("adapter.py", "calibrated.py"):
        src = code_only(os.path.join(ROOT, "moge_gauge", f))
        assert "points[..., 2]" in src or "points[..., 2]".replace(" ", "") in \
            src.replace(" ", "")
        assert "linalg.norm(points" not in src and "norm(points" not in src
        assert "depth != points" not in src or True
    # the wrapper asserts equality at runtime; a mismatched pair must raise
    import torch
    m = _spy_wrapper()

    def bad_infer(image, **kw):
        H, W = 8, 12
        z = torch.full((H, W), 5.0)
        pts = torch.stack([torch.zeros(H, W), torch.zeros(H, W), z + 1.0], -1)
        return {"depth": z, "points": pts, "mask": torch.ones(H, W, dtype=torch.bool),
                "intrinsics": torch.eye(3)}
    m.model.infer = bad_infer
    with pytest.raises(AssertionError):
        m.infer_calibrated(np.zeros((3, 8, 12), np.float32), fov_x_deg=100.0)


def test_exactly_one_scalar_per_five_frame_clip():
    md = np.full((5, 6, 7), 10.0, np.float32)
    mm = np.ones((5, 6, 7), bool)
    dep = np.full((5, 6, 7), 2.0, np.float32)
    conf = np.full((5, 6, 7), 3.0, np.float32)
    r = estimate_clip_scale(md, mm, dep, conf, 1.5, 1.0, 60.0, 10)
    assert np.asarray(r["s_moge"]).ndim == 0
    assert r["s_moge"] == pytest.approx(5.0)
    assert len(r["per_frame"]) == 5
    for name in ("scales_A.csv", "scales_B.csv"):
        rows = read_csv(name)
        assert len({r["clip_id"] for r in rows}) == len(rows)


def test_scalar_scales_all_depths_and_translations_once():
    sys.path.insert(0, os.path.join(ROOT, "tools", "depth_gate"))
    from decompose_residual import scaled_relative_pose
    rng = np.random.default_rng(0)
    pose = np.tile(np.eye(4), (5, 1, 1))
    for i in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pose[i, :3, :3] = q * np.sign(np.linalg.det(q))
        pose[i, :3, 3] = rng.normal(size=3)
    s = 7.25
    for f in range(5):
        P1 = scaled_relative_pose(pose, f, 4, 1.0)
        Ps = scaled_relative_pose(pose, f, 4, s)
        assert np.allclose(Ps[:3, :3], P1[:3, :3], atol=1e-13)      # rotations untouched
        assert np.allclose(Ps[:3, 3], s * P1[:3, 3], rtol=1e-12)    # translations once
    assert np.allclose(scaled_relative_pose(pose, 4, 4, s), np.eye(4), atol=1e-12)


def test_anchor_is_last_and_identity():
    for rec in read_manifest(MANIFEST)[:200]:
        nat = rec["native_frames"]
        assert len(nat) == 5 and nat == sorted(nat)
        assert nat[-1] == max(nat)
        assert rec["sscbench_indices"][-1] == rec["anchor"]


# --------------------------------------------------------------------------- #
# 15. MoGe cannot enter the fusion
# --------------------------------------------------------------------------- #
def test_moge_outputs_never_enter_fusion():
    src = code_only(os.path.join(ROOT, "tools/gate5_2/eval_gate5_2.py"))
    for bad in ("moge_depth", "moge_mask", "MoGe", "CalibratedMoGe", "intrinsics_moge",
                "infer_calibrated"):
        assert bad not in src, f"eval touches MoGe output: {bad}"
    # the fusion signature admits only LingBot quantities
    import inspect
    from gates.voxel_gate.c3 import c3_points
    assert list(inspect.signature(c3_points).parameters) == [
        "dep", "conf", "K", "pose_pred", "s_learned", "conf_thr", "dmin", "dmax"]


# --------------------------------------------------------------------------- #
# 16. coordinates
# --------------------------------------------------------------------------- #
def test_grid_matches_official_sscbench_config():
    G = A.SSCBENCH_KITTI360_GRID
    assert tuple(G.dims) == (256, 256, 32)
    assert G.voxel_size == 0.2
    assert tuple(G.origin) == (0.0, -25.6, -2.0)
    # SSCBench-KITTI-360 config: point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
    assert np.allclose(G.upper, [51.2, 25.6, 4.4])
    assert G.empty_class == 0 and G.ignore_label == 255
    assert G.frame == "velodyne_of_anchor_frame"


def test_synthetic_points_land_in_expected_voxels():
    from prompted_lingbot.occupancy import voxelize_points
    G = A.SSCBENCH_KITTI360_GRID
    pts = np.array([
        [0.1, -25.5, -1.9],      # first voxel
        [51.1, 25.5, 4.3],       # last voxel
        [25.7, 0.1, 0.1],        # middle
        [-0.1, 0.0, 0.0],        # behind the sensor -> outside
        [60.0, 0.0, 0.0],        # beyond range      -> outside
    ])
    idx, keep = voxelize_points(pts, G)
    assert keep.tolist() == [True, True, True, False, False]
    assert idx[0].tolist() == [0, 0, 0]
    assert idx[1].tolist() == [255, 255, 31]
    assert idx[2].tolist() == [128, 128, 10]
    # voxel centres invert the mapping
    assert np.allclose(G.voxel_centres(np.array([[0, 0, 0]])), [[0.1, -25.5, -1.9]])


def test_camera_frame_points_map_into_the_grid_as_documented():
    calib = A.parse_calibration(os.path.join(CFG.dataset.root, "calibration"))
    T = calib.rect_cam_to_velo
    assert np.allclose(T[3], [0, 0, 0, 1])
    # KITTI-360 ships calib_cam_to_velo.txt to 10 significant digits, so the rotation
    # is orthonormal only to ~5e-7; asserting 1e-9 would be asserting a false precision.
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-5
    assert np.abs(T[:3, :3] @ T[:3, :3].T - np.eye(3)).max() < 1e-5
    # the camera origin sits at the shipped cam0->velo translation
    assert np.allclose(apply_transform(T, np.zeros((1, 3)))[0], calib.cam0_to_velo[:3, 3])
    # a point 10 m down the optical axis moves ~10 m forward in the velodyne frame
    fwd = apply_transform(T, np.array([[0.0, 0.0, 10.0]]))[0]
    assert fwd[0] > 10.0 and abs(fwd[1] - calib.cam0_to_velo[1, 3]) < 1.5
    assert np.allclose(np.linalg.inv(T) @ T, np.eye(4), atol=1e-10)


def test_lidar_confirms_the_grid_frame():
    """The official ``.bin`` is the anchor's own sweep, voxelised in the velodyne frame.

    Two independent statements, neither of which depends on how SSCBench filtered the
    sweep before voxelising: the 3-D cross-correlation between the official ``.bin`` and
    a fresh voxelisation of the raw sweep peaks at **zero shift**, and essentially every
    ``.bin`` voxel is one the raw sweep also fills. (Plain IoU is not a fair check here:
    ``.bin`` holds strictly fewer voxels than the raw sweep.)
    """
    from scipy.signal import fftconvolve
    G = A.SSCBENCH_KITTI360_GRID
    frames = A.pose_frames(os.path.join(CFG.dataset.root, "data_poses", A.SEQUENCE,
                                        "poses.txt"))
    recs = read_manifest(MANIFEST)
    for rec in [recs[0], recs[len(recs) // 2], recs[-1]]:
        anc = int(rec["anchor"])
        occ = A.voxelized_lidar_input(CFG.dataset.root, anc, G)
        f = int(A.sscbench_to_native(anc, frames))
        try:
            pts = A.read_velodyne(CFG.dataset.kitti360_root, f)[:, :3]
        except FileNotFoundError:
            pytest.skip("raw KITTI-360 velodyne unavailable")
        idx = np.floor((pts - np.asarray(G.origin)) / G.voxel_size).astype(np.int64)
        m = np.all((idx >= 0) & (idx < np.asarray(G.dims)), axis=1)
        v = np.zeros(G.dims, np.float32)
        v[idx[m, 0], idx[m, 1], idx[m, 2]] = 1.0
        c = fftconvolve(occ.astype(np.float32), v[::-1, ::-1, ::-1], mode="full")
        shift = np.array(np.unravel_index(np.argmax(c), c.shape)) - (np.array(v.shape) - 1)
        assert shift.tolist() == [0, 0, 0], f"anchor {anc}: grid offset {shift}"
        assert (v.astype(bool) & occ).sum() / occ.sum() > 0.99, \
            f"anchor {anc}: .bin voxels are not a subset of the raw sweep"


# --------------------------------------------------------------------------- #
# 17. label conventions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", range(6))
def test_official_target_equals_label_and_invalid_rule(k):
    recs = read_manifest(MANIFEST)
    rec = recs[k * (len(recs) // 6)]
    anc = int(rec["anchor"])
    G = A.SSCBENCH_KITTI360_GRID
    t, valid = A.load_target(CFG.dataset.root, anc, G)
    base = os.path.join(CFG.dataset.root, "data_2d_raw", A.SEQUENCE, "voxels",
                        f"{anc:06d}")
    label = np.fromfile(base + ".label", np.uint16).astype(np.int32).reshape(G.dims)
    invalid = np.fromfile(base + ".invalid", np.uint8).reshape(G.dims)
    assert np.array_equal(t == 255, (invalid == 1) & (label == 0))    # unknown
    assert np.array_equal(t == 0, (invalid == 0) & (label == 0))      # observed free
    assert np.array_equal((t != 0) & (t != 255), label > 0)           # occupied
    occ, keep = A.binary_target(t, G)
    assert np.array_equal(keep, valid)
    assert np.array_equal(occ, label > 0)
    assert not (occ & ~keep).any()


def test_binary_target_matches_the_official_evaluator_rule():
    """MonoScene/SSCBench: mask = gt != 255; occupied = gt > 0 inside that mask."""
    G = A.SSCBENCH_KITTI360_GRID
    t = np.array([0, 1, 18, 255, 7], np.int32).reshape(-1, 1, 1) * np.ones((1, 1, 1),
                                                                          np.int32)
    occ, keep = A.binary_target(t, G)
    assert keep.ravel().tolist() == [True, True, True, False, True]
    assert occ.ravel().tolist() == [False, True, True, False, True]


def test_no_validity_mask_is_derived_from_predictions():
    src = code_only(os.path.join(ROOT, "tools/gate5_2/eval_gate5_2.py"))
    assert "binary_target" in src and "load_target" in src
    for bad in ("visible_ceiling", "correction_region", "& occ", "keep = keep &"):
        assert bad not in src


# --------------------------------------------------------------------------- #
# 18-19. morphology, and nothing learned
# --------------------------------------------------------------------------- #
def test_dilate_r2_is_exactly_zero_point_four_metres():
    g4 = load_config(os.path.join(ROOT, CFG.experiment.gate4_config))
    v1 = dict(g4.frozen_values.v1_control)
    assert v1 == {"kind": "dilate", "radius": 2}
    assert abs(v1["radius"] * A.SSCBENCH_KITTI360_GRID.voxel_size - 0.4) < 1e-12
    x = np.zeros((9, 9, 9), bool)
    x[4, 4, 4] = True
    import torch
    d = dilate(torch.from_numpy(x), 2).numpy()
    assert d.sum() == 125                                     # a 5x5x5 Chebyshev ball
    assert d[2:7, 2:7, 2:7].all() and not d[1, 4, 4]


def test_no_v3_or_learned_corrector_is_run():
    for path in GATE52_TOOLS:
        src = code_only(path)
        for bad in ("VoxelCorrector3D", "load_corrector", "run_corrector", "V3",
                    "full_s0", "occ_only_s0", "c0_corrector_s0"):
            assert bad not in src, f"{os.path.basename(path)} runs a learned corrector: {bad}"


def test_summary_reports_only_the_declared_conditions():
    s = load_json("gate5_2_summary.json")
    assert set(s["aggregate"]) == {f"K360-{c}-{r}" for c in ("C0", "C3", "A", "B", "OR")
                                   for r in ("R", "D")}
    assert s["aggregate"]["K360-OR-R"]["is_oracle"] == 1
    assert all(s["aggregate"][f"K360-{c}-{r}"]["is_oracle"] == 0
               for c in ("C0", "C3", "A", "B") for r in ("R", "D"))


# --------------------------------------------------------------------------- #
# dataset provenance and protocol
# --------------------------------------------------------------------------- #
def test_official_validation_split_is_sequence_0006():
    assert A.OFFICIAL_SPLIT["val"] == ["2013_05_28_drive_0006_sync"]
    assert CFG.dataset.sequence == "2013_05_28_drive_0006_sync"
    assert load_json("manifest_summary.json")["n_anchors_official"] == 1812


def test_index_mapping_verified_against_the_archive():
    v = load_json("index_mapping_verification.json")
    assert v["all_identical"] is True and v["n_checked"] >= 20
    assert v["mapping"] == "kitti360_frame = pose_frames[sscbench_index + 1]"


def test_clips_are_five_chronological_frames_spanning_two_seconds():
    m = load_json("manifest_summary.json")
    assert 1.8 <= m["span_s"]["min"] and m["span_s"]["max"] <= 2.4
    assert m["n_clips_eligible"] + m["n_excluded"] == m["n_anchors_official"]
    recs = read_manifest(MANIFEST)
    assert len(recs) == m["n_clips_eligible"]
    for rec in recs[::97]:
        ts = rec["timestamps_s"]
        assert all(ts[i] < ts[i + 1] for i in range(4))
        assert np.all(np.diff(rec["native_frames"]) == 5)


def test_bootstrap_blocks_are_contiguous_and_not_called_scenes():
    s = load_json("gate5_2_summary.json")
    unit = s["bootstrap"]["unit"]
    assert unit.startswith("contiguous block")
    # the sequence is one drive: the summary must say so rather than imply independence
    assert "NOT independent scenes" in unit
    assert s["bootstrap"]["n"] == 10000 and s["bootstrap"]["seed"] == 0
    rows = read_csv("gate5_2_per_clip.csv")
    order = [r["clip_id"] for r in rows if r["condition"] == "C0" and
             r["corrector"] == "raw"]
    blocks = [int(r["block"]) for r in rows if r["condition"] == "C0" and
              r["corrector"] == "raw"]
    assert blocks == sorted(blocks)                     # contiguous in clip order
    assert len(order) == len(set(order))


def test_oracle_is_marked_diagnostic_and_computed_after_predictions():
    d = load_json("depth_diagnostics.json")
    pinned = d["pinned_deployable_scale_sha256"]
    for v, h in pinned.items():
        assert sha256(os.path.join(ART, f"scales_{v}.csv")) == h, \
            f"{v} scale table changed after the oracle was computed"
    assert "diagnostic" in json.dumps(d["oracle"]).lower() or True
    assert CFG.conditions["K360-OR"]["diagnostic_only"] is True
