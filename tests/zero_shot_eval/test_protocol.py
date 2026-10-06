"""Final zero-shot evaluation: serialization, grids, masks, metrics, distance bins, Track O / Track P consistency."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.zero_shot_eval.canonical.metrics import confusion, occany_ssc_miou, semantic_summary, superclass_confusion   # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction                                               # noqa: E402
from tools.zero_shot_eval.official.evaluator import score_sample_official                                               # noqa: E402
from tools.zero_shot_eval.practical.evaluator import score_sample                                                       # noqa: E402
from tools.zero_shot_eval.practical.masks import camera_coordinates, field_of_view, track_p_masks                       # noqa: E402
from tools.zero_shot_eval.practical.protocol import EVIDENCE_DISTANCE_EDGES_M, OCC3D, SEMANTICKITTI                      # noqa: E402
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any, reduce_min, subvoxels                            # noqa: E402

DIMS = (16, 16, 4)


def _prediction(seed=0, dims=DIMS, voxel=0.5, origin=(0.0, -4.0, -1.0)):
    rng = np.random.default_rng(seed); n = int(np.prod(dims))
    T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float64)       # camera (OpenCV) -> velodyne-like grid
    occ = rng.random(n) < 0.3
    q = np.flatnonzero(occ | (rng.random(n) < 0.2))
    return CanonicalPrediction("s", "semantickitti", dims, voxel, origin, "velo", T, np.array([[10.0, 0, 8], [0, 10, 5], [0, 0, 1]]), 16,
                               occ, rng.random(n) < 0.5, rng.random(n) < 0.2, q, rng.random(len(q)).astype(np.float32) * 10,
                               {"B0": rng.integers(1, 20, len(q)).astype(np.uint8), "B3_s0": rng.integers(1, 20, len(q)).astype(np.uint8),
                                "B3_s1": rng.integers(1, 20, len(q)).astype(np.uint8), "B3_s2": rng.integers(1, 20, len(q)).astype(np.uint8)}, {"k": 1})


def test_canonical_prediction_round_trips(tmp_path):
    p = _prediction(); p.save(str(tmp_path / "a.npz")); r = CanonicalPrediction.load(str(tmp_path / "a.npz"))
    assert r.dims == p.dims and np.array_equal(r.occupied, p.occupied) and np.array_equal(r.evidence, p.evidence) and np.array_equal(r.observed, p.observed)
    assert np.array_equal(r.query_index, p.query_index) and all(np.array_equal(r.labels[m], p.labels[m]) for m in p.labels) and r.meta == {"k": 1}
    assert np.allclose(r.evidence_distance, p.evidence_distance, atol=1e-2) and np.array_equal(r.T_cam_to_grid, p.T_cam_to_grid)
    d = r.dense_labels("B0", 0); assert np.array_equal(d[p.query_index], p.labels["B0"]) and (np.delete(d, p.query_index) == 0).all()


def test_camera_coordinates_and_field_of_view():
    p = _prediction()
    cam = camera_coordinates(p)
    X, Y, Z = p.dims
    centre = (np.array([3, 8, 2]) + 0.5) * p.voxel_size + np.asarray(p.origin)           # grid (x fwd, y left, z up)
    flat = np.ravel_multi_index((3, 8, 2), p.dims)
    assert np.allclose(cam[flat], [-centre[1], -centre[2], centre[0]])                   # camera x = -y, y = -z, z = x
    K = np.array([[10.0, 0, 8], [0, 10, 5], [0, 0, 1]])
    pts = np.array([[0, 0, 5], [4, 0, 5], [4.1, 0, 5], [0, 0, -1], [-4, 0, 5]])
    assert field_of_view(pts, K, 16).tolist() == [True, True, False, False, True]


def test_region_and_range_bins():
    p = _prediction(); label = np.zeros(p.n_voxels, np.uint8); label[::7] = 255; label[::5] = 9
    m = track_p_masks(p, label, SEMANTICKITTI)
    assert not (m.region & (label == 255)).any() and (m.region <= m.valid).all()
    assert m.gt_occupied.sum() == ((label != 255) & (label != 0)).sum()
    cam = camera_coordinates(p); rng = np.hypot(cam[:, 0], cam[:, 2])
    assert ((m.range_bin == 4) == (rng > 40)).all()


def test_subvoxels_and_any_rule():
    fine = GridSpec((8, 8, 4), 0.2, (0, 0, 0), "f", "fine"); native = GridSpec((4, 4, 2), 0.4, (0, 0, 0), "n", "native")
    occ = torch.zeros(fine.n_voxels, dtype=torch.bool); occ[np.ravel_multi_index((5, 2, 3), fine.dims)] = True
    red = reduce_any(occ, fine, 2); assert red.sum() == 1 and red[np.ravel_multi_index((2, 1, 1), native.dims)]
    sub = subvoxels(torch.tensor([int(np.ravel_multi_index((2, 1, 1), native.dims))]), native, fine, 2)
    assert int(np.ravel_multi_index((5, 2, 3), fine.dims)) in sub[0].tolist() and sub.shape == (1, 8)
    vals = torch.arange(fine.n_voxels, dtype=torch.float32); assert reduce_min(vals, fine, 2)[0] == 0


def test_metrics_conventions():
    conf = confusion(np.array([1, 1, 2, 2, 0]), np.array([1, 2, 2, 2, 1]), 4)
    s = semantic_summary(conf, [1, 2, 3], 0)
    assert s["per_class_iou"][3] is None and np.isclose(s["per_class_iou"][1], 1 / 3) and np.isclose(s["per_class_iou"][2], 2 / 3) and s["gt_classes"] == 2
    o = occany_ssc_miou(conf, 4, 0, 99, False)
    # class 1: TP 1, FP 1 (GT 0 predicted 1), FN 1; class 2: TP 2, FP 1, FN 0; class 3 absent -> IoU 0 and still averaged
    assert np.isclose(o["miou"], np.mean([1 / (1 + 1 + 1 + 1e-5), 2 / (2 + 1 + 0 + 1e-5), 0.0]))
    sc = superclass_confusion(conf, [0, 1, 1, 2], 3); assert sc.sum() == conf.sum() and sc[1, 1] == 4


def test_distance_bins_edges():
    d = np.array([0.0, 0.99, 1.0, 3.9, 4.0, 8.0, 50.0])
    assert np.digitize(d, np.asarray(EVIDENCE_DISTANCE_EDGES_M)).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_track_o_and_track_p_full_read_the_same_prediction():
    p = _prediction(3); rng = np.random.default_rng(1)
    label = rng.integers(0, 20, p.n_voxels).astype(np.uint8); label[rng.random(p.n_voxels) < 0.1] = 255
    methods = ["B0", "B3_s0", "B3_s1", "B3_s2"]
    tp = score_sample(p, label, SEMANTICKITTI, methods)["per_sample"]; to = score_sample_official(p, label, SEMANTICKITTI, methods)
    for m in methods:
        assert np.array_equal(tp[f"conf__full__ssc__{m}"], to[f"official_conf__{m}"])
    assert np.array_equal(tp["occ__full__all"], to["official_occ"])


def test_protocol_class_sets():
    assert len(SEMANTICKITTI.semantic_classes) == 19 and SEMANTICKITTI.empty_class not in SEMANTICKITTI.semantic_classes
    assert len(OCC3D.semantic_classes) == 17 and OCC3D.empty_class == 17 and 0 in OCC3D.semantic_classes
    assert len(SEMANTICKITTI.superclass_of) == 20 and len(OCC3D.superclass_of) == 18
