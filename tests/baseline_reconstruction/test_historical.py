"""Historical record parsing, ceiling-fix detection constants, and the remembered-number check."""
from __future__ import annotations
import json, os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.historical import records as R, verify_ceiling_fix as V

_have_sky = os.path.isdir(C.HistoricalPaths().sky_eval_dir)
needs_records = pytest.mark.skipif(not _have_sky, reason="sky eval records not on this machine")


def test_parse_eval_json_uses_original_metric_names(tmp_path):
    d = {"dataset": "semantickitti", "mode": "past5", "seed": 1, "checkpoint": "x.pt", "threshold": -0.25, "n_clips": 7,
         "semantic": {"completion": {"binary_iou": 0.2271, "binary_precision": 0.3, "binary_recall": 0.4, "ssc_miou": 0.03}},
         "methods": {"completion": {"full": {"auroc": 0.8}}}, "baselines": {"mapper_native": {"binary_iou": 0.09}}}
    p = tmp_path / "eval_semantickitti_past5_seed1.json"; p.write_text(json.dumps(d))
    r = R.parse_eval_json(str(p), "sky")
    assert (r.sc_iou, r.seed, r.protocol, r.n_clips, r.threshold) == (0.2271, 1, "past5", 7, -0.25)
    assert abs(r.sc_iou_pct - 22.71) < 1e-9


def test_parse_result_logs(tmp_path):
    (tmp_path / "eval_seed0.log").write_text("RESULT seed0 SC_IoU=22.47 P=32.29 R=42.48 mIoU=3.35 clips=163\n")
    (tmp_path / "occ3d_sky_seed2.log").write_text("RESULT occ3d seed2 SC_IoU=33.88 P=75.24 R=38.13 mIoU=4.24 clips=1182\n")
    (tmp_path / "padfix_seed0.log").write_text("RESULT seed0 SC_IoU=10.63 P=1 R=1\n")      # ablation logs are ignored
    logs = R.parse_result_logs(str(tmp_path))
    assert logs == {"semantickitti": {0: 22.47}, "occ3d": {2: 33.88}}


def test_ceiling_layer_constants():
    assert V.TOP_LAYER_Z == 31 and V.ABOVE_2M_Z == 20        # +4.3 m layer and the +2 m line on the 0.2 m grid with origin z = -2


@needs_records
def test_canonical_numbers_do_not_contain_the_remembered_occ3d_value():
    results = R.load_variant_results()
    occ = [round(100 * r.sc_iou, 2) for r in R.select(results, "sky", "occ3d", "past5")]
    sk = [round(100 * r.sc_iou, 2) for r in R.select(results, "sky", "semantickitti", "past5")]
    assert 30.74 not in occ and sorted(occ)[1] == 33.88 and sorted(sk)[1] == 22.71


@needs_records
def test_ceiling_fix_checks_pass_without_gpu():
    p = C.HistoricalPaths()
    assert V.check_checkpoints(p).passed and V.check_eval_records(p).passed
