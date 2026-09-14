"""The target-evaluation freeze guard, protocol serialization, and the comparability classifier."""
from __future__ import annotations
import json, os, sys
import pytest, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.baseline_reconstruction import config as C, freeze as FZ
from tools.baseline_reconstruction.metrics.comparability import Property, classify_comparability, property_table
from tools.baseline_reconstruction.historical.provenance import historical_protocol_yaml


def test_guard_refuses_without_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(FZ, "FREEZE_PATH", str(tmp_path / "code_freeze.json"))
    with pytest.raises(FZ.SourceFreezeMissing):
        FZ.require_source_freeze()


def test_guard_refuses_on_changed_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(FZ, "FREEZE_PATH", str(tmp_path / "code_freeze.json"))
    monkeypatch.setattr(C, "REPO_ROOT", str(tmp_path))
    f = tmp_path / "frozen.py"; f.write_text("a = 1\n")
    json.dump({"frozen_at": "now", "sha256": {"frozen.py": FZ.sha256_file(str(f))}}, open(FZ.FREEZE_PATH, "w"))
    assert FZ.require_source_freeze()["frozen_at"] == "now"
    f.write_text("a = 2\n")
    with pytest.raises(FZ.SourceFreezeViolated):
        FZ.require_source_freeze()


def test_protocols_serialize():
    y = yaml.safe_dump({k: v.__dict__ for k, v in C.TARGET_PROTOCOLS.items()})
    back = yaml.safe_load(y)
    assert back["occ3d"]["anchors_expected"] == 1182 and back["semantickitti"]["anchors_expected"] == 163
    assert json.loads(json.dumps(historical_protocol_yaml()))["protocol_of_record"] == "past5"


def test_comparability_classes():
    same = [Property("metric", "SC IoU", "SC IoU", "evaluation")]
    assert classify_comparability(same, {"semantickitti": True, "occ3d": True})["occ3d"]["class"] == "fully comparable"
    pipe = same + [Property("semantics", "yes", "none", "pipeline")]
    assert classify_comparability(pipe, {"semantickitti": True, "occ3d": True})["occ3d"]["class"] == "partially comparable"
    ev = same + [Property("mask", "camera", "lidar", "evaluation")]
    assert classify_comparability(ev, {"semantickitti": True, "occ3d": True})["occ3d"]["class"] == "not comparable"
    assert classify_comparability(same, {"semantickitti": False, "occ3d": True})["semantickitti"]["class"] == "not comparable"


def test_real_property_table_is_partially_comparable():
    cls = classify_comparability(property_table(), {"semantickitti": True, "occ3d": True})
    assert all(v["class"] == "partially comparable" for v in cls.values())
    assert "semantics" in cls["occ3d"]["pipeline_differences"] and not cls["occ3d"]["evaluation_differences"]
