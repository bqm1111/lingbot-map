"""Shared setup for the Gate 8C-0 tools."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
import gates.gate8b.sources  # noqa: E402,F401  -- registers k360_train with gate8.sources

ART = os.path.join(REPO_ROOT, "artifacts", "gate8c0")
CFG = os.path.join(REPO_ROOT, "configs", "gate8c0", "audit.yaml")


def cfg():
    import yaml
    return yaml.safe_load(open(CFG))


def default_device() -> str:
    d = os.environ.get("GATE8_DEVICE")
    return d if d else f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"
