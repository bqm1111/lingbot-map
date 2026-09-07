"""Shared bits for the Gate 8B tools: GPU pool default and the Gate 8B artifact root."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
import gates.gate8b.sources  # noqa: E402,F401  -- registers k360_train with gate8.sources

ART = os.path.join(REPO_ROOT, "artifacts", "gate8b")


def default_device() -> str:
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"
