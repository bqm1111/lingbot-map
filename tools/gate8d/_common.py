"""Shared setup for the Gate 8D tools."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8d")
ART_ROOT = os.path.join(REPO_ROOT, "artifacts")
SEEDS = (0, 1, 2)


def default_device() -> str:
    d = os.environ.get("GATE8_DEVICE")
    return d if d else f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def sha256(p: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
