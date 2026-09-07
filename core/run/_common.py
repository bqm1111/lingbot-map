# extracted from tools/gate8c1/_common.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Shared setup for the Gate 8C-1 tools."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.config import REPO_ROOT                                          # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8c1")
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
