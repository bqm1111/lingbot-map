"""Gate 13 -- oracle observed-geometry completion ceiling. Paired M (monocular) vs O (oracle) inputs.

M: the frozen Gate 12 H input cache (deployable local-reset L, dense LingBot rays, hard mapper), [S, 0, F].
O: LiDAR of the five causal sweeps t-4..t, z-buffered into the calibrated camera lattice (Gate 9 LD), GT
   relative poses, through the same hard mapper (VariantMapper band 0.2 / margin 0.2 / carving == core
   IncrementalMapper), [S, 0, F]. Non-deployable; measures the completion model's geometry headroom.
Everything after the 3-channel input is Gate 12's frozen recipe (same anchors, draws, crops, targets,
loss weights, initial weights, optimiser, schedule, steps, final-step checkpoint, evaluation regions).
"""
from __future__ import annotations
import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G12_DIR = os.path.join(REPO_ROOT, "tools", "gate12_suf")
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
ART = os.path.join(REPO_ROOT, "artifacts", "gate13_oracle")
G12_ART = os.path.join(REPO_ROOT, "artifacts", "gate12_suf")
DATA_ROOT = os.environ.get("G13_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate13_oracle")
O_ROOT = os.path.join(DATA_ROOT, "inputs_O")
CONDITIONS = ("M", "O")
SEEDS = (0, 1, 2)
sys.path.insert(0, G12_DIR)
import g12                                                                                   # noqa: E402

TRAIN_DRIVES, VAL_DRIVE = g12.TRAIN_DRIVES, g12.VAL_DRIVE


def o_path(drive, t):
    return os.path.join(O_ROOT, drive, f"{t:05d}.npz")


def m_path(drive, t):
    return g12.input_path(drive, t)


__all__ = [k for k in dir() if not k.startswith("_")]
