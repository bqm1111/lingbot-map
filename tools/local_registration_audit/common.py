"""Local-registration audit: can LingBot-Map give geometrically consistent causal 3D
geometry once long-horizon streaming pose drift is removed? No training.

Conditions on the same five-frame causal clips t-4 .. t:
  S   streaming cache (depth, pose, frozen stream scale) -- current behaviour
  L   LingBot re-run on the five images alone (fresh KV cache, RoPE counter at 0), clip scale
  Lp  L's relative poses with S's depth (pose isolated; depth held fixed)
  G   S's depth with KITTI-360 GT relative poses (diagnostic ceiling for pose)
  GLd L's depth with GT poses (depth-only comparison of S vs L)
"""
from __future__ import annotations
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO_ROOT, "artifacts", "local_registration_audit")
DATA_ROOT = os.environ.get("LRA_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")   # current raw-LiDAR targets
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
ANCHOR_STRIDE = 3
OFFSETS = (-4, -3, -2, -1, 0)
CONDITIONS = ("S", "L", "Lp", "G", "GLd", "Gs")
DIAGNOSTIC_ORACLE = ("Gs",)   # uses LiDAR depth scale; not a deployable condition


def clip_path(drive, t):
    return os.path.join(DATA_ROOT, "local_lingbot", drive, f"{t:05d}.npz")


__all__ = [k for k in dir() if not k.startswith("_")]
