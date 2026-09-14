"""Depth-source comparison (Gate 9 scale correction). No training.

Conditions, all fed to the frozen Gate 9 mapper as METRIC depth with the mapper scale forced to 1
and KITTI-360 GT relative poses (metric translations, unchanged):
  LD    LiDAR z-buffer camera depth                                   (reference)
  LBM   local-reset LingBot depth x the causal MoGe-2 clip scale, applied once   (== Gate 9 "PD")
  MG    frozen MoGe-2 metric z-depth (calibrated-FOV B variant cache)
  LBO   LingBot depth x a per-clip LiDAR-fitted scalar that REPLACES the MoGe scale   (diagnostic)
LBO scalar objective: s* = median_i z_LiDAR,i / z_LB,i over the clip's mask pixels (all five frames),
i.e. a minimiser of sum_i |log(s z_LB,i) - log z_LiDAR,i|. It is a depth-error fit, not an IoU optimum.

Masks (identical for every condition within a mask):
  S1 (primary)   Gate 9 LiDAR support AND MoGe-2 valid AND LingBot conf >= 1.5 (frozen gate)
  S0 (secondary) Gate 9 LiDAR support AND MoGe-2 valid  (Gate 9 PD convention, no LingBot gate)
"""
from __future__ import annotations
import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
ART = os.path.join(REPO_ROOT, "artifacts", "depth_source_comparison")
DATA_ROOT = os.environ.get("DSC_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_depth_source_comparison")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
SCALE_AUDIT_CLIPS = (4, 16, 1549, 1747)          # first 0006 clip + Gate 9's good / median / bad clips

# run name -> (depth source, support)
RUNS = {"LD_g9": ("lidar", "g9"), "LBM_g9": ("LBM", "g9"), "LD_gate_g9": ("lidar", "g9gate"), "LBM_gate_g9": ("LBM", "g9gate"),
        "LD": ("lidar", "S1"), "LBM": ("LBM", "S1"), "MG": ("MG", "S1"), "LBO": ("LBO", "S1"),
        "LD_u": ("lidar", "S0"), "LBM_u": ("LBM", "S0"), "MG_u": ("MG", "S0"), "LBO_u": ("LBO_u", "S0")}
G9_REPRO = {"LD_g9": "m_LD", "LBM_g9": "m_PD", "LD_gate_g9": "m_LD_gate", "LBM_gate_g9": "m_PD_gate"}
PRIMARY_COND = ("LD", "LBM", "MG", "LBO")
PIX_SOURCES = ("LBM", "MG", "LBO", "LBO_u")
NAME = {"LD": "LD", "LBM": "LB-M", "MG": "MG", "LBO": "LB-O", "LD_u": "LD", "LBM_u": "LB-M", "MG_u": "MG", "LBO_u": "LB-O",
        "LD_g9": "Gate 9 LD", "LBM_g9": "Gate 9 PD (= LB-M)", "LD_gate_g9": "Gate 9 LD_gate", "LBM_gate_g9": "Gate 9 PD_gate (= LB-M, gated)"}
BOOT_BLOCK = 20; N_BOOT = 1000


def moge_loader(src):
    """Frozen MoGe-2 B-variant caches -- the same files the clip scale was computed from
    (tools/local_registration_audit/run_local_lingbot.py): clip-keyed Gate 5.2 cache for 0006,
    per-frame Gate 8B cache for the training drives. Returns (z-depth m float64, mask, fov_x)."""
    from core.mapping import depth as D7
    from core.datasets import sources as S
    if src == "kitti360":
        index = D7.moge_frame_index("kitti360", REPO_ROOT)
        def load(key):
            r = D7.load_moge_frame("kitti360", key, index)
            return None if r is None else (r[0].astype(np.float64), r[1], r[2])
    else:
        def load(key):
            p = S.moge_path(src, key)
            if not os.path.exists(p):
                return None
            with np.load(p) as z:
                return z["moge_depth"].astype(np.float32).astype(np.float64), D7.unpack_mask(z["moge_mask"], z["mask_shape"]), float(z["fov_x_deg"])
    return load


__all__ = [k for k in dir() if not k.startswith("_")]
