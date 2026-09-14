"""Gate 9 -- depth-to-occupancy geometry audit. No training.

Is the remaining raw geometry failure caused by LingBot depth, or by the depth -> occupancy
mapper? Same five-frame causal clips as the local-registration audit, KITTI-360 GT relative
poses throughout, anchor camera = identity, evaluation on the anchor's SSC velodyne grid.
"""
from __future__ import annotations
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO_ROOT, "artifacts", "gate9_depth_occupancy")
DATA_ROOT = os.environ.get("G9_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
NATIVE_HW = (376, 1408); LATTICE_HW = (140, 518)
DIMS = (256, 256, 32); ORIGIN = (0.0, -25.6, -2.0); VS = 0.2
SUPPORT_CONF = 1e3                         # confidence written at LiDAR-supported pixels: passes the gate, no other effect (semantics off)
RANGE_BINS = (0, 5, 10, 20, 30, 1e9); RANGE_NAMES = ("0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m")
EDGE_WIN = 5; EDGE_EDGES = (0.05, 0.20)    # max relative LiDAR depth jump to supported neighbours in a 5x5 lattice window
PLANAR_WIN = 5; PLANAR_MIN_PTS = 6; PLANAR_MAX = 0.01   # smallest-eigenvalue fraction of the local LiDAR point covariance
PIXEL_KEEP = 0.2                           # fraction of supported pixels stored per clip (deterministic)

# the oracle ladder and the mapper matrix: (depth source, support, band_half_m, carve, carve_margin_m, gate)
RUNS = {
    "LD":              ("lidar", "lidar", 0.2, True, 0.2, "none"),
    "PD":              ("lingbot", "lidar", 0.2, True, 0.2, "none"),
    "GF":              ("lingbot", "dense", 0.2, True, 0.2, "lingbot_conf"),
    "Gs":              ("lingbot_oracle_scale", "dense", 0.2, True, 0.2, "lingbot_conf"),
    # mapper matrix on LiDAR depth
    "LD_band0":        ("lidar", "lidar", 0.0, True, 0.2, "none"),
    "LD_band0.4":      ("lidar", "lidar", 0.4, True, 0.2, "none"),
    "LD_nocarve":      ("lidar", "lidar", 0.2, False, 0.2, "none"),
    "LD_band0_nocarve": ("lidar", "lidar", 0.0, False, 0.2, "none"),
    "LD_margin0":      ("lidar", "lidar", 0.2, True, 0.0, "none"),
    "LD_margin0.1":    ("lidar", "lidar", 0.2, True, 0.1, "none"),
    "LD_margin0.4":    ("lidar", "lidar", 0.2, True, 0.4, "none"),
    "LD_gate":         ("lidar", "lidar_and_conf", 0.2, True, 0.2, "none"),
    # the same key switches on LingBot depth (for endpoint-only vs carving and support effects)
    "PD_nocarve":      ("lingbot", "lidar", 0.2, False, 0.2, "none"),
    "PD_band0_nocarve": ("lingbot", "lidar", 0.0, False, 0.2, "none"),
    "PD_margin0.4":    ("lingbot", "lidar", 0.2, True, 0.4, "none"),
    "PD_gate":         ("lingbot", "lidar_and_conf", 0.2, True, 0.2, "none"),
    "GF_nocarve":      ("lingbot", "dense", 0.2, False, 0.2, "lingbot_conf"),
}

__all__ = [k for k in dir() if not k.startswith("_")]
