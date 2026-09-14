"""Gate 10 -- depth-source sufficiency test. No training.

Is the geometry failure specific to LingBot depth, or are frozen monocular depth sources
generally insufficient for exact 0.2 m occupancy? Everything except the depth source is
Gate 9's: the local-registration audit's five-frame causal clips, KITTI-360 GT relative
poses, the anchor SSC velodyne grid, the lattice camera model, the LiDAR z-buffer support,
the reference/evaluation mask and the frozen mapper (VariantMapper at band 0.2 m, carve
margin 0.2 m, carving on == core IncrementalMapper, bit-identical).

Depth sources, all written to the mapper as METRIC depth with the scale state forced to 1:
  lidar  LiDAR z-buffer depth                                    (LD, practical ceiling)
  LB     raw local-reset LingBot depth, canonical units x 1       (no scale at all)
  LBM    LB x the clip's deployable MoGe-2 scale (run_local_lingbot.py: median over the five
         frames of gates.gate7b.scale.frame_candidate; no LiDAR)  == Gate 9 "PD"
  M      frozen MoGe-2 (calibrated-FOV B variant) z-depth itself
  LBL    LB x per-clip LiDAR oracle scalar            DIAGNOSTIC ORACLE, non-deployable
  ML     M  x per-clip LiDAR oracle scalar            DIAGNOSTIC ORACLE, non-deployable
Oracle scalar = median over the clip's matched pixels (all five frames) of z_lidar / z_source.
"""
from __future__ import annotations
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
ART = os.path.join(REPO_ROOT, "artifacts", "gate10_depth_source")
DATA_ROOT = os.environ.get("G10_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate10_depth_source")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")

PRED = ("LB", "LBM", "M", "LBL", "ML")                       # non-LiDAR depth sources
ORACLES = ("LBL", "ML")                                       # use LiDAR; never feed anything else
LABEL = {"LD": "LD (LiDAR depth)", "LB": "LB (raw LingBot, canonical)", "LBM": "LB-M (LingBot x causal MoGe scale)",
         "M": "M (MoGe-2 depth)", "LBL": "LB-LiDAR (oracle scalar)", "ML": "M-LiDAR (oracle scalar)"}
# mapper runs: name -> (depth source, support). "gate9" = Gate 9's LiDAR z-buffer support (reproduction
# of Gate 9 LD / PD); "matched" = Gate 9 support AND MoGe-valid (the common pixel set of this gate).
RUNS = {"LD_g9": ("lidar", "gate9"), "LBM_g9": ("LBM", "gate9"),
        "LD": ("lidar", "matched"), "LB": ("LB", "matched"), "LBM": ("LBM", "matched"), "M": ("M", "matched"),
        "LBL": ("LBL", "matched"), "ML": ("ML", "matched")}
MAIN = ("LD", "LB", "LBM", "M", "LBL", "ML")
CONF_BINS = (0, 1.5, 3, 6, 12, 1e9); CONF_NAMES = ("<1.5 (gated)", "1.5-3", "3-6", "6-12", ">=12")   # Gate 9 analyze.py bins
EDGE_NAMES = {0: "non-edge (<5 %)", 1: "moderate (5-20 %)", 2: "strong (>=20 %)", 3: "isolated"}     # Gate 9 edge classes
THRESH = (0.1, 0.2, 0.4, 0.6, 1.0)
BOOT_BLOCK = 20; N_BOOT = 1000

__all__ = [k for k in dir() if not k.startswith("_")]
