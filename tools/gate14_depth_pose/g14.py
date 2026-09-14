"""Gate 14 -- can a stronger causal camera-only depth source move observed geometry toward the oracle?

Stage 1 (no training): a 2 x 2 depth x pose observed-map audit on the Gate 9 / Gate 13 five-frame clips.
  depth  LBM  = s_clip * z_LingBot (causal MoGe-2 clip scale, applied once; the deployed baseline)
         NEW  = MUSt3R-512 (official naver/must3r code, checkpoint MD5 8854f948a8674fb1740258c1872f80dc),
                run on frames t-4..t only, per-view z-depth (pts3d_local), resampled onto the LingBot lattice,
                metric scale from the same causal MoGe-2 clip-scale estimator
  pose   L    = local-reset LingBot relative poses (metric via s_clip) with LingBot pred_K  (deployable)
         G    = KITTI-360 GT relative poses with the calibrated lattice camera            (diagnostic)
  Conditions LBM-L, LBM-G, NEW-L, NEW-G, each on S_shared (primary) and S_native (secondary);
  LD-G (Gate 9 / Gate 13 LiDAR camera depth + GT pose) is reused as the accurate-geometry reference.
Everything else is Gate 9: anchor SSC grid (0.2 m), frozen hard mapper, five-sweep LiDAR reference and mask.
"""
from __future__ import annotations
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
ART = os.environ.get("G14_ART", os.path.join(REPO_ROOT, "artifacts", "gate14_depth_pose"))
DATA_ROOT = os.environ.get("G14_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r")
MUST3R_SRC = os.path.join(DATA_ROOT, "src", "must3r")
MUST3R_CKPT = "/media/SSD1/MINH_DATASETS/occany_checkpoints/MUSt3R_512.pth"
MUST3R_MD5 = "8854f948a8674fb1740258c1872f80dc"
MUST3R_PY = "/home/minh/workspace/third_party/occany_env/bin/python"          # torch 2.7.1+cu128, roma, xformers; nothing installed into it
MANIFEST_DIR = os.path.join(DATA_ROOT, "manifests")
NEW_CACHE = os.path.join(DATA_ROOT, "must3r_depth")
CLIP_ROOT = os.path.join(DATA_ROOT, "clips")
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
LR_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot"
G9_CLIPS = "/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy/clips"
O_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate13_oracle/inputs_O"
PROMOTION_RULE = os.path.join(ART, "promotion_rule.json")
RANGE_BINS = (0, 5, 10, 20, 30, 1e9); RANGE_NAMES = ("0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m")
CONDITIONS = ("LBM-L", "LBM-G", "NEW-L", "NEW-G")
SUPPORTS = ("shared", "native")


def clip_manifest(drive):
    return os.path.join(MANIFEST_DIR, f"{drive}.json")


def new_cache_path(drive, t):
    return os.path.join(NEW_CACHE, drive, f"{t:05d}.npz")


__all__ = [k for k in dir() if not k.startswith("_")]
