"""Gate 12 -- does uncertainty-aware causal geometry improve learned completion? Paired H vs S/U/F.

Input (deployable, no LiDAR, no GT pose): frozen local-reset five-frame LingBot inference (L) per
anchor t (frames t-4..t): dense LingBot depth x causal MoGe-2 clip scale, LingBot confidence,
LingBot local poses (anchor camera = identity), LingBot pred_K, integrated by the frozen mapper
(Gate 9 VariantMapper == core IncrementalMapper, semantics off) through Gate 11's ChannelMapper.

  H   : S_H = mapper occupied (+-0.2 m band, hard carving to z - 0.2 m), U_H = 0, F_H = mapper free.
  SUF : rays with metric depth < 20 m only; S = mapper occupied with conservative carving to
        z - u_front; F = mapper free; U = inside the frozen Gate 11 U4 interval, neither S nor F.
Both on the SSCBench-KITTI-360 anchor grid (256x256x32, 0.2 m). Target: unchanged raw-LiDAR
geometry target (lingbot_k360_7drives/targets). Everything after the 3-channel input is identical.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
G11_DIR = os.path.join(REPO_ROOT, "tools", "gate11_uncertainty")
ART = os.path.join(REPO_ROOT, "artifacts", "gate12_suf")
DATA_ROOT = os.environ.get("G12_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf")
L_ROOT = os.path.join(DATA_ROOT, "local_lingbot")                  # stride-1 L clips, training drives
LR_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot"   # stride-3 L clips (0006)
INPUT_ROOT = os.path.join(DATA_ROOT, "inputs")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")
TARGET_ROOT = os.path.join(os.environ["G8C1_DATA_ROOT"], "targets")
EVAL_MASK_ROOT = os.path.join(os.environ["G8C1_DATA_ROOT"], "eval_masks")
TRAIN_DRIVES = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
VAL_DRIVE = "2013_05_28_drive_0006_sync"
CONDITIONS = ("H", "SUF")
SEEDS = (0, 1, 2)

# ---- frozen Gate 11 U4 rule (read from Gate 11's frozen rules file, never refitted) ----
G11_RULES = os.path.join(REPO_ROOT, "artifacts", "gate11_uncertainty", "uncertainty_rules.json")
G11_FREEZE = os.path.join(REPO_ROOT, "artifacts", "gate11_uncertainty", "code_freeze.json")
PROMPT_ROUNDED = {"a": 0.066, "b": 0.127, "g": [1.50, 1.16, 0.98, 0.88], "u_front": 0.308, "u_back": 1.395}
RANGE_CUTOFF_M = 20.0
BAND_MARGIN_M = 0.2
RANGE_BINS = (0, 5, 10, 20, 30, 1e9); RANGE_NAMES = ("0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m")
VOLUMES = ("S_H", "F_H", "S", "U", "F")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_rules():
    """Gate 11's frozen rules, checked against Gate 11's code freeze and the rounded values in the brief."""
    rules = json.load(open(G11_RULES))
    frozen = json.load(open(G11_FREEZE))["sha256"]["artifacts/gate11_uncertainty/uncertainty_rules.json"]
    assert sha256_file(G11_RULES) == frozen, "Gate 11 rules file differs from its code freeze"
    assert rules["U4"]["include"]
    got = {"a": round(rules["U2"]["a"], 3), "b": round(rules["U2"]["b"], 3), "g": [round(x, 2) for x in rules["U3"]["g"]],
           "u_front": round(-rules["U4"]["q16"], 3), "u_back": round(rules["U4"]["q84"], 3)}
    assert got == PROMPT_ROUNDED, f"frozen U4 coefficients do not round to the brief's values: {got}"
    return rules


def clip_path(drive, t):
    root = LR_ROOT if drive == VAL_DRIVE else L_ROOT
    return os.path.join(root, drive, f"{t:05d}.npz")


def input_path(drive, t):
    return os.path.join(INPUT_ROOT, drive, f"{t:05d}.npz")


def target_path(drive, t):
    return os.path.join(TARGET_ROOT, drive, f"{t:05d}.npz")


def unpack(z, key, n):
    return np.unpackbits(z[key])[:n].astype(bool)


__all__ = [k for k in dir() if not k.startswith("_")]
