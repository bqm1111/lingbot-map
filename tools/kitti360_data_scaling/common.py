"""KITTI-360 data-scaling experiment: does more training coverage improve the existing
semantic-completion model under the same causal LingBot inputs, once the privileged
semantic targets are registered with trusted geometry instead of the drifting mapper?

Everything trainable, the objective, the causal input contract and the step budget are
the adopted Gate 8C-1 configuration. What changes: (i) the semantic targets are rebuilt
(``build_targets_gtsem.py``) by projecting the raw-LiDAR occupied voxels of the anchor
grid into each future camera with the ground-truth calibration/poses, a LiDAR z-buffer
visibility test, and direct sampling of the cached Trident-H probabilities -- the LingBot
future mapper is not used; (ii) the training pool is a frozen manifest (25 % nested
blocks vs 100 %).
"""
from __future__ import annotations
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO_ROOT, "artifacts", "kitti360_data_scaling")
#: the adopted (ceiling-fixed) samples: inputs are copied from here byte for byte
SOURCE_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky"
#: the regenerated samples (same layout, same input arrays, GT-registered semantic targets)
DATA_ROOT = os.environ.get("KDS_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky_gtsem")
FEATURE_STUDY_MASKS = "/media/SSD1/MINH_DATASETS/lingbot_feature_study/source_eval_masks"
SSCBENCH_LABELS = "/media/SSD1/MINH_DATASETS/sscbench_kitti360/preprocess/labels"
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "gate8c1_sky", "seed0.yaml")

NATIVE_HW = (376, 1408); LATTICE_HW = (140, 518)
FUTURE_FRAMES = 20
DEPTH_MIN, DEPTH_MAX = 1.5, 60.0
BORDER_PX = 2
ZBUF_WINDOW = 7
VIS_TOL = (0.3, 0.01)                 # |z - z_ref| < a + b z (the audit's strict rule)
SEM_W_MIN = 1.0                       # >= one visible future observation
REQUIRE_HALF_AGREEMENT = True         # only a contradiction between the two window halves disqualifies (adopted rule)
BLOCK_ANCHORS = 20                    # manifest / bootstrap block size (100 native frames, >= the 20-frame future horizon)
SUBSET_EVERY_BLOCK = 4                # 25 %: every 4th block, offset 0
RUNS = {"small": 0.25, "full": 1.0}
SEEDS = (0, 1)
CKPT_EVERY = 1000
CURVE_ANCHOR_STRIDE = 5               # intermediate checkpoints scored on every 5th 0006 anchor


def sample_path(drive: str, idx: int) -> str:
    return os.path.join(DATA_ROOT, "samples", drive, f"{idx:05d}.npz")


def source_sample_path(drive: str, idx: int) -> str:
    return os.path.join(SOURCE_ROOT, "samples", drive, f"{idx:05d}.npz")


def run_tag(run: str, seed: int) -> str:
    return f"kds_{run}_seed{seed}"


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha256_json(obj):
    import hashlib, json
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    import json
    import numpy as np
    def d(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=d)
    os.replace(tmp, path)


def environment():
    import subprocess, sys
    import numpy as np, torch
    def g(*a):
        try: return subprocess.check_output(a, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return ""
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git": {"commit": g("git", "rev-parse", "HEAD"), "branch": g("git", "rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(g("git", "status", "--porcelain"))}}


__all__ = [k for k in dir() if not k.startswith("_")]
