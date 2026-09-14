"""Shared paths, constants and provenance for the feature-supervision study.

One question: does predicting a frozen segmentation teacher's *dense features* complete
map-unobserved space better than predicting its *class probabilities*?

Three conditions, identical in every other respect:

* ``A`` -- ``L_occ``                         (occupancy loss only)
* ``B`` -- ``L_occ + 0.5 * L_KL``            (the adopted Gate 8C-1 objective, retrained)
* ``C`` -- ``L_occ + 0.5 * L_feature``       (cosine distance to SAM-H features)

Everything else is the adopted ceiling-fixed configuration
(``artifacts/gate8c1/frozen_manifest_sky.json``): replicate padding, the 256x256x32 grid at
0.2 m, the frozen five-frame metric scale, the 32-channel input encoding, the residual
occupancy lock at |log-odds| >= 2, bf16 autocast, and the ``past5`` inference protocol.

The data boundary is the gate's: KITTI-360 drives 0003/0007/0010 train, drive 0006
validates and selects, SemanticKITTI and Occ3D-nuScenes are opened once, at the end,
through one frozen manifest per condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, List, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Every artefact this study writes.
ART = os.path.join(REPO_ROOT, "artifacts", "feature_completion_study")
#: Bulk caches (SAM features, feature voxel targets, source evaluation masks).
DATA_ROOT = os.environ.get("FCS_DATA_ROOT",
                           "/media/SSD1/MINH_DATASETS/lingbot_feature_study")

#: The behavioural reference this study inherits its frozen conventions from.
REFERENCE_MANIFEST = os.path.join(REPO_ROOT, "artifacts", "gate8c1",
                                  "frozen_manifest_sky.json")
#: The occupancy targets and causal inputs are the adopted ceiling-fixed ones, reused as-is.
#:
#: ``core.datasets.kitti360_splits`` resolves this from ``G8C1_DATA_ROOT`` **at import time**,
#: and its default is the PRE-ceiling-fix root, whose targets predate the open-sky rule. The
#: two roots hold the same file names, so reading the wrong one fails silently rather than
#: loudly. Setting it here -- and importing this module before anything under ``core`` -- is
#: what keeps the study on the adopted configuration; :func:`require_data_root` then proves
#: it did.
G8C1_DATA_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky"
os.environ["G8C1_DATA_ROOT"] = G8C1_DATA_ROOT


def require_data_root() -> str:
    """Fail loudly if the gate's splits module did not pick up the adopted root."""
    from core.datasets import kitti360_splits as SRC
    if os.path.abspath(SRC.G8C1_ROOT) != os.path.abspath(G8C1_DATA_ROOT):
        raise SystemExit(
            f"wrong Gate 8C-1 data root: kitti360_splits resolved {SRC.G8C1_ROOT!r} but this "
            f"study requires {G8C1_DATA_ROOT!r} (the ceiling-fixed one the adopted manifest "
            f"declares). Import tools.feature_study.study before core.datasets.")
    return SRC.G8C1_ROOT

CONDITIONS = ("A", "B", "C")
SEEDS = (0, 1, 2)

#: SAM-H, as built inside the frozen Trident-H teacher stack.
SAM_CKPT = os.environ.get("GATE6_SAM_CKPT",
                          "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
#: ``sam.image_encoder`` output: 256 channels on a 64x64 lattice over the padded square.
SAM_NATIVE_DIM = 256
SAM_GRID = 64
SAM_IMG_SIZE = 1024

#: The PCA target dimension the feature head predicts.
FEATURE_DIM = 64
#: Fused teacher weight a voxel needs before it carries a feature target. Same rule and
#: same constant as the class-probability target (``core/run/build_samples.SEM_W_MIN``).
SEM_W_MIN = 1.0


def sam_feature_path(source: str, key: str) -> str:
    return f"{DATA_ROOT}/sam_vith/{source}/{key}.npz"


def pca_path() -> str:
    return f"{DATA_ROOT}/pca/sam_vith_pca{FEATURE_DIM}.npz"


def feature_target_path(drive: str, stream_index: int) -> str:
    return f"{DATA_ROOT}/feature_targets/{drive}/{stream_index:05d}.npz"


def eval_mask_path(drive: str, stream_index: int) -> str:
    return f"{DATA_ROOT}/source_eval_masks/{drive}/{stream_index:05d}.npz"


def run_tag(condition: str, seed: int) -> str:
    return f"fcs_{condition}_seed{seed}"


def checkpoint_path(condition: str, seed: int) -> str:
    return os.path.join(ART, "checkpoints", f"{run_tag(condition, seed)}_last.pt")


def manifest_path(condition: str) -> str:
    return os.path.join(ART, f"frozen_manifest_{condition}.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def git_revision() -> Dict[str, str]:
    def _run(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain"))}


def environment() -> Dict[str, object]:
    import numpy as np
    import torch
    return {"python": os.sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "git": git_revision()}


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_default)
    os.replace(tmp, path)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


__all__ = ["REPO_ROOT", "ART", "DATA_ROOT", "REFERENCE_MANIFEST", "G8C1_DATA_ROOT",
           "CONDITIONS", "SEEDS", "SAM_CKPT", "SAM_MODEL_TYPE", "SAM_NATIVE_DIM",
           "SAM_GRID", "SAM_IMG_SIZE", "FEATURE_DIM", "SEM_W_MIN", "sam_feature_path",
           "pca_path", "feature_target_path", "eval_mask_path", "run_tag",
           "checkpoint_path", "manifest_path", "sha256_file", "sha256_json",
           "git_revision", "environment", "write_json", "require_data_root"]
