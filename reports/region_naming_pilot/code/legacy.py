"""The adapter onto the legacy implementation. Nothing else in this experiment imports it directly.

Reuse policy: the frozen grid conventions, transforms, occupancy network, teacher readout, the B0 rule
and the scorer are **imported, not copied**, so they cannot silently diverge from the reference that
produced the published numbers. Isolation is enforced instead of achieved by copying:

  * ``PYTHONDONTWRITEBYTECODE=1`` (asserted in ``paths``) stops any ``__pycache__`` write;
  * no legacy *script* is executed -- only pure modules are imported;
  * every legacy file that ends up in ``sys.modules`` is hashed by :func:`record`, together with the
    repository commit, so the exact reused source is pinned in the manifest.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import Dict

import paths as P

if P.LEGACY_REPO not in sys.path:
    sys.path.insert(0, P.LEGACY_REPO)
assert P.REPRO_REPO not in sys.path, "the reproduction repository must never be importable here"

from core.datasets import kitti360 as K3, kitti360_splits as SRC                       # noqa: E402
from core.datasets.audit import FileAudit, ForbiddenAccess                             # noqa: E402
from tools.baseline_reconstruction.current.geometry import LocalClip, frame_inputs     # noqa: E402
from tools.semantic_completion import config as C                                      # noqa: E402
from tools.semantic_completion.evaluation import baseline_nearest as B0                # noqa: E402
from tools.semantic_completion.evaluation import semantic_metrics as SM                # noqa: E402
from tools.semantic_completion.evaluation.regions import (                             # noqa: E402
    PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS, build_regions)
from tools.semantic_completion.evidence import lift as LF                              # noqa: E402
from tools.semantic_completion.evidence import trident as TR                           # noqa: E402
from tools.semantic_completion.evidence.build import anchor_ids, camera_to_grid        # noqa: E402
from tools.semantic_completion.frozen_geometry import (                                # noqa: E402
    geometry_input, load_frozen_geometry, occupancy_and_trunk, sha256_file, state_digest)

GRID_DIMS = C.GRID_DIMS
N_VOXELS = C.N_VOXELS
VOXEL_SIZE_M = C.VOXEL_SIZE_M
GRID_ORIGIN_M = C.GRID_ORIGIN_M
N_CLASSES = C.N_CLASSES
TEACHER_DIM = C.TEACHER_DIM
GEOMETRY_SEED = 0

#: forbidden on the prediction side: target benchmarks, other datasets, and the human labels themselves
PREDICT_FORBIDDEN = tuple(C.EVAL_FORBIDDEN_TOKENS) + ("eval_masks", "semantickitti", "/voxels/", "velodyne")
#: forbidden during evaluation: the other benchmarks only (human labels are the measuring stick there)
EVAL_FORBIDDEN = tuple(C.EVAL_FORBIDDEN_TOKENS)


def load_hard_map_flat(t: int, device):
    """``(S, F)`` flat bool, the frozen hard map of anchor ``t`` -- read straight from the Gate 12 cache."""
    import numpy as np
    import torch
    with np.load(C.Paths().input_path(P.DRIVE, t)) as z:
        S = torch.from_numpy(np.unpackbits(z["S_H"])[:N_VOXELS].astype(bool)).to(device)
        F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:N_VOXELS].astype(bool)).to(device)
    return S, F_


def load_valid_and_gt_occ(t: int, device):
    """Gate 12 occupancy regions: ``(valid, gt_occupied)`` flat bool. Evaluation only."""
    import numpy as np
    import torch
    with np.load(os.path.join(P.GATE12_REGIONS, f"{t:05d}.npz")) as z:
        valid = torch.from_numpy(np.unpackbits(z["valid"])[:N_VOXELS].astype(bool)).to(device)
        occ = torch.from_numpy(np.unpackbits(z["occupied"])[:N_VOXELS].astype(bool)).to(device) & valid
    return valid, occ


def load_sem_gt(t: int, device):
    """Human SSCBench labels on the grid. Evaluation and debug reporting only, never an input."""
    import numpy as np
    import torch
    with np.load(C.Paths().eval_mask_path(t)) as z:
        return torch.from_numpy(z["sem_gt"].astype(np.int64)).to(device)


def _git(args):
    return subprocess.run(["git", "-C", P.LEGACY_REPO] + args, capture_output=True, text=True).stdout.strip()


def record() -> Dict:
    """Provenance of every legacy file actually imported, plus the repository state."""
    files = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.realpath(f).startswith(os.path.realpath(P.LEGACY_REPO) + os.sep):
            rel = os.path.relpath(os.path.realpath(f), os.path.realpath(P.LEGACY_REPO))
            with open(f, "rb") as fh:
                files[rel] = hashlib.sha256(fh.read()).hexdigest()
    return {"legacy_repo": P.LEGACY_REPO, "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": _git(["status", "--porcelain"]).splitlines(),
            "imported_files_sha256": dict(sorted(files.items()))}
