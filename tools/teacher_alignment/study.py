"""Study constants for the teacher-alignment control, layered on the feature study.

Importing this module *re-points* ``tools.feature_study.study`` at the new artefact
directory and the new pair of conditions. Every feature-study tool that is reused here
(source evaluation, transfer, prediction dump) reads those module attributes at call time,
so they run unchanged against this study's checkpoints and write into this study's
directory. Historical artefacts under ``artifacts/feature_completion_study`` are read only.

Two conditions, differing in one thing -- *where* the class-probability target sits:

* ``R`` -- aligned: the future-teacher vector at its own supervised voxel;
* ``S`` -- shuffled: the same vectors, permuted among supervised voxels within
  (z-layer, map-observed) strata of each training crop.

Both use ``L_occ + 0.5 * L_KL`` with the trunk gradient of the auxiliary term rescaled to a
fixed fraction ``rho`` of the occupancy gradient (``train.py``).
"""

from __future__ import annotations

import os

# import order matters: this pins G8C1_DATA_ROOT before anything under core is imported
from tools.feature_study import study as FS

ART = os.path.join(FS.REPO_ROOT, "artifacts", "teacher_alignment_study")
CONDITIONS = ("R", "S")
SEEDS = FS.SEEDS
#: the study this one is layered on, read-only
FEATURE_STUDY_ART = FS.ART
#: historical B, the checkpoint rho is calibrated from
HISTORICAL_B_SEED0 = os.path.join(FS.ART, "checkpoints", "fcs_B_seed0_last.pt")
#: numerical floor below which a gradient norm counts as degenerate
NORM_FLOOR = 1e-12


def run_tag(condition: str, seed: int) -> str:
    return f"tas_{condition}_seed{seed}"


def checkpoint_path(condition: str, seed: int) -> str:
    return os.path.join(ART, "checkpoints", f"{run_tag(condition, seed)}_last.pt")


def manifest_path(condition: str) -> str:
    return os.path.join(ART, f"frozen_manifest_{condition}.json")


def rho_path() -> str:
    return os.path.join(ART, "rho_calibration.json")


def activate() -> None:
    """Point the feature-study module at this study. Idempotent."""
    FS.ART = ART
    FS.CONDITIONS = CONDITIONS
    FS.run_tag = run_tag
    FS.checkpoint_path = checkpoint_path
    FS.manifest_path = manifest_path
    # both conditions train the 25-class head, so both get the semantic-threshold rule
    FS.SEMANTIC_CONDITIONS = CONDITIONS


activate()

# re-export what callers use
REPO_ROOT = FS.REPO_ROOT
require_data_root = FS.require_data_root
write_json = FS.write_json
sha256_file = FS.sha256_file
sha256_json = FS.sha256_json
environment = FS.environment
eval_mask_path = FS.eval_mask_path
G8C1_DATA_ROOT = FS.G8C1_DATA_ROOT

__all__ = ["ART", "CONDITIONS", "SEEDS", "FEATURE_STUDY_ART", "HISTORICAL_B_SEED0",
           "NORM_FLOOR", "run_tag", "checkpoint_path", "manifest_path", "rho_path",
           "activate", "REPO_ROOT", "require_data_root", "write_json", "sha256_file",
           "sha256_json", "environment", "eval_mask_path", "G8C1_DATA_ROOT", "FS"]
