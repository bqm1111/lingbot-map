"""Future-semantic-completion study: shared paths, constants and provenance.

One question: **can a small causal head predict the semantic representation of voxels
that are currently unobserved but later confirmed occupied, using future observations as
privileged supervision?** Geometry is frozen; nothing here touches occupancy.

The teacher representation is the one the repository already caches and lifts: Trident-H's
vocabulary-conditioned probabilities, fused per voxel over the twenty frames after the
anchor by the frozen future mapper (``core.supervision.targets``, key ``fut_p``). It is a
continuous 25-d vector on the union simplex -- the teacher's open-vocabulary output over a
*fixed* vocabulary. No language-aligned dense embedding is cached and the SAM features of
the feature study are not text-aligned, so this study predicts the probability vector and
says so plainly (``core.datasets.union_vocab`` flags the same limitation).

Data boundary, unchanged: KITTI-360 drives 0003/0007/0010 train, drive 0006 validates,
SemanticKITTI and Occ3D-nuScenes are opened once, after the manifest is frozen.
"""
from __future__ import annotations
import os
from typing import Dict

# feature-study module first: it pins G8C1_DATA_ROOT before core.datasets is imported
from tools.feature_study import study as FS

REPO_ROOT = FS.REPO_ROOT
ART = os.path.join(REPO_ROOT, "artifacts", "future_semantic_completion")
FEATURE_STUDY_DATA = FS.DATA_ROOT            # drive-0006 measured partition masks live here
require_data_root = FS.require_data_root
sha256_file, sha256_json, write_json, environment = (FS.sha256_file, FS.sha256_json,
                                                     FS.write_json, FS.environment)

SEEDS = (0, 1, 2)
#: The frozen geometry: the adopted Gate 8C-1 (ceiling-fixed) checkpoints, one per seed.
FROZEN_MANIFEST = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "frozen_manifest_sky.json")


def frozen_checkpoint(seed: int) -> str:
    import json
    m = json.load(open(FROZEN_MANIFEST))
    return os.path.join(REPO_ROOT, m["seeds"][str(seed)]["checkpoint"])


CONDITIONS = ("B0", "B1", "B2")
#: B1 falls back to B0 beyond this many voxels (0.2 m each) from the nearest map voxel
#: that carries teacher evidence.
B1_MAX_DIST_VOX = 15.0


def run_tag(seed: int) -> str:
    return f"fsc_seed{seed}"


def head_checkpoint(seed: int) -> str:
    return os.path.join(ART, "checkpoints", f"{run_tag(seed)}_head_last.pt")


def eval_mask_path(drive: str, idx: int) -> str:
    return FS.eval_mask_path(drive, idx)


def manifest_path() -> str:
    return os.path.join(ART, "frozen_manifest.json")


__all__ = ["REPO_ROOT", "ART", "SEEDS", "FROZEN_MANIFEST", "CONDITIONS", "B1_MAX_DIST_VOX",
           "frozen_checkpoint", "run_tag", "head_checkpoint", "eval_mask_path",
           "manifest_path", "require_data_root", "sha256_file", "sha256_json", "write_json",
           "environment"]
