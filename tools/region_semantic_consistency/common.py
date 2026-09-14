"""Region-semantic-consistency study: is the persistent semantic unit a coherent 3D region
rather than a voxel? Frozen everything; no training. Reuses the per-frame future teacher
evidence of the target-consistency study and its interleaved frame-disjoint groups.
"""
from __future__ import annotations
import os
from tools.future_semantic_completion import study as FSC
from tools.future_semantic_completion.target_consistency import common as TC

ART = os.path.join(FSC.REPO_ROOT, "artifacts", "region_semantic_consistency")
EVIDENCE_DIR = TC.EVIDENCE_DIR
#: cubic block side in voxels (0.2 m each); regions are 26-connected components of
#: LiDAR-occupied voxels inside one block
SCALES = {"small": 3, "medium": 5, "large": 9, "xlarge": 15}
GROUPING = "interleaved"
#: local-shuffle strata: z-layer bin height and xy cell side, in voxels
LOCAL_Z_BIN = 4
LOCAL_XY_CELL = 20
#: size strata for the within-anchor shuffle (number of teacher observations in A+B)
SIZE_EDGES = (1, 2, 4, 8, 16, 10 ** 9)

require_data_root = FSC.require_data_root
sha256_file, sha256_json, write_json, environment = (FSC.sha256_file, FSC.sha256_json,
                                                     FSC.write_json, FSC.environment)

__all__ = ["ART", "EVIDENCE_DIR", "SCALES", "GROUPING", "LOCAL_Z_BIN", "LOCAL_XY_CELL",
           "SIZE_EDGES", "require_data_root", "sha256_file", "sha256_json", "write_json",
           "environment"]
