"""Everything Gate 7A is allowed to decide, decided in one place.

``tools/gate7a/precommit.py`` writes the pinned YAML **from this module**, so the config
and the code cannot drift: change a radius here and the pinned SHA-256 changes with it.

Gate 7A is explicitly a **target-dependent oracle analysis**. It is not a deployable
prediction, it produces no new prediction artifact, and it does not alter Gate 6's
target-free claim or any Gate-6 prediction hash.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Predeclared analysis grid
# --------------------------------------------------------------------------- #
DATASETS: Tuple[str, ...] = ("semantickitti", "occ3d", "kitti360")

#: Metric completion radii, in metres. ``0.0`` must reproduce the base volume exactly.
RADII_M: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 2.0, 4.0)

#: Horizontal-range bands, in metres.
RANGE_BANDS: Tuple[Tuple[float, float], ...] = ((0, 10), (10, 20), (20, 30), (30, 40),
                                                (40, 60))

#: Base occupancy conditions. ``B-D`` is the primary, exactly as in Gate 6.
CONDITIONS: Tuple[str, ...] = ("B-R", "B-D")
PRIMARY_CONDITION = "B-D"

#: Constructions evaluated at every radius.
CONSTRUCTIONS: Tuple[str, ...] = ("oracle", "morph")

#: Propagation-distance intervals for the semantic-transport analysis, in metres.
#: Right-closed; the left edge of the first interval is exclusive so that a voxel already
#: in the base support (distance exactly 0) is never counted as "propagated".
TRANSPORT_INTERVALS_M: Tuple[Tuple[float, float], ...] = ((0.0, 0.4), (0.4, 0.8),
                                                          (0.8, 1.2), (1.2, 2.0),
                                                          (2.0, 4.0))

#: Miss-distance histogram: bin width in metres and the largest edge. Quantiles are read
#: off this histogram, so its resolution bounds the precision of a reported quantile.
MISS_HIST_BIN_M = 0.05
MISS_HIST_MAX_M = 80.0

BOOTSTRAP = {"n_boot": 10000, "seed": 0, "alpha": 0.05,
             "units": {"occ3d": "official nuScenes scene",
                       "semantickitti": "contiguous block of 20 clips (one drive)",
                       "kitti360": "contiguous block of 20 clips (one drive)"},
             "block_size": 20,
             "independence_caveat": ("blocks from a single drive are not independent "
                                     "scenes and are never described as such")}

# --------------------------------------------------------------------------- #
# Definitions, written out so they cannot be reinterpreted after the fact
# --------------------------------------------------------------------------- #
DISTANCE_DEFINITION = {
    "quantity": "Euclidean distance between voxel CENTRES, in metres",
    "grid": ("the dataset's NATIVE evaluation grid: 0.2 m for SemanticKITTI and "
             "SSCBench-KITTI-360, 0.4 m for the official Occ3D-nuScenes evaluation grid"),
    "method": ("exact Euclidean distance transform (scipy.ndimage.distance_transform_edt) "
               "on the voxel lattice with unit sampling, multiplied by the voxel size; "
               "voxel indices are never treated as metres"),
    "source_set": "the occupied voxels of the base condition (B-R or B-D)",
    "empty_base": ("a clip whose base condition occupies no voxel has distance +inf "
                   "everywhere; it adds nothing under either construction and is counted "
                   "and reported separately"),
    "range_band_distance": ("horizontal (xy) norm of the voxel centre in the benchmark's "
                            "own grid frame -- identical to gate6.metrics.band_masks"),
}

ORACLE_DEFINITION = {
    "formula": "P_oracle(r) = P_base OR (GT_occupied AND valid AND dist_to_P_base <= r)",
    "adds": "only ground-truth-occupied voxels inside the official evaluation mask",
    "removes": "nothing; P_base is always a subset of P_oracle(r)",
    "false_positive_count": "unchanged from the base at every radius, by construction",
    "status": ("OPTIMISTIC UPPER BOUND for any model restricted to that local correction "
               "region. It uses the target to decide which voxels are added and is NOT a "
               "deployable result."),
}

MORPHOLOGY_DEFINITION = {
    "formula": "P_morph(r) = P_base OR (valid AND dist_to_P_base <= r)",
    "adds": "every valid voxel inside the radius, true positive or not",
    "status": "deployable, non-learned, target-free spatial expansion",
    "valid_restriction": ("expansion is restricted to the official evaluation mask. This "
                          "is metric-neutral -- voxels outside the mask are scored by no "
                          "benchmark -- and keeps the added-volume ratio interpretable"),
    "b_d_incremental": ("for base B-D the radii are INCREMENTAL around the already-dilated "
                        "B-D support, not total radius from B-R. r = 0.4 m on top of B-D "
                        "is therefore a total of ~0.8 m of dilation around B-R"),
}

SEMANTIC_PROPAGATION_RULE = {
    "rule": ("every newly added voxel copies the complete fused probability vector of its "
             "nearest frozen semantic source in the base support; the label is the argmax "
             "of that vector"),
    "tie_convention": ("exact-distance ties are averaged over the tied sources' complete "
                       "probability vectors -- the Gate-6 dilation convention, extended to "
                       "the larger radii"),
    "source_vectors": ("the Gate-6 fused vectors, recomputed bit-identically from the "
                       "frozen LingBot cache, the frozen G51-B scale table and the frozen "
                       "Trident cache, and verified against the pinned Gate-6 prediction "
                       "files before use"),
    "existing_voxels": "the semantics of base voxels are never changed by any construction",
    "no_target": ("ground-truth semantic labels never enter a propagated class. The oracle "
                  "uses the target for OCCUPANCY only -- which voxels are added -- and the "
                  "class still comes only from the frozen teacher distribution"),
}

FRUSTUM_DEFINITION = {
    "term": ("'in-frustum', never 'visible'. Projection into an image does not establish "
             "visibility: occlusion is not tested"),
    "test": ("the voxel CENTRE has positive depth in the frame's camera, projects to "
             "0 <= u < W and 0 <= v < H on the LingBot processed lattice with the frozen "
             "predicted intrinsics, and its camera depth lies in the frozen metric depth "
             "range [1.0, 60.0] m"),
    "frames": "the exact five rectified input frames of the clip; the anchor is the last",
    "transforms": ("grid -> anchor camera by the inverse of the clip's frozen "
                   "anchor-to-grid transform; anchor -> frame f by the inverse of "
                   "decompose_residual.scaled_relative_pose(pose, f, anchor, s), the same "
                   "function the frozen reconstruction uses"),
    "residual": {
        "quantity": "voxel_camera_depth - scale * predicted_depth_at_projected_pixel",
        "frame_choice": ("the anchor frame when the voxel is in-frustum there, otherwise "
                         "the lowest-index input frame in which it is in-frustum"),
        "tolerance": "one voxel diagonal, sqrt(3) * voxel_size, predeclared",
        "classes": ("near_surface |residual| <= tol; behind residual > tol; "
                    "in_front residual < -tol; no_valid_depth where the frozen "
                    "confidence/range mask is False at the projected pixel"),
    },
}

AGGREGATION = {
    "primary": "pooled voxel counts over the whole benchmark (micro)",
    "secondary": ("mean-per-clip values are reported only where labelled "
                  "'mean-per-clip'; they are never mixed into a pooled table"),
    "miou": ("mean over classes with a non-zero TP+FP+FN denominator, exactly "
             "gate6.metrics.summarize"),
}

STOP_CONDITIONS = (
    "frozen Gate-6 prediction rollup or per-file hashes do not match",
    "B-R/B-D geometry cannot be reproduced exactly from the frozen caches",
    "the native Occ3D grid mapping is ambiguous",
    "the exact camera-frame transformations cannot be verified",
    "the frozen fused probability vectors are unavailable",
    "the analysis would require retraining or rerunning a foundation model",
    "a target-dependent quantity would enter a deployable prediction",
)

FORBIDDEN = ("no optimizer", "no backward pass", "no neural network created or trained",
             "no radius tuned on any target benchmark", "B-D unchanged",
             "Trident unchanged", "no teacher recaching", "no frozen prediction modified",
             "no new weights or datasets downloaded", "C3 and V3 remain retired")


def radii_voxels(voxel_size: float) -> Tuple[float, ...]:
    """The predeclared metric radii expressed in voxel units of a given grid."""
    return tuple(r / voxel_size for r in RADII_M)


def max_radius_m() -> float:
    return max(RADII_M)


__all__ = [n for n in dir() if not n.startswith("_")]
