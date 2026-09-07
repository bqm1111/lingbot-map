"""Everything Gate 7B is allowed to decide, decided once, before evaluation.

``tools/gate7b/precommit.py`` writes the pinned YAML from this module and from the
modules the experiment imports, so the configuration and the running code cannot drift.
"""

from __future__ import annotations

from typing import Dict, Tuple

DATASETS: Tuple[str, ...] = ("semantickitti", "occ3d", "kitti360")

#: Causal history horizons, in frames of the deduplicated stream.
HORIZONS: Tuple = (1, 5, 20, 50, "all")
PRIMARY_HORIZON = "all"

SCALE_POLICIES: Tuple[str, ...] = ("G-A", "G-B", "G-C")
PRIMARY_SCALE = "G-A"

VARIANTS: Tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")
PRIMARY_VARIANT = "S1"

#: The paired comparisons declared before any number was produced.
PAIRED_COMPARISONS = (
    ("S1|G-A|all", "S0", "streaming accumulation against the frozen five-frame map"),
    ("S2|G-A|all", "S1|G-A|all", "relaxed LingBot gate"),
    ("S3|G-A|all", "S1|G-A|all", "MoGe rescue on rejected rays"),
    ("S4|G-A|all", "S1|G-A|all", "map-consistency-gated complementary fusion"),
    ("S1|G-B|all", "S1|G-A|all", "causal running-median gauge"),
    ("S1|G-C|all", "S1|G-A|all", "per-frame gauge (ablation only)"),
)

BOOTSTRAP = {"n_boot": 10000, "seed": 0, "alpha": 0.05,
             "units": {"occ3d": "official nuScenes scene",
                       "semantickitti": "contiguous block of 20 clips (one drive)",
                       "kitti360": "contiguous block of 20 clips (one drive)"},
             "block_size": 20,
             "independence_caveat": ("blocks from a single drive are not independent "
                                     "scenes and are never described as such")}

SEEDS = {"global": 0, "bootstrap": 0}

#: Diagnostic future horizons for Phase 6. Never enters a causal map.
RECOVERY_HORIZONS: Tuple = (1, 5, 10, 20, 50, "any")

RANGE_BANDS: Tuple[Tuple[float, float], ...] = ((0, 10), (10, 20), (20, 30), (30, 40),
                                                (40, 60))

RESOURCE_LIMITS = {"max_gpu_hours": 4.0, "max_new_storage_gb": 100.0,
                   "action_if_exceeded": ("stop after the pilot and report the measured "
                                          "estimate; never launch an unbounded run")}

CAUSALITY = (
    "the map at timestamp t may use only frames with stream index <= t",
    "LingBot's own context is the native causal stream: its state is reset only at a "
    "genuine segment boundary, and its KV cache is causal by construction",
    "the temporal-recoverability analysis of Phase 6 looks forward and is diagnostic "
    "only; it is computed in a separate volume and never reaches a deployable output",
)

NOT_DONE = ("no optimizer", "no backward pass", "no network created or trained",
            "C3 and V3 remain retired", "no local 3D completion network",
            "no fixed-class semantic head", "no per-benchmark parameter tuning",
            "Trident is not re-run; its frame cache is read only",
            "no Gate-6 or Gate-7A artifact is modified", "no dataset file is written")

STOP_CONDITIONS = (
    "direct-mode streaming cannot be verified",
    "dense MoGe outputs are unavailable or are the wrong metric gauge",
    "a coordinate convention cannot be verified from the code",
    "the projected full run exceeds 4 GPU-hours or 100 GB of new storage",
    "a future frame could enter a causal prediction",
)

#: The final queryable system cannot ship these vectors; stated so it is not forgotten.
SEMANTIC_CAVEAT = (
    "the cached per-benchmark Trident probability vectors are used for evaluation only. "
    "A deployable open-vocabulary map must carry fixed-dimensional language-aligned "
    "descriptors, not permanent 17/18/19-class probability vectors, or the "
    "open-vocabulary transfer claim of Gate 6 is lost.")

__all__ = [n for n in dir() if not n.startswith("_")]
