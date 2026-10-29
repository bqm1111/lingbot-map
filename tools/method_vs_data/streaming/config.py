"""Single source of truth for the streaming-scale validation phase.

Two inference modes, decided here before any number exists:

``A`` -- **clip-reset reference**. The canonical procedure every historical result used:
    each evaluation anchor replays its own five-frame causal clip, LingBot's state is reset
    at the clip boundary, and the metric scale is estimated from that clip's own five
    frames. This is read from the frozen clip cache written by
    ``tools/occany_comparison/cli/build_clips.py``.

``B`` -- **persistent state, initialization-only scale**. One unbroken causal pass over
    each physical segment (SemanticKITTI sequence 08; each nuScenes validation scene),
    LingBot reset only at the genuine segment boundary, and ONE metric scale estimated from
    the segment's first five frames and then frozen for the whole segment. No later MoGe
    call and no scale refresh occurs.

What is deliberately held identical between the modes: the five frames whose depth enters
each anchor's occupancy map, the mapper, the voxel size, the free-space carving, the
prediction grid, the evaluation grid, the read-out, the completion checkpoints and the
occupancy threshold. What differs, and differs *together*: LingBot's reconstruction state
and the calibration policy. The A/B difference therefore cannot be attributed to either
one alone, and no table in this phase claims otherwise.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "method_vs_data", "streaming_scale_validation")
MANIFEST_DIR = os.path.join(ARTIFACT_DIR, "manifests")
SCORE_DIR = os.path.join(ARTIFACT_DIR, "scores")
RUNTIME_DIR = os.path.join(ARTIFACT_DIR, "runtime")
VISIBILITY_DIR = os.path.join(ARTIFACT_DIR, "visibility")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "figures")

#: the shared map caches are bulk data and live beside the other dataset caches
CACHE_ROOT = os.environ.get("STREAMING_VALIDATION_CACHE",
                            "/media/SSD1/MINH_DATASETS/lingbot_streaming_validation")

DATASETS: Tuple[str, ...] = ("semantickitti", "occ3d")
MODES: Tuple[str, ...] = ("A", "B")
MODE_NAMES: Dict[str, str] = {
    "A": "clip_reset_reference",
    "B": "persistent_state_startup_scale",
}

#: the three training sources and the seeds trained for each
SOURCES: Tuple[str, ...] = ("k360", "waymo", "once")
SEEDS: Tuple[int, ...] = (0, 1, 2)
SOURCE_CHECKPOINTS: Dict[str, str] = {
    "k360": os.path.join(REPO_ROOT, "artifacts", "baseline_reconstruction", "checkpoints",
                         "geo_seed{seed}_step06000.pt"),
    "waymo": os.path.join(REPO_ROOT, "artifacts", "waymo_arm", "checkpoints",
                          "geo_seed{seed}_step06000.pt"),
    "once": os.path.join(REPO_ROOT, "artifacts", "method_vs_data", "once_arm", "checkpoints",
                         "geo_seed{seed}_step06000.pt"),
}
#: ONCE stays exploratory for the reasons recorded in once_training_report.md: its original
#: occupied-prevalence gate failed and its supervision window is ~2 s, not ~10 s.
EXPLORATORY: Tuple[str, ...] = ("once",)

#: number of leading frames of a segment used to initialize the metric scale in mode B.
#: Five, because that is the frozen G-A rule (gates/gate7b/scale.N_ANCHOR_FRAMES) and the
#: number of frames LingBot itself processes as its bidirectional scale block.
N_INIT_FRAMES = 5
#: LingBot keyframe interval. 1 = every frame's KV is cached. The RoPE capacity rule in
#: gates/gate7b/replay.keyframe_interval_for decides whether 1 is affordable per segment.
DEFAULT_KEYFRAME_INTERVAL = 1

#: paired block bootstrap, unchanged from the canonical protocol
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
#: SemanticKITTI is ONE physical stream, so the canonical 25-anchor blocks are *within* a
#: single stream and are not independent streams. Recorded explicitly, and a stream-level
#: sensitivity check is reported beside every SemanticKITTI interval.
SK_BLOCK_ANCHORS = 25

IMAGE_SIZE = {"semantickitti": (1226, 370), "occ3d": (1600, 900)}


def map_cache_dir(dataset: str, mode: str) -> str:
    return os.path.join(CACHE_ROOT, "maps", dataset, f"mode{mode}")


def map_cache_path(dataset: str, mode: str, sample_id: str) -> str:
    return os.path.join(map_cache_dir(dataset, mode), f"{sample_id}.npz")


def stream_state_path(dataset: str, mode: str, segment: str) -> str:
    return os.path.join(CACHE_ROOT, "streams", dataset, f"mode{mode}", f"{segment}.npz")


def manifest_path(dataset: str) -> str:
    return os.path.join(MANIFEST_DIR, f"stream_manifest_{dataset}.json")


def checkpoint_path(source: str, seed: int) -> str:
    return SOURCE_CHECKPOINTS[source].format(seed=seed)


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, MANIFEST_DIR, SCORE_DIR, RUNTIME_DIR, VISIBILITY_DIR, LOG_DIR,
              FIGURE_DIR, CACHE_ROOT):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
