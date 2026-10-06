"""Operational settings of the B4 source-data scaling study. Research settings are NOT here: B4 keeps
``tools/semantic_field/config/schema.py`` and B3 keeps ``tools/delayed_semantic/config.py`` + ``gate16/config.py`` (both frozen).

Only the update budget is stretched: ``steps = ceil(original_steps * N_pool_reference / 1358)``, one value per method, identical
for the current and the expanded pool.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple

from tools.delayed_semantic import config as PILOT
from tools.semantic_field.config.schema import REPO_ROOT, TrainConfig as B4TrainConfig

STUDY_VERSION = "b4_source_scaling/2026-09-15"
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "semantic_field_scaling")
#: large new caches (tokens, clips, inputs, evidence, targets, B3 caches) for the added drives
DATA_ROOT = os.environ.get("B4_SCALING_DATA_ROOT", "/media/welf/SSD2/MINH_DATASETS/lingbot_b4_scaling")

METHODS: Tuple[str, ...] = ("B4", "B3")
POOLS: Tuple[str, ...] = ("current", "expanded")
SEEDS: Tuple[int, ...] = (0, 1, 2)
#: the pool both original schedules were declared on (0003 / 0007 / 0010)
REFERENCE_POOL_SIZE = 1358


@dataclass(frozen=True)
class OriginalBudgets:
    """Original update counts, read from the frozen configs (not retyped)."""
    b4_steps: int = B4TrainConfig().steps          # 6,000 updates x 4 anchors
    b3_steps: int = PILOT.TrainConfig().steps      # 3,000 updates x 8 anchors

    def of(self, method: str) -> int:
        return {"B4": self.b4_steps, "B3": self.b3_steps}[method]


def stretched_steps(original_steps: int, expanded_pool_size: int, reference_pool_size: int = REFERENCE_POOL_SIZE) -> int:
    """Update budget that keeps the original draws per anchor on the expanded pool: ceil(original * N_expanded / N_reference)."""
    if expanded_pool_size < reference_pool_size:
        raise ValueError(f"expanded pool ({expanded_pool_size}) is smaller than the reference pool ({reference_pool_size})")
    return math.ceil(original_steps * expanded_pool_size / reference_pool_size)


@dataclass(frozen=True)
class LoaderConfig:
    """Bounded loading. Anchors are read from disk on access with the unchanged per-anchor loaders; at most
    ``host_cache_bytes`` of decoded host tensors are kept (LRU). GPU memory holds only the current step's anchors."""
    host_cache_bytes: int = 8 * 1024 ** 3

    def validate(self) -> None:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if not 0 <= self.host_cache_bytes <= total // 4:
            raise ValueError(f"host_cache_bytes={self.host_cache_bytes} must be in [0, {total // 4}] (a quarter of host RAM)")


@dataclass(frozen=True)
class CheckpointConfig:
    """Resume checkpoints: model, optimizer, scheduler, both sampler generators, draw log and training log."""
    resume_every_steps: int = 1000

    def validate(self) -> None:
        if self.resume_every_steps <= 0:
            raise ValueError("resume_every_steps must be positive")


@dataclass(frozen=True)
class ResourceCaps:
    """Study-wide caps (readiness + preparation + source training + source evaluation)."""
    gpu_hours: float = 48.0
    extra_storage_gb: float = 750.0
    #: never fill a shared disk beyond this free-space floor
    min_free_gb_per_disk: float = 20.0


@dataclass(frozen=True)
class StudyPaths:
    artifacts: str = ARTIFACT_DIR
    data_root: str = DATA_ROOT

    @property
    def manifests(self) -> str:
        return os.path.join(self.artifacts, "manifests")

    @property
    def ledger(self) -> str:
        return os.path.join(self.artifacts, "resources", "ledger.jsonl")

    def run_dir(self, method: str, pool: str, seed: int) -> str:
        if method not in METHODS or pool not in POOLS or seed not in SEEDS:
            raise ValueError(f"unknown run {method}/{pool}/seed{seed}")
        return os.path.join(self.artifacts, "runs", f"{method}_{pool}_seed{seed}")

    def resume_checkpoint(self, method: str, pool: str, seed: int) -> str:
        return os.path.join(self.run_dir(method, pool, seed), "resume.pt")

    def final_checkpoint(self, method: str, pool: str, seed: int, steps: int) -> str:
        return os.path.join(self.run_dir(method, pool, seed), f"{method}_seed{seed}_step{steps:05d}.pt")

    def run_record(self, method: str, pool: str, seed: int) -> str:
        return os.path.join(self.run_dir(method, pool, seed), "run.json")


__all__ = ["STUDY_VERSION", "ARTIFACT_DIR", "DATA_ROOT", "METHODS", "POOLS", "SEEDS", "REFERENCE_POOL_SIZE", "OriginalBudgets",
           "stretched_steps", "LoaderConfig", "CheckpointConfig", "ResourceCaps", "StudyPaths"]
