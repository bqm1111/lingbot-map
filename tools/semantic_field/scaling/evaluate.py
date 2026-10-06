"""Source evaluation of the scaling runs with the FROZEN evaluator ``tools/semantic_field/evaluation/evaluate_source.evaluate``.

That function scores B0 + ``B3_s{0,1,2}`` + ``B4_s{0,1,2}`` on the 590 anchors of 0006 (frozen masks, regions, candidate sets,
occupancy, masked diagnostic). Two redirections select the scaling checkpoints of one pool, nothing else changes:
    B4 checkpoints: ``PoolPaths.checkpoint`` (the evaluator calls ``paths.checkpoint(seed, 6000)``) -> the pool's final B4 checkpoint
    B3 checkpoints: ``gate16.config.head_path`` (the evaluator calls ``G16.head_path("B3", seed)``) -> the pool's final B3 checkpoint,
                    replaced only for the duration of the call
    output:         ``PoolPaths.eval`` = artifacts/semantic_field_scaling/eval/<pool>/source_eval_shard<i>of<n>.npz

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.evaluate --pool expanded --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from typing import Iterator

from tools.delayed_semantic.gate16 import config as G16
from tools.semantic_field.config.schema import Paths
from tools.semantic_field.evaluation import evaluate_source as ES
from tools.semantic_field.scaling.config import ARTIFACT_DIR, POOLS, StudyPaths
from tools.semantic_field.scaling.run import study_steps


@dataclass(frozen=True)
class PoolPaths(Paths):
    """The frozen evaluator's paths with the scaling pool's B4 checkpoints and output directory."""
    pool: str = "current"

    def checkpoint(self, seed: int, step: int) -> str:            # ``step`` is the evaluator's hard-coded 6000; the pool's final step is used
        path = StudyPaths().final_checkpoint("B4", self.pool, seed, study_steps("B4"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"B4 {self.pool} seed {seed}: final checkpoint {path} missing")
        return path


def pool_paths(pool: str) -> PoolPaths:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}")
    return PoolPaths(eval=os.path.join(ARTIFACT_DIR, "eval", pool), pool=pool)


@contextlib.contextmanager
def b3_heads_of(pool: str) -> Iterator[None]:
    original = G16.head_path

    def head_path(method: str, seed: int) -> str:
        if method != "B3":
            return original(method, seed)
        path = StudyPaths().final_checkpoint("B3", pool, seed, study_steps("B3"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"B3 {pool} seed {seed}: final checkpoint {path} missing")
        return path

    G16.head_path = head_path
    try:
        yield
    finally:
        G16.head_path = original


def evaluate_pool(pool: str, device: str, shard: int, shards: int) -> dict:
    with b3_heads_of(pool):
        return ES.evaluate(device, shard, shards, paths=pool_paths(pool))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", choices=POOLS, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    print(evaluate_pool(a.pool, a.device, a.shard, a.shards))


if __name__ == "__main__":
    main()
