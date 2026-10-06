"""Build and run one training run (method x pool x seed) on disk-backed anchors, resumably.

``train_on_anchors`` is the core: given an explicit anchor list it initialises the method exactly as declared, wraps the anchors
in ``DiskAnchorSequence`` and runs ``train_resumable`` with the method's frozen step. ``run`` adds the pool, study paths, the
final checkpoint in the method's original format (so the frozen evaluators load it unchanged) and the run record.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PILOT, rules as PILOT_RULES
from tools.delayed_semantic.gate16 import config as G16, rules as G16_RULES
from tools.delayed_semantic.model import EvidenceAssociation
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, state_digest
from tools.semantic_field import freeze
from tools.semantic_field.config.schema import GEOMETRY_SEED, TARGET_TOKENS, TRAIN_FORBIDDEN, TrainConfig as B4TrainConfig
from tools.semantic_field.data.feature_mean import load_feature_mean
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead
from tools.semantic_field.scaling import loaders
from tools.semantic_field.scaling.anchor_store import AnchorKey, DiskAnchorSequence
from tools.semantic_field.scaling.b3_step import b3_step
from tools.semantic_field.scaling.b4_step import b4_step
from tools.semantic_field.scaling.config import (STUDY_VERSION, CheckpointConfig, LoaderConfig, OriginalBudgets, StudyPaths,
                                                 stretched_steps)
from tools.semantic_field.scaling.resumable import OptimSpec, RunState, draw_digest, train_resumable
from tools.semantic_field.scaling.source_pool import expanded_pool, pool_by_name


@dataclass(frozen=True)
class TrainedRun:
    model: torch.nn.Module
    state: RunState
    anchors: DiskAnchorSequence
    geometry_digest: Optional[str]
    peak_gpu_bytes: int


def check_frozen_rules(method: str) -> None:
    PILOT_RULES.check("reliability")
    if method == "B4":
        freeze.check("source_predeclaration"); freeze.check("centering")
    else:
        G16_RULES.check("predeclaration")


def train_on_anchors(method: str, keys: List[AnchorKey], seed: int, total_steps: int, device: str, resume_path: str,
                     loader: LoaderConfig = LoaderConfig(), ckpt: CheckpointConfig = CheckpointConfig(),
                     stop_after_step: Optional[int] = None, tag: str = "") -> TrainedRun:
    loader.validate(); ckpt.validate(); check_frozen_rules(method)
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.reset_peak_memory_stats(dev)
    tiers = loaders.delayed_tiers()
    anchors = DiskAnchorSequence(keys, loaders.host_loader(method, tiers), loaders.files_of(method), dev, loader.host_cache_bytes)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    digest = None
    if method == "B4":
        mu = load_feature_mean(dev)
        net = load_frozen_geometry(GEOMETRY_SEED, dev); digest = state_digest(net)
        torch.manual_seed(seed); model: torch.nn.Module = SemanticFieldHead().to(dev)             # the trainer's initialisation order
        cfg = B4TrainConfig(); step = lambda g, gd: b4_step(model, net, anchors, text, mu, g, gd)
    elif method == "B3":
        torch.manual_seed(seed); model = EvidenceAssociation(semantic_projection=G16.VARIANTS["B3"]["semantic_projection"]).to(dev)
        cfg = PILOT.TrainConfig(); step = lambda g, gd: b3_step(model, anchors, text, g, gd)
    else:
        raise ValueError(f"unknown method {method!r}")
    with FileAudit(patterns=TRAIN_FORBIDDEN + TARGET_TOKENS, strict=True) as audit:
        state = train_resumable(model, step, total_steps, seed, OptimSpec(cfg.lr, cfg.weight_decay, cfg.grad_clip), resume_path,
                                ckpt.resume_every_steps, cfg.log_every, stop_after_step, tag)
    if audit.violations:
        raise RuntimeError(f"[{tag}] training opened forbidden files: {audit.violations[:5]}")
    if method == "B4" and state_digest(net) != digest:
        raise RuntimeError(f"[{tag}] the frozen geometry changed during training")
    return TrainedRun(model, state, anchors, digest, torch.cuda.max_memory_allocated(dev))


def study_steps(method: str, paths: StudyPaths = StudyPaths()) -> int:
    """The declared update budget for both pools: stretched by the frozen expanded pool size."""
    return stretched_steps(OriginalBudgets().of(method), len(expanded_pool(paths).anchors))


def _final_checkpoint(method: str, model: torch.nn.Module, seed: int, steps: int, tiers: List[int]) -> Dict:
    if method == "B4":                                   # trainer.py format, loaded by evaluation/predict.load_b4
        return {"state_dict": model.state_dict(), "config": model.config(), "seed": seed, "step": steps}
    return {"state_dict": model.state_dict(), "model_config": model.config(), "step": steps, "seed": seed,    # pilot fit format (gate16 load_head)
            "semantic_projection": model.semantic_projection, "method": "B3", "tiers": tiers, "tau_target": G16.TAU_TARGET, "k_candidates": G16.K}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(method: str, pool_name: str, seed: int, device: str, stop_after_step: Optional[int] = None,
        loader: LoaderConfig = LoaderConfig(), ckpt: CheckpointConfig = CheckpointConfig(), paths: StudyPaths = StudyPaths()) -> Dict:
    pool = pool_by_name(pool_name, paths); steps = study_steps(method, paths)
    tag = f"{method}_{pool_name}_seed{seed}"
    trained = train_on_anchors(method, list(pool.anchors), seed, steps, device, paths.resume_checkpoint(method, pool_name, seed),
                               loader, ckpt, stop_after_step, tag)
    rec = {"study_version": STUDY_VERSION, "tag": tag, "method": method, "pool": pool_name, "pool_sha256": pool.digest(),
           "n_anchors": len(pool.anchors), "seed": seed, "total_steps": steps, "completed_steps": trained.state.step,
           "draw_digest_so_far": draw_digest(trained.state.draws), "geometry_digest": trained.geometry_digest,
           "seconds": trained.state.seconds, "peak_gpu_bytes_this_session": trained.peak_gpu_bytes,
           "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "cache": asdict(trained.anchors.stats),
           "loader": asdict(loader), "device": torch.cuda.get_device_name(torch.device(device)), "log": trained.state.log}
    if trained.state.step == steps:
        path = paths.final_checkpoint(method, pool_name, seed, steps)
        torch.save(_final_checkpoint(method, trained.model, seed, steps, loaders.delayed_tiers()), path)
        rec["final_checkpoint"] = {"path": path, "sha256": _sha256(path)}
    os.makedirs(paths.run_dir(method, pool_name, seed), exist_ok=True)
    json.dump(rec, open(paths.run_record(method, pool_name, seed), "w"), indent=1)
    return rec


__all__ = ["TrainedRun", "check_frozen_rules", "train_on_anchors", "study_steps", "run"]
