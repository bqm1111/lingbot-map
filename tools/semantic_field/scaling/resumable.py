"""Resumable optimisation loop shared by B4 and B3 (the original loops: AdamW, cosine over the run's own step count, grad clip).

A resume checkpoint holds everything that decides the remaining sequence: model / optimizer / scheduler state, both sampler
generators (``torch.Generator.get_state``), the draw records and the log. Resuming at step k therefore draws exactly the same
anchors, crops, masks and subsamples as an uninterrupted run. (GPU kernels are not forced deterministic, as in the original
runs, so losses after a resume agree to floating-point tolerance, not necessarily bit-for-bit.)
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from tools.semantic_field.scaling.b4_step import StepOutput

StepFn = Callable[[torch.Generator, torch.Generator], StepOutput]


@dataclass(frozen=True)
class OptimSpec:
    """The method's frozen optimiser settings (copied from its TrainConfig by the caller)."""
    lr: float
    weight_decay: float
    grad_clip: float


@dataclass
class RunState:
    step: int = 0
    draws: List[str] = field(default_factory=list)
    log: List[Dict] = field(default_factory=list)
    seconds: float = 0.0


def draw_digest(draws: List[str]) -> str:
    """sha256 over the draw records in order -- equal to the original trainers' ``draw_digest``."""
    h = hashlib.sha256()
    for r in draws:
        h.update(r.encode())
    return h.hexdigest()


def _atomic_save(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def train_resumable(model: torch.nn.Module, step_fn: StepFn, total_steps: int, seed: int, optim: OptimSpec, resume_path: str,
                    resume_every: int, log_every: int = 50, stop_after_step: Optional[int] = None, tag: str = "") -> RunState:
    """Run updates ``state.step + 1 .. min(total_steps, stop_after_step)``. Resumes from ``resume_path`` if it exists.

    The model must already be initialised exactly as the method declares (``torch.manual_seed(seed)`` then construction)."""
    opt = torch.optim.AdamW(model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    gen, gen_d = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)
    state = RunState()
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        if ck["total_steps"] != total_steps or ck["seed"] != seed:
            raise RuntimeError(f"{resume_path} belongs to total_steps={ck['total_steps']} seed={ck['seed']}, not {total_steps}/{seed}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        gen.set_state(ck["gen"]); gen_d.set_state(ck["gen_d"]); state = RunState(**ck["state"])
        print(f"[{tag}] resumed at step {state.step}/{total_steps}", flush=True)
    last = min(total_steps, stop_after_step or total_steps)
    t0 = time.time(); seconds0 = state.seconds
    for it in range(state.step + 1, last + 1):
        out = step_fn(gen, gen_d)
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip); opt.step(); sched.step()
        state.step = it; state.draws.extend(out.draws); state.log.append({"step": it, **out.row})
        state.seconds = seconds0 + time.time() - t0
        if it % log_every == 0 or it == last:
            print(f"  [{tag} {it}/{total_steps}] " + " ".join(f"{k} {v:.4f}" for k, v in out.row.items() if isinstance(v, float))
                  + f" {state.seconds / 60:.1f} min", flush=True)
        if it % resume_every == 0 or it == last:
            _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "gen": gen.get_state(),
                          "gen_d": gen_d.get_state(), "state": asdict(state), "total_steps": total_steps, "seed": seed}, resume_path)
    return state


__all__ = ["OptimSpec", "RunState", "draw_digest", "train_resumable"]
