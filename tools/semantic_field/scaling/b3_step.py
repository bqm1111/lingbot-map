"""One B3 training step: the pilot's ``anchor_step`` (unchanged, imported) aggregated exactly as ``tools/delayed_semantic/train.py::fit``.

Per step ``PILOT.TrainConfig().anchors_per_step`` = 8 anchors; per term (hidden evidence, delayed TRAINING-ONLY targets) the loss is
the query-weighted mean over anchors; loss = hidden + delayed. Generator order is the pilot's: gen picks all 8 anchors first,
then per anchor block mask + hidden-query randperm (gen) and delayed randperm (gen_d).
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from tools.delayed_semantic import config as PILOT
from tools.delayed_semantic.model import EvidenceAssociation
from tools.delayed_semantic.train import anchor_step
from tools.semantic_field.scaling.b4_step import StepOutput


class _DrawRecorder:
    """Stands in for the pilot's ``hashlib`` digest: keeps the records so a run can resume and still report the same digest."""

    def __init__(self) -> None:
        self.records: list[str] = []

    def update(self, data: bytes) -> None:
        self.records.append(data.decode())


def b3_step(model: EvidenceAssociation, anchors: Sequence[Dict], text: torch.Tensor, gen: torch.Generator, gen_d: torch.Generator,
            cfg: PILOT.TrainConfig = PILOT.TrainConfig()) -> StepOutput:
    picks = torch.randint(len(anchors), (cfg.anchors_per_step,), generator=gen).tolist()
    recorder = _DrawRecorder()
    parts = [anchor_step(model, anchors[i], text, cfg, gen, gen_d, True, recorder, "select") for i in picks]
    row: Dict[str, float] = {}; loss = 0.0
    for term in ("hidden", "delayed"):
        ps = [p[term] for p in parts if term in p]
        if ps:
            n = sum(p[1] for p in ps)
            lt = sum(p[0] * p[1] for p in ps) / n; loss = loss + lt
            row[f"loss_{term}"] = float(lt); row[f"acc_{term}"] = sum(p[2] * p[1] for p in ps) / n; row[f"w0_{term}"] = sum(p[3] * p[1] for p in ps) / n
            if len(ps[0]) > 4:
                for k in ps[0][4]:
                    row[f"sel_{k}"] = sum(p[4][k] * p[1] for p in ps) / n
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError(f"B3 step produced no loss term (anchor picks {picks}); the pilot loop would fail the same way")
    return StepOutput(loss, row, recorder.records)


__all__ = ["b3_step"]
