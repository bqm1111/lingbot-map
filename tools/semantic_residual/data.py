"""Bounded loading of B5 anchors: one small npz pair per anchor, nothing held open.

Per anchor the loader reads the query cache (context, ~250 KB plus the B0 features) and the teacher features it
indexes. Nothing else is touched during training, so host memory stays flat regardless of pool size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from tools.semantic_residual import config as B5
from tools.semantic_residual.model import QueryContext
from tools.semantic_residual.prepare import query_cache_path


class AnchorLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnchorKey:
    drive: str
    t: int

    def __str__(self) -> str:
        return f"{self.drive[-9:-5]}/{self.t:05d}"


def anchors_of(drives: Sequence[str]) -> List[AnchorKey]:
    """Every anchor with both a query cache and a target file, in a fixed order."""
    out: List[AnchorKey] = []
    for d in drives:
        qdir = os.path.join(B5.DATA_ROOT, "queries", d)
        if not os.path.isdir(qdir):
            continue
        for name in sorted(os.listdir(qdir)):
            if name.endswith(".npz"):
                t = int(name[:-4])
                if os.path.exists(B5.target_path(d, t)):
                    out.append(AnchorKey(d, t))
    return out


def load_anchor(key: AnchorKey, device) -> Tuple[QueryContext, torch.Tensor, dict]:
    """``(context, target [Q,1024] unit, extras)``. Raises ``AnchorLoadError`` naming the anchor and file."""
    qp, tp = query_cache_path(key.drive, key.t), B5.target_path(key.drive, key.t)
    try:
        with np.load(qp) as q:
            need = ("flat_index", "target_row", "trunk", "geometry", "delta_m", "distance_m", "nn_count", "b0_feature",
                    "hist_seen", "tier", "first_dt_s")
            missing = [k for k in need if k not in q.files]
            if missing:
                raise AnchorLoadError(f"{qp} is missing {missing} -- rebuild the query cache (tools.semantic_residual.prepare)")
            d = {k: q[k] for k in need}
        with np.load(tp) as z:
            target = z["feature"][d["target_row"]]
    except AnchorLoadError:
        raise
    except Exception as exc:                                  # a missing or corrupt cache must name itself
        raise AnchorLoadError(f"{key}: could not load {qp} / {tp}: {exc}") from exc

    if target.shape[1] != B5.ResidualConfig().teacher_dim:
        raise AnchorLoadError(f"{key}: teacher features are {target.shape[1]}-d, expected "
                              f"{B5.ResidualConfig().teacher_dim}-d -- this is a different teacher cache")

    r = B5.ResidualConfig().range_norm_m
    to = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    b0 = torch.nn.functional.normalize(to(d["b0_feature"]), dim=-1)          # fp16 storage -> re-normalise
    ctx = QueryContext(
        b0_feature=b0,
        trunk=to(d["trunk"]),
        geometry=to(d["geometry"]),
        delta=to(d["delta_m"]) / r,
        distance=to(d["distance_m"]).unsqueeze(1) / r,
        valid=(to(d["nn_count"]) > 0).float().unsqueeze(1),
        log_count=(torch.log1p(to(d["nn_count"])) / 5).unsqueeze(1),
    )
    extras = {"flat_index": to(d["flat_index"], torch.long), "hist_seen": to(d["hist_seen"], torch.bool),
              "tier": to(d["tier"], torch.long), "first_dt_s": to(d["first_dt_s"])}
    return ctx, torch.nn.functional.normalize(to(target), dim=-1), extras


def subsample(ctx: QueryContext, target: torch.Tensor, extras: dict, n: int, gen: torch.Generator):
    """Fixed-size draw, so a step's cost does not depend on how much of an anchor was revealed."""
    q = len(target)
    if q <= n:
        return ctx, target, extras
    idx = torch.randperm(q, generator=gen)[:n].to(target.device)
    return (QueryContext({k: v[idx] for k, v in ctx.items()}), target[idx],
            {k: v[idx] for k, v in extras.items()})


__all__ = ["AnchorKey", "AnchorLoadError", "anchors_of", "load_anchor", "subsample"]
