#!/usr/bin/env python
"""Measure how much supervision each target type actually delivers per teacher density.

    python tools/supervision_coverage.py --config configs/semantic_sidecar/feasibility.yaml

Pixelwise distillation can only supervise a token whose own frame carries a teacher
feature, so its supervised fraction is exactly the teacher frame density.  A
3D-consensus target reaches every observation of any track that has at least one
teacher observation, so it propagates supervision to frames the teacher never saw.
This script quantifies that gap directly from the cached tracks — no training needed.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

import _bootstrap  # noqa: F401

from semantic_sidecar.config import load_config, write_json
from semantic_sidecar.consensus import ConsensusTargets, segment_ids
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.metrics import format_table
from semantic_sidecar.tracks import TrackSet, tracks_are_cached


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--densities", nargs="*", type=float, default=[1.0, 0.5, 0.25, 0.1, 0.05])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from safetensors.torch import load_file

    train_scenes = [s.name for s in expand_scenes(cfg.scenes) if s.role == "train"]
    rows: List[Dict[str, object]] = []
    for density in args.densities:
        tag = f"density{int(round(density * 100)):03d}"
        root = os.path.join(cfg.paths.tracks_root, tag)
        obs_total = obs_pixel = obs_consensus = 0
        tracks_total = tracks_valid = 0
        for scene in train_scenes:
            if not tracks_are_cached(root, scene):
                continue
            tracks = TrackSet.load(root, scene)
            targets = ConsensusTargets.load(root, scene)
            obs = load_file(os.path.join(root, scene, "obs_features.safetensors"))
            has = obs["obs_has_teacher"].bool()
            seg = segment_ids(tracks.obs_ptr)
            obs_total += int(has.numel())
            obs_pixel += int(has.sum())
            obs_consensus += int(targets.valid[seg].sum())
            tracks_total += int(targets.valid.numel())
            tracks_valid += int(targets.valid.sum())
        if obs_total == 0:
            continue
        rows.append({
            "teacher frame density": 100.0 * density,
            "observations": obs_total,
            "pixelwise supervised %": 100.0 * obs_pixel / obs_total,
            "consensus supervised %": 100.0 * obs_consensus / obs_total,
            "propagation factor": (obs_consensus / max(obs_pixel, 1)),
            "tracks with a target %": 100.0 * tracks_valid / max(tracks_total, 1),
        })

    table = format_table(rows)
    print(table)
    if args.output:
        write_json(args.output, {"rows": rows, "train_scenes": train_scenes})
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
