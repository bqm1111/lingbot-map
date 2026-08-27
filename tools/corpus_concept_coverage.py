#!/usr/bin/env python
"""Post-hoc: what concepts does the frozen teacher actually see in a training corpus?

    python tools/corpus_concept_coverage.py --config <cfg> --label <name>

Assigns every cached training token to its best-matching evaluation prompt and reports
the distribution.  This is **analysis only**, run after all training and evaluation are
finished; no model was trained, selected or tuned using it.  It answers whether a
corpus that nominally contains a concept contains it *as the teacher represents it* —
which is what the sidecar can actually learn from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from semantic.eval_semantickitti import CLASS_NAMES, CLASS_PROMPTS
from semantic_sidecar.config import load_config, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.teacher_features import SemanticProjection, build_teacher, encode_text_queries

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_teacher_features import load_teacher_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max_scenes", type=int, default=24)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    proj = SemanticProjection.load(os.path.join(cfg.paths.cache_root, "pca.safetensors"))
    teacher = build_teacher(cfg.teacher)
    text = encode_text_queries(teacher, CLASS_PROMPTS, proj)  # [19, 64]
    del teacher
    torch.cuda.empty_cache()

    tea_root = os.path.join(cfg.paths.cache_root, "teacher")
    scenes = [s.name for s in expand_scenes(cfg.scenes) if s.role == "train"][: args.max_scenes]
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    total = 0
    for name in scenes:
        path = os.path.join(tea_root, name, "teacher_manifest.json")
        if not os.path.exists(path):
            continue
        feats = proj.project(load_teacher_features(tea_root, name).float().reshape(-1, 512))
        best = (feats @ text.T).argmax(dim=-1)
        counts += torch.bincount(best, minlength=len(CLASS_NAMES))
        total += int(best.numel())

    frac = (counts.float() / max(total, 1) * 100).tolist()
    rows = sorted(zip(CLASS_NAMES, frac), key=lambda x: -x[1])
    print(f"=== {args.label}: teacher-argmax concept distribution over {total:,} training tokens ===")
    for n, f in rows:
        if f >= 0.05:
            print(f"  {n:<16}{f:6.2f} %")
    payload = {"label": args.label, "config": os.path.abspath(args.config),
               "scenes": scenes, "total_tokens": total,
               "percent_by_class": dict(zip(CLASS_NAMES, frac)),
               "note": "post-hoc analysis only; not used for training or model selection"}
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
