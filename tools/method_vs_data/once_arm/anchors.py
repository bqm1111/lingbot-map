"""Deterministic anchor manifest for the ONCE arm.

Selection is a pure function of (eligible anchors, target count, seed), written to disk with a
digest so preprocessing is resumable and reproducible. The manifest records every item brief
§5.E asks for: independent source sequences, eligible anchors, selected anchors, the input
frame timestamps and the supervision frame timestamps with their actual sweep counts.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

import numpy as np

from tools.method_vs_data.once_arm import config as A, geometry as G
from tools.method_vs_data.once_reader import OnceSequence


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def build(target_n: int = None, seed: int = None) -> Dict:
    """Select up to ``target_n`` anchors, spread evenly over sequences."""
    target_n = A.TARGET_N_ANCHORS if target_n is None else int(target_n)
    seed = A.SAMPLE_SEED if seed is None else int(seed)
    seqs = A.sequences()
    sp = A.split(seqs)
    eligible, seq_objs = {}, {}
    for s in seqs:
        sq = OnceSequence(A.ONCE_DATA_ROOT, s)
        seq_objs[s] = sq
        eligible[s] = G.eligible_anchors(sq)

    # even per-sequence quota, deterministic stride (no RNG: stride keeps anchors spread in
    # time rather than clustered, which an RNG draw would not guarantee)
    train_seqs = sp["train"]
    per = max(target_n // max(len(train_seqs), 1), 1)
    selected: Dict[str, List[int]] = {}
    for s in seqs:
        el = eligible[s]
        quota = per if s in train_seqs else len(el)
        if len(el) <= quota:
            selected[s] = list(el)
        else:
            step = len(el) / float(quota)
            selected[s] = [el[int(i * step)] for i in range(quota)]

    rows = []
    for s in seqs:
        sq = seq_objs[s]
        for a in selected[s]:
            clip = G.input_clip_indices(a)
            win = G.supervision_window(sq, a)
            rows.append({
                "sequence": s, "anchor": int(a),
                "split": "val" if s in sp["val"] else "train",
                "input_frames": [int(j) for j in clip],
                "input_frame_ids": [sq.frame_ids[j] for j in clip],
                "input_timestamps_s": [float(sq.t_s[j]) for j in clip],
                "input_span_s": float(sq.t_s[clip[-1]] - sq.t_s[clip[0]]),
                "supervision_frames": [int(j) for j in win],
                "supervision_frame_ids": [sq.frame_ids[j] for j in win],
                "supervision_timestamps_s": [float(sq.t_s[j]) for j in win],
                "supervision_sweeps": len(win),
                "supervision_span_s": float(sq.t_s[win[-1]] - sq.t_s[win[0]]),
            })
    spans = np.array([r["supervision_span_s"] for r in rows])
    sweeps = np.array([r["supervision_sweeps"] for r in rows])
    ispans = np.array([r["input_span_s"] for r in rows])
    man = {
        "provenance": A.provenance(),
        "independent_source_sequences": len(seqs),
        "sequences": seqs,
        "n_eligible_per_sequence": {s: len(eligible[s]) for s in seqs},
        "n_eligible_total": int(sum(len(v) for v in eligible.values())),
        "n_selected_per_sequence": {s: len(selected[s]) for s in seqs},
        "n_selected_total": len(rows),
        "n_train": sum(1 for r in rows if r["split"] == "train"),
        "n_val": sum(1 for r in rows if r["split"] == "val"),
        "supervision_span_s": {"min": float(spans.min()), "p50": float(np.median(spans)),
                               "max": float(spans.max())},
        "supervision_sweeps": {"min": int(sweeps.min()), "p50": float(np.median(sweeps)),
                               "max": int(sweeps.max())},
        "input_span_s": {"min": float(ispans.min()), "p50": float(np.median(ispans)),
                         "max": float(ispans.max())},
        "anchors": rows,
    }
    man["digest"] = _digest({k: v for k, v in man.items() if k != "anchors"} |
                            {"anchor_keys": [f"{r['sequence']}/{r['anchor']}" for r in rows]})
    return man


def path(p: A.Paths = None) -> str:
    return os.path.join((p or A.Paths()).manifests, "anchors.json")


def load_or_build(p: A.Paths = None, target_n: int = None) -> Dict:
    p = p or A.Paths()
    f = path(p)
    if os.path.exists(f):
        return json.load(open(f))
    A.ensure_dirs(p)
    man = build(target_n)
    tmp = f + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, f)
    return man


__all__ = ["build", "path", "load_or_build"]
