#!/usr/bin/env python
"""Freeze the study before any target dataset is opened: head and geometry hashes per
seed, the B0 vector hash, the source results the decision was taken on, and the transfer
protocol. ``transfer.py`` refuses to run without this file and asserts every hash in it.

    python tools/future_semantic_completion/freeze_manifest.py --seeds 0 1 2
"""
from __future__ import annotations
import argparse, datetime, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(ST.SEEDS))
    a = ap.parse_args()
    if os.path.exists(ST.manifest_path()):
        raise SystemExit(f"{ST.manifest_path()} exists; a frozen manifest is not rewritten")
    with np.load(os.path.join(ST.ART, "b0_mean.npz")) as z:
        b0 = z["mean"]
    seeds = {}
    for s in a.seeds:
        tr = json.load(open(os.path.join(ST.ART, f"train_{ST.run_tag(s)}.json")))
        src = json.load(open(os.path.join(ST.ART, "source", f"source_{ST.run_tag(s)}.json")))
        hp = ST.head_checkpoint(s)
        assert ST.sha256_file(hp) == tr["checkpoint_sha256"] == src["head_checkpoint_sha256"]
        key = src["populations"]["key"]
        seeds[str(s)] = {"seed": s, "head_checkpoint": os.path.relpath(hp, ST.REPO_ROOT),
                         "head_checkpoint_sha256": tr["checkpoint_sha256"],
                         "frozen_checkpoint": tr["frozen_checkpoint"],
                         "frozen_checkpoint_sha256": tr["frozen_checkpoint_sha256"],
                         "steps": tr["steps"], "selected": "last",
                         "source_key": {c: {k: key[c][k] for k in ("n_voxels", "cosine_mean", "kl_mean",
                                                                 "top1_agreement", "miou_present_classes")}
                                        for c in key}}
    man = {"study": "future_semantic_completion",
           "claim": "a small head on the frozen adopted geometry predicts the future teacher's "
                    "class-probability vector for currently unobserved, later LiDAR-confirmed "
                    "occupied voxels; scored on oracle-occupied voxels, never through predicted occupancy",
           "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "b0_sha256": ST.sha256_json(b0.tolist()), "b0_argmax_union_index": int(b0.argmax()),
           "b1_max_dist_vox": ST.B1_MAX_DIST_VOX,
           "seeds": seeds,
           "transfer_protocol": {"mode": "past5", "populations": ["unobserved", "observed"],
                                 "voxels": "benchmark GT-occupied on the official grid and mask",
                                 "occ3d_reduction": "mean of eight subvoxel probability vectors; observed if any subvoxel observed",
                                 "expected_populations": {"semantickitti": 163, "occ3d": 1182},
                                 "nothing_selected_on_targets": True},
           "rule": "final-step head; no threshold; no checkpoint or hyperparameter selection on any target",
           "env": ST.environment()}
    ST.write_json(ST.manifest_path(), man)
    print(f"froze {ST.manifest_path()} for seeds {list(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
