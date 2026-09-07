#!/usr/bin/env python
"""Build deterministic train / val / smoke clip manifests."""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import build_clips, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke-clips", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "manifests")

    hashes = {}
    splits = {"train": cfg.dataset.train_sequences, "val": cfg.dataset.val_sequences}
    for split, seqs in splits.items():
        recs = build_clips(root, seqs, cfg.lingbot.clip_length, cfg.lingbot.frame_stride,
                           cfg.lingbot.clip_stride, cfg.dataset.camera,
                           cfg.dataset.max_clips_per_sequence)
        p = os.path.join(out_dir, f"{split}.jsonl")
        hashes[split] = {"path": os.path.relpath(p, REPO_ROOT), "n_clips": len(recs),
                         "sha256": write_manifest(p, recs)}
        print(f"{split:6s} {len(recs):5d} clips -> {os.path.relpath(p, REPO_ROOT)}")

    # Smoke: the first clips of the validation split, so it exercises the real path.
    smoke = build_clips(root, cfg.dataset.val_sequences, cfg.lingbot.clip_length,
                        cfg.lingbot.frame_stride, cfg.lingbot.clip_stride,
                        cfg.dataset.camera, a.smoke_clips)[: a.smoke_clips]
    p = os.path.join(out_dir, "smoke.jsonl")
    hashes["smoke"] = {"path": os.path.relpath(p, REPO_ROOT), "n_clips": len(smoke),
                       "sha256": write_manifest(p, smoke)}
    print(f"smoke  {len(smoke):5d} clips -> {os.path.relpath(p, REPO_ROOT)}")

    tr = set(cfg.dataset.train_sequences) & set(cfg.dataset.val_sequences)
    if tr:
        raise SystemExit(f"train/val sequence leakage: {sorted(tr)}")

    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "manifest_hashes.json"),
               {"provenance": collect_provenance(cfg, "prepare_manifest"), "manifests": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
