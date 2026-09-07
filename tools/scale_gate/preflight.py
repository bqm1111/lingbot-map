#!/usr/bin/env python
"""Validate config, assets and conventions before any GPU work.

    python tools/scale_gate/preflight.py --config configs/scale_gate/semantickitti.yaml --dry-run

Exits nonzero and names every missing asset. Performs no large writes.
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import Preprocess, build_clips, validate_sequence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    problems: list[str] = []
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    print(f"config hash {cfg.hash}   dataset root {cfg.dataset.root}")

    ck = os.path.join(REPO_ROOT, cfg.lingbot.checkpoint)
    if not os.path.isfile(ck):
        problems.append(f"missing checkpoint: {cfg.lingbot.checkpoint}")
    else:
        print(f"checkpoint OK ({os.path.getsize(ck) / 2**30:.2f} GiB)")

    if not os.path.isdir(root):
        problems.append(f"missing dataset root: {cfg.dataset.root}")

    inventory, splits = {}, {}
    if os.path.isdir(root):
        for split, seqs in (("train", cfg.dataset.train_sequences),
                            ("val", cfg.dataset.val_sequences)):
            n_clips = 0
            for s in seqs:
                if not os.path.isdir(os.path.join(root, "sequences", s)):
                    problems.append(f"missing sequence {s} ({split})")
                    continue
                inv = validate_sequence(root, s, cfg.dataset.camera)
                inventory[s] = inv.to_dict()
                if not inv.ok:
                    problems.extend(f"seq {s}: {p}" for p in inv.problems)
                try:
                    Preprocess.build(inv.image_hw, cfg.lingbot.inference_resolution,
                                     cfg.lingbot.patch_size)
                except RuntimeError as exc:
                    problems.append(f"seq {s}: {exc}")
                c = build_clips(root, [s], cfg.lingbot.clip_length, cfg.lingbot.frame_stride,
                                cfg.lingbot.clip_stride, cfg.dataset.camera,
                                cfg.dataset.max_clips_per_sequence)
                n_clips += len(c)
                print(f"  seq {s}: {inv.n_images:5d} frames  {inv.image_hw[1]}x{inv.image_hw[0]}"
                      f"  -> {len(c):4d} clips" + ("" if inv.ok else "   PROBLEMS"))
            splits[split] = {"sequences": list(seqs), "n_clips": n_clips,
                             "n_frames": n_clips * cfg.lingbot.clip_length}
            print(f"{split}: {n_clips} clips / {n_clips * cfg.lingbot.clip_length} frames")

    overlap = set(cfg.dataset.train_sequences) & set(cfg.dataset.val_sequences)
    if overlap:
        problems.append(f"train/val sequence leakage: {sorted(overlap)}")

    cache_root = os.path.join(REPO_ROOT, cfg.cache.root)
    try:
        os.makedirs(cache_root, exist_ok=True)
        probe = os.path.join(cache_root, ".writable")
        open(probe, "w").close(); os.remove(probe)
        print(f"cache writable: {cfg.cache.root}")
    except OSError as exc:
        problems.append(f"cache root not writable: {exc}")

    report = {"provenance": collect_provenance(cfg, "preflight"), "config": dict(cfg),
              "splits": splits, "sequence_inventory": inventory, "problems": problems,
              "ok": not problems}
    out = a.output or os.path.join(REPO_ROOT, cfg.experiment.output_dir, "data_inventory.json")
    if not a.dry_run:
        write_json(out, report)
        print(f"wrote {os.path.relpath(out, REPO_ROOT)}")

    if problems:
        print("\nBLOCKERS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\npreflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
