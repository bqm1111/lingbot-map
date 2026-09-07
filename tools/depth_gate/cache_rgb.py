#!/usr/bin/env python
"""Cache RGB at the Gate-0 processed resolution, so training never re-decodes PNGs.

Uses the identical ``load_and_preprocess_images`` transform validated in Gate 0, so RGB,
predicted depth, confidence and projected LiDAR all share one pixel grid.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, load_config
from gates.scale_gate.kitti import read_manifest
from lingbot_map.utils.load_fn import load_and_preprocess_images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/depth_gate/refine.yaml")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    dcfg = load_config(a.config)
    scfg = load_config(dcfg.data.scale_gate_config)
    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    out = os.path.join(REPO_ROOT, dcfg.data.rgb_cache)
    os.makedirs(out, exist_ok=True)
    n = 0
    for split in a.splits:
        recs = read_manifest(os.path.join(REPO_ROOT, scfg.experiment.output_dir,
                                          "manifests", f"{split}.jsonl"))
        for i, rec in enumerate(recs):
            p = os.path.join(out, f"{rec['clip_id']}.npz")
            if os.path.exists(p) and not a.overwrite:
                continue
            imgs = load_and_preprocess_images(
                [os.path.join(root, q) for q in rec["image_paths"]], mode="crop",
                image_size=scfg.lingbot.inference_resolution,
                patch_size=scfg.lingbot.patch_size)
            arr = (imgs.numpy() * 255).round().clip(0, 255).astype(np.uint8)
            np.savez_compressed(p + ".tmp.npz", rgb=arr)
            os.replace(p + ".tmp.npz", p)
            n += 1
            if n % 200 == 0:
                print(f"  {n} clips", flush=True)
        print(f"{split}: {len(recs)} clips")
    print(f"cached {n} clips -> {os.path.relpath(out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
