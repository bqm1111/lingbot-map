#!/usr/bin/env python
"""Cache a frozen external metric-depth model over the same frames LingbotMap used.

    python scripts/cache_external_depth_predictions.py \
      --model da_v2_metric_vkitti_vits \
      --dataset semantickitti --dataset-root data/kitti/dataset --sequence 08 \
      --lingbot-cache outputs/prompted_lingbot/cache_semkitti08 \
      --output-dir outputs/prompted_lingbot/depth_substitution/cache/semantickitti

    python scripts/cache_external_depth_predictions.py \
      --model da_v2_metric_vkitti_vits \
      --dataset occ3d_nuscenes --dataset-root /media/SSD1/MINH_DATASETS/nuscenes \
      --lingbot-cache outputs/prompted_lingbot/cache_occ3d \
      --output-dir outputs/prompted_lingbot/depth_substitution/cache/occ3d_nuscenes

Mirrors the LingbotMap cache one-for-one: same sequences, same frame order, same
depth lattice, so every downstream comparison is paired pixel-for-pixel.
Resumable -- completed sequences are skipped. RGB is never duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.datasets import discover_kitti_odometry, discover_occ3d_nuscenes
from prompted_lingbot.external_depth import (
    MODELS, FrozenExternalDepth, resample_to_cached_lattice,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="da_v2_metric_vkitti_vits", choices=sorted(MODELS))
    ap.add_argument("--checkpoint", default=None, help="override the registered checkpoint")
    ap.add_argument("--dataset", choices=["semantickitti", "occ3d_nuscenes"], required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--occ3d-root",
                    default="/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval")
    ap.add_argument("--lingbot-cache", required=True,
                    help="the LingbotMap cache whose sequences/lattice must be mirrored")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.checkpoint:
        MODELS[args.model] = dict(MODELS[args.model], checkpoint=args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    lb_manifest = json.load(open(os.path.join(args.lingbot_cache, "manifest.json")))
    lb_seqs = lb_manifest["sequences"]

    # Rebuild exactly the sequences the LingbotMap cache holds, in the same order.
    if args.dataset == "semantickitti":
        seqs = discover_kitti_odometry(args.dataset_root, [args.sequence],
                                       chunk_size=500, min_chunk=120, with_depth=False)
    else:
        seqs = discover_occ3d_nuscenes(args.dataset_root, args.occ3d_root, split="val")
    seqs = [s for s in seqs if s.name in lb_seqs]
    if not seqs:
        raise SystemExit("no sequences overlap the LingbotMap cache -- check --lingbot-cache")

    todo = [s for s in seqs
            if args.overwrite or not os.path.isfile(os.path.join(args.output_dir, f"{s.name}.npz"))]
    print(f"{len(seqs)} sequences mirrored, {len(todo)} to compute", flush=True)

    model = FrozenExternalDepth(name=args.model, device=args.device).load() if todo else None
    sha = model.checkpoint_sha256() if model else None
    spec = MODELS[args.model]

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {"sequences": {}}

    import cv2
    import torch

    for i, seq in enumerate(todo):
        lb = lb_seqs[seq.name]
        Ht, Wt = lb["cached_depth_hw"]
        src_hw = tuple(lb["source_image_hw"])
        stride = int(lb["depth_stride"])
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)
        out = np.zeros((len(seq), Ht, Wt), np.float16)
        native_med = np.zeros(len(seq), np.float32)
        for f, path in enumerate(seq.image_paths):
            img = cv2.imread(path)
            if img is None:
                raise SystemExit(f"cannot read {path}")
            d = model.infer_bgr(img)
            native_med[f] = float(np.median(d))
            out[f] = resample_to_cached_lattice(d, src_hw, (Ht, Wt),
                                                image_size=lb["model_image_hw"][1],
                                                depth_stride=stride).astype(np.float16)
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(args.device) / 1e9 if torch.cuda.is_available() else 0.0

        dst = os.path.join(args.output_dir, f"{seq.name}.npz")
        tmp = dst + ".part"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh,
                                frame_ids=np.arange(len(seq), dtype=np.int32),
                                timestamps=(seq.timestamps if seq.timestamps is not None
                                            else np.arange(len(seq), dtype=np.float64)),
                                external_depth=out,
                                native_median_depth=native_med)
        os.replace(tmp, dst)

        manifest["sequences"][seq.name] = {
            "name": seq.name, "dataset": seq.dataset, "scene": seq.scene,
            "num_frames": len(seq), "file": os.path.basename(dst),
            "bytes": os.path.getsize(dst),
            "cached_depth_hw": [Ht, Wt], "depth_stride": stride,
            "source_image_hw": list(src_hw), "model_image_hw": lb["model_image_hw"],
            "seconds_per_frame": wall / max(1, len(seq)), "peak_gpu_gb": peak,
            "mirrors_lingbot_cache": os.path.abspath(args.lingbot_cache),
            "extra": seq.extra,
        }
        manifest["model"] = args.model
        manifest["model_spec"] = spec
        manifest["checkpoint_sha256"] = sha
        manifest["is_substitute_for"] = (
            "Depth Anything 3 (OccAny's published baseline) -- DA3 package and "
            "checkpoint are NOT present on this machine; see "
            "docs/frozen_depth_substitution_feasibility.md")
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"[{i+1}/{len(todo)}] {seq.name}: {len(seq)} frames, "
              f"{wall/max(1,len(seq))*1000:.0f} ms/frame, {os.path.getsize(dst)/1e6:.1f} MB",
              flush=True)

    print(f"done. manifest: {manifest_path} ({len(manifest['sequences'])} sequences)")


if __name__ == "__main__":
    main()
