#!/usr/bin/env python
"""Run frozen LingbotMap over metrically supervised sequences and cache the result.

Nothing here trains or modifies LingbotMap.  The cache is the only interface
between the foundation model and every later phase of this prototype, so the
expensive inference is paid exactly once.

Example
-------
    python scripts/cache_lingbot_predictions.py \
        --dataset tartanair --dataset-root /media/minh/TartanAir/dataset \
        --checkpoint checkpoints/lingbot-map/204754b/lingbot-map.pt \
        --output-dir outputs/prompted_lingbot/cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot import conventions, preprocess
from prompted_lingbot.datasets import (discover_kitti_odometry, discover_occ3d_nuscenes,
                                       discover_tartanair_v2)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_model(checkpoint: str, device: torch.device, num_scale_frames: int, use_sdpa: bool):
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
        kv_cache_scale_frames=num_scale_frames, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=use_sdpa,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def run_sequence(model, seq, args, device):
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(
        seq.image_paths, image_size=args.image_size, patch_size=14
    ).to(device)
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model.inference_streaming(
            images, num_scale_frames=args.num_scale_frames, keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    pose_enc = preds["pose_enc"].float()[0]                # (S, 9)
    depth = preds["depth"].float()[0, ..., 0]              # (S, H, W)
    conf = preds["depth_conf"].float()[0]                  # (S, H, W)
    S, H, W = depth.shape
    extr_c2w, K_pred = conventions.pose_enc_to_c2w_and_K(pose_enc, (H, W))
    extr_c2w = extr_c2w.numpy().astype(np.float64)
    K_pred = K_pred.numpy().astype(np.float64)

    st = args.depth_stride
    depth_s = depth[:, ::st, ::st].numpy().astype(np.float16)
    conf_s = conf[:, ::st, ::st].numpy().astype(np.float16)

    def scale_K(K):
        K = K.copy()
        K[..., 0, 0] /= st
        K[..., 1, 1] /= st
        K[..., 0, 2] = (K[..., 0, 2] - 0.5 * (st - 1)) / st
        K[..., 1, 2] = (K[..., 1, 2] - 0.5 * (st - 1)) / st
        return K

    K_pred_s = scale_K(K_pred)

    # ---- ground truth on the same lattice ---------------------------------- #
    gt_K_full = preprocess.resample_gt_intrinsics(seq.K_gt, seq.image_hw, args.image_size, 14)
    gt_K_s = scale_K(gt_K_full)
    gt_depth_s = gt_valid_s = None
    if seq.has_depth:
        gds, gvs = [], []
        for i in range(len(seq)):
            d, v = seq.load_depth(i)
            d, v = preprocess.resample_depth(d, v, args.image_size, 14)
            gds.append(d[::st, ::st].astype(np.float16))
            gvs.append(v[::st, ::st])
        gt_depth_s = np.stack(gds)
        gt_valid_s = np.stack(gvs)

    payload = {
        "frame_ids": np.arange(len(seq), dtype=np.int32),
        "timestamps": (seq.timestamps if seq.timestamps is not None
                       else np.arange(len(seq), dtype=np.float64)).astype(np.float64),
        "pose_enc": pose_enc.numpy().astype(np.float32),
        "pred_pose_c2w": extr_c2w.astype(np.float32),
        "pred_K": K_pred_s.astype(np.float32),
        "pred_depth": depth_s,
        "pred_depth_conf": conf_s,
        "gt_pose_c2w": seq.poses_c2w.astype(np.float32),
        "gt_K": gt_K_s.astype(np.float32),
        "frame_type": preds["frame_type"][0].numpy().astype(np.uint8),
        "is_keyframe": preds["is_keyframe"][0].numpy(),
    }
    if gt_depth_s is not None:
        payload["gt_depth"] = gt_depth_s
        payload["gt_depth_valid"] = gt_valid_s

    meta = {
        "name": seq.name, "dataset": seq.dataset, "scene": seq.scene,
        "num_frames": len(seq),
        "source_image_hw": list(seq.image_hw),
        "model_image_hw": [H, W],
        "cached_depth_hw": list(depth_s.shape[1:]),
        "depth_stride": st,
        "has_gt_depth": gt_depth_s is not None,
        "gt_depth_is_densified": bool(seq.extra.get("depth_is_densified", False)),
        "gt_trajectory_length_m": seq.trajectory_length(),
        "inference_wall_s": wall,
        "inference_s_per_frame": wall / max(1, len(seq)),
        "peak_gpu_gb": peak_mem,
        "conventions": {
            "camera_axes": conventions.CAMERA_CONVENTION,
            "pred_pose_c2w": "camera_to_world (verified; upstream docstring says otherwise)",
            "gt_pose_c2w": "camera_to_world, metric metres",
            "depth": conventions.DEPTH_REPRESENTATION,
            "depth_conf_activation": "expp1 (1+exp(x)), so values are > 1",
            "pred_scale": "arbitrary (monocular); this is what the prompts must anchor",
        },
        "inference": {
            "mode": "streaming", "num_scale_frames": args.num_scale_frames,
            "keyframe_interval": args.keyframe_interval, "image_size": args.image_size,
            "autocast": "bfloat16",
        },
        "extra": seq.extra,
    }
    return payload, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["tartanair", "kitti", "occ3d_nuscenes"], required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="all", help="informational tag stored in the manifest")
    ap.add_argument("--environments", nargs="*", default=None)
    ap.add_argument("--kitti-sequences", nargs="*", default=["09", "10"])
    ap.add_argument("--occ3d-root", default=None,
                    help="Occupancy3D-nuScenes-trainval directory (annotations.json + gts/)")
    ap.add_argument("--nuscenes-scenes", nargs="*", default=None)
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--kitti-with-depth", action="store_true",
                    help="also cache the densified KITTI depth maps (provenance unverified)")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--min-chunk", type=int, default=120)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--depth-stride", type=int, default=2)
    ap.add_argument("--num-scale-frames", type=int, default=8)
    ap.add_argument("--keyframe-interval", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-sdpa", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.dataset == "tartanair":
        seqs = discover_tartanair_v2(args.dataset_root, args.environments,
                                     chunk_size=args.chunk_size, min_chunk=args.min_chunk)
    elif args.dataset == "kitti":
        seqs = discover_kitti_odometry(args.dataset_root, args.kitti_sequences,
                                       chunk_size=args.chunk_size, min_chunk=args.min_chunk,
                                       with_depth=args.kitti_with_depth)
    else:
        if not args.occ3d_root:
            raise SystemExit("--occ3d-root is required for --dataset occ3d_nuscenes")
        seqs = discover_occ3d_nuscenes(args.dataset_root, args.occ3d_root, split=args.split,
                                       scenes=args.nuscenes_scenes,
                                       limit_scenes=args.limit_scenes)
    if args.limit:
        seqs = seqs[: args.limit]
    if not seqs:
        raise SystemExit(f"no sequences found under {args.dataset_root!r}")

    todo = [s for s in seqs
            if args.overwrite or not os.path.isfile(os.path.join(args.output_dir, f"{s.name}.npz"))]
    print(f"{len(seqs)} sequences discovered, {len(todo)} to compute "
          f"({len(seqs) - len(todo)} already cached)")

    model = build_model(args.checkpoint, device, args.num_scale_frames, args.use_sdpa) if todo else None
    ckpt_sha = sha256_file(args.checkpoint)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {"sequences": {}}
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))

    for i, seq in enumerate(todo):
        out = os.path.join(args.output_dir, f"{seq.name}.npz")
        print(f"[{i + 1}/{len(todo)}] {seq.name}  ({len(seq)} frames)", flush=True)
        payload, meta = run_sequence(model, seq, args, device)
        tmp = out + ".part"
        # np.savez_compressed appends ".npz" unless the *file object* is passed.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        os.replace(tmp, out)
        meta["file"] = os.path.basename(out)
        meta["bytes"] = os.path.getsize(out)
        meta["checkpoint_sha256"] = ckpt_sha
        meta["split_tag"] = args.split
        manifest["sequences"][seq.name] = meta
        manifest["checkpoint"] = os.path.abspath(args.checkpoint)
        manifest["checkpoint_sha256"] = ckpt_sha
        manifest["seed"] = args.seed
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"    -> {meta['bytes'] / 1e6:.1f} MB, {meta['inference_s_per_frame'] * 1000:.0f} ms/frame, "
              f"peak {meta['peak_gpu_gb']:.1f} GB", flush=True)

    print(f"done. manifest: {manifest_path} ({len(manifest['sequences'])} sequences)")


if __name__ == "__main__":
    main()
