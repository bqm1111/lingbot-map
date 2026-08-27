"""Build an open-vocabulary feature field from saved LingBot-MAP predictions.

Consumes the per-frame NPZs written by ``demo_render/batch_demo.py
--save_predictions`` (depth, intrinsic, extrinsic, image), lifts dense CLIP
features into world space using the model's own geometry, and fuses them into
a sparse voxel grid.

    python -m semantic.build_field \
        --predictions output/kitti_seq08_render/image_2 \
        --output output/kitti_seq08_semantic/field.npz \
        --frame_stride 2 --sky_mask_dir output/kitti_seq08_sky_masks

Run from the repo root (or with it on PYTHONPATH).
"""

import argparse
import glob
import os
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from semantic.dense_clip import DenseCLIP
from semantic.feature_field import FieldAccumulator, fit_projection, unproject


def list_frames(predictions: str, keyframes_only: bool = False) -> List[str]:
    """Per-frame prediction NPZs, excluding the sidecar ``meta.npz``."""
    paths = sorted(glob.glob(os.path.join(predictions, "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no frame_*.npz predictions under {predictions}")

    if keyframes_only:
        meta_path = os.path.join(predictions, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"--keyframes_only needs {meta_path}")
        flags = np.load(meta_path)["is_keyframe"].astype(bool)
        if len(flags) != len(paths):
            raise ValueError(f"meta.npz has {len(flags)} flags but {len(paths)} frames found")
        paths = [p for p, k in zip(paths, flags) if k]
        if not paths:
            raise ValueError("no keyframes flagged in meta.npz")
    return paths


def load_frame(path: str, device: torch.device, intrinsic_override=None):
    d = np.load(path)
    depth = torch.from_numpy(d["depth"][..., 0]).to(device)  # [H, W]
    conf = torch.from_numpy(d["depth_conf"]).to(device)
    image = torch.from_numpy(d["images"]).to(device)  # [3, H, W] in [0, 1]
    intrinsic = torch.from_numpy(d["intrinsic"]).to(device)
    extrinsic = torch.from_numpy(d["extrinsic"]).to(device)
    if intrinsic_override is not None:
        # The model estimates its own focal length. When the camera is actually
        # calibrated, using the real intrinsic removes a systematic lateral
        # stretch in the reconstruction (~5% on KITTI).
        fx, fy, cx, cy = intrinsic_override
        intrinsic = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=device, dtype=intrinsic.dtype,
        )
    return depth, conf, image, intrinsic, extrinsic


def sky_mask_for(path: str, sky_mask_dir: Optional[str], shape, device) -> Optional[torch.Tensor]:
    """Load the cached sky mask for a frame; True where the pixel is *not* sky."""
    if not sky_mask_dir:
        return None
    name = os.path.splitext(os.path.basename(path))[0] + ".png"
    mask_path = os.path.join(sky_mask_dir, name)
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    keep = torch.from_numpy((mask > 0).astype(np.uint8)).to(device).bool()
    if keep.shape != shape:
        keep = (
            F.interpolate(keep[None, None].float(), size=shape, mode="nearest")[0, 0].bool()
        )
    return keep


@torch.no_grad()
def dense_features_for_batch(
    clip: DenseCLIP, images: torch.Tensor, scale: float, out_hw
) -> torch.Tensor:
    """Dense CLIP features for [B, 3, H, W] images, resampled to ``out_hw``.

    Returns [B, D, Hs, Ws] (unnormalized; callers normalize after projection).
    """
    if scale != 1.0:
        h, w = images.shape[-2:]
        images = F.interpolate(
            images, size=(int(round(h * scale)), int(round(w * scale))),
            mode="bilinear", align_corners=False,
        )
    feats = clip.encode_dense(images)  # [B, h, w, D]
    feats = feats.permute(0, 3, 1, 2).float()  # [B, D, h, w]
    return F.interpolate(feats, size=out_hw, mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="Directory of per-frame NPZs")
    p.add_argument("--output", required=True, help="Output .npz field path")
    p.add_argument("--sky_mask_dir", default=None, help="Cached sky masks; sky pixels are dropped")

    p.add_argument("--voxel_size", type=float, default=None,
                   help="World units per voxel (default: scene extent / --grid_resolution)")
    p.add_argument("--grid_resolution", type=int, default=768,
                   help="Target voxels across the scene's longest axis when auto-sizing")
    p.add_argument("--pca_dim", type=int, default=64, help="Compressed feature dimension")
    p.add_argument("--pca_frames", type=int, default=64, help="Frames sampled to fit the basis")
    p.add_argument("--pca_samples_per_frame", type=int, default=4096)

    p.add_argument("--keyframes_only", action="store_true",
                   help="Fuse only frames flagged as keyframes in meta.npz")
    p.add_argument("--frame_stride", type=int, default=1, help="Use every Nth frame")
    p.add_argument("--pixel_stride", type=int, default=4, help="Use every Nth pixel of each frame")
    p.add_argument("--first_k", type=int, default=None, help="Only the first K frames (debugging)")
    p.add_argument("--batch_size", type=int, default=8, help="Frames per CLIP forward pass")

    p.add_argument("--conf_threshold", type=float, default=1.3,
                   help="Drop pixels whose depth_conf is below this")
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=None)
    p.add_argument("--min_count", type=int, default=2,
                   help="Drop voxels seen fewer than this many times")

    p.add_argument("--clip_model", default="ViT-B-16-quickgelu")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--clip_variant", default="maskclip", choices=["maskclip", "vv_residual"])
    p.add_argument("--clip_scale", type=float, default=2.0,
                   help="Upsample factor before CLIP; higher = finer feature grid")
    p.add_argument("--intrinsic", type=float, nargs=4, default=None,
                   metavar=("FX", "FY", "CX", "CY"),
                   help="Override the model-predicted intrinsic with the real calibration, "
                        "expressed in the prediction's pixel grid.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    paths = list_frames(args.predictions, keyframes_only=args.keyframes_only)
    if args.first_k:
        paths = paths[: args.first_k]
    frames = paths[:: args.frame_stride]
    print(f"{len(paths)} predictions found; fusing {len(frames)} frames "
          f"(stride {args.frame_stride}, pixel stride {args.pixel_stride})")

    clip = DenseCLIP(
        model_name=args.clip_model, pretrained=args.clip_pretrained,
        device=args.device, variant=args.clip_variant,
    )
    print(f"CLIP: {args.clip_model}/{args.clip_pretrained} variant={args.clip_variant} "
          f"dim={clip.embed_dim} patch={clip.patch_size}")

    def valid_pixels(depth, conf, keep_sky):
        mask = depth > args.min_depth
        if args.max_depth:
            mask &= depth < args.max_depth
        if args.conf_threshold:
            mask &= conf >= args.conf_threshold
        if keep_sky is not None:
            mask &= keep_sky
        return mask

    # ── Pass 1: fit the projection basis on a spread of frames ──────────────
    t0 = time.time()
    sample_idx = np.linspace(0, len(frames) - 1, min(args.pca_frames, len(frames))).astype(int)
    samples, extents = [], []
    for i in tqdm(sample_idx, desc="Fitting basis"):
        depth, conf, image, intr, extr = load_frame(frames[i], device, args.intrinsic)
        keep = sky_mask_for(frames[i], args.sky_mask_dir, depth.shape, device)
        mask = valid_pixels(depth, conf, keep)
        if mask.sum() == 0:
            continue
        feats = dense_features_for_batch(clip, image[None], args.clip_scale, depth.shape)[0]
        feats = feats.permute(1, 2, 0)[mask]  # [n, D]
        n = min(args.pca_samples_per_frame, feats.shape[0])
        sel = torch.randperm(feats.shape[0], device=device)[:n]
        samples.append(F.normalize(feats[sel], dim=-1))

        world = unproject(depth, intr, extr)[mask]
        extents.append(torch.stack([world.min(0).values, world.max(0).values]))

    sample_mat = torch.cat(samples)
    components, explained = fit_projection(sample_mat, args.pca_dim)
    print(f"Basis: {sample_mat.shape[0]} samples -> {args.pca_dim}d, "
          f"explains {explained * 100:.1f}% of feature variance ({time.time() - t0:.0f}s)")
    del samples, sample_mat

    # ── Voxel size ──────────────────────────────────────────────────────────
    ext = torch.stack(extents)
    lo = torch.quantile(ext[:, 0], 0.02, dim=0)
    hi = torch.quantile(ext[:, 1], 0.98, dim=0)
    span = (hi - lo).max().item()
    voxel_size = args.voxel_size or span / args.grid_resolution
    print(f"Scene span {span:.2f} world units -> voxel_size {voxel_size:.5f}")

    # ── Pass 2: fuse ────────────────────────────────────────────────────────
    accum = FieldAccumulator(voxel_size, args.pca_dim, device)
    t0, kept, seen = time.time(), 0, 0
    for start in tqdm(range(0, len(frames), args.batch_size), desc="Fusing"):
        batch = frames[start : start + args.batch_size]
        loaded = [load_frame(fp, device, args.intrinsic) for fp in batch]
        images = torch.stack([l[2] for l in loaded])
        feats = dense_features_for_batch(clip, images, args.clip_scale, loaded[0][0].shape)
        feats = torch.einsum("bdhw,cd->bchw", feats, components)  # project to pca_dim

        for (depth, conf, _, intr, extr), feat, fp in zip(loaded, feats, batch):
            keep = sky_mask_for(fp, args.sky_mask_dir, depth.shape, device)
            mask = valid_pixels(depth, conf, keep)[:: args.pixel_stride, :: args.pixel_stride]
            seen += mask.numel()
            if mask.sum() == 0:
                continue
            world = unproject(depth, intr, extr)[:: args.pixel_stride, :: args.pixel_stride]
            feat = feat[:, :: args.pixel_stride, :: args.pixel_stride].permute(1, 2, 0)
            pts, vecs = world[mask], F.normalize(feat[mask], dim=-1)
            accum.add(pts, vecs)
            kept += pts.shape[0]

    field = accum.finalize(components, min_count=args.min_count)
    field.meta = {
        "predictions": os.path.abspath(args.predictions),
        "frames_fused": len(frames),
        "frame_stride": args.frame_stride,
        "pixel_stride": args.pixel_stride,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "clip_variant": args.clip_variant,
        "clip_scale": args.clip_scale,
        "pca_dim": args.pca_dim,
        "explained_variance": round(explained, 4),
        "conf_threshold": args.conf_threshold,
        "sky_masked": bool(args.sky_mask_dir),
        "min_count": args.min_count,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    field.save(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(
        f"\nFused {kept / 1e6:.1f}M points ({100 * kept / max(seen, 1):.0f}% of sampled pixels kept) "
        f"in {time.time() - t0:.0f}s\n"
        f"Field: {field.num_voxels:,} voxels, {args.pca_dim}d -> {args.output} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
