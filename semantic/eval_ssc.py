"""SemanticKITTI Semantic Scene Completion metrics for the feature field.

Reports the two standard SSC benchmark numbers:

    SC IoU    binary occupancy IoU — did we put geometry where geometry is?
    SSC mIoU  mean per-class IoU over the 19 classes on the completion grid.

    python -m semantic.eval_ssc \
        --field output/kitti_seq08_semantic/field.npz \
        --predictions output/kitti_seq08_render/image_2 \
        --kitti_root data/kitti/dataset --sequence 08 \
        --frame_stride 10 --output output/kitti_seq08_semantic/ssc.json

Read the numbers with the caveat in mind
----------------------------------------
SSC asks a method to *hallucinate* occupancy it cannot see, from a single frame.
This field does no completion: it only holds voxels that were actually observed,
though because it aggregates 4071 frames of a moving vehicle it does fill in a
lot of what is occluded at any one instant.  It is also monocular, so the volume
has to be placed using a fitted global depth scale.  These numbers therefore say
"how does an aggregated monocular map score on the completion benchmark", which
is not the task SSC methods are built for — treat them as a lower bound and a
diagnostic, not a like-for-like comparison with MonoScene or VoxFormer.

The grid: 256 x 256 x 32 at 0.2 m in the frame's velodyne coordinates, spanning
x 0..51.2 m, y -25.6..25.6 m, z -2..4.4 m.  Ground truth occupancy is
``label > 0`` — the ``.bin`` file is the sparse single-scan *input*, not the
target.  Voxels flagged ``.invalid`` are excluded, per the official protocol.
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from semantic.dense_clip import DenseCLIP
from semantic.eval_semantickitti import (
    CLASS_NAMES,
    CLASS_PROMPTS,
    build_lut,
    load_scan,
    parse_calib,
    project_to_image,
)
from semantic.feature_field import FeatureField, unproject

GRID_DIMS = (256, 256, 32)
VOXEL_SIZE = 0.2
GRID_ORIGIN = (0.0, -25.6, -2.0)


def load_ssc_gt(kitti_root: str, seq: str, idx: int, lut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (semantic labels [256,256,32] in 0..19, valid mask)."""
    base = os.path.join(kitti_root, "sequences", seq, "voxels", f"{idx:06d}")
    label = np.fromfile(base + ".label", dtype=np.uint16).reshape(GRID_DIMS)
    invalid = np.unpackbits(np.fromfile(base + ".invalid", dtype=np.uint8)).reshape(GRID_DIMS)
    sem = lut[np.clip(label, 0, len(lut) - 1)]
    return sem, invalid == 0


def grid_centres(device: torch.device) -> torch.Tensor:
    """Voxel centres of the completion grid, in velodyne coordinates. [N, 3]."""
    i = torch.arange(GRID_DIMS[0], device=device, dtype=torch.float32)
    j = torch.arange(GRID_DIMS[1], device=device, dtype=torch.float32)
    k = torch.arange(GRID_DIMS[2], device=device, dtype=torch.float32)
    gi, gj, gk = torch.meshgrid(i, j, k, indexing="ij")
    x = GRID_ORIGIN[0] + (gi + 0.5) * VOXEL_SIZE
    y = GRID_ORIGIN[1] + (gj + 0.5) * VOXEL_SIZE
    z = GRID_ORIGIN[2] + (gk + 0.5) * VOXEL_SIZE
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def cam0_to_cam2(p2: np.ndarray) -> np.ndarray:
    """Translation from the reference camera to the colour camera image_2."""
    k2 = p2[:3, :3]
    return np.linalg.inv(k2) @ p2[:, 3]


def fit_depth_scale(
    field: FeatureField, predictions: str, pred_files, kitti_root: str, seq: str,
    p2: np.ndarray, tr: np.ndarray, lut: np.ndarray, indices, device: torch.device,
) -> float:
    """Median ratio of LiDAR depth to predicted depth — the map has no metric scale."""
    ratios = []
    for i in tqdm(indices, desc="Fitting depth scale"):
        npz = np.load(os.path.join(predictions, pred_files[i]))
        depth = torch.from_numpy(npz["depth"][..., 0]).to(device)
        H, W = depth.shape
        pts, _ = load_scan(kitti_root, seq, i, lut)
        u, v, lidar_depth, valid = project_to_image(pts, p2, tr, 1226, 370)
        if valid.sum() == 0:
            continue
        uu = torch.from_numpy(u[valid] * (W / 1226)).to(device).long().clamp(0, W - 1)
        vv = torch.from_numpy(v[valid] * (H / 370)).to(device).long().clamp(0, H - 1)
        pred = depth[vv, uu]
        lz = torch.from_numpy(lidar_depth[valid].astype(np.float32)).to(device)
        good = pred > 1e-6
        if good.any():
            ratios.append((lz[good] / pred[good]).cpu())
    return float(torch.cat(ratios).median())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--kitti_root", default="data/kitti/dataset")
    p.add_argument("--sequence", default="08")
    p.add_argument("--frame_stride", type=int, default=10)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--scale_mode", default="per_frame", choices=["per_frame", "global"],
                   help="Monocular depth has no metric scale. 'per_frame' median-scales each "
                        "frame against its own LiDAR (the standard monocular protocol, and the "
                        "right choice here because scale drifts over a 3 km sequence); 'global' "
                        "fits one scale for the whole run.")
    p.add_argument("--depth_scale", type=float, default=None,
                   help="Fixed predicted-depth -> metres scale; implies --scale_mode global.")
    p.add_argument("--scale_frames", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    field = FeatureField.load(args.field)
    meta = field.meta or {}
    print(f"Field: {field.num_voxels:,} voxels @ {field.voxel_size:.5f}")

    clip = DenseCLIP(
        model_name=meta.get("clip_model", "ViT-B-16-quickgelu"),
        pretrained=meta.get("clip_pretrained", "openai"),
        device=args.device,
        variant=meta.get("clip_variant", "maskclip"),
    )
    text = clip.encode_text(CLASS_PROMPTS)
    sims = field.query(text, device)
    voxel_class = torch.softmax(sims / args.temperature, dim=-1).argmax(dim=-1) + 1  # 1..19
    keys_gpu = torch.from_numpy(field.keys).to(device)

    lut = build_lut()
    p2, tr = parse_calib(os.path.join(args.kitti_root, "sequences", args.sequence, "calib.txt"))
    t02 = torch.from_numpy(cam0_to_cam2(p2).astype(np.float32)).to(device)
    tr_t = torch.from_numpy(tr[:3, :4].astype(np.float32)).to(device)

    pred_files = sorted(
        f for f in os.listdir(args.predictions) if f.startswith("frame_") and f.endswith(".npz")
    )
    indices = list(range(len(pred_files)))
    if args.first_k:
        indices = indices[: args.first_k]
    indices = indices[:: args.frame_stride]

    scale_mode = "global" if args.depth_scale is not None else args.scale_mode
    scale = args.depth_scale
    if scale is None and scale_mode == "global":
        sub = indices[:: max(1, len(indices) // args.scale_frames)]
        scale = fit_depth_scale(
            field, args.predictions, pred_files, args.kitti_root, args.sequence,
            p2, tr, lut, sub, device,
        )
    if scale_mode == "global":
        print(f"Depth scale (predicted -> metres): {scale:.4f}")
    else:
        print("Depth scale: fitted per frame against that frame's LiDAR")
    print(f"Evaluating SSC on {len(indices)} frames of sequence {args.sequence}\n")

    centres = grid_centres(device)  # [N, 3] velodyne frame, metres
    homo = torch.cat([centres, torch.ones(len(centres), 1, device=device)], dim=1)
    cam0 = homo @ tr_t.T  # -> reference camera, metres
    cam2_metric = cam0 + t02  # -> image_2 camera, metres
    frame_scales = []

    n_cls = len(CLASS_NAMES) + 1
    confusion = torch.zeros(n_cls, n_cls, dtype=torch.int64, device=device)
    tp = fp = fn = 0

    for i in tqdm(indices, desc="SSC"):
        npz = np.load(os.path.join(args.predictions, pred_files[i]))
        extr = torch.from_numpy(npz["extrinsic"]).to(device)
        R, t = extr[:3, :3], extr[:3, 3]

        if scale_mode == "per_frame":
            depth_i = torch.from_numpy(npz["depth"][..., 0]).to(device)
            H, W = depth_i.shape
            pts, _ = load_scan(args.kitti_root, args.sequence, i, lut)
            u, v, ld, val = project_to_image(pts, p2, tr, 1226, 370)
            if val.sum() < 100:
                continue
            uu = torch.from_numpy(u[val] * (W / 1226)).to(device).long().clamp(0, W - 1)
            vv = torch.from_numpy(v[val] * (H / 370)).to(device).long().clamp(0, H - 1)
            dp = depth_i[vv, uu]
            lz = torch.from_numpy(ld[val].astype(np.float32)).to(device)
            good = dp > 1e-6
            if not good.any():
                continue
            frame_scale = float((lz[good] / dp[good]).median())
        else:
            frame_scale = scale
        frame_scales.append(frame_scale)
        cam2_units = cam2_metric / frame_scale

        sem, valid = load_ssc_gt(args.kitti_root, args.sequence, i, lut)
        gt = torch.from_numpy(sem.reshape(-1).astype(np.int64)).to(device)
        keep = torch.from_numpy(valid.reshape(-1)).to(device)

        # Camera -> world for this frame's predicted pose (world-to-camera extrinsic).
        world = (cam2_units[keep] - t) @ R
        idx = field.lookup(world, keys_gpu)
        pred = torch.where(idx >= 0, voxel_class[idx.clamp_min(0)], torch.zeros_like(idx))

        g = gt[keep]
        confusion += torch.bincount(g * n_cls + pred, minlength=n_cls**2).reshape(n_cls, n_cls)

        po, go = pred > 0, g > 0
        tp += int((po & go).sum())
        fp += int((po & ~go).sum())
        fn += int((~po & go).sum())

    # ── Scene completion (occupancy only) ───────────────────────────────────
    sc_iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # ── Semantic scene completion (19 classes) ──────────────────────────────
    diag = confusion.diag().float()
    iou = diag / (confusion.sum(0).float() + confusion.sum(1).float() - diag).clamp_min(1)
    present = confusion.sum(1) > 0
    scored = present.clone()
    scored[0] = False  # empty is not one of the 19 classes
    ssc_miou = float(iou[scored].mean() * 100)

    print(f"\n{'class':<16}{'IoU':>8}")
    for c in range(1, n_cls):
        if scored[c]:
            print(f"{CLASS_NAMES[c - 1]:<16}{iou[c] * 100:7.2f}%")

    print(f"\n{'SSC mIoU':<16}{ssc_miou:7.2f}%   over {int(scored.sum())} classes")
    print(f"{'SC IoU':<16}{sc_iou * 100:7.2f}%   (precision {precision * 100:.2f}%, "
          f"recall {recall * 100:.2f}%)")
    print("\nRecall is the completion term: voxels the ground truth fills that an "
          "observation-only map leaves empty.")
    if frame_scales:
        print(f"Per-frame depth scale spanned {np.min(frame_scales):.2f}-{np.max(frame_scales):.2f} "
              f"(median {np.median(frame_scales):.2f}) — monocular scale drift over the sequence.")

    results = {
        "field": os.path.abspath(args.field),
        "sequence": args.sequence,
        "frames_evaluated": len(indices),
        "frame_stride": args.frame_stride,
        "scale_mode": scale_mode,
        "depth_scale": scale if scale_mode == "global" else None,
        "depth_scale_per_frame": {
            "median": float(np.median(frame_scales)) if frame_scales else None,
            "min": float(np.min(frame_scales)) if frame_scales else None,
            "max": float(np.max(frame_scales)) if frame_scales else None,
        },
        "ssc_miou": ssc_miou,
        "sc_iou": sc_iou * 100,
        "sc_precision": precision * 100,
        "sc_recall": recall * 100,
        "per_class_iou": {
            CLASS_NAMES[c - 1]: float(iou[c] * 100) for c in range(1, n_cls) if scored[c]
        },
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
