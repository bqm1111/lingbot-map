"""Evaluate the open-vocabulary feature field against SemanticKITTI labels.

    python -m semantic.eval_semantickitti \
        --field output/kitti_seq08_semantic/field.npz \
        --predictions output/kitti_seq08_render/image_2 \
        --kitti_root data/kitti/dataset --sequence 08 \
        --frame_stride 5 --output output/kitti_seq08_semantic/eval.json

Protocol — image space, deliberately
------------------------------------
The obvious approach (align the reconstruction to the LiDAR frame with a Sim(3)
fit, then label 3D points) is wrong for a *monocular* map: scale is arbitrary and
pose drifts over kilometres, so alignment error would contaminate the semantic
score everywhere and we would be grading geometry, not semantics.

Instead every labelled LiDAR point is projected into its own camera frame using
the true KITTI calibration, and the field's prediction is read at that same
pixel.  Global drift cancels out — only per-frame depth ordering matters — so
what is measured is the semantics.

Geometry is then reported *separately* (`--report_geometry`): a single global
scale is fitted between predicted and LiDAR depth, and mIoU is recomputed over
only those pixels whose depth is accurate.  The gap between the two mIoUs is how
much of the semantic error is really geometric error.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from semantic.dense_clip import DenseCLIP
from semantic.feature_field import FeatureField, unproject

# Standard SemanticKITTI raw id -> 19-class benchmark id (0 = ignore).
LEARNING_MAP: Dict[int, int] = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

# Class names and the text prompts used to query the field.  Ordered by class id.
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]
CLASS_PROMPTS = [
    "a car", "a bicycle", "a motorcycle", "a truck", "a vehicle", "a person",
    "a person riding a bicycle", "a person riding a motorcycle", "a road",
    "a parking lot", "a sidewalk", "the ground", "a building", "a fence",
    "vegetation", "a tree trunk", "grass and terrain", "a pole", "a traffic sign",
]


def parse_calib(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P2 [3,4], Tr [4,4]) from a KITTI odometry calib.txt."""
    values = {}
    with open(path) as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            values[key.strip()] = np.array([float(x) for x in rest.split()])
    p2 = values["P2"].reshape(3, 4)
    tr = np.eye(4)
    tr[:3, :4] = values["Tr"].reshape(3, 4)
    return p2, tr


def build_lut() -> np.ndarray:
    """Lookup table mapping raw SemanticKITTI ids to 19-class ids."""
    lut = np.zeros(max(LEARNING_MAP) + 1, dtype=np.int32)
    for raw, learned in LEARNING_MAP.items():
        lut[raw] = learned
    return lut


def load_scan(kitti_root: str, seq: str, idx: int, lut: np.ndarray):
    """Labelled LiDAR scan: (points [N,3] in velodyne frame, class ids [N])."""
    base = os.path.join(kitti_root, "sequences", seq)
    pts = np.fromfile(os.path.join(base, "velodyne", f"{idx:06d}.bin"), dtype=np.float32)
    pts = pts.reshape(-1, 4)[:, :3]
    raw = np.fromfile(os.path.join(base, "labels", f"{idx:06d}.label"), dtype=np.uint32) & 0xFFFF
    raw = np.clip(raw, 0, len(lut) - 1)
    return pts, lut[raw]


def project_to_image(points: np.ndarray, p2: np.ndarray, tr: np.ndarray, width: int, height: int):
    """Project velodyne points into image_2. Returns (u, v, depth, valid mask)."""
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = homo @ tr.T  # velodyne -> camera 0
    pix = cam @ p2.T  # camera 0 -> image_2 (P2 carries the stereo baseline)
    depth = pix[:, 2]
    ok = depth > 1e-3
    u = np.full(len(points), -1.0)
    v = np.full(len(points), -1.0)
    u[ok] = pix[ok, 0] / depth[ok]
    v[ok] = pix[ok, 1] / depth[ok]
    valid = ok & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, depth, valid


class Confusion:
    """Confusion matrix over the 19 benchmark classes (id 0 = ignore)."""

    def __init__(self, num_classes: int, device: torch.device):
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        keep = (gt >= 0) & (gt < self.n) & (pred >= 0) & (pred < self.n)
        idx = gt[keep] * self.n + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self) -> torch.Tensor:
        tp = self.mat.diag().float()
        fp = self.mat.sum(0).float() - tp
        fn = self.mat.sum(1).float() - tp
        return tp / (tp + fp + fn).clamp_min(1)

    def present(self) -> torch.Tensor:
        return self.mat.sum(1) > 0

    def accuracy(self) -> float:
        total = self.mat.sum()
        return float(self.mat.diag().sum() / total) if total > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--kitti_root", default="data/kitti/dataset")
    p.add_argument("--sequence", default="08")
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--output", default=None, help="Write metrics to this JSON path")

    p.add_argument("--orig_width", type=int, default=1226)
    p.add_argument("--orig_height", type=int, default=370)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--conf_threshold", type=float, default=0.0,
                   help="Ignore pixels whose predicted depth_conf is below this")
    p.add_argument("--report_geometry", action="store_true",
                   help="Also fit depth scale and report mIoU on geometrically accurate pixels")
    p.add_argument("--baseline_2d", action="store_true",
                   help="Also score raw per-frame 2D CLIP on the same points, to isolate "
                        "what the 3D fusion contributes")
    p.add_argument("--depth_tolerance", type=float, default=0.1,
                   help="Relative depth error counted as accurate for the geometry-gated mIoU")
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

    # Closed-set assignment: every voxel takes the best-matching class prompt.
    text = clip.encode_text(CLASS_PROMPTS)
    sims = field.query(text, device)  # [M, 19]
    voxel_class = torch.softmax(sims / args.temperature, dim=-1).argmax(dim=-1) + 1  # 1..19
    keys_gpu = torch.from_numpy(field.keys).to(device)

    lut = build_lut()
    p2, tr = parse_calib(os.path.join(args.kitti_root, "sequences", args.sequence, "calib.txt"))

    pred_files = sorted(
        f for f in os.listdir(args.predictions) if f.startswith("frame_") and f.endswith(".npz")
    )
    indices = list(range(len(pred_files)))
    if args.first_k:
        indices = indices[: args.first_k]
    indices = indices[:: args.frame_stride]
    print(f"Evaluating {len(indices)} frames of sequence {args.sequence}\n")

    num_classes = len(CLASS_NAMES) + 1  # + ignore at 0
    conf_all = Confusion(num_classes, device)
    conf_geom = Confusion(num_classes, device)
    conf_2d = Confusion(num_classes, device)
    clip_scale = float(meta.get("clip_scale", 2.0))
    ratios: List[torch.Tensor] = []
    n_points = n_covered = 0

    # Pass 1 gathers depth ratios so a single global scale can be fitted; the
    # monocular map has no metric scale of its own.
    for pass_idx in range(2 if args.report_geometry else 1):
        scale = None
        if pass_idx == 1:
            scale = torch.cat(ratios).median().item()
            print(f"\nFitted depth scale (predicted -> metres): {scale:.4f}")

        for i in tqdm(indices, desc="Scoring" if pass_idx == 0 else "Geometry gate"):
            npz = np.load(os.path.join(args.predictions, pred_files[i]))
            depth = torch.from_numpy(npz["depth"][..., 0]).to(device)
            intr = torch.from_numpy(npz["intrinsic"]).to(device)
            extr = torch.from_numpy(npz["extrinsic"]).to(device)
            H, W = depth.shape

            pts, gt_raw = load_scan(args.kitti_root, args.sequence, i, lut)
            u, v, lidar_depth, valid = project_to_image(pts, p2, tr, args.orig_width, args.orig_height)
            if valid.sum() == 0:
                continue

            # The predictions were made on a pure resize of the original image,
            # so mapping pixel coordinates is a straight scale on each axis.
            uu = torch.from_numpy(u[valid] * (W / args.orig_width)).to(device).long().clamp(0, W - 1)
            vv = torch.from_numpy(v[valid] * (H / args.orig_height)).to(device).long().clamp(0, H - 1)
            gt = torch.from_numpy(gt_raw[valid].astype(np.int64)).to(device)
            lidar_z = torch.from_numpy(lidar_depth[valid].astype(np.float32)).to(device)

            flat = vv * W + uu
            world = unproject(depth, intr, extr).reshape(-1, 3)
            vox_idx = field.lookup(world, keys_gpu)[flat]
            pred_depth = depth.reshape(-1)[flat]

            covered = vox_idx >= 0
            if args.conf_threshold:
                conf_map = torch.from_numpy(npz["depth_conf"]).to(device).reshape(-1)
                covered &= conf_map[flat] >= args.conf_threshold

            # Points the field cannot see are excluded from the confusion matrix
            # rather than scored as wrong: coverage is reported on its own, and
            # folding it into mIoU would conflate "not mapped" with "mislabelled".
            pred_cls = voxel_class[vox_idx.clamp_min(0)]
            if pass_idx == 0:
                n_points += int(gt.numel())
                n_covered += int(covered.sum())
                conf_all.update(gt[covered], pred_cls[covered])
                good = covered & (pred_depth > 1e-6)
                if good.any():
                    ratios.append((lidar_z[good] / pred_depth[good]).cpu())

                if args.baseline_2d:
                    # Same prompts, same points — the only difference is that this
                    # reads a single frame's 2D features instead of the fused field.
                    img = torch.from_numpy(npz["images"]).to(device)[None]
                    if clip_scale != 1.0:
                        img = torch.nn.functional.interpolate(
                            img, scale_factor=clip_scale, mode="bilinear", align_corners=False
                        )
                    feat = clip.encode_dense(img).float()[0]  # [h, w, D]
                    scores2d = (feat @ text.T).permute(2, 0, 1)[None]  # [1, 19, h, w]
                    scores2d = torch.nn.functional.interpolate(
                        scores2d, size=(H, W), mode="bilinear", align_corners=False
                    )[0]
                    pred2d = scores2d.reshape(len(CLASS_NAMES), -1).argmax(dim=0) + 1
                    conf_2d.update(gt[covered], pred2d[flat][covered])
            else:
                rel_err = (pred_depth * scale - lidar_z).abs() / lidar_z.clamp_min(1e-3)
                accurate = covered & (rel_err < args.depth_tolerance)
                conf_geom.update(gt[accurate], pred_cls[accurate])

    # ── Report ──────────────────────────────────────────────────────────────
    iou = conf_all.iou()
    present = conf_all.present()
    evaluated = present.clone()
    evaluated[0] = False  # never score the ignore class

    print(f"\n{'class':<16}{'IoU':>8}   (blank = absent from the evaluated frames)")
    for c in range(1, num_classes):
        if evaluated[c]:
            print(f"{CLASS_NAMES[c - 1]:<16}{iou[c] * 100:7.2f}%")
    miou = float(iou[evaluated].mean() * 100)
    print(f"\n{'mIoU':<16}{miou:7.2f}%   over {int(evaluated.sum())} classes present")
    print(f"{'point acc':<16}{conf_all.accuracy() * 100:7.2f}%")
    print(f"{'coverage':<16}{100 * n_covered / max(n_points, 1):7.2f}%   "
          f"({n_covered:,}/{n_points:,} LiDAR points landed in a populated voxel)")
    print("mIoU and accuracy are over covered points only; coverage is the separate metric.")

    results = {
        "field": os.path.abspath(args.field),
        "sequence": args.sequence,
        "frames_evaluated": len(indices),
        "frame_stride": args.frame_stride,
        "miou": miou,
        "point_accuracy": conf_all.accuracy() * 100,
        "coverage": 100 * n_covered / max(n_points, 1),
        "per_class_iou": {
            CLASS_NAMES[c - 1]: float(iou[c] * 100) for c in range(1, num_classes) if evaluated[c]
        },
        "prompts": dict(zip(CLASS_NAMES, CLASS_PROMPTS)),
    }

    if args.baseline_2d:
        iou_2d = conf_2d.iou()
        miou_2d = float(iou_2d[evaluated].mean() * 100)
        print(f"\nAblation — raw per-frame 2D CLIP on the identical points:")
        print(f"{'mIoU (2D)':<16}{miou_2d:7.2f}%")
        print(f"{'mIoU (fused 3D)':<16}{miou:7.2f}%")
        print(f"  3D fusion contributes {miou - miou_2d:+.2f} mIoU points.")
        results.update({
            "miou_2d_baseline": miou_2d,
            "fusion_gain": miou - miou_2d,
            "per_class_iou_2d": {
                CLASS_NAMES[c - 1]: float(iou_2d[c] * 100) for c in range(1, num_classes) if evaluated[c]
            },
        })

    if args.report_geometry:
        iou_g = conf_geom.iou()
        miou_g = float(iou_g[evaluated].mean() * 100)
        scale = torch.cat(ratios).median().item()
        rel = None
        print(f"\nGeometry-gated (relative depth error < {args.depth_tolerance:.0%}):")
        print(f"{'mIoU':<16}{miou_g:7.2f}%")
        print(f"  The gap of {miou_g - miou:+.2f} points is semantic error attributable to depth.")
        results.update({
            "depth_scale": scale,
            "miou_geometry_gated": miou_g,
            "depth_tolerance": args.depth_tolerance,
            "per_class_iou_geometry_gated": {
                CLASS_NAMES[c - 1]: float(iou_g[c] * 100) for c in range(1, num_classes) if evaluated[c]
            },
        })
        _ = rel

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
