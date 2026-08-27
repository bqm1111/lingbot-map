"""Query an open-vocabulary feature field with free text.

    python -m semantic.query_field \
        --field output/kitti_seq08_semantic/field.npz \
        --queries "a car" "the road" "vegetation" \
        --predictions output/kitti_seq08_render/image_2 \
        --output_dir output/kitti_seq08_semantic

Produces, per query, a relevancy-coloured point cloud (PLY) and optionally an
overlay video where every pixel is tinted by the relevancy of the voxel it
falls into.

Scoring modes
    relevancy  (default) LERF-style: how much more this voxel matches the query
               than it matches canonical background prompts.  Absolute cosine
               similarities from CLIP sit in a narrow band (~0.2) and are not
               comparable across prompts; this contrast is.
    softmax    Softmax across the supplied queries — treats them as a closed
               label set, i.e. semantic segmentation.
    cosine     Raw cosine, percentile-normalized for display.
"""

import argparse
import os
import subprocess
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from semantic.build_field import list_frames, load_frame, sky_mask_for
from semantic.dense_clip import DenseCLIP
from semantic.feature_field import FeatureField, unproject

# Generic prompts that any scene content matches to some degree.  Contrasting
# against them is what turns a flat ~0.2 cosine into a usable score.
CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture", "background")
# C biet 
# Distinct, colour-blind-friendly hues for closed-set segmentation, BGR.
CLASS_COLORS = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
    (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23),
]

def relevancy_scores(sim_pos: torch.Tensor, sim_neg: torch.Tensor, temperature: float) -> torch.Tensor:
    """LERF relevancy: min over negatives of the pairwise positive softmax.

    Args:
        sim_pos: [M, T] similarity to each query.
        sim_neg: [M, N] similarity to each canonical negative.

    Returns:
        [M, T] in (0, 1); > 0.5 means the query beats every negative.
    """
    # A 2-way softmax between one positive and one negative logit is exactly
    # sigmoid of their scaled difference, so skip building the stacked tensor.
    margin = (sim_pos.unsqueeze(2) - sim_neg.unsqueeze(1)) / temperature  # [M, T, N]
    return torch.sigmoid(margin).min(dim=-1).values

def compute_scores(
    field: FeatureField,
    clip: DenseCLIP,
    queries: Sequence[str],
    mode: str,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """[M, T] display scores in [0, 1] for every voxel."""
    text = clip.encode_text(queries)
    sim = field.query(text, device)  # [M, T]

    if mode == "relevancy":
        neg = clip.encode_text(CANONICAL_NEGATIVES)
        sim_neg = field.query(neg, device)
        return relevancy_scores(sim, sim_neg, temperature)
    if mode == "softmax":
        if len(queries) < 2:
            raise ValueError("--mode softmax needs at least two queries")
        return torch.softmax(sim / temperature, dim=-1)
    if mode == "cosine":
        lo = torch.quantile(sim, 0.02, dim=0, keepdim=True)
        hi = torch.quantile(sim, 0.98, dim=0, keepdim=True)
        return ((sim - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    raise ValueError(f"unknown mode {mode!r}")


def display_normalize(scores: torch.Tensor, lo_pct: float, hi_pct: float) -> torch.Tensor:
    """Rescale each query's scores to [0, 1] for display.

    Relevancy distributions are heavily skewed — typically only the top few
    percent of voxels match a given query at all — so a fixed threshold either
    shows nothing or everything.  Stretching a high percentile band instead
    keeps the visualization readable regardless of how common the query is.
    """
    lo = torch.quantile(scores, lo_pct / 100.0, dim=0, keepdim=True)
    hi = torch.quantile(scores, hi_pct / 100.0, dim=0, keepdim=True)
    return ((scores - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def colorize(scores: torch.Tensor, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """[N] scores in [0,1] -> [N, 3] BGR uint8."""
    vals = (scores.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().reshape(-1, 1, 1)
    return cv2.applyColorMap(vals, colormap).reshape(-1, 3)


def export_ply(path: str, points: np.ndarray, colors_bgr: np.ndarray) -> None:
    """Write an ASCII-header binary PLY (RGB order)."""
    rgb = colors_bgr[:, ::-1].astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(len(points), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


class FFmpegWriter:
    """Pipe raw BGR frames to ffmpeg (matches the repo's libx264 settings)."""

    def __init__(self, path: str, width: int, height: int, fps: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
            stdin=subprocess.PIPE,
        )
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def label_frame(img: np.ndarray, lines: Sequence[str], colors: Optional[Sequence] = None) -> None:
    """Draw a small legend in the top-left corner, in place."""
    for i, text in enumerate(lines):
        y = 18 + 18 * i
        color = tuple(int(c) for c in colors[i]) if colors else (255, 255, 255)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


@torch.no_grad()
def render_overlay_video(
    field: FeatureField,
    scores: torch.Tensor,
    queries: Sequence[str],
    frames: List[str],
    out_path: str,
    args,
    device: torch.device,
) -> None:
    """Project per-voxel scores back into every camera view and write an MP4."""
    keys_gpu = torch.from_numpy(field.keys).to(device)
    segmentation = args.mode == "softmax"
    palette = torch.tensor(
        [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(queries))],
        device=device, dtype=torch.float32,
    )

    writer = None
    for fp in tqdm(frames, desc=f"Rendering {os.path.basename(out_path)}"):
        depth, conf, image, intr, extr = load_frame(fp, device)
        H, W = depth.shape

        world = unproject(depth, intr, extr).reshape(-1, 3)
        idx = field.lookup(world, keys_gpu)
        hit = idx >= 0
        if args.conf_threshold:
            hit &= (conf.reshape(-1) >= args.conf_threshold)
        keep_sky = sky_mask_for(fp, args.sky_mask_dir, depth.shape, device)
        if keep_sky is not None:
            hit &= keep_sky.reshape(-1)

        rgb = (image.permute(1, 2, 0)[..., [2, 1, 0]] * 255).clamp(0, 255)  # BGR
        gray = rgb.mean(-1, keepdim=True).expand(-1, -1, 3).reshape(-1, 3)

        vox = scores[idx.clamp_min(0)]  # [HW, T]
        if segmentation:
            conf_val, cls = vox.max(dim=-1)
            color = palette[cls]
            alpha = ((conf_val - 1.0 / len(queries)) / (1.0 - 1.0 / len(queries))).clamp(0, 1)
        else:
            val = vox[:, 0]  # already display-normalized by the caller
            color = torch.from_numpy(colorize(val)).to(device).float()
            alpha = ((val - args.alpha_floor) / max(1e-6, 1.0 - args.alpha_floor)).clamp(0, 1)
        alpha = (alpha * hit.float() * args.overlay_strength).unsqueeze(-1)

        out = (gray * (1 - alpha) + color * alpha).reshape(H, W, 3)
        frame = out.clamp(0, 255).to(torch.uint8).cpu().numpy()

        if args.video_scale != 1:
            frame = cv2.resize(
                frame, (W * args.video_scale, H * args.video_scale), interpolation=cv2.INTER_LINEAR
            )
        frame = np.ascontiguousarray(frame)
        if segmentation:
            label_frame(frame, list(queries), [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(queries))])
        else:
            label_frame(frame, [f'"{queries[0]}"'])

        if writer is None:
            writer = FFmpegWriter(out_path, frame.shape[1], frame.shape[0], args.video_fps)
        writer.write(frame)

    if writer is not None:
        writer.close()
        print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True, help="Field .npz from semantic.build_field")
    p.add_argument("--queries", nargs="+", required=True, help="Free-text queries")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--mode", default="relevancy", choices=["relevancy", "softmax", "cosine"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--score_threshold", type=float, default=0.5,
                   help="Voxels scoring above this count as a hit in the printed summary")

    p.add_argument("--export_ply", action="store_true", help="Write a coloured point cloud per query")
    p.add_argument("--ply_threshold", type=float, default=None,
                   help="Keep only voxels above this score in the PLY (default: keep all)")

    p.add_argument("--predictions", default=None, help="Prediction NPZs; enables the overlay video")
    p.add_argument("--sky_mask_dir", default=None)
    p.add_argument("--render_video", action="store_true")
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--video_scale", type=int, default=2)
    p.add_argument("--overlay_strength", type=float, default=0.85)
    p.add_argument("--alpha_floor", type=float, default=0.15,
                   help="Display-normalized score below this stays greyscale in the overlay")
    p.add_argument("--display_lo", type=float, default=90.0,
                   help="Score percentile mapped to 0 when colouring")
    p.add_argument("--display_hi", type=float, default=99.5,
                   help="Score percentile mapped to 1 when colouring")
    p.add_argument("--conf_threshold", type=float, default=1.3)

    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    field = FeatureField.load(args.field)
    meta = field.meta or {}
    print(f"Field: {field.num_voxels:,} voxels, voxel_size={field.voxel_size:.5f}, "
          f"{field.features.shape[1]}d (explains {meta.get('explained_variance', '?')} of variance)")

    clip = DenseCLIP(
        model_name=meta.get("clip_model", "ViT-B-16-quickgelu"),
        pretrained=meta.get("clip_pretrained", "openai"),
        device=args.device,
        variant=meta.get("clip_variant", "maskclip"),
    )

    scores = compute_scores(field, clip, args.queries, args.mode, args.temperature, device)
    print(f"\nScoring mode: {args.mode}")
    for i, q in enumerate(args.queries):
        s = scores[:, i]
        frac = (s > args.score_threshold).float().mean().item()
        print(f'  "{q}": max={s.max():.3f} mean={s.mean():.3f} '
              f'above {args.score_threshold} = {frac * 100:.2f}% of voxels')

    os.makedirs(args.output_dir, exist_ok=True)

    # Colouring uses a stretched percentile band; thresholds stay in raw score units.
    display = (
        scores if args.mode == "softmax"
        else display_normalize(scores, args.display_lo, args.display_hi)
    )

    if args.export_ply:
        centres = field.centres()
        for i, q in enumerate(args.queries):
            slug = "".join(c if c.isalnum() else "_" for c in q).strip("_")
            s = scores[:, i]
            sel = torch.ones_like(s, dtype=torch.bool) if args.ply_threshold is None else s > args.ply_threshold
            n = int(sel.sum())
            if n == 0:
                print(f'  no voxels above {args.ply_threshold} for "{q}"; skipping PLY')
                continue
            path = os.path.join(args.output_dir, f"query_{slug}.ply")
            export_ply(path, centres[sel.cpu().numpy()], colorize(display[sel, i]))
            print(f"  wrote {path} ({n:,} points)")

    if args.render_video:
        if not args.predictions:
            raise SystemExit("--render_video needs --predictions")
        frames = list_frames(args.predictions)
        if args.first_k:
            frames = frames[: args.first_k]
        frames = frames[:: args.frame_stride]

        if args.mode == "softmax":
            out = os.path.join(args.output_dir, "segmentation.mp4")
            render_overlay_video(field, display, args.queries, frames, out, args, device)
        else:
            for i, q in enumerate(args.queries):
                slug = "".join(c if c.isalnum() else "_" for c in q).strip("_")
                out = os.path.join(args.output_dir, f"relevancy_{slug}.mp4")
                render_overlay_video(field, display[:, i : i + 1], [q], frames, out, args, device)


if __name__ == "__main__":
    main()
