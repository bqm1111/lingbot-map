"""SemanticKITTI / KITTI-odometry layer for the scale gate.

Responsibilities kept deliberately narrow: validate the raw layout, parse calibration
with the transformation chain named explicitly, build deterministic clip manifests, and
project LiDAR into each image to produce **sparse metric depth** at LingBot's output
resolution.

Transformation chain, stated once and used everywhere::

    p_velo --Tr--> p_cam0 --(+t_cam2)--> p_cam2 --K2--> p_pixel(original)
                                                   --resize--> p_pixel(processed)

Two conventions that are easy to get wrong and are therefore pinned here:

* ``P2`` is a *projection* matrix, not an intrinsic matrix: its 4th column carries the
  rectified stereo baseline. The intrinsics are ``P2[:3, :3]``; the baseline becomes the
  cam0 -> cam2 translation ``t_cam2 = (P2[0,3]/fx, P2[1,3]/fy, P2[2,3])``.
* ``poses.txt`` holds **cam0-to-world** 3x4 matrices in metres. Frames observed by
  ``image_2`` therefore need the cam0 -> cam2 offset applied before comparison.

Depth stored here is **optical-axis z depth** in the camera frame, matching LingBot's
verified convention (``occ_datasets.camera_points_from_depth``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class KittiCalibration:
    """Rectified calibration for one sequence."""

    K: np.ndarray          # [3, 3] cam2 intrinsics at native resolution
    t_cam2: np.ndarray     # [3] cam0 -> cam2 translation, metres
    Tr: np.ndarray         # [4, 4] velodyne -> cam0
    P2: np.ndarray         # [3, 4] raw projection matrix, kept for provenance

    def velo_to_cam2(self, pts: np.ndarray) -> np.ndarray:
        """``[N, 3]`` velodyne points -> cam2 frame (OpenCV axes, metres)."""
        cam0 = pts @ self.Tr[:3, :3].T + self.Tr[:3, 3]
        return cam0 + self.t_cam2

    def to_dict(self) -> Dict[str, object]:
        return {"K": self.K.tolist(), "t_cam2": self.t_cam2.tolist(),
                "Tr": self.Tr.tolist(), "P2": self.P2.tolist()}


def parse_calibration(path: str) -> KittiCalibration:
    """Parse a KITTI-odometry ``calib.txt``."""
    vals: Dict[str, np.ndarray] = {}
    with open(path) as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals[k.strip()] = np.fromstring(v, sep=" ")
    for req in ("P2", "Tr"):
        if req not in vals:
            raise ValueError(f"{path}: missing '{req}'")
    P2 = vals["P2"].reshape(3, 4)
    if P2.shape != (3, 4) or not np.isfinite(P2).all():
        raise ValueError(f"{path}: P2 malformed")
    fx, fy = P2[0, 0], P2[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{path}: non-positive focal length")
    Tr = np.eye(4)
    Tr[:3, :4] = vals["Tr"].reshape(3, 4)
    return KittiCalibration(
        K=P2[:3, :3].copy(),
        t_cam2=np.array([P2[0, 3] / fx, P2[1, 3] / fy, P2[2, 3]], dtype=np.float64),
        Tr=Tr, P2=P2)


def load_poses_cam2_c2w(poses_path: str, calib: KittiCalibration) -> np.ndarray:
    """``poses.txt`` (cam0-to-world) -> ``[N, 4, 4]`` **cam2-to-world**, metres.

    ``X_world = R (X_cam2 - t_cam2) + t``, so the cam2 centre sits at
    ``t - R t_cam2`` with the rotation unchanged.
    """
    raw = np.loadtxt(poses_path).reshape(-1, 3, 4)
    out = np.tile(np.eye(4), (len(raw), 1, 1))
    out[:, :3, :3] = raw[:, :3, :3]
    out[:, :3, 3] = raw[:, :3, 3] - np.einsum("nij,j->ni", raw[:, :3, :3], calib.t_cam2)
    return out


# --------------------------------------------------------------------------- #
# Preprocessing geometry (LingBot's resize)
# --------------------------------------------------------------------------- #
@dataclass
class Preprocess:
    """LingBot's ``mode='crop'`` resize, expressed as an explicit pixel mapping.

    For every KITTI odometry resolution the height after resizing stays below
    ``image_size``, so the crop branch never fires and the mapping is a pure
    anisotropic scale. :meth:`check` refuses to guess if that ever stops holding.
    """

    orig_hw: Tuple[int, int]
    proc_hw: Tuple[int, int]
    image_size: int = 518
    patch_size: int = 14

    @staticmethod
    def build(orig_hw: Tuple[int, int], image_size: int = 518,
              patch_size: int = 14) -> "Preprocess":
        oh, ow = orig_hw
        nw = image_size
        nh = int(round(oh * (nw / ow) / patch_size) * patch_size)
        p = Preprocess(orig_hw=(oh, ow), proc_hw=(nh, nw),
                       image_size=image_size, patch_size=patch_size)
        p.check()
        return p

    def check(self) -> None:
        nh, nw = self.proc_hw
        if nw != self.image_size:
            raise RuntimeError(f"unexpected processed width {nw} != {self.image_size}")
        if nh > self.image_size:
            raise RuntimeError(
                f"processed height {nh} exceeds {self.image_size}: the demo preprocessing "
                "would centre-crop, which this pure-scale mapping does not model")

    @property
    def sx(self) -> float:
        return self.proc_hw[1] / self.orig_hw[1]

    @property
    def sy(self) -> float:
        return self.proc_hw[0] / self.orig_hw[0]

    def scale_intrinsics(self, K: np.ndarray) -> np.ndarray:
        Ks = np.asarray(K, dtype=np.float64).copy()
        Ks[0, :] *= self.sx
        Ks[1, :] *= self.sy
        return Ks

    def map_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Original-image pixels ``[N, 2]`` -> processed-image pixels."""
        uv = np.asarray(uv, dtype=np.float64)
        return np.stack([uv[:, 0] * self.sx, uv[:, 1] * self.sy], axis=-1)

    def to_dict(self) -> Dict[str, object]:
        return {"orig_hw": list(self.orig_hw), "proc_hw": list(self.proc_hw),
                "image_size": self.image_size, "patch_size": self.patch_size,
                "sx": self.sx, "sy": self.sy, "mode": "crop:pure_resize"}


# --------------------------------------------------------------------------- #
# LiDAR projection
# --------------------------------------------------------------------------- #
def project_lidar_to_depth(
    points_velo: np.ndarray, calib: KittiCalibration, out_hw: Tuple[int, int],
    K_out: np.ndarray, min_depth: float = 1.0, max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR into an image plane, z-buffered, as sparse metric **z depth**.

    Points behind the camera, non-finite points and points outside the frame are
    dropped. When several points fall in one pixel the **nearest** is kept, which is
    what a real depth sensor would report and what prevents background bleeding
    through foreground.

    Args:
        points_velo: ``[N, 3]`` raw velodyne XYZ.
        calib: sequence calibration.
        out_hw: target ``(H, W)``.
        K_out: ``[3, 3]`` intrinsics matching ``out_hw``.

    Returns:
        ``(depth [H, W] float32, valid [H, W] bool)`` -- depth is 0 where invalid.
    """
    H, W = out_hw
    cam = calib.velo_to_cam2(np.asarray(points_velo, dtype=np.float64))
    z = cam[:, 2]
    keep = np.isfinite(cam).all(axis=1) & (z > min_depth) & (z < max_depth)
    cam, z = cam[keep], z[keep]
    if cam.shape[0] == 0:
        return np.zeros((H, W), np.float32), np.zeros((H, W), bool)

    u = cam[:, 0] * K_out[0, 0] / z + K_out[0, 2]
    v = cam[:, 1] * K_out[1, 1] / z + K_out[1, 2]
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    if ui.size == 0:
        return np.zeros((H, W), np.float32), np.zeros((H, W), bool)

    # Z-buffer: sort far->near so the nearest point wins the final write.
    order = np.argsort(-z, kind="mergesort")
    depth = np.zeros((H, W), np.float32)
    valid = np.zeros((H, W), bool)
    depth[vi[order], ui[order]] = z[order].astype(np.float32)
    valid[vi[order], ui[order]] = True
    return depth, valid


# --------------------------------------------------------------------------- #
# Layout validation and manifests
# --------------------------------------------------------------------------- #
@dataclass
class SequenceInventory:
    sequence: str
    n_images: int
    n_lidar: int
    n_poses: int
    n_times: int
    image_hw: Tuple[int, int]
    ok: bool
    problems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"sequence": self.sequence, "n_images": self.n_images,
                "n_lidar": self.n_lidar, "n_poses": self.n_poses,
                "n_times": self.n_times, "image_hw": list(self.image_hw),
                "ok": self.ok, "problems": self.problems}


def sequence_dir(root: str, seq: str) -> str:
    return os.path.join(root, "sequences", seq)


def validate_sequence(root: str, seq: str, camera: str = "image_2") -> SequenceInventory:
    """Check one sequence's layout without loading the whole thing."""
    from PIL import Image

    d = sequence_dir(root, seq)
    problems: List[str] = []
    img_dir, velo_dir = os.path.join(d, camera), os.path.join(d, "velodyne")
    imgs = sorted(f for f in os.listdir(img_dir)) if os.path.isdir(img_dir) else []
    velos = sorted(f for f in os.listdir(velo_dir)) if os.path.isdir(velo_dir) else []
    if not imgs:
        problems.append(f"missing or empty {camera}/")
    if not velos:
        problems.append("missing or empty velodyne/")

    img_ids = {os.path.splitext(f)[0] for f in imgs}
    velo_ids = {os.path.splitext(f)[0] for f in velos}
    if img_ids and velo_ids and img_ids != velo_ids:
        problems.append(f"image/LiDAR index mismatch: {len(img_ids ^ velo_ids)} unmatched")

    calib_p, poses_p, times_p = (os.path.join(d, "calib.txt"),
                                 os.path.join(d, "poses.txt"), os.path.join(d, "times.txt"))
    if not os.path.isfile(calib_p):
        problems.append("missing calib.txt")
    else:
        try:
            parse_calibration(calib_p)
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"calib.txt unparseable: {exc}")

    n_poses = 0
    if os.path.isfile(poses_p):
        try:
            n_poses = len(np.loadtxt(poses_p).reshape(-1, 3, 4))
            if imgs and n_poses != len(imgs):
                problems.append(f"pose count {n_poses} != image count {len(imgs)}")
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"poses.txt unreadable: {exc}")
    else:
        problems.append("missing poses.txt")

    n_times = 0
    if os.path.isfile(times_p):
        t = np.loadtxt(times_p).ravel()
        n_times = len(t)
        if n_times and np.any(np.diff(t) < 0):
            problems.append("times.txt is not monotonically non-decreasing")
    else:
        problems.append("missing times.txt")

    hw = (0, 0)
    if imgs:
        w, h = Image.open(os.path.join(img_dir, imgs[0])).size
        hw = (h, w)

    return SequenceInventory(sequence=seq, n_images=len(imgs), n_lidar=len(velos),
                             n_poses=n_poses, n_times=n_times, image_hw=hw,
                             ok=not problems, problems=problems)


def build_clips(
    root: str, sequences: Sequence[str], clip_length: int, frame_stride: int,
    clip_stride: Optional[int] = None, camera: str = "image_2",
    max_clips_per_sequence: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Deterministic chronological clip manifest records.

    A clip is ``clip_length`` frames spaced ``frame_stride`` native frames apart; the
    first frames of consecutive clips are ``clip_stride`` native frames apart. Clips
    that would run past the end of a sequence are dropped rather than truncated, so
    every clip has identical temporal structure.
    """
    if clip_length < 2:
        raise ValueError("clip_length must be >= 2 for a pose-derived scale")
    span = (clip_length - 1) * frame_stride
    step = clip_stride if clip_stride is not None else span + frame_stride
    out: List[Dict[str, object]] = []
    for seq in sorted(sequences):
        d = sequence_dir(root, seq)
        img_dir = os.path.join(d, camera)
        if not os.path.isdir(img_dir):
            continue
        n = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        starts = range(0, max(n - span, 0), step)
        if max_clips_per_sequence is not None:
            starts = list(starts)[:max_clips_per_sequence]
        for s0 in starts:
            ids = [s0 + i * frame_stride for i in range(clip_length)]
            out.append({
                "dataset": "semantickitti",
                "sequence": seq,
                "clip_id": f"{seq}_{ids[0]:06d}_{ids[-1]:06d}_s{frame_stride}",
                "frame_ids": ids,
                "image_paths": [f"sequences/{seq}/{camera}/{i:06d}.png" for i in ids],
                "lidar_paths": [f"sequences/{seq}/velodyne/{i:06d}.bin" for i in ids],
                "calibration_path": f"sequences/{seq}/calib.txt",
                "poses_path": f"sequences/{seq}/poses.txt",
                "times_path": f"sequences/{seq}/times.txt",
            })
    return out


def write_manifest(path: str, records: Iterable[Dict[str, object]]) -> str:
    """Write JSONL and return its sha256, so downstream artifacts can pin it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h = hashlib.sha256()
    with open(path, "w") as fh:
        for r in records:
            line = json.dumps(r, sort_keys=True)
            fh.write(line + "\n")
            h.update(line.encode())
    return h.hexdigest()


def read_manifest(path: str) -> List[Dict[str, object]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_velodyne(path: str) -> np.ndarray:
    """``[N, 4]`` XYZI float32 records; raises if the file is not a clean multiple."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"{path}: {raw.size} floats is not a multiple of 4")
    pts = raw.reshape(-1, 4)
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts
