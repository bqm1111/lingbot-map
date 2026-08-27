"""Readers for the metrically supervised sequences available on this machine.

Both readers return ground truth in a common form:

    * ``poses_c2w``   (N, 3, 4)  camera-to-world, OpenCV camera axes, metres
    * ``depth(i)``    (H, W)     metric Z-depth in the camera frame, or None
    * ``valid(i)``    (H, W)     bool validity mask for the depth

Conventions were established empirically (see ``docs/prompted_lingbot_feasibility.md``),
never assumed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# TartanAir V2
# --------------------------------------------------------------------------- #
# Empirically determined (48-way brute force over signed axis permutations x
# {c2w, w2c} x {RGBA, BGRA} depth decodes, scored by cross-view metric-depth
# reprojection inlier ratio):
#
#     decode = BGRA (i.e. what cv2.IMREAD_UNCHANGED yields), viewed as float32
#     direction = camera-to-world
#     R_c2w_opencv = quat_to_R(qx, qy, qz, qw) @ NED2CV
#
#   score (inlier ratio @ rel<2%): 0.608 for the winner, 0.132 for the runner-up.
NED2CV = np.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])

# TartanAir V2 pinhole cameras are 640x640 with a 90 degree field of view.
TARTANAIR_V2_K = np.array([[320.0, 0.0, 320.0],
                           [0.0, 320.0, 320.0],
                           [0.0, 0.0, 1.0]])

# Sky and other "infinite" pixels are written with a very large depth value.
TARTANAIR_MAX_VALID_DEPTH = 200.0


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` quaternion in ``(x, y, z, w)`` order -> ``(..., 3, 3)``."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def decode_tartanair_depth(path: str) -> np.ndarray:
    """Decode a TartanAir V2 ``*_depth.png`` into metric float32 Z-depth.

    The file is a 4-channel 8-bit PNG whose bytes are a little-endian float32 in
    **BGRA** order.  PIL hands back RGBA, so the channels must be swapped first;
    reading it as RGBA silently produces a plausible-looking but wrong depth map.
    """
    rgba = np.array(Image.open(path))
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(f"{path}: expected an 8-bit RGBA PNG, got {rgba.shape}/{rgba.dtype}")
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    return bgra.view("<f4").squeeze(-1)


@dataclass
class Sequence:
    """One contiguous, metrically supervised image sequence."""

    name: str
    dataset: str
    scene: str                      # physical scene identity, used for splitting
    image_paths: List[str]
    poses_c2w: np.ndarray           # (N, 3, 4) metric, OpenCV camera axes
    K_gt: np.ndarray                # (3, 3) intrinsics of the source images
    image_hw: Tuple[int, int]
    depth_paths: Optional[List[str]] = None
    depth_kind: str = "none"        # "tartanair_png" | "npy" | "none"
    max_valid_depth: float = np.inf
    timestamps: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def has_depth(self) -> bool:
        return self.depth_paths is not None

    def load_depth(self, i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(depth, valid)`` for frame ``i``, or ``(None, None)``."""
        if self.depth_paths is None:
            return None, None
        p = self.depth_paths[i]
        if self.depth_kind == "tartanair_png":
            d = decode_tartanair_depth(p)
        elif self.depth_kind == "npy":
            d = np.load(p).astype(np.float32)
        else:
            raise ValueError(f"unknown depth_kind {self.depth_kind!r}")
        valid = np.isfinite(d) & (d > 1e-3) & (d < self.max_valid_depth)
        return d, valid

    def trajectory_length(self) -> float:
        c = self.poses_c2w[:, :3, 3]
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def _chunk(seq: Sequence, chunk_size: Optional[int], min_chunk: int) -> List[Sequence]:
    if chunk_size is None or len(seq) <= chunk_size:
        return [seq]
    out = []
    for start in range(0, len(seq), chunk_size):
        stop = min(start + chunk_size, len(seq))
        if stop - start < min_chunk:
            break
        out.append(Sequence(
            name=f"{seq.name}_c{start:06d}",
            dataset=seq.dataset,
            scene=seq.scene,
            image_paths=seq.image_paths[start:stop],
            poses_c2w=seq.poses_c2w[start:stop].copy(),
            K_gt=seq.K_gt.copy(),
            image_hw=seq.image_hw,
            depth_paths=None if seq.depth_paths is None else seq.depth_paths[start:stop],
            depth_kind=seq.depth_kind,
            max_valid_depth=seq.max_valid_depth,
            timestamps=None if seq.timestamps is None else seq.timestamps[start:stop].copy(),
            extra=dict(seq.extra, source_start=start),
        ))
    return out


def discover_tartanair_v2(
    root: str,
    environments: Optional[Sequence[str]] = None,
    camera: str = "lcam_front",
    chunk_size: Optional[int] = 500,
    min_chunk: int = 120,
) -> List[Sequence]:
    """Find TartanAir V2 trajectories that have both images, depth and poses."""
    seqs: List[Sequence] = []
    if not os.path.isdir(root):
        return seqs
    for env in sorted(os.listdir(root)):
        if environments is not None and env not in environments:
            continue
        env_dir = os.path.join(root, env)
        if not os.path.isdir(env_dir):
            continue
        for diff in sorted(os.listdir(env_dir)):
            diff_dir = os.path.join(env_dir, diff)
            if not os.path.isdir(diff_dir):
                continue
            for traj in sorted(os.listdir(diff_dir)):
                t_dir = os.path.join(diff_dir, traj)
                img_dir = os.path.join(t_dir, f"image_{camera}")
                dep_dir = os.path.join(t_dir, f"depth_{camera}")
                pose_f = os.path.join(t_dir, f"pose_{camera}.txt")
                if not (os.path.isdir(img_dir) and os.path.isdir(dep_dir) and os.path.isfile(pose_f)):
                    continue
                imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
                deps = sorted(f for f in os.listdir(dep_dir) if f.endswith(".png"))
                pose = np.loadtxt(pose_f)
                n = min(len(imgs), len(deps), len(pose))
                if n < min_chunk:
                    continue
                R = quat_xyzw_to_matrix(pose[:n, 3:7]) @ NED2CV
                t = pose[:n, :3]
                poses = np.concatenate([R, t[:, :, None]], axis=-1)
                seq = Sequence(
                    name=f"tartanair_{env}_{diff}_{traj}_{camera}",
                    dataset="tartanair_v2",
                    scene=f"tartanair::{env}",
                    image_paths=[os.path.join(img_dir, f) for f in imgs[:n]],
                    poses_c2w=poses,
                    K_gt=TARTANAIR_V2_K.copy(),
                    image_hw=(640, 640),
                    depth_paths=[os.path.join(dep_dir, f) for f in deps[:n]],
                    depth_kind="tartanair_png",
                    max_valid_depth=TARTANAIR_MAX_VALID_DEPTH,
                    extra={"environment": env, "difficulty": diff, "trajectory": traj},
                )
                seqs.extend(_chunk(seq, chunk_size, min_chunk))
    return seqs


# --------------------------------------------------------------------------- #
# KITTI odometry
# --------------------------------------------------------------------------- #
def _read_kitti_calib(path: str) -> np.ndarray:
    """Return ``P2`` as a ``(3, 4)`` projection matrix."""
    with open(path) as f:
        for line in f:
            if line.startswith("P2:"):
                return np.fromstring(line[3:], sep=" ").reshape(3, 4)
    raise ValueError(f"no P2 in {path}")


def discover_kitti_odometry(
    root: str,
    sequences: Sequence[str] = ("09", "10"),
    chunk_size: Optional[int] = 500,
    min_chunk: int = 120,
    with_depth: bool = False,
) -> List[Sequence]:
    """KITTI odometry sequences with ground-truth metric poses.

    GT poses in ``poses/<seq>.txt`` are row-major 3x4 **camera-to-world** matrices
    in the left colour camera frame, which is already OpenCV-oriented, so no axis
    remap is needed.  ``with_depth`` enables the derived ``depth/sequences/<seq>``
    maps -- these are densified, not raw LiDAR, so they are off by default.
    """
    out: List[Sequence] = []
    for s in sequences:
        img_dir = os.path.join(root, "sequences", s, "image_2")
        pose_f = os.path.join(root, "poses", f"{s}.txt")
        calib_f = os.path.join(root, "sequences", s, "calib.txt")
        if not (os.path.isdir(img_dir) and os.path.isfile(pose_f) and os.path.isfile(calib_f)):
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        poses = np.loadtxt(pose_f).reshape(-1, 3, 4)
        n = min(len(imgs), len(poses))
        P2 = _read_kitti_calib(calib_f)
        K = P2[:3, :3].copy()
        with Image.open(os.path.join(img_dir, imgs[0])) as im:
            W, H = im.size
        dep_dir = os.path.join(root, "depth", "sequences", s)
        dep = None
        if with_depth and os.path.isdir(dep_dir):
            cand = [os.path.join(dep_dir, f"{i:06d}.npy") for i in range(n)]
            if all(os.path.isfile(c) for c in cand):
                dep = cand
        times_f = os.path.join(root, "sequences", s, "times.txt")
        ts = np.loadtxt(times_f)[:n] if os.path.isfile(times_f) else None
        seq = Sequence(
            name=f"kitti_{s}",
            dataset="kitti_odometry",
            scene=f"kitti::{s}",
            image_paths=[os.path.join(img_dir, f) for f in imgs[:n]],
            poses_c2w=poses[:n],
            K_gt=K,
            image_hw=(H, W),
            depth_paths=dep,
            depth_kind="npy" if dep else "none",
            max_valid_depth=80.0,
            timestamps=ts,
            extra={"sequence": s, "depth_is_densified": bool(dep)},
        )
        out.extend(_chunk(seq, chunk_size, min_chunk))
    return out


def scene_disjoint_split(
    sequences: Sequence[Sequence],
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
) -> dict:
    """Partition sequences by an explicit scene-key assignment.

    Raises if any scene key appears in more than one split, so that frames from
    the same physical scene can never leak across the split boundary.
    """
    keys = {"train": set(train), "val": set(val), "test": set(test)}
    for a in keys:
        for b in keys:
            if a < b and keys[a] & keys[b]:
                raise ValueError(f"scene keys shared between {a} and {b}: {sorted(keys[a] & keys[b])}")
    out = {k: [] for k in keys}
    unassigned = []
    for seq in sequences:
        for split, ks in keys.items():
            if seq.scene in ks or seq.extra.get("trajectory") in ks or seq.name in ks:
                out[split].append(seq)
                break
        else:
            unassigned.append(seq.name)
    out["unassigned"] = unassigned
    return out


# --------------------------------------------------------------------------- #
# Occ3D-nuScenes (CAM_FRONT chronological sequences)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_matrix(q) -> np.ndarray:
    """nuScenes stores quaternions in ``(w, x, y, z)`` order (pyquaternion)."""
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rt(translation, rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(rotation)
    T[:3, 3] = np.asarray(translation, float)
    return T


def order_scene_tokens(scene_info: dict) -> List[str]:
    """Chronological token order for one scene, following ``prev``/``next``."""
    starts = [t for t, v in scene_info.items() if v.get("prev") in (None, "", "EOF")
              or v.get("prev") not in scene_info]
    token = starts[0] if starts else sorted(scene_info)[0]
    out, seen = [], set()
    while token in scene_info and token not in seen:
        out.append(token)
        seen.add(token)
        token = scene_info[token].get("next")
    for t in sorted(scene_info):          # anything the chain missed
        if t not in seen:
            out.append(t)
    return out


def discover_occ3d_nuscenes(
    nuscenes_root: str,
    occ3d_root: str,
    split: str = "val",
    camera: str = "CAM_FRONT",
    scenes: Optional[Sequence[str]] = None,
    limit_scenes: Optional[int] = None,
) -> List[Sequence]:
    """One :class:`Sequence` per nuScenes scene, single camera, chronological.

    Everything comes from Occ3D's own ``annotations.json`` -- ``camera_sensor``
    gives the intrinsics and the sensor-to-ego extrinsic, ``ego_pose`` gives
    ego-to-world -- so the nuScenes devkit is not required.  Ground-truth camera
    poses are ``ego_to_world @ sensor_to_ego``.
    """
    import json

    ann_path = os.path.join(occ3d_root, "annotations.json")
    if not os.path.isfile(ann_path):
        return []
    ann = json.load(open(ann_path))
    wanted = list(ann.get(f"{split}_split", []))
    if scenes is not None:
        wanted = [s for s in wanted if s in set(scenes)]
    if limit_scenes:
        wanted = wanted[:limit_scenes]

    out: List[Sequence] = []
    for scene_name in wanted:
        info = ann["scene_infos"].get(scene_name)
        if not info:
            continue
        tokens = order_scene_tokens(info)
        paths, poses, gt_dirs, sensor_to_ego, K = [], [], [], None, None
        for tok in tokens:
            frame = info[tok]
            cam = frame.get("camera_sensor", {}).get(camera)
            if cam is None:
                continue
            img = os.path.join(nuscenes_root, "samples", os.path.basename(os.path.dirname(cam["img_path"])),
                               os.path.basename(cam["img_path"]))
            if not os.path.isfile(img):
                img = os.path.join(nuscenes_root, cam["img_path"])
            if not os.path.isfile(img):
                continue
            s2e = _rt(cam["extrinsic"]["translation"], cam["extrinsic"]["rotation"])
            e2w = _rt(cam["ego_pose"]["translation"], cam["ego_pose"]["rotation"])
            paths.append(img)
            poses.append((e2w @ s2e)[:3, :4])
            gt_dirs.append(os.path.join(occ3d_root, os.path.dirname(frame["gt_path"])))
            if K is None:
                K = np.asarray(cam["intrinsics"], float)
                sensor_to_ego = s2e
        if len(paths) < 3:
            continue
        with Image.open(paths[0]) as im:
            W, H = im.size
        out.append(Sequence(
            name=f"occ3d_{scene_name}_{camera}",
            dataset="occ3d_nuscenes",
            scene=f"nuscenes::{scene_name}",
            image_paths=paths,
            poses_c2w=np.stack(poses),
            K_gt=K,
            image_hw=(H, W),
            depth_paths=None,
            depth_kind="none",
            max_valid_depth=60.0,
            extra={"scene": scene_name, "camera": camera, "split": split,
                   "gt_dirs": gt_dirs,
                   "sensor_to_ego": sensor_to_ego.tolist()},
        ))
    return out
