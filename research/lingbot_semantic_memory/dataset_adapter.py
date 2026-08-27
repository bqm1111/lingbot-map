"""Dataset backends: deterministic chunk splits, images, oracle geometry, labels.

Two datasets are supported behind one interface so the experiment code path is
identical for both and the KITTI/Replica comparison is not confounded by plumbing:

``kitti``
    SemanticKITTI odometry. Oracle depth from the derived metric ``depth/`` maps,
    oracle poses from ``poses.txt`` (cam0-to-world, converted to cam2), semantic
    labels projected from the LiDAR ``labels/``.

``replica``
    Replica in the Nice-SLAM / iMAP layout: ``<scene>/results/frame%06d.jpg`` +
    ``depth%06d.png`` (uint16, divide by 6553.5 for metres) and ``<scene>/traj.txt``.
    **The pose convention was determined by measurement, not assumption** -- see
    :class:`ReplicaBackend`. This release carries no semantic labels.

Preprocessing is byte-identical to ``demo.py`` (``load_and_preprocess_images`` with
``mode="crop"``). For both datasets at their native resolutions that reduces to a pure
anisotropic resize with **no crop**, so oracle intrinsics are the native matrix scaled
by ``(new_w/old_w, new_h/old_h)`` and there is no principal-point offset to track.
``assert_pure_resize`` fails loudly if that ever stops holding.

Pose convention throughout is **world-to-camera** (``X_cam = R X_world + t``) in OpenCV
axes, matching what ``pose_encoding_to_extri_intri`` returns for predicted poses, so
predicted and oracle geometry are interchangeable in :mod:`reprojection`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from lingbot_map.utils.load_fn import load_and_preprocess_images

from research.lingbot_semantic_memory.config import ChunkSpec, DataConfig, REPO_ROOT

#: SemanticKITTI raw label id -> contiguous 19-class learning id (evaluation only).
LEARNING_MAP: Dict[int, int] = {
    0: 255, 1: 255, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5, 31: 6,
    32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 255, 60: 8, 70: 14,
    71: 15, 72: 16, 80: 17, 81: 18, 99: 255, 252: 0, 253: 6, 254: 5, 255: 7,
    256: 4, 257: 4, 258: 3, 259: 4,
}
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
    "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]
#: Replica depth PNGs are uint16 millimetre-like units with this divisor (cam_params.json).
REPLICA_DEPTH_SCALE = 6553.5


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def scale_intrinsics(K: np.ndarray, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> np.ndarray:
    """Scale a pinhole matrix under a pure resize (no crop)."""
    (oh, ow), (nh, nw) = orig_hw, new_hw
    Ks = K.copy().astype(np.float64)
    Ks[0, :] *= nw / ow
    Ks[1, :] *= nh / oh
    return Ks


def assert_pure_resize(cfg: DataConfig, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> None:
    """Fail loudly if the demo preprocessing cropped, invalidating ``scale_intrinsics``."""
    oh, ow = orig_hw
    nh, nw = new_hw
    exp_w = cfg.image_size
    exp_h = int(round(oh * (exp_w / ow) / cfg.patch_size) * cfg.patch_size)
    if (nh, nw) != (exp_h, exp_w) or exp_h > cfg.image_size:
        raise RuntimeError(
            f"preprocessing is not a pure resize for {ow}x{oh} -> {nw}x{nh}; "
            "oracle intrinsics would need a crop offset")


def _resize_nearest(arr: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize so no depth is invented across a discontinuity."""
    oh, ow = arr.shape
    nh, nw = new_hw
    yi = np.clip((np.arange(nh) + 0.5) * oh / nh, 0, oh - 1).astype(np.int64)
    xi = np.clip((np.arange(nw) + 0.5) * ow / nw, 0, ow - 1).astype(np.int64)
    return arr[np.ix_(yi, xi)]


def invert_se3(c2w: np.ndarray) -> np.ndarray:
    """Invert a 4x4 camera-to-world transform into a 3x4 world-to-camera one."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    out = np.zeros((3, 4), dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class DatasetBackend:
    """Common interface the experiment uses; one implementation per dataset."""

    has_labels: bool = False

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def sequence_length(self, seq: str) -> int:
        raise NotImplementedError

    def image_paths(self, chunk: ChunkSpec) -> List[str]:
        raise NotImplementedError

    def original_hw(self, seq: str) -> Tuple[int, int]:
        raise NotImplementedError

    def intrinsics(self, seq: str) -> np.ndarray:
        """Native-resolution pinhole matrix for the frames returned by ``image_paths``."""
        raise NotImplementedError

    def oracle_poses_w2c(self, chunk: ChunkSpec) -> np.ndarray:
        raise NotImplementedError

    def oracle_depth(self, chunk: ChunkSpec, new_hw: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

    def patch_labels(self, chunk: ChunkSpec, new_hw, grid_hw) -> Optional[np.ndarray]:
        return None


class KittiBackend(DatasetBackend):
    """SemanticKITTI odometry, colour-left camera (``image_2``)."""

    has_labels = True

    def __init__(self, cfg: DataConfig):
        super().__init__(cfg)
        self._calib: Dict[str, Dict[str, np.ndarray]] = {}

    def _seq_dir(self, seq: str) -> str:
        return os.path.join(REPO_ROOT, self.cfg.root, "sequences", seq)

    def calib(self, seq: str) -> Dict[str, np.ndarray]:
        """Rectified calibration: ``K``, the cam0->cam2 offset, and velodyne->cam0."""
        if seq in self._calib:
            return self._calib[seq]
        vals: Dict[str, np.ndarray] = {}
        with open(os.path.join(self._seq_dir(seq), "calib.txt")) as fh:
            for line in fh:
                if line.strip():
                    k, v = line.split(":", 1)
                    vals[k.strip()] = np.fromstring(v, sep=" ")
        P2 = vals["P2"].reshape(3, 4)
        Tr = np.eye(4)
        Tr[:3, :4] = vals["Tr"].reshape(3, 4)
        # Rectified stereo: X_cam2 = X_cam0 + t_cam2, from the P2 offset column.
        out = {"K": P2[:3, :3].copy(),
               "t_cam2": np.array([P2[0, 3] / P2[0, 0], P2[1, 3] / P2[1, 1], P2[2, 3]]),
               "Tr": Tr}
        self._calib[seq] = out
        return out

    def sequence_length(self, seq: str) -> int:
        d = os.path.join(self._seq_dir(seq), self.cfg.image_dirname)
        return len([f for f in os.listdir(d) if f.endswith(".png")])

    def image_paths(self, chunk: ChunkSpec) -> List[str]:
        d = os.path.join(self._seq_dir(chunk.sequence), self.cfg.image_dirname)
        return [os.path.join(d, f"{i:06d}.png") for i in chunk.frame_indices]

    def original_hw(self, seq: str) -> Tuple[int, int]:
        w, h = Image.open(os.path.join(self._seq_dir(seq), self.cfg.image_dirname,
                                       "000000.png")).size
        return h, w

    def intrinsics(self, seq: str) -> np.ndarray:
        return self.calib(seq)["K"]

    def oracle_poses_w2c(self, chunk: ChunkSpec) -> np.ndarray:
        """GT world-to-camera2 extrinsics ``[S, 3, 4]``.

        ``poses.txt`` gives cam0-to-world; cam2 is offset by the rectified baseline
        ``X_cam2 = X_cam0 + t_cam2``, and the result is inverted to world-to-camera.
        """
        cal = self.calib(chunk.sequence)
        raw = np.loadtxt(os.path.join(self._seq_dir(chunk.sequence), "poses.txt")).reshape(-1, 3, 4)
        out = np.zeros((chunk.length, 3, 4), dtype=np.float64)
        for i, fi in enumerate(chunk.frame_indices):
            R, t = raw[fi][:3, :3], raw[fi][:3, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t - R @ cal["t_cam2"]
            out[i] = invert_se3(c2w)
        return out

    def oracle_depth(self, chunk: ChunkSpec, new_hw: Tuple[int, int]) -> np.ndarray:
        d = os.path.join(REPO_ROOT, self.cfg.root, "depth", "sequences", chunk.sequence)
        out = np.zeros((chunk.length, *new_hw), dtype=np.float32)
        for i, fi in enumerate(chunk.frame_indices):
            out[i] = _resize_nearest(np.load(os.path.join(d, f"{fi:06d}.npy")).astype(np.float32), new_hw)
        return out

    def patch_labels(self, chunk: ChunkSpec, new_hw, grid_hw) -> np.ndarray:
        """Majority SemanticKITTI class per patch, ``[S, gh, gw]``, 255 = unlabelled.

        LiDAR points go velodyne -> cam0 -> cam2, project with the scaled ``P2``, and
        vote into the patch grid. Evaluation only; never enters a training loss.
        """
        cal = self.calib(chunk.sequence)
        seq_dir = self._seq_dir(chunk.sequence)
        nh, nw = new_hw
        gh, gw = grid_hw
        ph, pw = nh / gh, nw / gw
        K = scale_intrinsics(cal["K"], self.original_hw(chunk.sequence), new_hw)
        lut = np.full(260, 255, dtype=np.uint8)
        for k, v in LEARNING_MAP.items():
            lut[k] = v

        out = np.full((chunk.length, gh, gw), 255, dtype=np.uint8)
        for i, fi in enumerate(chunk.frame_indices):
            pts = np.fromfile(os.path.join(seq_dir, "velodyne", f"{fi:06d}.bin"),
                              dtype=np.float32).reshape(-1, 4)[:, :3]
            lab = np.fromfile(os.path.join(seq_dir, "labels", f"{fi:06d}.label"),
                              dtype=np.uint32) & 0xFFFF
            cls = lut[np.clip(lab, 0, 259)]
            cam2 = (cal["Tr"][:3, :3] @ pts.T).T + cal["Tr"][:3, 3] + cal["t_cam2"]
            z = cam2[:, 2]
            keep = (z > 0.5) & (cls != 255)
            cam2, cls, z = cam2[keep], cls[keep], z[keep]
            if len(cls) == 0:
                continue
            uv = (K @ cam2.T).T
            gx = np.floor(uv[:, 0] / z / pw).astype(np.int64)
            gy = np.floor(uv[:, 1] / z / ph).astype(np.int64)
            ok = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
            gx, gy, c = gx[ok], gy[ok], cls[ok]
            if len(c) == 0:
                continue
            votes = np.zeros((gh * gw, 19), dtype=np.int32)
            np.add.at(votes, (gy * gw + gx, c.astype(np.int64)), 1)
            cell = np.full(gh * gw, 255, dtype=np.uint8)
            has = votes.max(1) > 0
            cell[has] = votes.argmax(1)[has].astype(np.uint8)
            out[i] = cell.reshape(gh, gw)
        return out


class ReplicaBackend(DatasetBackend):
    """Replica in the Nice-SLAM / iMAP layout.

    **Pose convention (measured, not assumed).** ``traj.txt`` holds row-major 4x4
    **camera-to-world** matrices that are already in OpenCV axes (x right, y down,
    z forward). This was established by warping frame *i*'s ground-truth depth into
    frame *j* and comparing against frame *j*'s own depth: the identity convention
    gives a median relative depth error of **0.0004** with 99.6 % of points agreeing
    within 5 %, whereas the widely copied Nice-SLAM loader flip
    (``c2w[:3, 1] *= -1; c2w[:3, 2] *= -1``) gives 0.098 and only 25 %. That flip
    serves Nice-SLAM's internal OpenGL raycasting and is wrong for OpenCV projection.
    ``tests/test_replica.py`` pins this on real data.

    This release ships no semantic annotation, so :meth:`patch_labels` returns ``None``
    and boundary preservation falls back to the depth-discontinuity definition.
    """

    has_labels = False

    def _scene_dir(self, seq: str) -> str:
        return os.path.join(REPO_ROOT, self.cfg.root, seq)

    def sequence_length(self, seq: str) -> int:
        d = os.path.join(self._scene_dir(seq), "results")
        return len([f for f in os.listdir(d) if f.startswith("frame") and f.endswith(".jpg")])

    def image_paths(self, chunk: ChunkSpec) -> List[str]:
        d = os.path.join(self._scene_dir(chunk.sequence), "results")
        return [os.path.join(d, f"frame{i:06d}.jpg") for i in chunk.frame_indices]

    def original_hw(self, seq: str) -> Tuple[int, int]:
        w, h = Image.open(os.path.join(self._scene_dir(seq), "results", "frame000000.jpg")).size
        return h, w

    def intrinsics(self, seq: str) -> np.ndarray:
        import json
        p = os.path.join(REPO_ROOT, self.cfg.root, "cam_params.json")
        c = json.load(open(p))["camera"]
        return np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])

    def depth_scale(self) -> float:
        import json
        p = os.path.join(REPO_ROOT, self.cfg.root, "cam_params.json")
        return float(json.load(open(p))["camera"].get("scale", REPLICA_DEPTH_SCALE))

    def oracle_poses_w2c(self, chunk: ChunkSpec) -> np.ndarray:
        traj = np.loadtxt(os.path.join(self._scene_dir(chunk.sequence), "traj.txt")).reshape(-1, 4, 4)
        return np.stack([invert_se3(traj[fi]) for fi in chunk.frame_indices], axis=0)

    def oracle_depth(self, chunk: ChunkSpec, new_hw: Tuple[int, int]) -> np.ndarray:
        d = os.path.join(self._scene_dir(chunk.sequence), "results")
        scale = self.depth_scale()
        out = np.zeros((chunk.length, *new_hw), dtype=np.float32)
        for i, fi in enumerate(chunk.frame_indices):
            raw = np.array(Image.open(os.path.join(d, f"depth{fi:06d}.png"))).astype(np.float32) / scale
            out[i] = _resize_nearest(raw, new_hw)
        return out


BACKENDS = {"kitti": KittiBackend, "replica": ReplicaBackend}


def get_backend(cfg: DataConfig) -> DatasetBackend:
    if cfg.dataset not in BACKENDS:
        raise ValueError(f"unknown dataset {cfg.dataset!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[cfg.dataset](cfg)


# --------------------------------------------------------------------------- #
# Splits and images
# --------------------------------------------------------------------------- #
def build_splits(cfg: DataConfig, seed: int) -> Tuple[List[ChunkSpec], List[ChunkSpec]]:
    """Deterministically choose contiguous chunks for training and validation.

    Chunk starts come from a ``numpy`` generator seeded with ``seed`` and a stable
    per-sequence index, so the split depends only on ``(seed, cfg)`` and never on
    iteration order or filesystem listing order.
    """
    backend = get_backend(cfg)
    span = cfg.chunk_length * cfg.stride

    def chunks_for(seqs: Sequence[str], n_per_seq: int, tag: str) -> List[ChunkSpec]:
        out: List[ChunkSpec] = []
        for si, seq in enumerate(sorted(seqs)):
            n = backend.sequence_length(seq)
            hi = n - span
            if hi <= 0:
                raise ValueError(f"sequence {seq} has {n} frames, shorter than span {span}")
            rng = np.random.default_rng([seed, si, len(tag)])
            for k in range(n_per_seq):
                base = int(round((k + 0.5) / n_per_seq * hi))
                jitter = int(rng.integers(-span // 2, span // 2 + 1))
                start = int(np.clip(base + jitter, 0, hi))
                out.append(ChunkSpec(seq, start, cfg.chunk_length, tag, cfg.stride))
        return out

    return (chunks_for(cfg.train_sequences, cfg.train_chunks_per_sequence, "train"),
            chunks_for(cfg.val_sequences, cfg.val_chunks, "val"))


def load_chunk_images(cfg: DataConfig, chunk: ChunkSpec) -> torch.Tensor:
    """Load one chunk as ``[S, 3, H, W]`` in ``[0, 1]`` using the demo preprocessing."""
    paths = get_backend(cfg).image_paths(chunk)
    return load_and_preprocess_images(
        paths, mode="crop", image_size=cfg.image_size, patch_size=cfg.patch_size)
