# extracted from occ3d_zeroshot/nuscenes_adapter.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Occ3D-nuScenes adapter: clips, calibration and coordinate frames.

Everything is read from the official Occ3D ``annotations.json`` plus the images it points
at. The nuScenes devkit is not required: the annotation file already carries per-camera
intrinsics, the camera-to-ego extrinsic, the camera's own ego pose and the keyframe ego
pose, which is all the calibration this experiment is allowed to use.

**Named transforms** (all 4x4, metres, right-handed, camera looks down +z):

    T_camera_to_ego_cam     static calibrated extrinsic of the camera in its own ego frame
    T_ego_cam_to_world      ego pose at the *camera's* exposure timestamp
    T_ego_keyframe_to_world ego pose at the LiDAR keyframe -- the frame Occ3D labels live in
    T_anchor_camera_to_ego  inv(T_ego_keyframe_to_world) @ T_ego_cam_to_world
                            @ T_camera_to_ego_cam

``T_anchor_camera_to_ego`` is evaluated **only at the clip's anchor frame**. It is a
single-timestamp calibration chain that places the finished reconstruction into the
official evaluation frame. It never replaces LingBot's relative poses between the five
input frames, and no other frame's ego pose is read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def quat_to_R(q: Sequence[float]) -> np.ndarray:
    """nuScenes quaternion ``[w, x, y, z]`` -> rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rt_to_T(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(rotation)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


@dataclass
class Frame:
    """One official keyframe, single camera."""
    scene: str
    token: str
    index_in_scene: int
    timestamp_ns: int
    camera: str
    image_path: str
    K: np.ndarray                       # 3x3 native intrinsics
    T_camera_to_ego_cam: np.ndarray     # static extrinsic
    T_ego_cam_to_world: np.ndarray      # ego pose at the camera timestamp
    T_ego_keyframe_to_world: np.ndarray  # ego pose at the LiDAR keyframe
    gt_path: str                        # relative path of labels.npz (evaluator only)

    @property
    def T_camera_to_ego(self) -> np.ndarray:
        """Camera -> ego frame of *this* keyframe, i.e. the Occ3D grid frame."""
        return (np.linalg.inv(self.T_ego_keyframe_to_world)
                @ self.T_ego_cam_to_world @ self.T_camera_to_ego_cam)


def load_annotations(occ3d_root: str) -> dict:
    return json.load(open(os.path.join(occ3d_root, "annotations.json")))


def scene_frames(ann: dict, scene: str, camera: str, nuscenes_root: str) -> List[Frame]:
    """Temporally ordered official keyframes of one scene for one camera."""
    info = ann["scene_infos"][scene]
    out: List[Frame] = []
    for i, (token, f) in enumerate(info.items()):
        cams = f.get("camera_sensor", {})
        if camera not in cams:
            continue
        c = cams[camera]
        out.append(Frame(
            scene=scene, token=token, index_in_scene=i,
            timestamp_ns=int(f["timestamp"]) * 1000,          # nuScenes stores microseconds
            camera=camera,
            image_path=os.path.join(nuscenes_root, "samples", c["img_path"]),
            K=np.asarray(c["intrinsics"], dtype=np.float64),
            T_camera_to_ego_cam=rt_to_T(c["extrinsic"]["translation"],
                                        c["extrinsic"]["rotation"]),
            T_ego_cam_to_world=rt_to_T(c["ego_pose"]["translation"],
                                       c["ego_pose"]["rotation"]),
            T_ego_keyframe_to_world=rt_to_T(f["ego_pose"]["translation"],
                                            f["ego_pose"]["rotation"]),
            gt_path=f["gt_path"]))
    out.sort(key=lambda fr: fr.timestamp_ns)
    return out


@dataclass
class Clip:
    scene: str
    camera: str
    frames: List[Frame]

    @property
    def anchor(self) -> Frame:
        return self.frames[-1]                    # frozen: Gate 0-3.1 fuse into the last

    @property
    def clip_id(self) -> str:
        return (f"{self.scene}_{self.camera}_"
                f"{self.frames[0].token[:8]}_{self.frames[-1].token[:8]}")

    @property
    def duration_s(self) -> float:
        return (self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns) / 1e9

    def spacings_s(self) -> List[float]:
        return [(b.timestamp_ns - a.timestamp_ns) / 1e9
                for a, b in zip(self.frames, self.frames[1:])]


def build_clips(frames: Sequence[Frame], n_frames: int, keyframe_stride: int,
                clip_stride: int, occ3d_root: str) -> tuple[List[Clip], Dict[str, int]]:
    """Non-overlapping five-keyframe clips inside one scene. No label is opened."""
    clips, rej = [], {"short_scene": 0, "missing_image": 0, "missing_gt": 0,
                      "nonmonotonic_time": 0}
    span = (n_frames - 1) * keyframe_stride
    if len(frames) < span + 1:
        rej["short_scene"] += 1
        return clips, rej
    for start in range(0, len(frames) - span, clip_stride):
        sel = [frames[start + k * keyframe_stride] for k in range(n_frames)]
        if any(not os.path.exists(f.image_path) for f in sel):
            rej["missing_image"] += 1
            continue
        # presence check only -- the label is never opened here
        if not os.path.exists(os.path.join(occ3d_root, sel[-1].gt_path)):
            rej["missing_gt"] += 1
            continue
        ts = [f.timestamp_ns for f in sel]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            rej["nonmonotonic_time"] += 1
            continue
        clips.append(Clip(scene=sel[0].scene, camera=sel[0].camera, frames=sel))
    return clips, rej


def val_scenes(ann: dict) -> List[str]:
    return [s for s in ann["val_split"] if s in ann["scene_infos"]]
