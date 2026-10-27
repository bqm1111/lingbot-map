"""Minimal ONCE reader — exactly what the frozen carve and the probe need, nothing more.

ONCE's own documentation fixes the two facts this module depends on:

  * *"The LiDAR coordinate is placed at the center of the LiDAR sensor, with x positive to the
    left, y-axis positive to the back and the z-axis positive upwards."* So the point cloud is
    already in the sensor's own frame and the carve rule's ray origin is published. §G1 checks
    that empirically before anything trusts it.
  * *"relative pose to the first frame of the scene"*, as `(qx, qy, qz, qw, tx, ty, tz)`.

**The axis rotation.** ONCE is x-left / y-back / z-up; the SSCBench-KITTI-360 grid this project
uses is x-forward / y-left / z-up with origin (0, −25.6, −2). The map between them is an exact
90° yaw, not a fitted quantity:

    ours_x (forward) = −once_y      ours_y (left) = once_x      ours_z (up) = once_z

    R = [[0, −1, 0],
         [1,  0, 0],
         [0,  0, 1]]          det(R) = +1

Points are rotated by `R`; pose transforms are conjugated, `T_ours = A T_once A⁻¹` with
`A = diag(R, 1)`, which is the only composition that leaves relative geometry unchanged.
§G2 verifies the rotation against the vehicle's own direction of travel rather than asserting
it.

This module reads **no labels** and no images. It exposes the three members
`core.supervision.rawtarget.build` touches, so the frozen builder runs unmodified:
`cam0_to_world` (membership only), `read_velodyne`, `velo_to_velo`.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

#: ONCE LiDAR frame -> this project's x-forward grid frame. Exact, not fitted.
R_ONCE_TO_OURS = np.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]], np.float64)
A_ONCE_TO_OURS = np.eye(4)
A_ONCE_TO_OURS[:3, :3] = R_ONCE_TO_OURS

CAMERAS = ("cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09")


def _quat_to_R(q) -> np.ndarray:
    """ONCE stores (qx, qy, qz, qw)."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
                    np.float64)


def _pose_to_T(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(p[:4])
    T[:3, 3] = [float(v) for v in p[4:]]
    return T


class OnceSequence:
    """One ONCE sequence, in this project's x-forward convention.

    Frame indices are the sequence's own 0-based order, which is also time order (the audit
    confirmed every sequence's timestamps are strictly increasing).
    """

    def __init__(self, data_root: str, seq: str, rotate: bool = True):
        self.data_root, self.seq, self.rotate = data_root, seq, bool(rotate)
        j = json.load(open(os.path.join(data_root, seq, f"{seq}.json")))
        self.frames = j["frames"]
        self.calib = j["calib"]
        self.frame_ids: List[str] = [str(f["frame_id"]) for f in self.frames]
        self.t_s = np.array([float(f["frame_id"]) for f in self.frames], np.float64) / 1000.0
        A = A_ONCE_TO_OURS if rotate else np.eye(4)
        Ai = np.linalg.inv(A)
        # sensor -> scene-first-frame, expressed in our convention
        self.T_sensor_to_world: Dict[int, np.ndarray] = {
            i: A @ _pose_to_T(f["pose"]) @ Ai
            for i, f in enumerate(self.frames) if f.get("pose") is not None}
        #: ``rawtarget.build`` only tests membership of this mapping
        self.cam0_to_world = self.T_sensor_to_world

    # ---- the three members the frozen builder uses -------------------------
    def read_velodyne(self, nf: int) -> np.ndarray:
        p = os.path.join(self.data_root, self.seq, "lidar_roof",
                         f"{self.frame_ids[int(nf)]}.bin")
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        return pts @ R_ONCE_TO_OURS.T if self.rotate else pts

    def velo_to_velo(self, a: int, b: int) -> np.ndarray:
        return np.linalg.inv(self.T_sensor_to_world[int(b)]) @ self.T_sensor_to_world[int(a)]

    # ---- helpers -----------------------------------------------------------
    def intensity(self, nf: int) -> np.ndarray:
        p = os.path.join(self.data_root, self.seq, "lidar_roof",
                         f"{self.frame_ids[int(nf)]}.bin")
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, 3]

    def has_cloud(self, nf: int) -> bool:
        return os.path.exists(os.path.join(self.data_root, self.seq, "lidar_roof",
                                           f"{self.frame_ids[int(nf)]}.bin"))

    def K(self, cam: str) -> np.ndarray:
        return np.asarray(self.calib[cam]["cam_intrinsic"], np.float64)

    def distortion(self, cam: str) -> np.ndarray:
        return np.asarray(self.calib[cam]["distortion"], np.float64).ravel()

    def cam_to_velo(self, cam: str) -> np.ndarray:
        """camera -> LiDAR, with the LiDAR side in our convention when ``rotate``.

        **Left-multiplication only, NOT conjugation.** This transform's two endpoints are
        different frames: the source is ONCE's camera (OpenCV pinhole, x-right / y-down /
        z-forward) and the target is ONCE's LiDAR. Only the target is being re-expressed in
        our x-forward convention, so the map is ``A @ T``. Conjugating it as ``A T A^-1`` --
        which is correct for :meth:`velo_to_velo` and for the poses, where BOTH endpoints are
        ONCE frames -- would additionally rotate the camera frame by -90 deg of yaw. That bug
        was present until 2026-10-27 and was caught by the arm's input/target alignment check
        (map-vs-GT IoU 0.2 % instead of ~9 %); the gates could not catch it because G1 and G2
        use LiDAR only and the distortion gate compared two projections of the same points,
        so an extrinsic error cancelled.
        """
        T = np.asarray(self.calib[cam]["cam_to_velo"], np.float64)
        if not self.rotate:
            return T
        return A_ONCE_TO_OURS @ T

    def image_path(self, nf: int, cam: str) -> str:
        return os.path.join(self.data_root, self.seq, cam,
                            f"{self.frame_ids[int(nf)]}.jpg")

    def window(self, anchor: int, span_s: float | None = None,
               n_sweeps: int | None = None, max_gap_s: float = 0.75) -> List[int]:
        """Sweeps from ``anchor`` forward, by time span or by count. No interpolation.

        Exactly one of ``span_s`` / ``n_sweeps`` is used. Both modes consume only real frames
        at ONCE's native 0.5 s cadence and reject any window containing an interval longer
        than ``max_gap_s`` -- 1.7 % of ONCE intervals are gaps of up to 13 s, so this cannot
        be assumed away.

        ``n_sweeps=21`` reproduces the frozen recipe: KITTI-360 uses 21 sweeps at 0.523 s
        spacing (span 10.46 s) and Waymo 21 at 0.500 s (span 10.0 s). ONCE's native 0.5 s
        cadence matches that spacing, so 21 ONCE sweeps span 10.0 s.
        """
        assert (span_s is None) != (n_sweeps is None), "give exactly one of span_s / n_sweeps"
        idx = [int(anchor)]
        t0 = self.t_s[anchor]
        for i in range(int(anchor) + 1, len(self.t_s)):
            if self.t_s[i] - self.t_s[i - 1] > max_gap_s:
                return []
            if span_s is not None and self.t_s[i] - t0 > span_s:
                break
            idx.append(i)
            if n_sweeps is not None and len(idx) >= int(n_sweeps):
                break
        if n_sweeps is not None and len(idx) < int(n_sweeps):
            return []
        if not all(i in self.T_sensor_to_world and self.has_cloud(i) for i in idx):
            return []
        return idx


def sequences(data_root: str) -> List[str]:
    return sorted(d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))


__all__ = ["OnceSequence", "sequences", "R_ONCE_TO_OURS", "A_ONCE_TO_OURS", "CAMERAS"]
