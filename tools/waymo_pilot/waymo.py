"""Waymo Open Dataset reader for this pilot. Reads real files or refuses to run.

**Provenance of every convention below.** Field names and enum values were read from the
installed SDK's generated protos (``waymo-open-dataset-tf-2-12-0`` 1.6.7), not from memory:

    Frame                 context, timestamp_micros, pose, images[], lasers[]
    Context               name, camera_calibrations[], laser_calibrations[]
    CameraCalibration     name, intrinsic[9], extrinsic(4x4), width, height,
                          rolling_shutter_direction
    CameraImage           name, image(jpeg), pose(4x4), pose_timestamp, shutter,
                          camera_trigger_time, camera_readout_done_time
    CameraName.Name       FRONT=1 FRONT_LEFT=2 FRONT_RIGHT=3 SIDE_LEFT=4 SIDE_RIGHT=5
    intrinsic             (f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3)
    extrinsic             camera -> VEHICLE, and CameraImage.pose is vehicle -> world at
                          that image's own timestamp (so it is already rolling-shutter
                          compensated per image; the per-Frame ``pose`` is not)

Two conventions are *not* guessed either, and both are load-bearing:

* **Waymo camera axes are x forward, y left, z up** -- not the pinhole convention. A
  projection therefore needs the fixed axis permutation :data:`AXES_CAM_TO_PINHOLE`
  before the intrinsics are applied. The same matrix appears in the local reference
  implementation at ``OccAny/dataset_setup/waymo/preprocess_waymo.py``.
* **LiDAR points come back in the VEHICLE frame** from
  ``frame_utils.convert_range_image_to_point_cloud``, and ``cp_points`` is ``[N, 6]`` =
  ``(camera_name, x, y, camera_name_2, x_2, y_2)`` -- two candidate projections per point.

Nothing in this module is reachable from a causal input path: the extractor writes a cache,
and the causal side reads only the RGB image out of it.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.waymo_pilot import config as C

#: Waymo camera (x fwd, y left, z up) -> pinhole (x right, y down, z fwd).
AXES_CAM_TO_PINHOLE = np.array([[0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
CAMERA_ID = {"FRONT": 1, "FRONT_LEFT": 2, "FRONT_RIGHT": 3, "SIDE_LEFT": 4, "SIDE_RIGHT": 5}
CAMERA_NAME = {v: k for k, v in CAMERA_ID.items()}


class WaymoUnavailable(RuntimeError):
    """Raised with the exact missing prerequisite. Never swallowed into a default."""


@dataclass(frozen=True)
class Availability:
    root: str
    root_exists: bool
    n_tfrecords: int
    sdk_importable: bool
    sdk_error: Optional[str]
    tensorflow_importable: bool
    segments: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.root_exists and self.n_tfrecords > 0 and self.sdk_importable

    def missing(self) -> List[str]:
        out = []
        if not self.root_exists:
            out.append(f"dataset root {self.root} does not exist")
        elif self.n_tfrecords == 0:
            out.append(f"no *.tfrecord under {self.root} (expected {C.WAYMO_EXPECTED_VERSION})")
        if not self.tensorflow_importable:
            out.append("tensorflow is not importable in this environment")
        if not self.sdk_importable:
            out.append(f"{C.WAYMO_SDK_PACKAGE} is not importable ({self.sdk_error})")
        return out

    def as_dict(self) -> Dict:
        return {"root": self.root, "root_exists": self.root_exists,
                "n_tfrecords": self.n_tfrecords, "sdk_importable": self.sdk_importable,
                "tensorflow_importable": self.tensorflow_importable,
                "sdk_error": self.sdk_error, "expected_version": C.WAYMO_EXPECTED_VERSION,
                "n_segments": len(self.segments), "ok": self.ok, "missing": self.missing()}


def probe(root: str = C.WAYMO_ROOT) -> Availability:
    """What is present and what is not. Opens nothing and downloads nothing."""
    exists = os.path.isdir(root)
    recs = sorted(glob.glob(os.path.join(root, "**", "*.tfrecord"), recursive=True)) if exists else []
    sdk_ok, sdk_err = True, None
    try:
        import waymo_open_dataset  # noqa: F401
    except Exception as exc:                                        # noqa: BLE001
        sdk_ok, sdk_err = False, f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow  # noqa: F401
        tf_ok = True
    except Exception:                                               # noqa: BLE001
        tf_ok = False
    return Availability(root=root, root_exists=exists, n_tfrecords=len(recs),
                        sdk_importable=sdk_ok, sdk_error=sdk_err,
                        tensorflow_importable=tf_ok,
                        segments=tuple(os.path.basename(r) for r in recs))


def require(root: str = C.WAYMO_ROOT) -> Availability:
    a = probe(root)
    if not a.ok:
        raise WaymoUnavailable(
            "the Waymo arm of this pilot needs real data and the official SDK.\n  "
            + "\n  ".join(a.missing())
            + f"\n\nminimal acquisition step:\n{acquisition_instructions(root)}")
    return a


def acquisition_instructions(root: str = C.WAYMO_ROOT) -> str:
    """The smallest sequence of steps that unblocks this arm. The licence step is the user's."""
    n = C.MAX_SEGMENTS
    return (
        f"1. accept the Waymo Open Dataset licence at https://waymo.com/open/licensing/\n"
        f"   (a human must do this; it is not automatable and this pilot does not do it)\n"
        f"2. install the gcloud SDK, then `gcloud auth login`\n"
        f"3. copy {n} segments only (~{n * 1.1:.0f} GiB), not the full split:\n"
        f"   gsutil -m cp $(gsutil ls gs://waymo_open_dataset_v_1_4_2/individual_files/training/ \\\n"
        f"       | head -{n}) {root}/\n"
        f"4. create the reader environment (kept apart from cu128; TF 2.12 pins numpy<2):\n"
        f"   conda create -y -n waymo python=3.10 && conda run -n waymo pip install \\\n"
        f"       {C.WAYMO_SDK_PACKAGE}==1.6.7\n"
        f"5. WAYMO_ROOT={root} <waymo-env>/bin/python \\\n"
        f"       tools/waymo_pilot/cli/waymo_extract.py --segments {C.SMOKE_SEGMENTS}\n"
        f"   (whole segments: an anchor needs "
        f"{C.anchor_frame_budget()['span']} native frames around it, so a short extraction "
        f"supports none)")


# --------------------------------------------------------------------------- #
# calibration and projection -- pure numpy, usable without the SDK
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaymoCamera:
    """One camera's calibration, in the pilot's own terms."""
    name: str
    width: int
    height: int
    K: np.ndarray                 # [3, 3] pinhole intrinsics
    distortion: np.ndarray        # [5] (k1, k2, p1, p2, k3)
    cam_to_vehicle: np.ndarray    # [4, 4] Waymo camera axes -> vehicle
    rolling_shutter: int

    @property
    def pinhole_to_vehicle(self) -> np.ndarray:
        """Pinhole-convention camera -> vehicle, i.e. what a projection pipeline needs."""
        return self.cam_to_vehicle @ np.linalg.inv(AXES_CAM_TO_PINHOLE)

    @property
    def vehicle_to_pinhole(self) -> np.ndarray:
        return np.linalg.inv(self.pinhole_to_vehicle)

    def to_dict(self) -> Dict:
        return {"name": self.name, "width": self.width, "height": self.height,
                "K": self.K.tolist(), "distortion": self.distortion.tolist(),
                "cam_to_vehicle": self.cam_to_vehicle.tolist(),
                "rolling_shutter_direction": self.rolling_shutter}


def camera_from_calibration(name: str, width: int, height: int, intrinsic: Sequence[float],
                            extrinsic: Sequence[float], rolling_shutter: int) -> WaymoCamera:
    """Build a :class:`WaymoCamera` from the nine-element Waymo intrinsic vector."""
    f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = (float(x) for x in intrinsic)
    K = np.array([[f_u, 0.0, c_u], [0.0, f_v, c_v], [0.0, 0.0, 1.0]], np.float64)
    T = np.asarray(extrinsic, np.float64).reshape(4, 4)
    return WaymoCamera(name=name, width=int(width), height=int(height), K=K,
                       distortion=np.array([k1, k2, p1, p2, k3], np.float64),
                       cam_to_vehicle=T, rolling_shutter=int(rolling_shutter))


def project(points_vehicle: np.ndarray, cam: WaymoCamera,
            min_depth_m: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, z)`` of vehicle-frame points in ``cam``, plus nothing hidden.

    No distortion is applied: the pilot uses the projection only to decide which pixel a
    LiDAR return falls in, and Waymo's rectified-pinhole residual is far below the 0.2 m
    voxel at the ranges used. This is recorded rather than assumed away -- ``checks.check_3``
    measures the reprojection residual against the SDK's own ``cp_points`` for real data,
    and the pilot does not proceed if it exceeds a voxel.
    """
    T = cam.vehicle_to_pinhole
    p = np.asarray(points_vehicle, np.float64) @ T[:3, :3].T + T[:3, 3]
    z = p[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.K[0, 0] * p[:, 0] / safe + cam.K[0, 2]
    v = cam.K[1, 1] * p[:, 1] / safe + cam.K[1, 2]
    return u, v, z


# --------------------------------------------------------------------------- #
# the one SDK-dependent step
# --------------------------------------------------------------------------- #
def read_segment(path: str, frame_stride: int = 1, max_frames: Optional[int] = None):
    """Yield ``(frame_index, parsed dict)`` for one ``.tfrecord``. Needs TF and the SDK.

    Returned per frame, and nothing else: the five camera JPEGs, their per-image
    vehicle-to-world poses and timestamps, the vehicle-frame LiDAR points of the first
    return, and each point's camera projection as the SDK computed it. Human annotations
    (``laser_labels``, ``camera_labels``, ``projected_lidar_labels``) are deliberately not
    read -- §3 forbids them in target construction.
    """
    import tensorflow.compat.v1 as tf                               # noqa: PLC0415
    from waymo_open_dataset import dataset_pb2                       # noqa: PLC0415
    from waymo_open_dataset.utils import frame_utils                 # noqa: PLC0415

    tf.enable_eager_execution()
    ds = tf.data.TFRecordDataset(path, compression_type="")
    for i, data in enumerate(ds):
        if i % frame_stride:
            continue
        if max_frames is not None and i // frame_stride >= max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        parsed = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _seg, top_pose = parsed
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, top_pose)
        cams = {}
        for cal in frame.context.camera_calibrations:
            nm = CAMERA_NAME.get(int(cal.name))
            if nm is None:
                continue
            cams[nm] = camera_from_calibration(nm, cal.width, cal.height, cal.intrinsic,
                                               cal.extrinsic.transform,
                                               cal.rolling_shutter_direction)
        images, poses, stamps = {}, {}, {}
        for im in frame.images:
            nm = CAMERA_NAME.get(int(im.name))
            if nm is None:
                continue
            images[nm] = bytes(im.image)
            poses[nm] = np.asarray(im.pose.transform, np.float64).reshape(4, 4)
            stamps[nm] = {"pose_timestamp": float(im.pose_timestamp),
                          "trigger": float(im.camera_trigger_time),
                          "readout_done": float(im.camera_readout_done_time),
                          "shutter": float(im.shutter)}
        yield i, {
            "segment": frame.context.name,
            "frame_timestamp_s": frame.timestamp_micros * 1e-6,
            "vehicle_to_world": np.asarray(frame.pose.transform, np.float64).reshape(4, 4),
            "points_vehicle": np.concatenate(points, axis=0).astype(np.float32),
            "cp_points": np.concatenate(cp_points, axis=0).astype(np.int32),
            "cameras": cams, "images": images, "image_pose": poses,
            "image_timing": stamps,
        }


def sync_error_s(timing: Dict[str, Dict], lidar_timestamp_s: float) -> Dict[str, float]:
    """Measured |image pose time - frame time| per camera. Recorded, never assumed."""
    return {k: abs(float(v["pose_timestamp"]) - float(lidar_timestamp_s))
            for k, v in timing.items()}


__all__ = ["AXES_CAM_TO_PINHOLE", "CAMERA_ID", "CAMERA_NAME", "WaymoUnavailable",
           "Availability", "probe", "require", "acquisition_instructions", "WaymoCamera",
           "camera_from_calibration", "project", "read_segment", "sync_error_s"]
