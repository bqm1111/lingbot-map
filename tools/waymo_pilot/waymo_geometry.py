"""The frozen causal pipeline, pointed at Waymo. Nothing in the pipeline itself is changed.

What is reused byte for byte: the LingBot replay and its preprocessing
(``gates.gate7b.replay``), the MoGe-2 clip-scale rule (``gates.gate7b.scale.frame_candidate``,
median over the clip's own five frames), the incremental hard mapper
(``tools.baseline_reconstruction.current.geometry.build_local_geometry``) and the frozen
CompletionUNet (``tools.semantic_completion.frozen_geometry``).

What is new, and only because Waymo is not KITTI-360:

* **the stream.** Waymo runs at 10 Hz; the causal contract is five frames ending at the
  anchor at KITTI-360's 0.52 s spacing, so the stream is the native frames decimated by
  :data:`config.WAYMO_STREAM_STRIDE` (5 -> 0.50 s). Recorded, not inherited silently.
* **the grid frame.** KITTI-360 anchors the volume to a velodyne 1.76 m above the road;
  Waymo's vehicle frame is on the road. The dims, voxel size and x/y extent are the frozen
  ones and only the z offset is chosen, so that the grid floor sits the same distance below
  the MEASURED road on both datasets (:func:`config.waymo_grid_origin`).
* **the camera.** ``T_cam_to_grid`` is the selected camera's pinhole-convention pose in the
  vehicle frame, which is the exact analogue of KITTI-360's ``rect_cam_to_velo``.

Causal integrity: this module reads the RGB of the selected camera and the calibration, and
nothing else. It never opens the LiDAR, the other cameras, a future frame or a pose beyond
the clip -- the clip's poses are LingBot's own predictions, not Waymo's.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.datasets.occupancy import VoxelGrid
from tools.waymo_pilot import config as C, waymo as W


def waymo_grid(road_z_m: float = None) -> VoxelGrid:
    """The anchor grid: frozen dims/voxel/extent, z aligned to the measured road."""
    rz = C.WAYMO_ROAD_Z_IN_VEHICLE_FRAME_M if road_z_m is None else float(road_z_m)
    return VoxelGrid(dims=C.GRID_DIMS, voxel_size=C.VOXEL_SIZE_M,
                     origin=C.waymo_grid_origin(rz),
                     frame="vehicle_of_anchor_frame", empty_class=0, ignore_label=255,
                     name="waymo")


@dataclass
class WaymoFrame:
    segment: str
    native: int                 # native 10 Hz frame index
    index: int                  # position in the decimated stream
    time_s: float               # seconds relative to the segment's first cached frame
    path: str                   # cached npz


class WaymoSegment:
    """The cached frames of one segment, as a causal stream at the pilot's stride."""

    def __init__(self, segment: str, paths: C.Paths = None,
                 stride: int = C.WAYMO_STREAM_STRIDE):
        self.segment = segment
        self.paths = paths or C.Paths()
        self.stride = int(stride)
        d = os.path.join(self.paths.waymo_frames, segment)
        files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        self.natives = [int(f[:-4]) for f in files]
        self._file = {n: os.path.join(d, f) for n, f in zip(self.natives, files)}
        t0 = self._stamp(self.natives[0])
        kept = self.natives[::self.stride]
        self.frames: List[WaymoFrame] = [
            WaymoFrame(segment=segment, native=n, index=i, time_s=self._stamp(n) - t0,
                       path=self._file[n]) for i, n in enumerate(kept)]
        self._by_native = {f.native: f for f in self.frames}

    def _stamp(self, native: int) -> float:
        with np.load(self._file[native], allow_pickle=False) as z:
            return float(z["frame_timestamp_s"])

    def __len__(self) -> int:
        return len(self.frames)

    def calibration(self, camera: str = C.WAYMO_SELECTED_CAMERA) -> W.WaymoCamera:
        with np.load(self.frames[0].path, allow_pickle=False) as z:
            cal = json.loads(str(z["calibration_json"]))[camera]
        return W.camera_from_calibration(camera, cal["width"], cal["height"],
                                         [cal["K"][0][0], cal["K"][1][1], cal["K"][0][2],
                                          cal["K"][1][2], *cal["distortion"]],
                                         np.asarray(cal["cam_to_vehicle"]).ravel(),
                                         cal["rolling_shutter_direction"])

    def road_z(self, n_frames: int = 10) -> float:
        """Road height in the vehicle frame, measured from this segment's own returns."""
        zs = []
        for f in self.frames[:n_frames]:
            with np.load(f.path, allow_pickle=False) as z:
                p = z["points_vehicle"]
            near = p[(np.abs(p[:, 1]) < 3) & (p[:, 0] > 3) & (p[:, 0] < 15)]
            if len(near) > 100:
                zs.append(float(np.percentile(near[:, 2], 2)))
        return float(np.median(zs)) if zs else C.WAYMO_ROAD_Z_IN_VEHICLE_FRAME_M

    def image_path(self, frame: WaymoFrame, camera: str = C.WAYMO_SELECTED_CAMERA) -> str:
        """Write the camera's JPEG out once and return its path (LingBot/Trident read files)."""
        out = os.path.join(self.paths.data_root, "images", self.segment, camera,
                           f"{frame.native:05d}.jpg")
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with np.load(frame.path, allow_pickle=False) as z:
                raw = z[f"image_{camera}"].tobytes()
            tmp = out + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, out)
        return out

    def anchors(self) -> List[int]:
        """Stream indices with a full causal clip and the full privileged window behind them."""
        b = C.anchor_frame_budget()
        back = (C.CLIP_LENGTH - 1)
        fwd = C.EXISTING_TARGET_CFG.horizon_frames
        return [f.index for f in self.frames
                if f.index >= back and f.index + fwd < len(self.frames)]


# --------------------------------------------------------------------------- #
# the causal clip
# --------------------------------------------------------------------------- #
def clip_fov_x_deg(cam: W.WaymoCamera) -> float:
    """Calibrated horizontal FOV for MoGe, from Waymo's own intrinsics."""
    return float(2 * np.degrees(np.arctan(cam.K[0, 2] / cam.K[0, 0])))


@torch.no_grad()
def build_clip(seg: WaymoSegment, t: int, model, moge, cam: W.WaymoCamera, device) -> Dict:
    """One local-reset five-frame clip ``t-4..t`` plus its causal MoGe-2 metric scale.

    Identical in every step to ``tools/local_registration_audit/run_local_lingbot.py``: the
    KV cache is cleared before the clip, the RoPE counter is asserted at zero, the five
    frames run as one batched block, and the scale is the median of the five per-frame MoGe
    candidates. Only the image source and the FOV differ.
    """
    from gates.gate7b import replay, scale as SC
    idx = [t + o for o in C.CLIP_OFFSETS]
    frames = [seg.frames[j] for j in idx]
    paths = [seg.image_path(f) for f in frames]
    imgs = replay.load_images(paths)
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "the RoPE counter did not reset"
    out = replay.replay_segment(model, imgs, 1, device)
    dep = out["pred_depth"].astype(np.float32)
    cnf = out["pred_depth_conf"].astype(np.float32)
    fov = clip_fov_x_deg(cam)
    logs = []
    for q, pth in enumerate(paths):
        md, mm = moge(pth, fov)
        logs.append(SC.frame_candidate(md, mm, dep[q], cnf[q])["log_s"]
                    if md is not None else float("nan"))
    logs = np.asarray(logs, np.float64)
    ls = float(np.nanmedian(logs)) if np.isfinite(logs).any() else float("nan")
    return {"t": np.int64(t), "stream_indices": np.asarray(idx, np.int32),
            "keys": np.asarray([f"{seg.segment}_{f.native:05d}" for f in frames]),
            "natives": np.asarray([f.native for f in frames], np.int32),
            "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"],
            "pred_pose_c2w": out["pred_pose_c2w"], "pred_K": out["pred_K"],
            "proc_hw": out["proc_hw"], "log_s_frames": logs,
            "scale": np.float64(np.exp(ls)), "fov_x_deg": np.float64(fov),
            "rope_slots_used": np.int64(int(out["rope_slots_used"]))}


def camera_to_grid(cam: W.WaymoCamera) -> np.ndarray:
    """Anchor camera (pinhole convention) -> the grid frame. KITTI-360's ``rect_cam_to_velo``."""
    return np.asarray(cam.pinhole_to_vehicle, np.float64)


__all__ = ["waymo_grid", "WaymoFrame", "WaymoSegment", "clip_fov_x_deg", "build_clip",
           "camera_to_grid"]
