"""ONCE geometry for the training arm: anchor eligibility, causal clips, camera->grid.

Three jobs, and nothing else:

1. :func:`eligible_anchors` -- which frames can be an anchor. An anchor needs four real past
   frames for the causal input clip AND a valid forward supervision window, with no temporal
   gap in either; 1.7 % of ONCE intervals are gaps of up to 13 s.
2. :func:`build_clip` -- a transcription of ``waymo_pilot.waymo_geometry.build_clip`` with the
   image source and FOV swapped. Every substantive call is the frozen one: the KV cache is
   cleared, the RoPE counter is asserted at zero, the five frames run as one batched block
   through ``gates.gate7b.replay.replay_segment``, and the clip scale is the median of the
   five per-frame MoGe-2 candidates from ``gates.gate7b.scale.frame_candidate``. **No LiDAR,
   no second camera, no future frame and no ONCE pose enter a clip** -- the poses are
   LingBot's own predictions.
3. :func:`camera_to_grid` -- ONCE's published ``cam_to_velo``, conjugated into our x-forward
   convention by the reader. ONCE's camera frame is OpenCV pinhole (x-right, y-down,
   z-forward), confirmed by the distortion gate, which projected LiDAR through it and matched
   image gradients in 6 of 6 sequences. So this is the pinhole->grid transform directly, with
   no extra convention flip.

The grid frame is the anchor sensor's own frame, which is what ``rawtarget.build`` carves in,
and ONCE's road plane sits 1.73-1.76 m below the roof LiDAR -- KITTI-360's velodyne height --
so the frozen origin z of -2.0 m applies unchanged.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from tools.method_vs_data.once_arm import config as A
from tools.method_vs_data.once_reader import OnceSequence


#: 2.1 s at ONCE's native 0.500 s cadence: anchor + 4 forward sweeps
EXPECTED_SWEEPS = 5


def supervision_window(seq: OnceSequence, anchor: int) -> List[int]:
    """The authorized forward window: real native sweeps within ``SUPERVISION_SPAN_S``."""
    return seq.window(anchor, span_s=A.SUPERVISION_SPAN_S, max_gap_s=A.MAX_SWEEP_GAP_S)


def input_clip_indices(anchor: int) -> List[int]:
    """``t-4..t`` at ONCE's native cadence; stride 1 because 0.500 s already is a stream step."""
    return [anchor + o * A.STREAM_STRIDE for o in A.CLIP_OFFSETS]


def eligible_anchors(seq: OnceSequence, min_sep: int = None) -> List[int]:
    """Anchors with a complete causal input clip AND a valid supervision window."""
    min_sep = A.MIN_ANCHOR_SEPARATION_FRAMES if min_sep is None else int(min_sep)
    import os
    out, last = [], None
    n = len(seq.frame_ids)
    for a in range(n):
        idx = input_clip_indices(a)
        if idx[0] < 0 or idx[-1] >= n:
            continue
        # the causal clip must be contiguous in time and have every image on disk
        gaps = np.diff(seq.t_s[idx])
        if not np.all(gaps <= A.MAX_SWEEP_GAP_S):
            continue
        if not all(os.path.exists(seq.image_path(j, A.SELECTED_CAMERA)) for j in idx):
            continue
        # require the FULL authorized window, so the supervision policy is uniform across
        # anchors; 2 of 1,582 candidates were truncated by a sequence end and are dropped
        if len(supervision_window(seq, a)) != EXPECTED_SWEEPS:
            continue
        if last is not None and a - last < min_sep:
            continue
        out.append(a)
        last = a
    return out


def clip_fov_x_deg(seq: OnceSequence, cam: str = None) -> float:
    """Horizontal FOV from the published intrinsics and the real image width."""
    cam = cam or A.SELECTED_CAMERA
    K = seq.K(cam)
    import cv2
    img = cv2.imread(seq.image_path(0, cam), cv2.IMREAD_GRAYSCALE)
    W = int(img.shape[1]) if img is not None else 1920
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * float(K[0, 0])))))


def _undistorted_path(seq: OnceSequence, nf: int, cam: str, cache_dir: str) -> str:
    """ONCE images are RAW (docs are wrong; resolved in once_metadata_audit.md §5).

    `cv2.undistort` is applied exactly ONCE here and the result cached, so the clip builder
    and any later projection see the same rectified pixels and nothing is undistorted twice.
    """
    import os

    import cv2
    out = os.path.join(cache_dir, seq.seq, f"{seq.frame_ids[int(nf)]}.jpg")
    if os.path.exists(out):
        return out
    img = cv2.imread(seq.image_path(nf, cam))
    if img is None:
        raise FileNotFoundError(seq.image_path(nf, cam))
    K, d = seq.K(cam), seq.distortion(cam)
    assert d.size == 5, f"expected 5 distortion coefficients, got {d.size}"
    und = cv2.undistort(img, K, d.reshape(1, -1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.jpg"
    cv2.imwrite(tmp, und, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
    os.replace(tmp, out)
    return out


def build_clip(seq: OnceSequence, t: int, model, moge, device, fov: float,
               undistort_cache: str, cam: str = None, undistort: bool = False) -> Dict:
    """One local-reset five-frame clip ``t-4..t`` plus its causal MoGe-2 metric scale.

    **The arm uses ``undistort=False``: the shipped JPEGs go straight through.** ONCE's
    documentation is correct that the released images are already rectified, and the earlier
    conclusion in ``once_metadata_audit.md`` §5 that they are raw is **retracted**. Two
    measurements, both on real data, settle it:

    * projecting the anchor sweep and correlating LiDAR depth against the monocular depth map,
      plain pinhole scores **0.857** against **0.698** for a distortion-applied projection,
      in **12 of 12** frames;
    * building clips both ways and scoring the hard map against the LiDAR targets, the
      shipped images give **IoU 3.26 %** against **1.44 %** for ``cv2.undistort``-ed ones, in
      **11 of 12** anchors.

    The gradient test that produced the wrong answer (``once_distortion_gate.py`` M2) is
    biased: a barrel-distorted projection pulls points toward the image centre, where street
    scenes carry more gradient, so it rewards the distorted model regardless of alignment.
    """
    from gates.gate7b import replay, scale as SC
    cam = cam or A.SELECTED_CAMERA
    idx = input_clip_indices(t)
    paths = [(_undistorted_path(seq, j, cam, undistort_cache) if undistort
              else seq.image_path(j, cam)) for j in idx]
    imgs = replay.load_images(paths)
    model.clean_kv_cache()
    assert replay.frame_index_trace(model) == 0, "the RoPE counter did not reset"
    out = replay.replay_segment(model, imgs, 1, device)
    dep = out["pred_depth"].astype(np.float32)
    cnf = out["pred_depth_conf"].astype(np.float32)
    logs = []
    for q, pth in enumerate(paths):
        md, mm = moge(pth, fov)
        logs.append(SC.frame_candidate(md, mm, dep[q], cnf[q])["log_s"]
                    if md is not None else float("nan"))
    logs = np.asarray(logs, np.float64)
    ls = float(np.nanmedian(logs)) if np.isfinite(logs).any() else float("nan")
    return {"t": np.int64(t), "stream_indices": np.asarray(idx, np.int32),
            "keys": np.asarray([f"{seq.seq}_{seq.frame_ids[j]}" for j in idx]),
            "natives": np.asarray(idx, np.int32),
            "frame_ids": np.asarray([seq.frame_ids[j] for j in idx]),
            "timestamps_s": np.asarray([seq.t_s[j] for j in idx], np.float64),
            "pred_depth": out["pred_depth"], "pred_depth_conf": out["pred_depth_conf"],
            "pred_pose_c2w": out["pred_pose_c2w"], "pred_K": out["pred_K"],
            "proc_hw": out["proc_hw"], "log_s_frames": logs,
            "scale": np.float64(np.exp(ls)), "fov_x_deg": np.float64(fov),
            "undistorted": np.bool_(undistort),
            "rope_slots_used": np.int64(int(out["rope_slots_used"]))}


def camera_to_grid(seq: OnceSequence, cam: str = None) -> np.ndarray:
    """Anchor camera (OpenCV pinhole) -> grid frame (the anchor sensor frame), metres."""
    return np.asarray(seq.cam_to_velo(cam or A.SELECTED_CAMERA), np.float64)


__all__ = ["supervision_window", "input_clip_indices", "eligible_anchors", "clip_fov_x_deg",
           "build_clip", "camera_to_grid"]
