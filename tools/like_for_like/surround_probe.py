"""Can our streaming model register six simultaneous surround views the way it registers five consecutive frames?

OccAny reconstructs the six nuScenes cameras jointly and pose-free: it stacks them as views and binds them through the
decoder's shared memory (``occany/must3r_inference.py``). Our model binds frames through its KV cache over TIME. This probe
asks whether the same mechanism survives when the axis is viewpoint instead of time.

Method: the frozen local-reset LingBot runs on one timestep's six images, ordered around the ring so neighbours share an
edge. Its predicted camera-to-camera poses are compared with the rig geometry from ``calibrated_sensor.json``. The rig is
used ONLY to grade the prediction -- never as input -- so this stays pose-free. Rotation error is scale-free and decisive;
translation is reported after a single global scale fit, since our poses are canonical.

    PYTHONPATH=$PWD python -m tools.like_for_like.surround_probe --device cuda:0 --samples 3
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from tools.like_for_like import config as LC

NUSCENES_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes"
#: ring order: each camera overlaps the next at its edge (the best case for a sequence model)
RING = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]


def quaternion_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def rig_poses(nusc, sample_token: str) -> Dict[str, Dict]:
    """Per camera: image path and the true camera-to-ego transform (grading only)."""
    sample = nusc.get("sample", sample_token)
    out = {}
    for cam in RING:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        T = np.eye(4); T[:3, :3] = quaternion_to_matrix(cs["rotation"]); T[:3, 3] = np.asarray(cs["translation"], float)
        out[cam] = {"path": os.path.join(NUSCENES_ROOT, sd["filename"]), "T_cam_to_ego": T}
    return out


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    c = (np.trace(R_pred.T @ R_gt) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def probe_sample(model, replay, sample_token: str, nusc, device) -> Dict:
    rig = rig_poses(nusc, sample_token)
    imgs = replay.load_images([rig[c]["path"] for c in RING])
    model.clean_kv_cache(); assert replay.frame_index_trace(model) == 0
    out = replay.replay_segment(model, imgs, 1, device)
    pred = np.asarray(out["pred_pose_c2w"], np.float64)                     # camera -> clip origin (view 0)
    gt0 = rig[RING[0]]["T_cam_to_ego"]
    rows, pred_t, gt_t = [], [], []
    for i, cam in enumerate(RING):
        P = np.linalg.inv(pred[0]) @ pred[i]
        G = np.linalg.inv(gt0) @ rig[cam]["T_cam_to_ego"]
        rows.append({"camera": cam, "rotation_error_deg": rotation_error_deg(P[:3, :3], G[:3, :3]),
                     "gt_rotation_from_front_deg": rotation_error_deg(np.eye(3), G[:3, :3])})
        pred_t.append(P[:3, 3]); gt_t.append(G[:3, 3])
    pred_t, gt_t = np.stack(pred_t), np.stack(gt_t)
    scale = float((np.linalg.norm(gt_t, axis=1).sum() + 1e-9) / (np.linalg.norm(pred_t, axis=1).sum() + 1e-9))
    trans_err = [float(np.linalg.norm(scale * p - g)) for p, g in zip(pred_t, gt_t)]
    for r, e in zip(rows, trans_err):
        r["translation_error_m_after_global_scale"] = e
    return {"sample_token": sample_token, "per_camera": rows, "fitted_scale": scale,
            "mean_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in rows[1:]])),
            "mean_translation_error_m": float(np.mean(trans_err[1:]))}


def run(device: str, n_samples: int) -> Dict:
    from nuscenes.nuscenes import NuScenes
    from gates.gate7b import replay
    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_ROOT, verbose=False)
    dev = torch.device(device); torch.cuda.set_device(dev)
    model, info = replay.build_model(dev)
    tokens = [s["token"] for s in nusc.sample[:: max(len(nusc.sample) // n_samples, 1)]][:n_samples]
    rec = {"ring_order": RING, "model_load": info, "samples": [probe_sample(model, replay, t, nusc, dev) for t in tokens]}
    rec["mean_rotation_error_deg"] = float(np.mean([s["mean_rotation_error_deg"] for s in rec["samples"]]))
    rec["mean_translation_error_m"] = float(np.mean([s["mean_translation_error_m"] for s in rec["samples"]]))
    rec["reference"] = "neighbouring surround cameras sit ~55-70 deg apart; a useless prediction scores ~60 deg on average"
    LC.ensure_dirs()
    json.dump(rec, open(os.path.join(LC.ARTIFACT_DIR, "surround_probe.json"), "w"), indent=1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    rec = run(a.device, a.samples)
    for s in rec["samples"]:
        print(s["sample_token"][:16], " ".join(f"{r['camera'][4:]}:{r['rotation_error_deg']:.0f}deg(gt {r['gt_rotation_from_front_deg']:.0f})" for r in s["per_camera"][1:]))
    print(f"mean rotation error {rec['mean_rotation_error_deg']:.1f} deg, mean translation error {rec['mean_translation_error_m']:.2f} m")


if __name__ == "__main__":
    main()
