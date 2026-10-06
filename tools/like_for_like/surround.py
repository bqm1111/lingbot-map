"""Six-camera (surround) setting for our method on Occ3D-nuScenes, matching OccAny's surround coverage.

OccAny's surround preset reconstructs the six cameras of ONE timestep jointly and pose-free, binding them through its
decoder memory (``occany/must3r_inference.py``). Our streaming model binds frames through its KV cache; the probe in
``surround_probe.py`` showed it also registers the six views from image content alone (mean rotation error 5.8 deg over
12 timesteps, against 55-180 deg for a degenerate prediction), so the same setting is available to us without feeding
any rig calibration.

What is used from the dataset: the camera images, and ``cam0_to_ego`` to place the finished map in the benchmark grid --
the same placement transform OccAny uses (``extract_output_occany.py``: ``T_cam_to_voxel = data['cam0_to_ego']``).

Samples: the 4,819 Occ3D samples of our existing evaluation, so front-camera and surround rows describe the same scenes.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List

import numpy as np

from tools.like_for_like import config as LC
from tools.occany_comparison import samples as OS
from tools.zero_shot_eval import config as ZC

NUSCENES_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes"
#: ring order: neighbouring views share an edge, the best case for a sequence model (CAM_FRONT first = anchor)
RING = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]
CLIP_ROOT = os.path.join(LC.DATA_ROOT, "surround_clips")
PREDICTION_ROOT = os.path.join(LC.DATA_ROOT, "surround_predictions")
FRAME_LIST = os.path.join(LC.DATA_ROOT, "frames_occ3d_surround.json")


@lru_cache(maxsize=1)
def nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-trainval", dataroot=NUSCENES_ROOT, verbose=False)


def quaternion_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def sample_token(sample) -> str:
    """Our Occ3D sample ids are ``<scene>_<nuScenes sample token>``."""
    return sample.sample_id.split("_", 1)[1]


def ring_views(token: str) -> List[Dict]:
    """Per camera of one timestep: key (the file stem, as the token cache uses), image path, camera-to-ego transform."""
    nusc = nuscenes(); sample = nusc.get("sample", token)
    out = []
    for cam in RING:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        T = np.eye(4); T[:3, :3] = quaternion_to_matrix(cs["rotation"]); T[:3, 3] = np.asarray(cs["translation"], float)
        out.append({"camera": cam, "key": os.path.splitext(os.path.basename(sd["filename"]))[0],
                    "path": os.path.join(NUSCENES_ROOT, sd["filename"]), "T_cam_to_ego": T})
    return out


def samples() -> List:
    return [s for s in OS.build_samples("occ3d")]


def write_frame_list() -> Dict:
    """Frames whose Trident tokens are still missing: the five non-front cameras (CAM_FRONT is already cached)."""
    seen, missing = set(), []
    ss = samples()
    for s in ss:
        for v in ring_views(sample_token(s)):
            if v["key"] in seen:
                continue
            seen.add(v["key"])
            if not os.path.exists(ZC.token_path("occ3d", v["key"])):
                missing.append({"key": v["key"], "path": v["path"]})
    rec = {"dataset": "occ3d", "setting": "surround", "n_samples": len(ss), "n_frames": len(seen),
           "n_missing_tokens": len(missing), "frames": missing}
    os.makedirs(LC.DATA_ROOT, exist_ok=True)
    json.dump(rec, open(FRAME_LIST, "w"))
    return {k: v for k, v in rec.items() if k != "frames"}


def surround_label(sample) -> np.ndarray:
    """The 360-degree Occ3D label, i.e. OccAny's ``use_surround_label=True`` ground truth.

    OccAny's nuScenes loader (``occany/datasets/nuscenes.py``) applies ``mask_camera`` and then, only when the setting is
    NOT surround, ignores the rear half (``voxel_label[:100] = 255``). The saved single-camera outputs on this machine are
    the ``5frames`` setting, so their stored ``voxel_label`` carries that rear-half cut; the surround setting keeps the
    full volume. This rebuilds the surround version from the official files with the same rule as
    ``occany_comparison.scoring.independent_label``, minus the cut.
    """
    from core.evaluation import official_targets as OT
    from tools.occany_comparison import config as OC
    with np.load(os.path.join(OT.OCC3D_ROOT, sample.gt_ref["anchor_gt_path"])) as z:
        label = z["semantics"].astype(np.int32)
        valid = z["mask_camera"].astype(bool) & (label != OC.IGNORE)
    label[~valid] = OC.IGNORE
    return label.astype(np.uint8)


__all__ = ["RING", "CLIP_ROOT", "PREDICTION_ROOT", "FRAME_LIST", "nuscenes", "sample_token", "ring_views", "samples", "surround_label", "write_frame_list"]
