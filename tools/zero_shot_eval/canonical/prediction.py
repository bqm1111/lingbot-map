"""Canonical prediction: one file per evaluated sample, written once and read by every evaluator (Track P, Track O).

All arrays live on the official **evaluation grid** of the dataset (SemanticKITTI 256x256x32 @ 0.2 m, velodyne frame;
Occ3D-nuScenes 200x200x16 @ 0.4 m, ego frame), flat C-order indices.

    occupied          [N] bool   predicted occupancy (Occ3D: any of the eight 0.2 m subvoxels occupied)
    observed          [N] bool   geometrically observed at prediction time (any subvoxel in hard-map S or F)
    evidence          [N] bool   semantic evidence at prediction time (any subvoxel is an evidence voxel)
    query_index       [M] int32  voxels carrying a semantic label: predicted occupied or GT occupied, within the valid label mask
    evidence_distance [M] float16  metres to the nearest evidence voxel (min over subvoxels)
    labels[method]    [M] uint8  official label id per query voxel, for each method (B0, B3_s0, B3_s1, B3_s2, ...)
    meta              sample id, dataset, grid dims / origin / voxel size / frame, T_cam_to_grid [4, 4], K [3, 3], image width,
                      vocabulary label ids, method checkpoints, evidence voxel count

Boolean volumes are stored bit-packed; the file is compressed npz.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class CanonicalPrediction:
    sample_id: str
    dataset: str
    dims: tuple
    voxel_size: float
    origin: tuple
    frame: str
    T_cam_to_grid: np.ndarray
    K: np.ndarray
    image_width: int
    occupied: np.ndarray
    observed: np.ndarray
    evidence: np.ndarray
    query_index: np.ndarray
    evidence_distance: np.ndarray
    labels: Dict[str, np.ndarray]
    meta: Dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.dims))

    def dense_labels(self, method: str, fill: int) -> np.ndarray:
        out = np.full(self.n_voxels, fill, np.int64); out[self.query_index] = self.labels[method]; return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, sample_id=self.sample_id, dataset=self.dataset, dims=np.asarray(self.dims, np.int64), voxel_size=np.float64(self.voxel_size),
                            origin=np.asarray(self.origin, np.float64), frame=self.frame, T_cam_to_grid=self.T_cam_to_grid.astype(np.float64),
                            K=self.K.astype(np.float64), image_width=np.int64(self.image_width), occupied=np.packbits(self.occupied), observed=np.packbits(self.observed),
                            evidence=np.packbits(self.evidence), query_index=self.query_index.astype(np.int32), evidence_distance=self.evidence_distance.astype(np.float16),
                            methods=np.asarray(sorted(self.labels)), meta=json.dumps(self.meta), **{f"label__{m}": v.astype(np.uint8) for m, v in self.labels.items()})
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "CanonicalPrediction":
        with np.load(path) as z:
            dims = tuple(int(d) for d in z["dims"]); n = int(np.prod(dims))
            unpack = lambda k: np.unpackbits(z[k])[:n].astype(bool)
            return CanonicalPrediction(str(z["sample_id"]), str(z["dataset"]), dims, float(z["voxel_size"]), tuple(z["origin"].tolist()), str(z["frame"]),
                                       z["T_cam_to_grid"], z["K"], int(z["image_width"]), unpack("occupied"), unpack("observed"), unpack("evidence"),
                                       z["query_index"].astype(np.int64), z["evidence_distance"].astype(np.float32),
                                       {str(m): z[f"label__{m}"] for m in z["methods"]}, json.loads(str(z["meta"])))


__all__ = ["CanonicalPrediction"]
