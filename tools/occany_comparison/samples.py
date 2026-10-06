"""OccAny's sample lists, mapped onto our frame streams.

A :class:`Sample` is one OccAny output directory: the target frame, the whole ordered frame list of
its sequence / scene (so both the forward and the causal window can be addressed), our camera-to-grid
transform at the target, and the reference the independent label loader needs.

* SemanticKITTI 08: OccAny directories ``08_000765`` are raw frame ids at step 5 -- exactly the keys
  of our frozen stride-5 stream (815 frames), so position = id / 5.
* Occ3D: directories ``scene-XXXX_<sample token>``. OccAny uses every sample with eight more after
  it (4,819 of 6,019). Our frozen stream holds the first 40 samples of each scene; the full ordered
  scene (39-41 samples) is rebuilt from ``annotations.json`` with the same frozen loader, and the
  prefix is asserted identical to the stream (keys and image paths).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.datasets import nuscenes as NU, sources as S
from tools.occany_comparison import config as C


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str                  # OccAny output directory name
    segment: str
    target_pos: int                 # position of the target in ``frame_keys``
    frame_keys: Tuple[str, ...]     # the whole ordered sequence / scene
    frame_paths: Tuple[str, ...]
    T_cam_to_grid_ours: np.ndarray = field(repr=False)   # target camera -> grid frame, metres
    occany_pkl: str = ""
    gt_ref: Dict = field(default_factory=dict)
    official_anchor: bool = False   # one of our reported 163 / 1,182 anchors
    n_stream_frames: int = 0        # frames of this segment in our frozen stream cache
    K_native: Optional[np.ndarray] = field(default=None, repr=False)  # [N, 3, 3], Occ3D only

    def window(self, protocol: C.Protocol) -> List[int]:
        return [self.target_pos + o for o in protocol.offsets]

    def protocol_ok(self, protocol: C.Protocol) -> bool:
        w = self.window(protocol)
        return min(w) >= 0 and max(w) < len(self.frame_keys)


def _occany_dirs(dataset: str) -> List[str]:
    run = C.OCCANY[dataset]
    names = sorted(n for n in os.listdir(run.pred_dir) if os.path.isfile(os.path.join(run.pred_dir, n, "voxel_predictions.pkl")))
    assert len(names) == run.expected_samples, f"{dataset}: {len(names)} OccAny outputs, expected {run.expected_samples}"
    return names


def semantickitti_samples(repo_root: str = C.REPO_ROOT) -> List[Sample]:
    run = C.OCCANY["semantickitti"]
    (seg,) = S.segments("semantickitti", repo_root)
    keys, paths = tuple(f.key for f in seg.frames), tuple(f.path for f in seg.frames)
    T = next(np.asarray(f.T_cam_to_grid, np.float64) for f in seg.frames if f.T_cam_to_grid is not None)
    pos_of = {k: i for i, k in enumerate(keys)}
    anchors = set(seg.anchors)
    out = []
    for name in _occany_dirs("semantickitti"):
        j = pos_of[name]
        fid = int(name.split("_")[1])
        assert fid == 5 * j, (name, j)
        out.append(Sample("semantickitti", name, seg.name, j, keys, paths, T,
                          os.path.join(run.pred_dir, name, "voxel_predictions.pkl"),
                          {"sequence": seg.name, "frame_ids": [fid]}, j in anchors, len(keys)))
    return out


def occ3d_samples(repo_root: str = C.REPO_ROOT) -> List[Sample]:
    run = C.OCCANY["occ3d"]
    ann = NU.load_annotations(S.OCC3D_ROOT)
    segs = {s.name: s for s in S.segments("occ3d", repo_root)}
    by_scene: Dict[str, List[Tuple[str, str]]] = {}
    for name in _occany_dirs("occ3d"):
        scene, token = name.split("_", 1)
        by_scene.setdefault(scene, []).append((token, name))
    out = []
    for scene in sorted(by_scene):
        frs = NU.scene_frames(ann, scene, "CAM_FRONT", C.NUSCENES_ROOT)
        keys = tuple(os.path.splitext(os.path.basename(f.image_path))[0] for f in frs)
        paths = tuple(f.image_path for f in frs)
        seg = segs[scene]
        n = len(seg.frames)
        assert list(keys[:n]) == [f.key for f in seg.frames] and list(paths[:n]) == [f.path for f in seg.frames], scene
        pos_of = {f.token: i for i, f in enumerate(frs)}
        K = np.stack([f.K for f in frs])
        anchors = set(seg.anchors)
        for token, name in by_scene[scene]:
            j = pos_of[token]
            T = np.asarray(frs[j].T_camera_to_ego, np.float64)
            if j in anchors:      # the frozen evaluator read this transform from the per-clip cache
                assert np.allclose(T, seg.frames[j].T_cam_to_grid, atol=1e-5), (scene, j)
            out.append(Sample("occ3d", name, scene, j, keys, paths, T,
                              os.path.join(run.pred_dir, name, "voxel_predictions.pkl"),
                              {"anchor_gt_path": frs[j].gt_path}, j in anchors, n, K))
    return out


def build_samples(dataset: str, repo_root: str = C.REPO_ROOT) -> List[Sample]:
    samples = semantickitti_samples(repo_root) if dataset == "semantickitti" else occ3d_samples(repo_root)
    return sorted(samples, key=lambda s: s.sample_id)


__all__ = ["Sample", "build_samples", "semantickitti_samples", "occ3d_samples"]
