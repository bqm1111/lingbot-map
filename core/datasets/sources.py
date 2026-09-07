# extracted from gate8/sources.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Every stream Gate 8 reads, train and validation alike, behind one interface.

Validation sources are the three frozen benchmarks and reuse the Gate-7B stream caches,
the Gate-6 Trident caches and the Gate-7B MoGe-B caches untouched. Training sources are
**genuine train splits** -- SemanticKITTI odometry sequences 00/05/07 and the first 100
official nuScenes *train* scenes -- so seq 08, the Occ3D val scenes and KITTI-360 stay
clean. Nothing here opens a target; the GT reference is a plain dict that only the
evaluator and the target builder ever resolve.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.datasets import frames as G6F
from core.mapping import depth as D7
from core.datasets import streams as ST

G8_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate8"
OCC3D_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval"
SK_TRAIN_SEQUENCES = ("00", "05", "07")
SK_STRIDE = 5                       # the frozen SemanticKITTI stream stride (Gate 6)
OCC3D_TRAIN_SCENES = 100
VAL_SOURCES = ("semantickitti", "occ3d", "kitti360")
TRAIN_SOURCES = ("sk_train", "occ3d_train")
#: which benchmark vocabulary / grid / target loader a source uses
DATASET_OF = {"semantickitti": "semantickitti", "occ3d": "occ3d", "kitti360": "kitti360",
              "sk_train": "semantickitti", "occ3d_train": "occ3d"}

#: extra sources registered by later gates: name -> {"dataset", "root", "segments"}.
#: Additive only -- nothing above changes behaviour for the sources it already knows.
EXTRA_SOURCES: Dict[str, dict] = {}


def register_source(name: str, dataset: str, root: str, builder) -> None:
    """Make ``name`` visible to every Gate-8 cache path helper and to ``segments``."""
    EXTRA_SOURCES[name] = {"dataset": dataset, "root": root, "segments": builder}
    DATASET_OF[name] = dataset


@dataclass
class SFrame:
    key: str
    path: str
    order: int
    index: int
    gt_ref: Optional[dict] = None          # a rec dict for gate6.targets.semantic_target
    T_cam_to_grid: Optional[np.ndarray] = None
    K_native: Optional[np.ndarray] = None
    native_hw: Optional[Tuple[int, int]] = None


@dataclass
class Seg:
    source: str
    name: str
    frames: List[SFrame] = field(default_factory=list)

    @property
    def dataset(self) -> str:
        return DATASET_OF[self.source]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def anchors(self) -> List[int]:
        return [f.index for f in self.frames if f.gt_ref is not None]


# --------------------------------------------------------------------------- #
# cache paths
# --------------------------------------------------------------------------- #
def stream_path(source: str, seg: str) -> str:
    if source in EXTRA_SOURCES:
        return f"{EXTRA_SOURCES[source]['root']}/stream/{source}/{seg}.npz"
    if source in VAL_SOURCES:
        return f"/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream/{source}/{seg}.npz"
    return f"{G8_ROOT}/stream/{source}/{seg}.npz"


def scale_path(source: str, seg: str) -> str:
    if source in EXTRA_SOURCES:
        return f"{EXTRA_SOURCES[source]['root']}/scale/{source}/{seg}.npz"
    if source in VAL_SOURCES:
        return f"/media/SSD1/MINH_DATASETS/lingbot_gate7b/scale/{source}/{seg}.npz"
    return f"{G8_ROOT}/scale/{source}/{seg}.npz"


def trident_path(source: str, key: str) -> str:
    if source in EXTRA_SOURCES:
        return f"{EXTRA_SOURCES[source]['root']}/semantics/{source}/{key}.npz"
    if source in VAL_SOURCES:
        return G6F.semantic_cache_path(source, key)
    return f"{G8_ROOT}/semantics/{source}/{key}.npz"


def moge_path(source: str, key: str) -> str:
    if source in EXTRA_SOURCES:
        return f"{EXTRA_SOURCES[source]['root']}/moge_b/{source}/{key}.npz"
    if source in VAL_SOURCES:
        return D7.moge_frame_path(source, key)
    return f"{G8_ROOT}/moge_b/{source}/{key}.npz"


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def _sk_calib(repo_root: str, seq: str):
    from core.datasets.occ_datasets import SemanticKittiOccSpec
    from core.datasets.kitti_odometry import parse_calibration
    root = os.path.join(repo_root, G6F.SEMANTICKITTI_ROOT)
    spec = SemanticKittiOccSpec.build(root, seq)
    calib = parse_calibration(os.path.join(root, "sequences", seq, "calib.txt"))
    return spec.cam_to_velo, np.asarray(calib.K, np.float64)


def _val_segments(source: str, repo_root: str) -> List[Seg]:
    recs = {r.clip_id: r.raw for r in G6F.read_manifest(source, repo_root)}
    out = []
    for s in ST.build(source, repo_root):
        seg = Seg(source=source, name=s.name)
        anchor_of = {t: cid for cid, t in s.anchors.items()}
        Tsk = None
        if source == "semantickitti":
            Tsk, _K = _sk_calib(repo_root, s.name)
        for f in s.frames:
            cid = anchor_of.get(f.index)
            gt = recs[cid] if cid else None
            T = None
            if cid:
                if source == "semantickitti":
                    T = Tsk
                else:
                    with np.load(G6F.lingbot_cache_path(source, cid, repo_root)) as z:
                        T = (z["rect_cam_to_velo"] if source == "kitti360"
                             else z["T_camera_to_ego"][-1]).astype(np.float64)
            seg.frames.append(SFrame(key=f.key, path=f.path, order=f.order,
                                     index=f.index, gt_ref=gt, T_cam_to_grid=T))
        out.append(seg)
    return out


def _sk_train_segments(repo_root: str) -> List[Seg]:
    root = os.path.join(repo_root, G6F.SEMANTICKITTI_ROOT)
    out = []
    for seq in SK_TRAIN_SEQUENCES:
        T, K = _sk_calib(repo_root, seq)
        imgs = sorted(os.listdir(os.path.join(root, "sequences", seq, "image_2")))
        seg = Seg(source="sk_train", name=seq)
        for i, name in enumerate(imgs[::SK_STRIDE]):
            fid = int(os.path.splitext(name)[0])
            vox = os.path.join(root, "sequences", seq, "voxels", f"{fid:06d}")
            has_gt = os.path.isfile(vox + ".label") and os.path.isfile(vox + ".invalid")
            gt = {"sequence": seq, "frame_ids": [fid], "clip_id": f"{seq}_{fid:06d}"} \
                if has_gt else None
            seg.frames.append(SFrame(
                key=f"{seq}_{fid:06d}",
                path=os.path.join(root, "sequences", seq, "image_2", name),
                order=fid, index=i, gt_ref=gt, T_cam_to_grid=T, K_native=K,
                native_hw=None))
        out.append(seg)
    return out


def _occ3d_train_segments(repo_root: str) -> List[Seg]:
    from core.datasets.nuscenes import load_annotations, scene_frames
    ann = load_annotations(OCC3D_ROOT)
    scenes = sorted(ann["train_split"])[:OCC3D_TRAIN_SCENES]
    out = []
    for sc in scenes:
        frs = scene_frames(ann, sc, "CAM_FRONT", G6F.OCC3D_ROOT)
        if len(frs) < 10:
            continue
        seg = Seg(source="occ3d_train", name=sc)
        for i, fr in enumerate(frs):
            gt = None
            if fr.gt_path and os.path.isfile(os.path.join(OCC3D_ROOT, fr.gt_path)):
                gt = {"anchor_gt_path": fr.gt_path, "clip_id": f"{sc}_{fr.token[:8]}",
                      "scene": sc}
            seg.frames.append(SFrame(
                key=os.path.splitext(os.path.basename(fr.image_path))[0],
                path=fr.image_path, order=fr.timestamp_ns, index=i, gt_ref=gt,
                T_cam_to_grid=np.asarray(fr.T_camera_to_ego, np.float64),
                K_native=np.asarray(fr.K, np.float64), native_hw=(900, 1600)))
        out.append(seg)
    return out


def segments(source: str, repo_root: str) -> List[Seg]:
    if source in EXTRA_SOURCES:
        return EXTRA_SOURCES[source]["segments"](repo_root)
    if source in VAL_SOURCES:
        return _val_segments(source, repo_root)
    if source == "sk_train":
        return _sk_train_segments(repo_root)
    if source == "occ3d_train":
        return _occ3d_train_segments(repo_root)
    raise KeyError(source)


def summarize(source: str, repo_root: str) -> dict:
    segs = segments(source, repo_root)
    n = [len(s) for s in segs]
    return {"source": source, "dataset": DATASET_OF[source], "n_segments": len(segs),
            "n_frames": int(sum(n)), "n_anchors": int(sum(len(s.anchors) for s in segs)),
            "frames_per_segment": [min(n), max(n)] if n else [0, 0]}


__all__ = ["Seg", "SFrame", "segments", "summarize", "stream_path", "scale_path",
           "trident_path", "moge_path", "DATASET_OF", "VAL_SOURCES", "TRAIN_SOURCES",
           "G8_ROOT", "SK_TRAIN_SEQUENCES", "OCC3D_TRAIN_SCENES", "EXTRA_SOURCES",
           "register_source"]
