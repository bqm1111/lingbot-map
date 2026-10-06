"""Track P (practical causal evaluation) and Track O (official benchmark) parameters. Frozen by hash before any method
comparison is scored; ``practical_evaluation_predeclaration.md`` explains every value.

Coordinates and units
    Evaluation grids are the official ones: SemanticKITTI 256x256x32 @ 0.2 m in the velodyne frame of the target frame;
    Occ3D-nuScenes 200x200x16 @ 0.4 m in the ego frame of the target keyframe. Distances are metres. The target camera frame
    is OpenCV (x right, y down, z forward); ``T_cam_to_grid`` maps it to the grid frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

PROTOCOL_VERSION = "final_zero_shot/track_p_v1"


@dataclass(frozen=True)
class Region:
    """Track P operational region = valid label ∩ input-camera horizontal field of view ∩ horizontal range <= max_range_m."""
    max_range_m: float = 40.0                                      # common forward extent of both benchmarks; urban stopping range
    range_edges_m: Tuple[float, ...] = (10.0, 20.0, 30.0)          # bins 0-10, 10-20, 20-30, 30-40 m
    range_names: Tuple[str, ...] = ("0-10 m", "10-20 m", "20-30 m", "30-40 m")
    fov_uses_native_image_width: bool = True                       # wedge bounded by the image columns 0 and W at every depth


REGION = Region()

#: distance from a voxel to the nearest semantic evidence voxel (metres), Gates 15/16 bins
EVIDENCE_DISTANCE_EDGES_M: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
EVIDENCE_DISTANCE_NAMES: Tuple[str, ...] = ("0-1 m", "1-2 m", "2-4 m", "4-8 m", ">8 m")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_classes: int                  # label ids 0..n_classes-1 including the empty class
    empty_class: int
    semantic_classes: Tuple[int, ...]
    class_names: Tuple[str, ...]    # index = label id
    dynamic_classes: Tuple[int, ...]
    superclass_of: Tuple[int, ...]  # OccAny's official superclass mapping, index = label id
    superclass_names: Tuple[str, ...]
    superclass_empty: int
    image_width: int
    occany_other_class: int         # OccAny SSCMetrics ``other_class``
    occany_ignore_other: bool       # OccAny SSCMetrics ``ignore_other_class_in_mIoU``


SEMANTICKITTI = DatasetSpec(
    name="semantickitti", n_classes=20, empty_class=0, semantic_classes=tuple(range(1, 20)),
    class_names=("empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", "parking",
                 "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"),
    dynamic_classes=(1, 2, 3, 4, 5, 6, 7, 8),
    superclass_of=(0, 3, 3, 3, 3, 3, 5, 5, 5, 1, 1, 1, 1, 2, 6, 4, 4, 4, 6, 6),
    superclass_names=("empty", "ground", "structure", "vehicle", "nature", "human", "object"), superclass_empty=0,
    image_width=1226, occany_other_class=20, occany_ignore_other=True)          # sequence 08 image_2 is 1226 x 370

OCC3D = DatasetSpec(
    name="occ3d", n_classes=18, empty_class=17, semantic_classes=tuple(range(0, 17)),
    class_names=("others", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer",
                 "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation", "free"),
    dynamic_classes=(2, 3, 4, 5, 6, 7, 9, 10),
    superclass_of=(0, 2, 3, 3, 3, 3, 3, 5, 6, 3, 3, 1, 1, 1, 4, 2, 4, 7),
    superclass_names=("other", "ground", "structure", "vehicle", "nature", "human", "object", "empty"), superclass_empty=7,
    image_width=1600, occany_other_class=0, occany_ignore_other=False)

DATASET_SPECS: Dict[str, DatasetSpec] = {"semantickitti": SEMANTICKITTI, "occ3d": OCC3D}

#: methods scored with the practical and official evaluators
METHODS: Tuple[str, ...] = ("B0", "B3_s0", "B3_s1", "B3_s2")
B3_SEED_METHODS: Tuple[str, ...] = ("B3_s0", "B3_s1", "B3_s2")


@dataclass(frozen=True)
class Statistics:
    resamples: int = 2000
    seed: int = 0
    semantickitti_block_samples: int = 25     # contiguous runs of 25 evaluated samples (sequence 08 order)
    occ3d_block: str = "scene"                 # one block per nuScenes scene


STATISTICS = Statistics()


@dataclass(frozen=True)
class PaperCriteria:
    central_metric: str = "completed_no_evidence_miou"             # GT occupied ∩ predicted occupied ∩ no semantic evidence, Track P region
    meaningful_gain_points: float = 1.0                            # mIoU points, B3 seed mean - B0, on each dataset
    occany_published_miou: Tuple[Tuple[str, float], ...] = (("semantickitti", 7.28), ("occ3d", 6.66))   # OccAny Table 4 (GSAM2)


PAPER = PaperCriteria()

__all__ = [k for k in dir() if not k.startswith("_")]
