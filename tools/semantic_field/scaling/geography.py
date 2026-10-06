"""Source/development independence check for the candidate drives 0000 / 0002 / 0004 / 0005 (CPU, poses and calibration only).

Pool rule (declared in ``artifacts/semantic_field_data_audit/data_scaling_proposal.md``, kept unchanged): a candidate anchor is kept
iff its sensor position at t is more than 2 x grid reach (114.4 m) from EVERY 0006 pose. Beyond that distance two grids cannot
intersect whatever their headings, so the rule is conservative.

This module re-checks the rule with the actual geometry:
    footprint(anchor) = XY convex hull of the 8 corners of the anchor's grid box (velodyne frame of t: x 0..51.2 m, y -25.6..25.6 m,
                        z -2..4.4 m) mapped to world by ``DriveGeometry.velo_to_world``; evidence (history cameras) and delayed targets
                        (future LiDAR/cameras) of an anchor all lie inside this box, so the footprint bounds its spatial content;
    overlap          = separating-axis test against the footprints of the 590 development anchors of 0006;
    windows          = the history t-4..t and future t+1..t+20 (<= 10 s) frames: every one must exist and resolve to a pose.
Not modelled: vertical overlap, occlusion, scene changes between recordings; drive 0009 cannot be checked (no local poses).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from tools.delayed_semantic import config as PILOT
from tools.semantic_completion.evidence import build as G15B
from tools.semantic_field.config.schema import DEV_DRIVE
from tools.semantic_field_data_audit.audit import K7, stream_natives

CANDIDATE_DRIVES: Tuple[str, ...] = tuple(f"2013_05_28_drive_{d}_sync" for d in ("0000", "0002", "0004", "0005"))
GRID_REACH_M = math.hypot(51.2, 25.6)
SCREEN_DISTANCE_M = 2 * GRID_REACH_M
_BOX = np.array([[x, y, z, 1.0] for x in (0.0, 51.2) for y in (-25.6, 25.6) for z in (-2.0, 4.4)])


def footprint(geo: TF.DriveGeometry, native: int) -> np.ndarray:
    """[H, 2] world-XY convex hull (counter-clockwise) of the anchor grid box."""
    pts = (geo.velo_to_world(native) @ _BOX.T).T[:, :2]
    return convex_hull(pts)


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: List[tuple] = []; upper: List[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (touching counts as overlap)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for n in np.stack([-edges[:, 1], edges[:, 0]], 1):
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


@dataclass
class DriveCheck:
    drive: str
    eligible: int
    kept: List[int]                             # stream indices kept by the declared screen (poses.txt position at t)
    excluded_by_screen: int
    kept_footprint_overlaps_dev: int            # exact velodyne-grid footprint overlap with any 0006 development grid
    screened_out_footprint_overlaps_dev: int
    min_kept_distance_to_dev_pose_m: float      # poses.txt position at t vs every 0006 poses.txt position
    min_kept_window_distance_to_dev_pose_m: float   # any t-4..t+20 (<= 10 s) frame's velodyne position vs every 0006 velodyne position
    kept_future_window_under_5_frames: int      # fewer than 5 future stream frames within 10 s (stream gaps; targets still built from those present)
    kept_window_frames_without_pose: int        # window frames that the target builder skips for lack of a pose


def dev_footprints() -> List[np.ndarray]:
    geo = TF.DriveGeometry(DEV_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    out = []
    for t in G15B.anchor_ids(DEV_DRIVE):
        with np.load(PILOT.Paths().g15.evidence_path(DEV_DRIVE, t)) as z:
            native = int(str(z["keys"][-1]).split("_")[-1])
        out.append(footprint(geo, native))
    return out


def check_drive(drive: str, anchors: Sequence[int], dev_polys: Sequence[np.ndarray], dev_pose_tree: cKDTree, dev_velo_tree: cKDTree) -> DriveCheck:
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    natives = stream_natives(drive); stamps = geo.timestamps("image"); pose_xy = pose_positions_xy(drive)
    centre_tree = cKDTree(np.stack([p.mean(0) for p in dev_polys]))
    kept, n_screen, ov_kept, ov_out, dmin, wmin, short_future, no_pose = [], 0, 0, 0, np.inf, np.inf, 0, 0
    horizon = PILOT.TargetConfig().horizon_s + 1e-6
    for t in anchors:
        d, _ = dev_pose_tree.query(pose_xy[natives[t]])
        poly = footprint(geo, natives[t])
        overlaps = any(polygons_overlap(poly, dev_polys[j]) for j in centre_tree.query_ball_point(poly.mean(0), 2 * SCREEN_DISTANCE_M))
        future = [j for j in range(t + 1, min(t + 21, len(natives))) if stamps[natives[j]] - stamps[natives[t]] <= horizon]
        window = [t + o for o in PILOT.HISTORY_OFFSETS] + future
        if d > SCREEN_DISTANCE_M:
            kept.append(int(t)); ov_kept += overlaps; dmin = min(dmin, float(d))
            posed = [j for j in window if natives[j] in geo.cam0_to_world]
            wd, _ = dev_velo_tree.query(np.stack([geo.velo_to_world(natives[j])[:2, 3] for j in posed]))
            wmin = min(wmin, float(wd.min())); short_future += int(len(future) < 5); no_pose += len(window) - len(posed)
        else:
            n_screen += 1; ov_out += overlaps
    return DriveCheck(drive, len(anchors), kept, n_screen, ov_kept, ov_out, dmin, wmin, short_future, no_pose)


def pose_positions_xy(drive: str) -> Dict[int, np.ndarray]:
    a = np.loadtxt(f"{SRC.SSCBENCH_ROOT}/data_poses/{drive}/poses.txt")
    return {int(r[0]): r[1:].reshape(3, 4)[:2, 3] for r in a}


def run(drives: Sequence[str] = CANDIDATE_DRIVES) -> Dict[str, DriveCheck]:
    """Candidate drives use the 7-drive manifest's eligible anchors; any other training drive uses its cached anchors."""
    polys = dev_footprints()
    dev_geo = TF.DriveGeometry(DEV_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    dev_pose_tree = cKDTree(np.stack(list(pose_positions_xy(DEV_DRIVE).values())))
    dev_velo_tree = cKDTree(np.stack([dev_geo.velo_to_world(n)[:2, 3] for n in sorted(dev_geo.cam0_to_world)]))
    manifest = json.load(open(f"{K7}/manifests/full.json"))["drives"]
    return {d: check_drive(d, manifest[d] if d in CANDIDATE_DRIVES else G15B.anchor_ids(d), polys, dev_pose_tree, dev_velo_tree) for d in drives}


__all__ = ["CANDIDATE_DRIVES", "GRID_REACH_M", "SCREEN_DISTANCE_M", "footprint", "convex_hull", "polygons_overlap", "DriveCheck", "run"]
