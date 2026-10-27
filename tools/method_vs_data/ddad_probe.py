#!/usr/bin/env python
"""DDAD feasibility probe — can DDAD support the frozen occupancy-supervision recipe?

This is a **probe**, not an adapter. It exists to answer one question with measurements
rather than with the dataset's prose: when the frozen Gate 8C-1 carve rule is handed real
DDAD sweeps, does it produce a target with statistics comparable to KITTI-360's and Waymo's?

What it reads is one complete DDAD scene recovered from a byte-range read of the public
archive (scene metadata, calibration, and every LiDAR sweep) -- no 256 GiB download. The
frozen ``core.supervision.rawtarget.build`` is called unchanged, through a duck-typed
geometry object exposing the three members it uses: ``cam0_to_world`` (membership only),
``read_velodyne`` and ``velo_to_velo``.

The decisive fact this probe quantifies: DDAD publishes **one aggregated point cloud per
sample, in a vehicle frame whose origin sits on the ground below the rear axle**, with
per-point channels X/Y/Z/INTENSITY only. Four physically separated Luminar-H2 sensors are
merged before release, so the carve rule's ray origin -- the thing that decides which
voxels are free -- is not recoverable from the published data.

    PYTHONPATH=$PWD python tools/method_vs_data/ddad_probe.py --scene-dir <extracted dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import kitti360_transforms as TF                                     # noqa: E402
from core.supervision import rawtarget as RT                                            # noqa: E402


def quat_to_R(q: Dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("qw", "qx", "qy", "qz"))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
                    np.float64)


def pose_to_T(p: Dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p["rotation"])
    T[:3, 3] = [float(p["translation"][k]) for k in "xyz"]
    return T


class DdadSceneGeometry:
    """The three members ``rawtarget.build`` touches, backed by one DDAD scene.

    ``mount`` shifts the assumed sensor origin within the vehicle frame. The published data
    implies (0, 0, 0) -- the ground point below the rear axle. Passing a roof-height mount
    re-expresses the same world points in a frame whose origin is there instead, which is
    how the probe measures the target's sensitivity to an origin DDAD does not publish.
    """

    def __init__(self, scene_json: str, cloud_dir: str, mount=(0.0, 0.0, 0.0)):
        d = json.load(open(scene_json))
        self.name = d["name"]
        lid = [x for x in d["data"] if x["id"]["name"] == "LIDAR"]
        self.mount = np.asarray(mount, np.float64)
        self.files, self.T_sensor_to_world = {}, {}
        for i, x in enumerate(lid):
            pc = x["datum"]["point_cloud"]
            assert pc["point_format"] == ["X", "Y", "Z", "INTENSITY"], pc["point_format"]
            base = os.path.basename(pc["filename"])
            cand = glob.glob(os.path.join(cloud_dir, f"*point_cloud__LIDAR__{base}"))
            if not cand:
                continue
            self.files[i] = cand[0]
            # vehicle frame -> world, then shift the origin to the assumed mount
            T_vw = pose_to_T(pc["pose"])
            M = np.eye(4); M[:3, 3] = self.mount
            self.T_sensor_to_world[i] = T_vw @ M
        self.cam0_to_world = {k: self.T_sensor_to_world[k] for k in self.files}

    def read_velodyne(self, nf: int) -> np.ndarray:
        with np.load(self.files[int(nf)]) as z:
            pts = z["data"][:, :3].astype(np.float64)
        return pts - self.mount                      # into the assumed sensor frame

    def velo_to_velo(self, a: int, b: int) -> np.ndarray:
        return np.linalg.inv(self.T_sensor_to_world[int(b)]) @ self.T_sensor_to_world[int(a)]


def describe(geo: DdadSceneGeometry) -> dict:
    """Corpus-level facts, and where the frame origin sits relative to the road."""
    frames = sorted(geo.files)
    tr = np.stack([geo.T_sensor_to_world[f][:3, 3] for f in frames])
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    pts = geo.read_velodyne(frames[len(frames) // 2])
    r = np.linalg.norm(pts[:, :2], axis=1)
    near = pts[r < 6.0]
    return {"scene": geo.name, "n_samples_with_cloud": len(frames),
            "points_per_sweep": int(len(pts)),
            "path_length_m": float(step.sum()),
            "median_step_m": float(np.median(step)),
            "range_m": {"p50": float(np.percentile(np.linalg.norm(pts, axis=1), 50)),
                        "p90": float(np.percentile(np.linalg.norm(pts, axis=1), 90)),
                        "max": float(np.linalg.norm(pts, axis=1).max())},
            "z_all": {"min": float(pts[:, 2].min()), "max": float(pts[:, 2].max())},
            "z_within_6m_horizontal": {
                "n": int(len(near)),
                "p01": float(np.percentile(near[:, 2], 1)),
                "p50": float(np.percentile(near[:, 2], 50)),
                "mode_bin": float(np.histogram(near[:, 2], bins=200)[1][
                    int(np.argmax(np.histogram(near[:, 2], bins=200)[0]))])}}


def target_stats(geo: DdadSceneGeometry, anchors, n_sweeps: int, device) -> list:
    out = []
    frames = sorted(geo.files)
    for a in anchors:
        if a not in geo.files:
            continue
        fut = [f for f in frames if f >= a][:n_sweeps]
        if len(fut) < n_sweeps:
            continue
        t = RT.build(geo, a, fut, device, grid=TF.GRID)
        s = dict(t.stats); s["anchor"] = a
        s["z_voxels_occupied"] = int((t.state.reshape(TF.DIMS) == RT.OCCUPIED).any((0, 1)).sum())
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, help="dir holding the extracted members")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sweeps", type=int, default=21)
    ap.add_argument("--anchors", default="0,20,40")
    ap.add_argument("--out", default="artifacts/method_vs_data/results/ddad_probe.json")
    a = ap.parse_args()
    sj = glob.glob(os.path.join(a.scene_dir, "*scene_*.json"))
    assert len(sj) == 1, sj
    dev = torch.device(a.device)
    anchors = [int(x) for x in a.anchors.split(",")]

    res = {"scene_json": os.path.basename(sj[0]), "grid": TF.GRID.name,
            "dims": list(TF.DIMS), "voxel_size": TF.VOXEL,
            "grid_origin": list(map(float, TF.ORIGIN)), "sweeps": a.sweeps, "arms": {}}
    for tag, mount in (("published_vehicle_origin", (0.0, 0.0, 0.0)),
                       ("assumed_roof_mount", (1.6, 0.0, 1.6))):
        geo = DdadSceneGeometry(sj[0], a.scene_dir, mount=mount)
        rec = {"mount_in_vehicle_frame": list(mount), "describe": describe(geo),
               "targets": target_stats(geo, anchors, a.sweeps, dev)}
        res["arms"][tag] = rec
        d = rec["describe"]
        print(f"\n=== {tag}  mount={mount}")
        print(f"  scene {d['scene']}: {d['n_samples_with_cloud']} sweeps, "
              f"{d['points_per_sweep']:,} pts/sweep, path {d['path_length_m']:.1f} m, "
              f"median step {d['median_step_m']:.3f} m")
        print(f"  range p50 {d['range_m']['p50']:.1f} p90 {d['range_m']['p90']:.1f} "
              f"max {d['range_m']['max']:.1f} m;  z all "
              f"[{d['z_all']['min']:.2f}, {d['z_all']['max']:.2f}]")
        z = d["z_within_6m_horizontal"]
        print(f"  z within 6 m horizontal: p01 {z['p01']:.2f} p50 {z['p50']:.2f} "
              f"mode {z['mode_bin']:.2f}  (road level relative to the frame origin)")
        for s in rec["targets"]:
            print(f"  anchor {s['anchor']:>3}: valid {s['valid_fraction']:.4f}  "
                  f"prevalence {s['prevalence_in_valid']:.4f}  "
                  f"conflict/touched {s['conflict_fraction_of_touched']:.4f}  "
                  f"occ-z-levels {s['z_voxels_occupied']}/{TF.DIMS[2]}  "
                  f"pts_in_grid {s['n_points_in_grid']:,}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
