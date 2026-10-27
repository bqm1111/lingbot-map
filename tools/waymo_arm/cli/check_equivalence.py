#!/usr/bin/env python
"""Gate: the multi-sensor accumulator must reproduce the frozen single-sensor rule exactly.

``core/supervision/rawtarget_multisensor.py`` is only admissible if, given one sweep per
frame with that frame's single sensor origin, it returns the same four arrays as
``core/supervision/rawtarget.build`` -- the function that produced every KITTI-360 target.
If it does, the Waymo arm's supervision differs from the KITTI-360 arm's in the DATA and in
nothing else, which is the only way the two arms can be compared.

Check 1 (the real gate): KITTI-360 anchors. Sweeps are constructed exactly as
``rawtarget.build`` constructs them internally, and all four arrays plus ``frames_used`` must
match bit for bit.

Check 2: a Waymo anchor, TOP laser only, run through the same two paths via a duck-typed
geometry -- so the gate also covers the data this arm actually trains on.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/check_equivalence.py --anchors 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.datasets import kitti360_splits as SRC                                       # noqa: E402
from core.datasets import kitti360_transforms as TF                                    # noqa: E402
from core.supervision import rawtarget as RT                                           # noqa: E402
from core.supervision import rawtarget_multisensor as MS                               # noqa: E402
from tools.waymo_arm import config as A                                                # noqa: E402
from tools.waymo_arm import geometry as WAG                                            # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                      # noqa: E402

ARRAYS = ("state", "n_obs", "n_frames_occ", "n_free")


def _sweeps_from_drive_geometry(geo, anchor_native: int, natives) -> list:
    """Exactly how ``rawtarget.build`` turns a DriveGeometry into sweeps."""
    out = []
    for nf in natives:
        nf = int(nf)
        if nf not in geo.cam0_to_world:
            continue
        try:
            pts = geo.read_velodyne(nf)
        except (FileNotFoundError, OSError):
            continue
        T = np.eye(4) if nf == anchor_native else geo.velo_to_velo(nf, anchor_native)
        out.append(MS.Sweep(frame=nf, sensor=np.asarray(T, np.float64)[:3, 3],
                            points=TF.apply(T, pts), laser="velodyne"))
    return out


def _compare(ref: RT.RawTarget, got: RT.RawTarget) -> dict:
    d = {"frames_used_equal": list(map(int, ref.frames_used)) == list(map(int, got.frames_used))}
    for k in ARRAYS:
        a, b = getattr(ref, k), getattr(got, k)
        same = bool(a.shape == b.shape and np.array_equal(a, b))
        d[k] = {"equal": same, "n_differing": int((a != b).sum()) if a.shape == b.shape else -1}
    d["all_equal"] = d["frames_used_equal"] and all(d[k]["equal"] for k in ARRAYS)
    return d


def check_k360(n_anchors: int, device) -> dict:
    drive = SRC.TRAIN_DRIVES[0]
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    ancs = SRC.anchors(drive, A.REPO_ROOT)
    picked = ancs[:: max(1, len(ancs) // max(n_anchors, 1))][:n_anchors]
    rows = []
    for anc in picked:
        natives = [anc.native_frame] + list(anc.future_natives)
        ref = RT.build(geo, anc.native_frame, natives, device)
        sw = _sweeps_from_drive_geometry(geo, anc.native_frame, natives)
        got = MS.build(sw, TF.GRID.dims, TF.GRID.origin, TF.GRID.voxel_size, device)
        rows.append({"drive": drive, "stream_index": int(anc.stream_index),
                     "native": int(anc.native_frame), "n_sweeps": len(sw),
                     "n_occupied": int(ref.occupied.sum()),
                     "n_valid": int(ref.valid.sum()), **_compare(ref, got)})
    return {"check": "k360_single_sensor_equivalence", "n": len(rows),
            "all_passed": all(r["all_equal"] for r in rows), "rows": rows}


class _WaymoAsDriveGeometry:
    """One Waymo laser presented through ``rawtarget.build``'s DriveGeometry interface.

    ``read_velodyne`` must return points in the SENSOR's own frame, because ``build`` takes
    the sensor origin from the transform's translation. So the laser origin is folded into
    the transform and subtracted from the points -- the identity that makes the two paths
    comparable at all.

    ``build`` also short-circuits ``T`` to the identity when ``nf == anchor_native``, which
    assumes the anchor sweep's sensor sits at the grid origin. That assumption is true for
    KITTI-360 and is precisely what fails on Waymo, so the check calls ``build`` with a
    sentinel ``anchor_native`` that matches no frame; ``anchor_native`` is used for nothing
    else in that function, and every sweep then goes through ``velo_to_velo``.
    """

    #: a frame index no Waymo stream frame can take, so the identity short-circuit never fires
    SENTINEL_ANCHOR = -1

    def __init__(self, wag: "WAG.WaymoAnchorGeometry", anchor: int, laser: str):
        self.w, self.anchor, self.laser = wag, int(anchor), laser
        self.frames = wag.target_stream_frames(self.anchor)
        self.cam0_to_world = {t: wag.vehicle_to_world(t) for t in self.frames}
        self._o = np.asarray(wag.lasers.origins[laser], np.float64)

    def read_velodyne(self, t: int) -> np.ndarray:
        fr = self.w.frame(int(t))
        counts = self.w.lasers.row(fr.native)
        with np.load(fr.path, allow_pickle=False) as z:
            pts = np.asarray(z["points_vehicle"], np.float64)
        lo = 0
        for name, c in zip(self.w.lasers.names, counts.tolist()):
            hi = lo + int(c)
            if name == self.laser:
                return pts[lo:hi] - self._o                     # sensor-centred
            lo = hi
        return np.zeros((0, 3))

    def velo_to_velo(self, src: int, anchor: int) -> np.ndarray:
        """sensor frame of ``src`` -> grid frame, so the translation IS the sensor origin."""
        assert int(anchor) == self.SENTINEL_ANCHOR, "call build() with the sentinel anchor"
        T = self.w.vehicle_to_vehicle(int(src), self.anchor)
        out = np.eye(4)
        out[:3, :3] = T[:3, :3]
        out[:3, 3] = T[:3, :3] @ self._o + T[:3, 3]
        return out


def check_waymo(n_anchors: int, device, laser: str = "TOP") -> dict:
    p = A.Paths()
    segs = [s for s in sorted(os.listdir(p.clips))
            if os.path.isfile(p.laser_sidecar(s))][:n_anchors]
    if not segs:
        return {"check": "waymo_single_laser_equivalence", "n": 0,
                "all_passed": None, "note": "no segment has both clips and a laser sidecar yet"}
    grid = WG.waymo_grid()
    rows = []
    for seg in segs:
        wag = WAG.WaymoAnchorGeometry(seg)
        ts = sorted(int(f[:-4]) for f in os.listdir(os.path.join(p.clips, seg))
                    if f.endswith(".npz"))
        anchor = ts[0]
        geo = _WaymoAsDriveGeometry(wag, anchor, laser)
        natives = geo.frames
        ref = RT.build(geo, geo.SENTINEL_ANCHOR, natives, device, grid=grid)
        sw = wag.sweeps(anchor, lasers=[laser])
        got = MS.build(sw, grid.dims, grid.origin, grid.voxel_size, device)
        rows.append({"segment": seg, "anchor": anchor, "laser": laser, "n_sweeps": len(sw),
                     "n_occupied": int(ref.occupied.sum()), "n_valid": int(ref.valid.sum()),
                     "laser_shares": wag.laser_shares(anchor), **_compare(ref, got)})
    return {"check": "waymo_single_laser_equivalence", "n": len(rows),
            "all_passed": all(r["all_equal"] for r in rows), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-k360", action="store_true")
    a = ap.parse_args()
    A.ensure_dirs()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = {"version": A.VERSION, "device": str(dev), "checks": []}
    if not a.skip_k360:
        out["checks"].append(check_k360(a.anchors, dev))
    out["checks"].append(check_waymo(a.anchors, dev))
    out["all_passed"] = all(c["all_passed"] for c in out["checks"]
                            if c["all_passed"] is not None)
    path = os.path.join(A.Paths().logs, "check_equivalence.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    for c in out["checks"]:
        print(f"[{'PASS' if c['all_passed'] else 'FAIL' if c['all_passed'] is not None else 'SKIP'}] "
              f"{c['check']}: {c['n']} anchors")
        for r in c.get("rows", []):
            if not r["all_equal"]:
                print("   ", json.dumps({k: v for k, v in r.items() if k in ARRAYS}))
    print(f"\nall_passed={out['all_passed']} -> {path}")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
