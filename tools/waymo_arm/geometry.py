"""Privileged geometry of one Waymo segment: the sweeps that build the occupancy target.

This is the training-only half of the arm. It opens the LiDAR, the Waymo vehicle poses and
the frames after the anchor -- all three forbidden on the causal path, which reads only the
selected camera's RGB (``waymo_pilot/waymo_geometry.py``).

The grid frame is the ANCHOR's vehicle frame, exactly as KITTI-360's grid frame is the
anchor's velodyne frame, so ``vehicle_to_vehicle`` is the analogue of ``velo_to_velo``.

Per-point laser identity comes from the sidecar written by
``waymo_pilot/cli/extract_laser_sidecar.py``: the SDK returns one array per laser in
``sorted(..., key=lambda c: c.name)`` order and the extractor concatenated them, so the
per-frame counts slice the cached cloud back apart exactly. Each slice becomes a
:class:`Sweep` with that laser's own origin, which is what the carve needs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.supervision.rawtarget_multisensor import Sweep
from tools.waymo_arm import config as A
from tools.waymo_pilot import waymo_geometry as WG


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ np.asarray(T, np.float64)[:3, :3].T + np.asarray(T, np.float64)[:3, 3]


@dataclass
class LaserSidecar:
    names: List[str]
    counts: np.ndarray                 # [n_frames, n_lasers], rows keyed by frame_index
    frame_index: np.ndarray            # [n_frames]
    origins: Dict[str, np.ndarray]     # laser -> [3] origin in the VEHICLE frame

    @staticmethod
    def load(path: str) -> "LaserSidecar":
        with np.load(path) as z:
            cal = json.loads(str(z["laser_calibration_json"]))
            names = [str(x) for x in z["laser_names"]]
            org = {k: np.asarray(v["extrinsic"], np.float64).reshape(4, 4)[:3, 3]
                   for k, v in cal.items()}
            return LaserSidecar(names, z["counts"].astype(np.int64),
                                z["frame_index"].astype(np.int64), org)

    def row(self, frame_index: int) -> Optional[np.ndarray]:
        hit = np.flatnonzero(self.frame_index == int(frame_index))
        return self.counts[int(hit[0])] if len(hit) else None


class WaymoAnchorGeometry:
    """Privileged sweeps of one segment, expressed in a chosen anchor's grid frame."""

    def __init__(self, segment: str, paths: A.Paths = None):
        self.segment = segment
        self.paths = paths or A.Paths()
        self.stream = WG.WaymoSegment(segment)
        side = self.paths.laser_sidecar(segment)
        if not os.path.isfile(side):
            raise FileNotFoundError(
                f"laser sidecar missing for {segment}; run "
                f"tools/waymo_pilot/cli/extract_laser_sidecar.py first")
        self.lasers = LaserSidecar.load(side)
        self._pose: Dict[int, np.ndarray] = {}

    # -- frames ------------------------------------------------------------
    def frame(self, stream_index: int):
        return self.stream.frames[int(stream_index)]

    def target_stream_frames(self, anchor: int,
                             horizon: int = A.TARGET_FUTURE_STREAM_FRAMES) -> List[int]:
        """The anchor and the later stream frames inside the frozen privileged window."""
        last = len(self.stream.frames) - 1
        return [t for t in range(int(anchor), int(anchor) + horizon + 1) if t <= last]

    # -- poses -------------------------------------------------------------
    def vehicle_to_world(self, stream_index: int) -> np.ndarray:
        s = int(stream_index)
        if s not in self._pose:
            with np.load(self.frame(s).path, allow_pickle=False) as z:
                self._pose[s] = np.asarray(z["vehicle_to_world"], np.float64)
        return self._pose[s]

    def vehicle_to_vehicle(self, src: int, anchor: int) -> np.ndarray:
        """vehicle frame of ``src`` -> vehicle frame of ``anchor`` (the grid frame)."""
        if int(src) == int(anchor):
            return np.eye(4)
        return np.linalg.inv(self.vehicle_to_world(anchor)) @ self.vehicle_to_world(src)

    # -- sweeps ------------------------------------------------------------
    def sweeps(self, anchor: int, horizon: int = A.TARGET_FUTURE_STREAM_FRAMES,
               lasers: Sequence[str] = None) -> List[Sweep]:
        """Every (frame, laser) sweep of the privileged window, in the anchor's grid frame."""
        want = set(lasers) if lasers else set(self.lasers.names)
        out: List[Sweep] = []
        for t in self.target_stream_frames(anchor, horizon):
            fr = self.frame(t)
            counts = self.lasers.row(fr.native)
            if counts is None:
                continue
            with np.load(fr.path, allow_pickle=False) as z:
                pts = np.asarray(z["points_vehicle"], np.float64)
            if int(counts.sum()) != len(pts):
                raise AssertionError(
                    f"{self.segment} frame {fr.native}: sidecar {int(counts.sum())} points "
                    f"vs cache {len(pts)} -- the slice would be wrong")
            T = self.vehicle_to_vehicle(t, anchor)
            lo = 0
            for name, c in zip(self.lasers.names, counts.tolist()):
                hi = lo + int(c)
                if name in want and c > 0:
                    out.append(Sweep(frame=t, laser=name,
                                     sensor=_apply(T, self.lasers.origins[name][None])[0],
                                     points=_apply(T, pts[lo:hi])))
                lo = hi
        return out

    def laser_shares(self, anchor: int, horizon: int = A.TARGET_FUTURE_STREAM_FRAMES) -> Dict[str, float]:
        """Point share per laser over the window, for the record."""
        tot = np.zeros(len(self.lasers.names), np.int64)
        for t in self.target_stream_frames(anchor, horizon):
            c = self.lasers.row(self.frame(t).native)
            if c is not None:
                tot += c
        s = max(int(tot.sum()), 1)
        return {n: float(v / s) for n, v in zip(self.lasers.names, tot.tolist())}


__all__ = ["LaserSidecar", "WaymoAnchorGeometry"]
