"""Stream frames, the causal history H_t, and the (target-only) future window.

``DriveStream`` resolves stream index -> native KITTI-360 frame, image, Trident token map and image
timestamp from the frozen stream definition (``core.datasets.sources.segments``). Two accessors:

* ``history(t)`` -> :class:`CausalHistory`: the only object the student pipeline receives. It holds
  frames t-4..t and raises :class:`FutureAccess` for any index > t, so a code path that tries to read
  a future frame fails instead of silently leaking.
* ``future(t)`` -> frames t+1..t+H within ``horizon_s``: used by target construction only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from core.datasets import kitti360_folds  # noqa: F401  (registers the k360_train source)
from core.datasets import kitti360_splits as SRC, sources as S
from core.datasets import kitti360_transforms as TF
from tools.delayed_semantic import config as C


class FutureAccess(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamFrame:
    index: int          # stream index
    key: str            # token / cache key
    native: int         # KITTI-360 native frame
    image_path: str
    token_path: str
    time_s: float       # image timestamp, seconds since the drive's first image


class CausalHistory:
    """Frames t-4..t. Everything the student may see at prediction time t."""

    def __init__(self, frames: Sequence[StreamFrame], t: int):
        self._frames = {f.index: f for f in frames}
        self.t = t
        assert sorted(self._frames) == [t + o for o in C.HISTORY_OFFSETS], "history must be exactly t-4..t"

    @property
    def indices(self) -> List[int]:
        return sorted(self._frames)

    def frame(self, index: int) -> StreamFrame:
        if index > self.t:
            raise FutureAccess(f"stream frame {index} is in the future of t={self.t}")
        return self._frames[index]

    def frames(self) -> List[StreamFrame]:
        return [self._frames[i] for i in self.indices]


class DriveStream:
    def __init__(self, drive: str, repo_root: str = C.REPO_ROOT):
        src = SRC.source_of(drive)
        seg = [s for s in S.segments(src, repo_root) if s.name == drive]
        assert seg, drive
        ts = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT).timestamps("image")
        tok = C.Paths().g15.token_path
        self.drive = drive
        self.frames: List[StreamFrame] = [StreamFrame(f.index, f.key, int(f.order), f.path, tok(drive, f.key), float(ts[int(f.order)]))
                                          for f in seg[0].frames]
        assert [f.index for f in self.frames] == list(range(len(self.frames)))

    def __len__(self) -> int:
        return len(self.frames)

    def history(self, t: int) -> CausalHistory:
        return CausalHistory([self.frames[t + o] for o in C.HISTORY_OFFSETS], t)

    def future(self, t: int, cfg: C.TargetConfig = C.TargetConfig()) -> List[StreamFrame]:
        """TARGET CONSTRUCTION ONLY: t+1..t+horizon_frames, no later than horizon_s after t."""
        t0 = self.frames[t].time_s
        return [f for f in self.frames[t + 1: t + 1 + cfg.horizon_frames] if f.time_s - t0 <= cfg.horizon_s + 1e-6]


__all__ = ["FutureAccess", "StreamFrame", "CausalHistory", "DriveStream"]


if __name__ == "__main__":   # quick listing
    for d in C.TRAIN_DRIVES + (C.DEV_DRIVE,):
        s = DriveStream(d); print(d, len(s), s.frames[0], os.path.exists(s.frames[0].token_path))
