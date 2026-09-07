# extracted from gate8c1/data.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Sample reader for Gate 8C-1: KITTI-360 drives with rebuilt raw-LiDAR supervision.

Deliberately thin. The crop sampler, the batch layout and the tensor unpacking are Gate
8A's and Gate 8's frozen ones (:class:`tools.gate8a.train.AblationSamples`,
:func:`gate8.targets.unpack_sample`); the only new thing is that files are enumerated per
*drive* from the Gate 8C-1 sample root, and that a drive list is checked against the
firewall before anything opens.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Sequence

from core.datasets import kitti360_splits as SRC
from core.run._gate8a_train import AblationSamples


class DriveSamples(AblationSamples):
    """Gate 8A's sampler over an explicit list of KITTI-360 drives."""

    def __init__(self, drives: Sequence[str], crop, device, seed: int = 0,
                 per_drive_limit: int = None, **kw):
        AblationSamples.__init__(self, [], crop, device, seed, None, 1, **kw)
        SRC.assert_no_target_access(list(drives))
        self.drives = list(drives)
        self.by_drive: Dict[str, List[str]] = {}
        for d in self.drives:
            fs = sorted(glob.glob(os.path.join(os.path.dirname(SRC.sample_path(d, 0)), "*.npz")))
            if per_drive_limit:
                fs = fs[:per_drive_limit]
            assert fs, f"no Gate 8C-1 samples for {d}"
            self.by_drive[d] = fs
        self._reindex()

    def _reindex(self):
        self.files, self._offset, o = [], {}, 0
        for d in self.drives:
            self._offset[d] = o
            self.files += self.by_drive[d]
            o += len(self.by_drive[d])

    def verify(self) -> int:
        n_bad = AblationSamples.verify(self)
        keep = set(self.files)
        self.by_drive = {d: [f for f in fs if f in keep] for d, fs in self.by_drive.items()}
        self._reindex()
        return n_bad

    def draw(self) -> int:
        """One index: drive uniformly, then a file uniformly inside it."""
        d = self.drives[self.rng.randrange(len(self.drives))]
        return self._offset[d] + self.rng.randrange(len(self.by_drive[d]))

    def drive_of(self, i: int) -> str:
        for d in self.drives:
            if self._offset[d] <= i < self._offset[d] + len(self.by_drive[d]):
                return d
        raise IndexError(i)


__all__ = ["DriveSamples"]
