"""A file-access auditor that proves targets and LiDAR never reach the deployable path.

Gate 5.2's central claim is that C0, C3, A and B are computed from RGB, calibration and
LingBot alone. Asserting that by reading the code is weaker than observing it, so the
scale stage runs inside :class:`FileAudit`, which records every path opened through
``builtins.open``, ``numpy.load`` and ``numpy.fromfile`` and raises the moment a forbidden
one is touched.
"""

from __future__ import annotations

import builtins
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Anything that encodes the occupancy target, its validity mask or a LiDAR measurement.
FORBIDDEN_PATTERNS: Tuple[str, ...] = (".label", ".invalid", "_1_1.npy", "_1_2.npy",
                                       "_1_8.npy", "velodyne", "/voxels/", "\\voxels\\")


class ForbiddenAccess(RuntimeError):
    pass


class FileAudit:
    """Context manager recording file opens and rejecting forbidden ones."""

    def __init__(self, patterns: Sequence[str] = FORBIDDEN_PATTERNS, strict: bool = True):
        self.patterns = tuple(patterns)
        self.strict = strict
        self.opened: List[str] = []
        self.violations: List[str] = []
        self._saved = {}

    def _check(self, path) -> None:
        try:
            p = os.fspath(path)
        except TypeError:
            return
        if not isinstance(p, str):
            return
        self.opened.append(p)
        low = p.lower()
        if any(pat in low for pat in self.patterns):
            self.violations.append(p)
            if self.strict:
                raise ForbiddenAccess(f"deployable path opened a forbidden file: {p}")

    def __enter__(self) -> "FileAudit":
        audit = self

        self._saved = {"open": builtins.open, "load": np.load, "fromfile": np.fromfile}

        def _open(file, *a, **kw):
            audit._check(file)
            return audit._saved["open"](file, *a, **kw)

        def _load(file, *a, **kw):
            audit._check(file)
            return audit._saved["load"](file, *a, **kw)

        def _fromfile(file, *a, **kw):
            audit._check(file)
            return audit._saved["fromfile"](file, *a, **kw)

        builtins.open, np.load, np.fromfile = _open, _load, _fromfile
        return self

    def __exit__(self, *exc) -> bool:
        builtins.open = self._saved["open"]
        np.load = self._saved["load"]
        np.fromfile = self._saved["fromfile"]
        return False

    def summary(self) -> dict:
        return {"n_opened": len(self.opened), "n_unique": len(set(self.opened)),
                "violations": self.violations,
                "forbidden_patterns": list(self.patterns)}
