"""The Gate-6 target-access auditor.

Gate 5.2 established the pattern: asserting target-independence by reading the code is
weaker than observing it. This extends :class:`sscbench_kitti360.audit.FileAudit` with the
paths Gate 6 additionally must not touch during prediction -- Occ3D ``labels.npz`` and its
``gts/`` tree, the SemanticKITTI ``voxels/`` volumes, every LiDAR sweep, and every oracle
or LiDAR-derived scale table -- and additionally intercepts ``cv2.imread``, which the
teacher uses and which bypasses ``builtins.open`` entirely.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from sscbench_kitti360.audit import FileAudit as _FileAudit, ForbiddenAccess

# Anything encoding an occupancy or semantic target, its validity/visibility mask, a LiDAR
# measurement, or a scale derived from one.
GATE6_FORBIDDEN: Tuple[str, ...] = (
    ".label", ".invalid", "_1_1.npy", "_1_2.npy", "_1_8.npy",
    "labels.npz", "/gts/", "\\gts\\", "/voxels/", "\\voxels\\", "/labels/", "\\labels\\",
    "velodyne", "lidar_surface", "oracle", "scale_targets", "visible_ceiling",
)


class Gate6Audit(_FileAudit):
    """``FileAudit`` plus ``cv2.imread`` interception."""

    def __init__(self, patterns: Sequence[str] = GATE6_FORBIDDEN, strict: bool = True):
        super().__init__(patterns=patterns, strict=strict)
        self._cv2 = None

    def __enter__(self) -> "Gate6Audit":
        super().__enter__()
        try:
            import cv2
        except Exception:                                     # pragma: no cover
            return self
        self._cv2 = (cv2, cv2.imread)
        audit = self

        def _imread(filename, *a, **kw):
            audit._check(filename)
            return audit._cv2[1](filename, *a, **kw)

        cv2.imread = _imread
        return self

    def __exit__(self, *exc) -> bool:
        if self._cv2 is not None:
            self._cv2[0].imread = self._cv2[1]
            self._cv2 = None
        return super().__exit__(*exc)


__all__ = ["Gate6Audit", "ForbiddenAccess", "GATE6_FORBIDDEN"]
