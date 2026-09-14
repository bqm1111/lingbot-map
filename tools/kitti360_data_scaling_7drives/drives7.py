"""Seven-drive registration shim for the KITTI-360 data-scaling experiment.

The frozen cache/sample tools (``tools/gate8b/stream_source.py``, ``cache_trident.py``,
``core/run/build_targets.py``, ``core/run/build_samples.py``) enumerate the training drives
from a module-level tuple that holds only 0003/0007/0010. Importing this module extends
that tuple -- in both the ``core`` copy and the ``gates`` copy the gate tools import -- to
the full SSCBench-KITTI-360 training split, and redirects the tools' artifact records into
this experiment's directory so historical ``artifacts/gate8b`` / ``artifacts/gate8c1``
records are never overwritten. Nothing about inference, eligibility or target rules is
changed; the stream is still defined by the pose file at SSCBench-index stride 5.

Drive 0006 stays the development drive; 0009 is not registered anywhere.
"""
from __future__ import annotations
import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TRAIN_DRIVES7 = tuple(f"2013_05_28_drive_{d}_sync" for d in ("0000", "0002", "0003", "0004", "0005", "0007", "0010"))
NEW_DRIVES = tuple(d for d in TRAIN_DRIVES7 if d[17:21] in ("0000", "0002", "0004", "0005"))
VAL_DRIVE = "2013_05_28_drive_0006_sync"
RESERVED = ("2013_05_28_drive_0009_sync",)
ART = os.path.join(REPO_ROOT, "artifacts", "kitti360_data_scaling_7drives")
DATA_ROOT = os.environ.get("KDS7_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")


def apply(redirect_artifacts: bool = True) -> None:
    os.environ["G8C1_DATA_ROOT"] = DATA_ROOT                      # targets/ and samples/ of this experiment
    import core.datasets.kitti360 as K3
    assert all(d in K3.OFFICIAL_SPLIT["train"] for d in TRAIN_DRIVES7) and VAL_DRIVE not in TRAIN_DRIVES7
    assert not set(RESERVED) & set(TRAIN_DRIVES7)
    import core.datasets.kitti360_folds as KF
    KF.K360_TRAIN_DRIVES = TRAIN_DRIVES7
    import core.datasets.kitti360_splits as SRC
    assert os.path.abspath(SRC.G8C1_ROOT) == os.path.abspath(DATA_ROOT), (SRC.G8C1_ROOT, "import drives7 before core.datasets.kitti360_splits")
    SRC.TRAIN_DRIVES = TRAIN_DRIVES7
    try:                                                          # the gates copy used by tools/gate8b/*
        import gates.gate8b.sources as GS
        GS.K360_TRAIN_DRIVES = TRAIN_DRIVES7
    except Exception:
        pass
    if redirect_artifacts:
        import core.run._common as CC
        CC.ART = os.path.join(ART, "prep", "core")
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "gate8b"))
        try:
            import _common as G8BC                                # tools/gate8b/_common.py
            G8BC.ART = os.path.join(ART, "prep", "gate8b")
        except Exception:
            pass


def run_tool(path: str, argv) -> int:
    """Execute an unchanged tool script with the shim applied."""
    import runpy
    apply()
    sys.argv = [path] + list(argv)
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tool(sys.argv[1], sys.argv[2:]))
