"""Resolved identities and every path this experiment touches. Single source of truth.

Isolation contract (asserted by :func:`assert_isolated`, called at import):
  * EXPERIMENT_ROOT is disjoint from LEGACY_REPO and REPRO_REPO;
  * REPRO_REPO is never read, imported or written;
  * LEGACY_REPO is imported read-only, with bytecode writes disabled;
  * every write this experiment performs is under EXPERIMENT_ROOT.
"""
from __future__ import annotations

import os
import sys

LEGACY_REPO = "/home/minh/workspace/lingbot-map_fork"
REPRO_REPO = "/home/minh/workspace/StreamingSSC"
EXPERIMENT_ROOT = "/media/SSD1/region_naming_pilot"

CODE = os.path.join(EXPERIMENT_ROOT, "code")
CACHE = os.path.join(EXPERIMENT_ROOT, "cache")
ARTIFACTS = os.path.join(EXPERIMENT_ROOT, "artifacts")
FIGURES = os.path.join(EXPERIMENT_ROOT, "figures")
LOGS = os.path.join(EXPERIMENT_ROOT, "logs")

# ---- read-only inputs -----------------------------------------------------------------
DRIVE = "2013_05_28_drive_0006_sync"
RGB_ROOT = f"/media/welf/MINH/datasets/kitti360/KITTI-360/data_2d_raw/{DRIVE}/image_00/data_rect"
SAM_CKPT = "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth"
SAM_PKG = "/home/minh/workspace/third_party/Trident"          # vendored segment_anything (class-agnostic AMG)
GATE12_REGIONS = f"/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/regions/{DRIVE}"

QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"

# ---- experiment outputs ---------------------------------------------------------------
PROPOSALS = os.path.join(CACHE, "proposals")      # per anchor: masks, lifted support, evidence ownership
CROPS = os.path.join(CACHE, "crops")              # per region: the images Qwen was shown
QWEN_CACHE = os.path.join(CACHE, "qwen")          # per anchor: raw responses
MEMBERSHIP = os.path.join(CACHE, "membership")    # per anchor: region id per occupied voxel
PREDICTIONS = os.path.join(CACHE, "predictions")  # per anchor: the five label sets
SCORES = os.path.join(ARTIFACTS, "scores")

#: a read-only copy of the deliverables, placed beside the legacy repository's other reports on
#: explicit instruction. Writing here is the one exception to "everything under EXPERIMENT_ROOT";
#: it is a record, not an integration -- nothing under reports/ is importable by that system.
DELIVERABLE_COPY = os.path.join(LEGACY_REPO, "reports", "region_naming_pilot")

MANIFEST = os.path.join(EXPERIMENT_ROOT, "experiment_manifest.json")
RESULTS_CSV = os.path.join(EXPERIMENT_ROOT, "results.csv")
REPORT = os.path.join(EXPERIMENT_ROOT, "report.md")


def assert_isolated() -> None:
    exp = os.path.realpath(EXPERIMENT_ROOT)
    for p in (LEGACY_REPO, REPRO_REPO):
        rp = os.path.realpath(p)
        assert not exp.startswith(rp + os.sep) and exp != rp, f"experiment inside {p}"
        assert not rp.startswith(exp + os.sep), f"{p} inside the experiment"
    assert sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE"), \
        "run with PYTHONDONTWRITEBYTECODE=1: imports must not write into the legacy repo"
    assert os.environ.get("HF_HOME", "").startswith(exp), "HF_HOME must point inside the experiment"


def under_experiment(path: str) -> str:
    """Assert a path is an experiment output, create its directory, return it."""
    rp = os.path.realpath(path)
    assert rp.startswith(os.path.realpath(EXPERIMENT_ROOT) + os.sep), f"write outside the experiment: {path}"
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


assert_isolated()
