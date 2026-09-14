"""Direct semantic-correspondence audit: is the cross-view inconsistency Trident-H's, or
the 2D->3D lifting's? Physical 3D points from raw KITTI-360 LiDAR, projected with the
calibrated cameras, visibility-checked against the LiDAR itself, and the dense Trident-H
feature sampled directly in image space -- the semantic mapper is never consulted to define
a correspondence. Nothing trained.
"""
from __future__ import annotations
import os
from tools.future_semantic_completion import study as FSC
from tools.feature_first_3d_semantics import common as FF

ART = os.path.join(FSC.REPO_ROOT, "artifacts", "direct_semantic_correspondence_audit")
DATA_ROOT = os.environ.get("DSCA_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_direct_correspondence_audit")
PAIR_DIR = os.path.join(DATA_ROOT, "pairs")            # geometry-only correspondence sets
ROW_DIR = os.path.join(DATA_ROOT, "rows")              # per-pair similarity scalars
TOKEN_DIR = FF.TOKEN_DIR; TEXT_PATH = FF.TEXT_PATH; FEAT_DIM = FF.FEAT_DIM; LOGIT_SCALE = FF.LOGIT_SCALE

NATIVE_HW = (376, 1408)
TOKEN_HW = (34, 128)
LATTICE_HW = (140, 518)
#: source frames A: every k-th stream frame; partner frames B: stream offsets
A_STRIDE = {"2013_05_28_drive_0006_sync": 5, "default": 10}
B_OFFSETS = (-12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12)
N_LANDMARKS = 1500
DEPTH_MIN, DEPTH_MAX = 3.0, 50.0
BORDER_PX = 12
ZBUF_WINDOW = 7                                        # pixels, min-depth window of the sweep projection
#: visibility tolerances |z_proj - z_ref| < a + b * z  (strict is the primary)
TOLS = {"strict": (0.3, 0.01), "loose": (0.6, 0.02)}
#: local-control point: another accepted landmark of the same (A, B) set at this range
LOCAL_R = (0.3, 1.5); LOCAL_DZ = 0.3; LOCAL_DEPTH_REL = 0.10
PERTURB_EVERY = 20
PERTURB_PX = {"1 token (11 px)": 11.0, "2 px": 2.0, "5 px": 5.0}
BASELINE_EDGES = (0, 5, 15, 1e9); BASELINE_NAMES = ("<5 m", "5-15 m", ">15 m")
ANGLE_EDGES = (0, 10, 30, 1e9); ANGLE_NAMES = ("<10 deg", "10-30 deg", ">30 deg")
DEPTH_EDGES = (0, 10, 20, 35, 1e9); DEPTH_NAMES = ("3-10 m", "10-20 m", "20-35 m", "35-50 m")

require_data_root = FSC.require_data_root
sha256_file, sha256_json, write_json, environment = FSC.sha256_file, FSC.sha256_json, FSC.write_json, FSC.environment


def pair_path(drive, native_a): return os.path.join(PAIR_DIR, drive, f"{native_a:010d}.npz")
def row_path(drive, native_a): return os.path.join(ROW_DIR, drive, f"{native_a:010d}.npz")


__all__ = [k for k in dir() if not k.startswith("_")]
