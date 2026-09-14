"""Gate 11 -- uncertainty-aware observed-geometry representation. No training.

Can frozen causal monocular depth (LB-M = s_clip x z_LingBot) become useful 0.2 m geometry if a
predicted depth is an interval along the ray rather than an exact surface?

Fixed from Gate 9 / the depth-source comparison: five-frame local-reset clips, KITTI-360 GT relative
poses, lattice camera, anchor SSC grid, reference and evaluation mask, LiDAR z-buffer support, frozen
LingBot confidence gate (>= 1.5). Rays = Gate 9 support AND gate (Gate 9 "PD_gate" / "LD_gate").

Per voxel, four states (predeclared resolution, identical for every condition):
  S  surface evidence : Gate 9 mapper log-odds over the +-0.2 m band samples and carving samples > 0
  F  definite free    : that log-odds <= 0 and free weight > 0            (exactly the mapper's free)
  U  uncertain        : not S, not F, and inside at least one ray's uncertainty interval
  -  unknown          : no evidence
Carving: "conservative" stops each ray at z_pred - u_front; "hard" stops at z_pred - 0.2 m (current).
H is the unchanged mapper (no interval, hard carving) == Gate 9 PD_gate.
Binary conversions for secondary metrics: occupied := S (primary); optimistic occupied := S or U.
"""
from __future__ import annotations
import os
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G9_DIR = os.path.join(REPO_ROOT, "tools", "gate9_depth_occupancy")
ART = os.environ.get("G11_ART", os.path.join(REPO_ROOT, "artifacts", "gate11_uncertainty"))   # override only for verification re-derivation
DATA_ROOT = os.environ.get("G11_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate11_uncertainty")
os.environ.setdefault("G8C1_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives")
PRIMARY = "2013_05_28_drive_0006_sync"
ROBUST = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
RULES = os.path.join(ART, "uncertainty_rules.json")

# ---- predeclared width-derivation rules (depth residuals only; written before any occupancy run) ----
DERIVATION = {
    "data": "Gate 9 0006 pixel rows (deterministic 20 % subsample of LiDAR support) with LingBot conf >= 1.5",
    "residual": "r = (z_LiDAR - z_LBM) x |ray|  (metres along the ray; r > 0 = true surface behind the prediction)",
    "coverage_quantile": 0.68,
    "u_min_m": 0.2,
    "U2": "u2(z) = max(u_min, a + b z), z = predicted optical-axis depth; (a, b) = count-weighted least squares of per-bin P68(|r|) "
          "on per-bin mean z over 2 m z bins from 1 to 41 m plus [41, 60) m, bins with >= 1000 rows",
    "U3": "u3(z, c) = max(u_min, u2(z) g_k), k = LingBot confidence bucket [1.5,3) [3,6) [6,12) [12,inf) (Gate 9 bins); "
          "g_k = P68(|r| / u2(z)) in bucket k, then made non-increasing in confidence by count-weighted pool-adjacent-violators",
    "U4": "q16, q84 = quantiles of r / u3(z, c); include U4 only if max(|q16|,|q84|) / min(|q16|,|q84|) >= 1.5; "
          "u_front = max(u_min, -q16 u3), u_back = max(u_min, q84 u3)",
    "U1": "fixed u in {0.2, 0.4, 0.6} m (given)",
}
U1_WIDTHS = (0.2, 0.4, 0.6)
CONF_EDGES = (1.5, 3.0, 6.0, 12.0, np.inf)
ZBIN_EDGES = tuple(float(x) for x in np.r_[np.arange(1, 42, 2), 60.0])
RANGE_BINS = (0, 5, 10, 20, 30, 1e9); RANGE_NAMES = ("0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m")   # Gate 9 bins
EDGE_NAMES = {0: "non-edge (<5 %)", 1: "moderate (5-20 %)", 2: "strong (>=20 %)", 3: "isolated"}

# ---- predeclared decision guidance (fixed before any occupancy number) ----
DECISION_GUIDE = {
    "substantial_coverage_gain": "band coverage (S or U contains the LiDAR-occupied voxel) >= H surface coverage + 0.15",
    "excessive_uncertainty": "uncertain voxels > 25 % of the evaluation mask",
    "useful": "band coverage >= 0.50 AND containment efficiency >= H precision AND false-free rate <= H AND uncertain <= 25 % of mask",
    "useful_in_range_bin": "the same three criteria inside a range bin (voxel range = anchor-camera z of the voxel centre)",
    "confidence_benefit": "U3 uncertain volume <= 0.9 x U2 with band coverage loss <= 0.02",
    "range_benefit": "U2 containment efficiency above the U1 curve linearly interpolated at U2's band coverage",
    "carving_benefit": "conservative false-free rate <= 0.8 x hard false-free rate for the same width rule",
    "consistency": "the same qualitative outcome on 0006, 0003, 0007 and 0010",
}


def conditions(rules, with_ld):
    """name -> (depth source, width rule, carving). Order is fixed."""
    c = {"H": ("LBM", None, "hard")}
    rs = [f"U1_{u}" for u in U1_WIDTHS] + ["U2", "U3"] + (["U4"] if rules["U4"]["include"] else [])
    for r in rs:
        c[r] = ("LBM", r, "conservative")
    for r in rs:
        if r != "U1_0.2":                                   # margin 0.2 either way: hard == conservative
            c[r + "h"] = ("LBM", r, "hard")
    if with_ld:
        c["H_LD"] = ("lidar", None, "hard")
        for r in rs:
            c[r + "_LD"] = ("lidar", r, "conservative")
    return c


def widths(rule, z, conf, rules):
    """(u_front, u_back) in metres along the ray for predicted optical-axis depth z and LingBot confidence."""
    z = np.asarray(z, np.float64); zero = np.zeros_like(z)
    if rule is None:
        return zero, zero
    umin = rules["u_min_m"]
    if rule.startswith("U1_"):
        u = np.full_like(z, float(rule[3:])); return u, u
    u2 = np.maximum(umin, rules["U2"]["a"] + rules["U2"]["b"] * np.nan_to_num(z))
    if rule == "U2":
        return u2, u2
    k = np.clip(np.digitize(np.asarray(conf, np.float64), CONF_EDGES[1:-1]), 0, 3)
    u3 = np.maximum(umin, u2 * np.asarray(rules["U3"]["g"])[k])
    if rule == "U3":
        return u3, u3
    return np.maximum(umin, -rules["U4"]["q16"] * u3), np.maximum(umin, rules["U4"]["q84"] * u3)


__all__ = [k for k in dir() if not k.startswith("_")]
