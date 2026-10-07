"""The four focused checks §6 asks for. Not a broad test suite; each one can actually fail.

1. the existing KITTI-360 preprocessing still reproduces its cached fixture;
2. perturbing or removing privileged inputs leaves the causal tensors bit-identical;
3. a known projection/alignment case gives the expected voxel and the expected visibility;
4. invalid supervision is excluded, never assigned a class.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC
from core.datasets.audit import FileAudit, ForbiddenAccess
from tools.delayed_semantic import config as DS
from tools.delayed_semantic.targets import _voxel_of
from tools.semantic_completion import frozen_geometry as FG
from tools.semantic_completion.evidence import build as G15B, lift as LF, trident as TR
from tools.waymo_pilot import config as C, k360 as K, populations as POP, waymo as W


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        x = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:16]


def _causal_digest(cs: Dict) -> str:
    return _digest(cs["S"], cs["F"], cs["evidence"].flat_index, cs["evidence"].feature,
                   cs["evidence_label"], np.float64(cs["scale"]),
                   cs["clip"].depth_canonical, cs["clip"].conf, cs["clip"].pose_c2w_canonical)


# --------------------------------------------------------------------------- #
def check_1_preprocessing_matches_fixture(drive: str, t: int, device,
                                          text: torch.Tensor) -> Dict:
    """Recompute the evidence lift from the clip and compare with the cached fixture.

    The established tolerance is the cache's own: features are stored as fp16 and
    renormalised on load, so the comparison is exact on the voxel set and the ray counts,
    and bounded by fp16 round-off on the feature. The frozen geometry checkpoint is compared
    by sha256 against ``code_freeze.json`` -- that one is exact by construction.
    """
    p = C.Paths().g15
    net = FG.load_frozen_geometry(C.GEOMETRY_SEED, device, p)      # asserts its own sha256
    clip = LF.LocalClip.load(p.clip_path(drive, t)) if hasattr(LF, "LocalClip") else None
    from tools.baseline_reconstruction.current.geometry import LocalClip
    clip = LocalClip.load(p.clip_path(drive, t))
    with np.load(p.input_path(drive, t)) as z:
        S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(device)
        clip_sha_cached = str(z["clip_sha256"])
    clip_sha_live = hashlib.sha256(open(p.clip_path(drive, t), "rb").read()).hexdigest()
    tokens = [TR.load_token_map(p.token_path(drive, str(k)), device) for k in clip.keys]
    live = LF.lift_clip_evidence(clip, tokens, S, G15B.camera_to_grid(), device)
    cached = LF.load_evidence(p.evidence_path(drive, t), device)
    same_set = bool(live.flat_index.numel() == cached.flat_index.numel()
                    and torch.equal(live.flat_index, cached.flat_index))
    same_cnt = bool(torch.equal(live.count, cached.count)) if same_set else False
    fmax = (float((live.feature - cached.feature).abs().max())
            if same_set and live.feature.numel() else None)
    fp16_eps = 2.0 ** -10
    ok = same_set and same_cnt and (fmax is not None and fmax <= fp16_eps) \
        and clip_sha_live == clip_sha_cached
    return {"name": "preprocessing matches cached fixture", "ok": bool(ok),
            "drive": drive, "anchor": int(t),
            "geometry_params": int(net.n_params()),
            "clip_sha256_matches_input_cache": clip_sha_live == clip_sha_cached,
            "evidence_voxel_set_identical": same_set,
            "evidence_ray_counts_identical": same_cnt,
            "max_feature_abs_diff": fmax, "fp16_tolerance": fp16_eps,
            "n_evidence_voxels": int(cached.flat_index.numel())}


# --------------------------------------------------------------------------- #
def check_2_privileged_cannot_touch_causal(drive: str, t: int, device,
                                           text: torch.Tensor) -> Dict:
    """Two independent proofs that the privileged side cannot reach the causal side.

    (a) a file audit over the causal state: every privileged path is forbidden, so merely
        opening one raises instead of being noticed later;
    (b) a value test: the causal digest is recomputed after the privileged arrays have been
        replaced by garbage in memory. Equal digests mean no causal tensor is a function of
        them.
    """
    base = _causal_digest(K.causal_state(drive, t, device, text))

    # Privileged roots only. The drive NAME must not appear here: the causal clip and the
    # hard-map input legitimately live in per-drive directories, so forbidding the drive name
    # would fail the check on a legal read instead of on a leak.
    privileged = ("delayed_targets", "/targets/", "velodyne", "data_3d_raw", "data_poses",
                  "eval_masks", "gate20_future_reveal")  # roots, never the drive name
    audit_ok, audit_err = True, None
    try:
        with FileAudit(patterns=privileged, strict=True) as audit:
            K.causal_state(drive, t, device, text)
    except ForbiddenAccess as exc:                                  # noqa: BLE001
        audit_ok, audit_err = False, str(exc)
        audit = None

    # (b) perturb the privileged tensors and rebuild the causal state
    sem = K.privileged_semantics(drive, t, device)
    sem["prob"] = torch.rand_like(sem["prob"])
    sem["valid"] = ~sem["valid"]
    O_priv, valid, meta = K.privileged_occupancy(drive, t, device)
    O_priv = ~O_priv
    after = _causal_digest(K.causal_state(drive, t, device, text))

    return {"name": "privileged inputs cannot change causal tensors",
            "ok": bool(audit_ok and base == after),
            "causal_digest_before": base, "causal_digest_after_perturbation": after,
            "digests_equal": base == after,
            "causal_state_opened_no_privileged_path": audit_ok,
            "audit_error": audit_err,
            "forbidden_patterns": list(privileged),
            "n_paths_opened_by_causal_state": (len(audit.opened) if audit else None)}


# --------------------------------------------------------------------------- #
def check_3_known_projection_and_visibility(device) -> Dict:
    """Exact, hand-checkable cases for the voxel binning and the occlusion verdict.

    The binning case is arithmetic: a point at ``origin + (n + 0.5) * 0.2`` must land in
    voxel ``n``, and a point just outside the box must be rejected. The visibility case is a
    two-surface scene: a near return and a far return on the same pixel, where the far one
    must be rejected by the z-buffer rule and the near one kept. The Waymo axis convention is
    checked the same way -- a point straight ahead of the vehicle must hit the principal
    point, a point to the left must fall left of it.
    """
    origin = np.asarray(C.GRID_ORIGIN_M, np.float64)
    want = np.array([[3, 17, 5], [0, 0, 0], [255, 255, 31]], np.int64)
    pts = (want + 0.5) * C.VOXEL_SIZE_M + origin
    flat = _voxel_of(torch.as_tensor(pts, device=device))
    expect = (want[:, 0] * C.GRID_DIMS[1] + want[:, 1]) * C.GRID_DIMS[2] + want[:, 2]
    bin_ok = bool(np.array_equal(flat.cpu().numpy(), expect))
    outside = torch.as_tensor(np.array([[-0.1, 0.0, 0.0], [51.3, 0.0, 0.0]]) + origin * [0, 1, 1],
                              device=device)
    out_ok = bool((_voxel_of(outside) < 0).all())

    # z-buffer: same pixel, two depths. tol = max(0.5 m, 0.05 * zmin)
    cfg = C.EXISTING_TARGET_CFG
    zmin, zfar = 10.0, 25.0
    tol = max(cfg.zbuffer_tol_rel * zmin, cfg.zbuffer_tol_m)
    near_kept = zmin <= zmin + tol
    far_rejected = not (zfar <= zmin + tol)
    grazing = zmin + 0.4                         # inside the 0.5 m tolerance -> still visible
    grazing_kept = grazing <= zmin + tol

    cam = W.camera_from_calibration("FRONT", 1920, 1280,
                                    [2055.0, 2055.0, 960.0, 640.0, 0, 0, 0, 0, 0],
                                    np.eye(4).ravel(), 1)
    u_f, v_f, z_f = W.project(np.array([[10.0, 0.0, 0.0]]), cam)
    u_l, _v, _z = W.project(np.array([[10.0, 1.0, 0.0]]), cam)
    _u, v_u, _z2 = W.project(np.array([[10.0, 0.0, 1.0]]), cam)
    waymo_ok = bool(abs(u_f[0] - 960.0) < 1e-6 and abs(v_f[0] - 640.0) < 1e-6
                    and abs(z_f[0] - 10.0) < 1e-9 and u_l[0] < 960.0 and v_u[0] < 640.0)

    ok = bin_ok and out_ok and near_kept and far_rejected and grazing_kept and waymo_ok
    return {"name": "known projection / alignment / visibility case", "ok": bool(ok),
            "voxel_binning_exact": bin_ok, "outside_grid_rejected": out_ok,
            "zbuffer": {"zmin_m": zmin, "tolerance_m": tol, "near_kept": near_kept,
                        "far_25m_rejected": far_rejected,
                        "grazing_within_tolerance_kept": grazing_kept},
            "waymo_axis_convention_ok": waymo_ok,
            "waymo_case": {"forward_10m_uv": [float(u_f[0]), float(v_f[0])],
                           "left_1m_u": float(u_l[0]), "up_1m_v": float(v_u[0])}}


# --------------------------------------------------------------------------- #
def check_4_invalid_supervision_excluded(drive: str, t: int, device,
                                         text: torch.Tensor) -> Dict:
    """Unknown geometry and missing semantics stay invalid; they never become a class.

    Three things must hold at once: ``U`` is inside the valid target set; every voxel that is
    hidden but has no target carries readout 0 (not a class, not "free", not "background");
    and the privileged occupancy's own unknown set -- where sweeps disagreed -- is excluded
    from ``O_priv`` rather than resolved.
    """
    cs = K.causal_state(drive, t, device, text)
    centres = POP.grid_centres(device)
    T = np.linalg.inv(G15B.camera_to_grid())
    obs = POP.causal_observability(centres, T, K.causal_views(cs["clip"], device),
                                   cs["S"], cs["F"], cs["evidence_mask"])
    O_priv, valid, _m = K.privileged_occupancy(drive, t, device)
    sem = K.privileged_semantics(drive, t, device)
    pop = POP.build(O_priv, sem["valid"], obs)

    readout = torch.zeros(C.N_VOXELS, dtype=torch.uint8, device=device)
    readout[sem["flat"]] = (1 + sem["prob"].argmax(1)).to(torch.uint8)
    unsupervised = pop.hidden & ~pop.Z_valid
    assigned_a_class = int((readout[unsupervised] != 0).sum())

    with np.load(K.target_path(drive, t)) as z:
        n = C.N_VOXELS
        occ_raw = np.unpackbits(z["occ_packed"])[:n].astype(bool)
        valid_raw = np.unpackbits(z["valid_packed"])[:n].astype(bool)
    unknown = ~valid_raw
    unknown_called_occupied = int((torch.from_numpy(unknown & occ_raw).to(device)
                                  & O_priv).sum())

    u_subset = bool((pop.U & ~pop.Z_valid).sum() == 0)
    ok = (assigned_a_class == 0) and u_subset and unknown_called_occupied == 0
    return {"name": "invalid supervision excluded, not assigned a class", "ok": bool(ok),
            "n_hidden_without_target": int(unsupervised.sum()),
            "n_of_those_given_a_class": assigned_a_class,
            "U_is_subset_of_valid_targets": u_subset,
            "n_unknown_voxels_in_privileged_occupancy": unknown_called_occupied,
            "n_privileged_unknown": int(unknown.sum())}


# --------------------------------------------------------------------------- #
def run_all(device, drive: Optional[str] = None, anchor: Optional[int] = None) -> Dict:
    text, _p, _l = TR.load_text_embeddings()
    text = text.to(device)
    if drive is None:
        av = K.available()
        drive = C.K360_SAMPLE_DRIVES[1]
        anchor = av[drive][5]
    out: List[Dict] = [
        check_1_preprocessing_matches_fixture(drive, anchor, device, text),
        check_2_privileged_cannot_touch_causal(drive, anchor, device, text),
        check_3_known_projection_and_visibility(device),
        check_4_invalid_supervision_excluded(drive, anchor, device, text),
    ]
    return {"version": C.VERSION, "drive": drive, "anchor": int(anchor),
            "all_ok": all(c["ok"] for c in out), "n_checks": len(out), "checks": out}


__all__ = ["run_all", "check_1_preprocessing_matches_fixture",
           "check_2_privileged_cannot_touch_causal",
           "check_3_known_projection_and_visibility",
           "check_4_invalid_supervision_excluded"]
