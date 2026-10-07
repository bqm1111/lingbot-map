"""The KITTI-360 reference arm: the existing recipe, measured on the pilot's populations.

Nothing here is a new supervision recipe. The privileged semantic target is the cached
output of ``tools/delayed_semantic/targets.py`` -- the existing temporal recipe, with its
own static-support and conflict rules -- and the privileged occupancy is the cached raw
LiDAR target of ``core/supervision/rawtarget.py``. This module reads both, rebuilds the
causal observability state from the causal clip alone, and reports the §4/§5 decomposition.

Causal side (allowed at deployment): the five-frame local-reset LingBot clip ``t-4..t`` of
``image_00``, its causal MoGe-2 clip scale, the frozen hard map ``(S, F)``, the frozen
Trident-H evidence lifted from those five frames, and the calibration. Nothing else.

Privileged side (training only): raw Velodyne sweeps, KITTI-360 ground-truth poses and the
future frames of the frozen window. Human annotations are never opened here; the one
annotation-assisted number in this pilot lives in ``diagnostic.py`` and is marked there.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.datasets import kitti360_folds  # noqa: F401  registers the k360_train source
from core.datasets import kitti360_splits as SRC
from tools.baseline_reconstruction import config as BR
from core.mapping import depth as D7
from core.run.build_targets import load as load_raw_target
from tools.baseline_reconstruction.current.geometry import LocalClip, frame_inputs
from tools.delayed_semantic import config as DS
from tools.semantic_completion.evaluation import baseline_nearest as BN
from tools.semantic_completion.evidence import build as G15B, lift as LF, trident as TR
from tools.waymo_pilot import config as C, populations as POP, provenance as PR

ARM = "k360"
VARIANT = "k360_reference"


def target_path(drive: str, t: int) -> str:
    """The frozen baseline's own occupancy-target root.

    Not ``kitti360_splits.target_path``: that points at the superseded Gate 8C-1 cache,
    which predates the open-sky fix and covers only four drives. The two caches agree
    exactly on ``occupied`` -- the fix changed which voxels are *resolved*, not which are
    full -- so ``O_priv`` is the same either way, but the current root is the one the frozen
    baseline trains against and it is the only one that covers all seven training drives.
    """
    return BR.CurrentPaths().target_path(drive, t)


def has_clip(drive: str, t: int) -> bool:
    """Is the local-reset causal clip on disk for this anchor?

    Only the three frozen training drives have one. The other four have hard-map inputs,
    lifted evidence and both privileged targets cached, but their clips were not retained,
    so their causal frusta cannot be rebuilt without re-running LingBot. Coverage statistics
    do not need the clip; the §4 sub-case decomposition does.
    """
    p = C.Paths().g15
    return os.path.exists(p.clip_path(drive, t))


def eligible_anchors(drive: str, require_clip: bool = True) -> List[int]:
    """Anchors with every cache this arm needs. Opens no target.

    With ``require_clip=False`` an anchor qualifies on the coverage caches alone, which is
    what the scene-count control uses: it measures how much supervision each scene yields,
    and reports the sub-case decomposition as unavailable rather than guessing it.
    """
    p = C.Paths().g15
    ids = G15B.anchor_ids(drive, p)
    out = []
    for t in ids:
        if require_clip and not has_clip(drive, t):
            continue
        # the coverage caches: hard-map input, lifted evidence, privileged occupancy and
        # privileged semantic target.
        if (os.path.exists(p.input_path(drive, t))
                and os.path.exists(p.evidence_path(drive, t))
                and os.path.exists(target_path(drive, t))
                and os.path.exists(DS.Paths().target_path(drive, t))):
            out.append(t)
    return out


def available(drives: Optional[Sequence[str]] = None,
              require_clip: bool = True) -> Dict[str, List[int]]:
    return {d: eligible_anchors(d, require_clip)
            for d in (drives or C.K360_SAMPLE_DRIVES)}


# --------------------------------------------------------------------------- #
# causal side
# --------------------------------------------------------------------------- #
def causal_views(clip: LocalClip, device) -> List[Tuple[np.ndarray, np.ndarray,
                                                        torch.Tensor, torch.Tensor]]:
    """The five allowed views in the anchor camera's METRIC frame.

    The scale, the acceptance gate and the pose rebasing are the frozen ones: ``s`` is the
    clip's own causal MoGe-2 scale, applied once to depth and once to the pose translation,
    and acceptance is ``conf >= 1.5 & 1 m < s*z < 60 m`` -- byte-identical to
    ``IncrementalMapper._integrate`` and to the evidence lifter.
    """
    s = float(clip.scale_m_per_unit)
    out = []
    for f in frame_inputs(clip, device):
        T = np.asarray(f.pose_c2w_canonical, np.float64).copy()
        T[:3, 3] *= s
        d_m = (s * f.depth_canonical.to(torch.float64))
        acc = (f.conf.to(torch.float64) >= D7.CONF_THRESHOLD) & torch.isfinite(d_m) \
            & (d_m > D7.MIN_DEPTH_M) & (d_m < D7.MAX_DEPTH_M)
        out.append((T, np.asarray(f.K, np.float64), d_m, acc))
    return out


def coverage_state(drive: str, t: int, device, text: torch.Tensor) -> Dict:
    """The causal side WITHOUT the clip: hard map and cached evidence only.

    Enough for ``hidden`` and ``U``; not enough for the frustum or the occlusion test, which
    is why ``measure_anchor`` marks ``subcases_available = False`` for these anchors instead
    of filling the four sub-cases with a default.
    """
    p = C.Paths().g15
    with np.load(p.input_path(drive, t)) as z:
        S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(device)
        F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:C.N_VOXELS].astype(bool)).to(device)
    ev = LF.load_evidence(p.evidence_path(drive, t), device)
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    ev_label[ev.flat_index] = TR.classify(ev.feature, text).long()
    return {"clip": None, "S": S, "F": F_, "evidence": ev, "evidence_mask": ev.dense_mask(),
            "evidence_label": ev_label, "scale": float("nan")}


def causal_state(drive: str, t: int, device, text: torch.Tensor) -> Dict:
    """Everything the deployed system has at ``t``: clip, hard map, evidence, readout."""
    p = C.Paths().g15
    clip = LocalClip.load(p.clip_path(drive, t))
    with np.load(p.input_path(drive, t)) as z:
        S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(device)
        F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:C.N_VOXELS].astype(bool)).to(device)
    ev = LF.load_evidence(p.evidence_path(drive, t), device)
    ev_mask = ev.dense_mask()
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    ev_label[ev.flat_index] = TR.classify(ev.feature, text).long()
    return {"clip": clip, "S": S, "F": F_, "evidence": ev, "evidence_mask": ev_mask,
            "evidence_label": ev_label, "scale": float(clip.scale_m_per_unit)}


# --------------------------------------------------------------------------- #
# privileged side -- cached, not recomputed
# --------------------------------------------------------------------------- #
def privileged_occupancy(drive: str, t: int, device) -> Tuple[torch.Tensor, torch.Tensor,
                                                              Dict]:
    """``(O_priv, valid, meta)`` from the cached raw-LiDAR occupancy target.

    ``valid`` is the sweeps' own resolved set: a voxel occupied in one sweep and traversed
    free by another is UNKNOWN there and therefore not ``valid`` -- the existing rule's way
    of refusing to adjudicate dynamic disagreement. ``O_priv = valid & occupied``.
    """
    d = load_raw_target(target_path(drive, t))
    valid = torch.from_numpy(d["valid"]).to(device)
    occ = torch.from_numpy(d["occupied"]).to(device)
    n_fr = torch.from_numpy(d["n_frames_occ"].astype(np.int16)).to(device)
    meta = {"n_sweeps_used": int(len(d["future_natives"]) + 1),
            "native_frame": d["native_frame"]}
    return valid & occ, valid, {"support": n_fr, **meta}


def privileged_semantics(drive: str, t: int, device) -> Dict:
    """The cached existing-recipe delayed semantic target, as dense masks plus provenance."""
    with np.load(DS.Paths().target_path(drive, t)) as z:
        flat = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device)
        prob = torch.from_numpy(z["prob"].astype(np.float32)).to(device)
        fields = {k: torch.from_numpy(np.asarray(z[k])).to(device) for k in
                  ("n_views", "first_dt_s", "second_dt_s", "view0_top1", "view1_top1",
                   "tier", "in_evidence", "hist_seen", "n_future_sweeps")}
        stats = json.loads(str(z["stats_json"]))
    valid = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=device)
    valid[flat] = True
    return {"flat": flat, "prob": prob, "valid": valid, "stats": stats, **fields}


# --------------------------------------------------------------------------- #
# one anchor
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_anchor(drive: str, t: int, device, text: torch.Tensor,
                   centres: Optional[torch.Tensor] = None) -> Dict:
    """The full §4/§5 record of one KITTI-360 anchor under the existing recipe."""
    if centres is None:
        centres = POP.grid_centres(device)
    full = has_clip(drive, t)
    if full:
        cs = causal_state(drive, t, device, text)
        obs = POP.causal_observability(centres, np.linalg.inv(G15B.camera_to_grid()),
                                       causal_views(cs["clip"], device),
                                       cs["S"], cs["F"], cs["evidence_mask"])
    else:
        cs = coverage_state(drive, t, device, text)
        obs = POP.unknown_observability(cs["S"], cs["F"], cs["evidence_mask"])
    O_priv, priv_valid, occ_meta = privileged_occupancy(drive, t, device)
    sem = privileged_semantics(drive, t, device)
    pop = POP.build(O_priv, sem["valid"], obs)

    nearest_flat, dist_m = BN.nearest_evidence(cs["evidence_mask"])
    dist = torch.from_numpy(dist_m).to(device)
    rng = centres[:, :2].norm(dim=1).float()
    b0 = BN.nearest_label(cs["evidence_label"], nearest_flat)

    # readout of the privileged target, dense over its own rows only
    readout = torch.zeros(C.N_VOXELS, dtype=torch.uint8, device=device)
    readout[sem["flat"]] = (1 + sem["prob"].argmax(1)).to(torch.uint8)

    prov = PR.summarize_single_source(sem, pop.U, C.SOURCE_FUTURE)
    rec = {
        "arm": ARM, "variant": VARIANT, "scene": drive, "anchor": int(t),
        "scale_m_per_unit": cs["scale"],
        "subcases_available": bool(full),
        "causal": obs.stats(),
        "privileged": {"n_valid": int(priv_valid.sum()), **{k: v for k, v in
                                                            occ_meta.items()
                                                            if k != "support"}},
        "counts": pop.counts(),
        "clip_span_s": _clip_span(sem["stats"]),
        "exclusions": _exclusions(sem["stats"]),
        "provenance": prov,
        "distance": _binned(dist[pop.U], C.DISTANCE_EDGES_M, C.DISTANCE_NAMES),
        "distance_hidden": _binned(dist[pop.hidden], C.DISTANCE_EDGES_M, C.DISTANCE_NAMES),
        "range": _binned(rng[pop.U], C.RANGE_EDGES_M, C.RANGE_NAMES),
        "range_hidden": _binned(rng[pop.hidden], C.RANGE_EDGES_M, C.RANGE_NAMES),
        "readout_histogram": _class_hist(readout[pop.U]),
        "b0_histogram": _class_hist(b0[pop.U].to(torch.uint8)),
        "agreement": PR.agreement(sem, pop.U),
        "repeat_vs_new": PR.repeat_vs_new(sem, pop.U),
    }
    return rec


def _clip_span(stats: Dict) -> Dict:
    """The causal clip's measured temporal extent, read from the cache's own frame table."""
    dts = [f["dt_s"] for f in stats["frames"] if not f["future"]]
    fut = [f["dt_s"] for f in stats["frames"] if f["future"]]
    return {"causal_first_dt_s": min(dts) if dts else None,
            "causal_span_s": (max(dts) - min(dts)) if dts else None,
            "n_causal_frames": len(dts), "n_future_frames": len(fut),
            "future_last_dt_s": max(fut) if fut else None,
            "stream_dt_s": (abs(min(dts)) / (len(dts) - 1)) if len(dts) > 1 else None}


def _exclusions(stats: Dict) -> Dict:
    """Why a future-hit voxel did not become a reliable target, in the recipe's own terms."""
    return {"n_future_hit": int(stats["n_future_hit"]),
            "n_conflict_dynamic": int(stats["n_conflict"]),
            "n_not_static_1s": int(stats["n_future_hit_not_static"]),
            "n_tierA": int(stats["n_tierA"]), "n_tierB": int(stats["n_tierB"]),
            "n_target": int(stats["n_target"])}


def _binned(values: torch.Tensor, edges: Sequence[float], names: Sequence[str]) -> Dict:
    v = values.detach().float().cpu().numpy()
    idx = np.digitize(v, np.asarray(edges, np.float64))
    return {n: int((idx == i).sum()) for i, n in enumerate(names)}


def _class_hist(labels: torch.Tensor) -> Dict[str, int]:
    h = torch.bincount(labels.reshape(-1).long(), minlength=C.N_CLASSES + 1).cpu().numpy()
    return {C.CLASS_NAMES[i - 1]: int(h[i]) for i in range(1, C.N_CLASSES + 1)}


__all__ = ["ARM", "VARIANT", "target_path", "has_clip", "eligible_anchors", "available",
           "causal_views", "causal_state", "coverage_state",
           "privileged_occupancy", "privileged_semantics", "measure_anchor"]
