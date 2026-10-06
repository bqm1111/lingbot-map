"""Six-panel 3D maps of the frozen baseline vs OccAny on identical samples, raw and pooled.

Same panel design as the Gate 8C-1 figures (``tools/gate8c1/figure_3d_compare.py``, reused for the
colours and layout), redrawn for the **current** frozen baseline under the like-for-like setting of
this comparison: OccAny's five forward frames, OccAny's stored label, OccAny's camera-to-grid
transform, and the same post-processing for both.

Scene choice is not curated. Samples are ranked by the per-sample primary difference (seed-mean IoU
of ours minus OccAny, pooled) and one sample is taken at each fixed quantile, walking to the nearest
admissible rank (a new Occ3D scene; >= 50 stream frames from other SemanticKITTI picks). Rows
therefore include samples where OccAny is better. Our volumes come from the median seed on the
primary variant; every drawn volume is re-counted and asserted equal to the evaluation shard.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
from matplotlib.patches import Patch

from core.datasets import grids as G
from core.evaluation.regions import occ_to_eval_grid
from core.model.net import load_checkpoint
from gates.gate8c1 import render3d as R3
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction import freeze as FZ
from tools.baseline_reconstruction.current.evaluate_source import predict
from tools.baseline_reconstruction.current.geometry import network_input
from tools.gate8c1.figure_3d_compare import C_BOTH, C_FP, C_GT, C_MISS, C_OCC, C_OURS, C_TP, GHOST, SOLID, compose, write_pairs
from tools.occany_comparison import config as C
from tools.occany_comparison import scoring as SC
from tools.occany_comparison.analysis import load_analysis, load_dataset
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels
from tools.occany_comparison.samples import build_samples

NAMES = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}
QUANTILES = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)
SK_MIN_GAP = 50                                   # stream positions = 250 raw frames
SK_FOV_DEG = (81.8, 29.1)                         # KITTI cam2, fx 718.8 at 1226x370
OCC3D_NATIVE_WH = (1600, 900)
VIEW = {"semantickitti": dict(eye_offset=(-34.0, -26.0, 30.0), fov=46.0),
        "occ3d": dict(eye_offset=(-26.0, -20.0, 26.0), fov=48.0)}
FRUSTUM = {"semantickitti": dict(depth=6.0, radius=0.16), "occ3d": dict(depth=10.0, radius=0.24)}
#: Gate 8C-1 wording (``tools/gate8c1/figure_3d_compare.py``), kept identical so the two generations of figures read
#: the same way: a short plain title, panel titles in its "what it invented" phrasing, and a bare sample id per row.
#: The per-sample IoU / precision / recall that an earlier version printed into each row is not lost -- it is the
#: per-sample table in ``report.md``, which is where numbers belong.
PANEL_TITLES = ("1 · ground truth", "2 · OccAny — what it recovered", "3 · ours — what it recovered",
                "4 · OccAny — + what it invented", "5 · ours — + what it invented", "6 · who recovers what")
HANDLES = [Patch(color=C_GT, label="ground truth — the shape to be reproduced"), Patch(color=C_TP, label="recovered correctly"),
           Patch(color=C_MISS, label="left out by this method"), Patch(color=C_FP, label="predicted where the ground truth says free"),
           Patch(color=C_OURS, label="only ours recovers it"), Patch(color=C_OCC, label="only OccAny recovers it"),
           Patch(color=C_BOTH, label="both recover it"), Patch(color=(0.05, 0.05, 0.05), label="camera frustum at the target frame")]


def placement(ds: str) -> str:
    return C.PRIMARY["calibration"][ds]


def median_seed(a: Dict, ds: str) -> int:
    r = a[ds]["subsets"]["occany_all"]["ours"][f"forward|{placement(ds)}|pooled"]["per_seed"]
    ious = {s: r[f"seed{s}"]["iou"] for s in C.SEEDS}
    return sorted(ious, key=ious.get)[len(ious) // 2]


def pick_samples(a: Dict, ds: str, d: Dict[str, np.ndarray]) -> List[Dict]:
    p = a[ds]["per_sample_primary"]
    assert p["sample_id"] == d["sample_id"].tolist()
    delta = np.asarray(p["ours_iou_seed_mean"]) - np.asarray(p["occany_iou"])
    order = np.argsort(delta, kind="stable"); n = len(order)
    picks: List[Dict] = []
    for q in QUANTILES:
        target = int(round(q * (n - 1)))
        for k in range(n):
            hit = None
            for r in (target + k, target - k):
                if not 0 <= r < n:
                    continue
                i = int(order[r])
                if any(x["index"] == i for x in picks):
                    continue
                if ds == "occ3d" and any(x["segment"] == str(d["segment"][i]) for x in picks):
                    continue
                if ds == "semantickitti" and any(abs(x["target_pos"] - int(d["target_pos"][i])) < SK_MIN_GAP for x in picks):
                    continue
                hit = (r, i); break
            if hit:
                r, i = hit
                picks.append({"index": i, "sample_id": str(d["sample_id"][i]), "segment": str(d["segment"][i]), "target_pos": int(d["target_pos"][i]),
                              "rank_pct": 100.0 * r / (n - 1), "delta_seed_mean": float(delta[i])})
                break
    return picks


def _mesh(mask: np.ndarray, colour, grid, shrink: float = SOLID):
    idx = np.argwhere(mask)
    return None if not len(idx) else R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)), float(grid.voxel_size), grid.origin, shrink=shrink)


RATE_KEYS = ("tp", "fp", "fn", "iou", "precision", "recall")


def _rates(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    tp = int((pred & gt & valid).sum()); fp = int((pred & ~gt & valid).sum()); fn = int((~pred & gt & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def _fov(ds: str, s) -> tuple:
    if ds == "semantickitti":
        return SK_FOV_DEG
    K = s.K_native[s.target_pos]
    return (float(np.degrees(2 * np.arctan(OCC3D_NATIVE_WH[0] / 2 / K[0, 0]))), float(np.degrees(2 * np.arctan(OCC3D_NATIVE_WH[1] / 2 / K[1, 1]))))


#: Rendered panels are cached per (dataset, sample, post) so that changing a title, a label or the legend is a
#: text-only operation: rendering is the expensive step and restyling should not pay for it twice. The cached file
#: also stores the (tp, fp, fn) that were asserted equal to the evaluation shard when the panels were drawn, so a
#: cache hit carries the same guarantee as a fresh render rather than silently dropping it.
RENDER_CACHE = os.path.join(C.DATA_ROOT, "figure_render_cache")
CACHE_VERSION = 1


def _cache_path(ds: str, sample_id: str, post: str, seed: int) -> str:
    return os.path.join(RENDER_CACHE, f"v{CACHE_VERSION}__{ds}__{sample_id}__{post}__seed{seed}.npz")


def _load_panels(ds: str, sample_id: str, post: str, seed: int, expect_occany, expect_ours):
    """Cached panels, or None. The stored counts must still match the evaluation shard."""
    path = _cache_path(ds, sample_id, post, seed)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if z["counts_occany"].tolist() != list(expect_occany) or z["counts_ours"].tolist() != list(expect_ours):
            return None                                    # evaluation moved under the cache: re-render
        return [z[f"p{i}"] for i in range(int(z["n_panels"]))], {k: float(z[f"rate_{k}"]) for k in RATE_KEYS}


def _store_panels(ds: str, sample_id: str, post: str, seed: int, imgs, counts_occany, counts_ours, rates) -> None:
    os.makedirs(RENDER_CACHE, exist_ok=True)
    path = _cache_path(ds, sample_id, post, seed); tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, n_panels=np.int32(len(imgs)), counts_occany=np.asarray(counts_occany),
                        counts_ours=np.asarray(counts_ours), **{f"p{i}": im for i, im in enumerate(imgs)},
                        **{f"rate_{k}": np.float64(v) for k, v in rates.items()})
    os.replace(tmp, path)


def render_dataset(ds: str, device: str) -> Dict:
    FZ.require_source_freeze()
    a = load_analysis(); d = load_dataset(ds)
    keys = [str(k) for k in d["keys"]]; col = {k: i for i, k in enumerate(keys)}
    seed = median_seed(a, ds)
    pgrid, egrid = G.PREDICTION_GRID[ds], G.EVAL_GRID[ds]
    edims = tuple(int(x) for x in egrid.dims)
    samples = {s.sample_id: s for s in build_samples(ds)}
    picks = pick_samples(a, ds, d)

    def expected(pk, post):
        """The (tp, fp, fn) the evaluation recorded for this sample, per method."""
        return (d["official"][pk["index"], col[f"occany|{post}"]].tolist(),
                d["official"][pk["index"], col[f"ours|forward|{placement(ds)}|{post}|seed{seed}"]].tolist())

    rows = {post: [] for post in C.POSTPROCESS}
    net = dev = None
    for pk in picks:
        sid = pk["sample_id"]; s = samples[sid]
        hits = {post: _load_panels(ds, sid, post, seed, *expected(pk, post)) for post in C.POSTPROCESS}
        if all(h is not None for h in hits.values()):                     # nothing to draw: restyle only
            for post, (imgs, rates) in hits.items():
                rows[post].append({"images": imgs, "label": sid})
                pk[f"{post}_ours"] = rates
            print(f"  [{ds}] {sid} from render cache", flush=True)
            continue
        if net is None:                                                   # load the network only if something must be drawn
            dev = torch.device(device); torch.cuda.set_device(dev)
            net = load_checkpoint(BR.CurrentPaths().checkpoint_path(seed, C.CHECKPOINT_STEP), dev)
        oc = SC.load_occany(ds, s.occany_pkl, edims)
        T = s.T_cam_to_grid_ours if ds == "semantickitti" else oc.T_cam_to_grid
        mapper = build_mapper(WindowClip.load(C.clip_path(ds, "forward", sid)), dev)
        S_, F_ = hard_channels(mapper, pgrid, T); del mapper
        ours_raw = occ_to_eval_grid(predict(net, network_input(S_, F_, pgrid.dims)), ds).reshape(edims)
        vols = {"raw": (oc.pred, ours_raw.cpu().numpy()),
                "pooled": (SC.pool(torch.from_numpy(oc.pred)).numpy(), SC.pool(ours_raw).cpu().numpy())}
        valid = oc.label != C.IGNORE
        gt = valid & (oc.label != C.OCCANY[ds].empty_class)
        g = gt & valid
        idx = np.argwhere(g)
        ctr = ((idx + 0.5) * float(egrid.voxel_size) + np.asarray(egrid.origin)).mean(0) if len(idx) else None
        fx, fy = _fov(ds, s)
        fr = R3.frustum_mesh(T, fx, fy, **FRUSTUM[ds])
        for post, (po, pu) in vols.items():
            r_o, r_u = _rates(po, gt, valid), _rates(pu, gt, valid)
            exp_o, exp_u = expected(pk, post)
            # the drawn volumes are exactly the ones the evaluation counted
            assert [r_o[k] for k in ("tp", "fp", "fn")] == exp_o, (sid, post, "occany")
            assert [r_u[k] for k in ("tp", "fp", "fn")] == exp_u, (sid, post, "ours")
            scenes = [[_mesh(g, C_GT, egrid)],
                      [_mesh(~po & g, C_MISS, egrid, GHOST), _mesh(po & g, C_TP, egrid)],
                      [_mesh(~pu & g, C_MISS, egrid, GHOST), _mesh(pu & g, C_TP, egrid)],
                      [_mesh(~po & g, C_MISS, egrid, GHOST), _mesh(po & g, C_TP, egrid), _mesh(po & ~gt & valid, C_FP, egrid)],
                      [_mesh(~pu & g, C_MISS, egrid, GHOST), _mesh(pu & g, C_TP, egrid), _mesh(pu & ~gt & valid, C_FP, egrid)],
                      [_mesh(g & po & pu, C_BOTH, egrid, 0.55), _mesh(g & pu & ~po, C_OURS, egrid), _mesh(g & po & ~pu, C_OCC, egrid)]]
            imgs = [R3.render(ms + [fr], size=(980, 700), look_at=ctr, **VIEW[ds]) for ms in scenes]
            _store_panels(ds, sid, post, seed, imgs, exp_o, exp_u, r_u)
            rows[post].append({"images": imgs, "label": sid})
            pk[f"{post}_occany"], pk[f"{post}_ours"] = r_o, r_u
        print(f"  [{ds}] rendered {sid} (Δ-rank {pk['rank_pct']:.0f} %)", flush=True)
    out = {}
    for post in C.POSTPROCESS:
        title = f"Visualization on {NAMES[ds]}" + ("" if post == "pooled" else " (no majority pooling)")
        base = os.path.join(C.FIGURE_DIR, f"5_maps_{ds}_{post}.png")
        compose(rows[post], base, title, HANDLES, titles=PANEL_TITLES, dpi=100)
        write_pairs(rows[post], base, title, HANDLES, per_fig=2, titles=PANEL_TITLES, dpi=100)
        out[post] = base
    return {"dataset": ds, "seed": seed, "quantiles": list(QUANTILES), "picks": picks, "figures": out}


def render_all(device: str) -> Dict:
    os.makedirs(C.FIGURE_DIR, exist_ok=True)
    rec = {ds: render_dataset(ds, device) for ds in C.DATASETS}
    json.dump(rec, open(os.path.join(C.FIGURE_DIR, "maps_index.json"), "w"), indent=1)
    return rec


__all__ = ["pick_samples", "median_seed", "render_dataset", "render_all"]
