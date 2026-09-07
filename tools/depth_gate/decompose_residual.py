#!/usr/bin/env python
"""Gate 2 diagnostic — what did the `depth_cnn` residual head actually learn?

The Gate-2 head predicts a dense log residual ``r`` and applies ``D = s0 * D_lingbot * exp(r)``.
A single per-clip scalar hidden inside ``r`` is indistinguishable from a metric-scale
correction, so this tool splits ``r`` into that scalar and the rest::

    a_clip     = median over the clip's unchanged fusion support of r[t, p]
    r_shape    = r - a_clip                      (median-zero by construction)
    s_learned  = s0 * exp(a_clip)

and re-runs the *frozen* Gate-2 occupancy and depth protocols over ten configurations that
turn the two components on and off independently, and — crucially — also fix the
pose-scale coupling that the original Gate-2 implementation left inconsistent
(C1 scales depth by ``s0*exp(r)`` while still scaling pose translation by ``s0``).

No training happens here. The checkpoint, preprocessing, normalisation statistics, clip
manifests, calibration, voxeliser and evaluation mask are all reused verbatim.

    python tools/depth_gate/decompose_residual.py --run depth_cnn
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci
from gates.depth_gate.metrics import binned_metrics, depth_metrics
from gates.depth_gate.models import build_inputs, build_model

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "geometry_gate"))
from factorize import grid_fractions, relative_metric                       # noqa: E402


# --------------------------------------------------------------------------- #
# Configurations.  depth = s_depth * D_lingbot * exp(r_used);  translation = s_pose * t.
# `scale` names the scalar multiplying BOTH depth and (for predicted poses) translation,
# except where the two deliberately disagree (C1, C2) — that disagreement is the thing
# under test.
# --------------------------------------------------------------------------- #
CONFIGS = {
    "C0": dict(depth_scale="const",   resid="none",  pose="pred", pose_scale="const",
               label="constant coupled baseline (deployable)"),
    "C1": dict(depth_scale="const",   resid="full",  pose="pred", pose_scale="const",
               label="existing Gate-2 refinement (depth-only, pose left at s0)"),
    "C2": dict(depth_scale="learned", resid="none",  pose="pred", pose_scale="const",
               label="scalar correction, depth only"),
    "C3": dict(depth_scale="learned", resid="none",  pose="pred", pose_scale="learned",
               label="scalar correction, correctly coupled"),
    "C4": dict(depth_scale="const",   resid="shape", pose="pred", pose_scale="const",
               label="shape residual under the deployable constant"),
    "C5": dict(depth_scale="learned", resid="shape", pose="pred", pose_scale="learned",
               label="full coupled correction (depth identical to C1)"),
    "O0": dict(depth_scale="oracle",  resid="none",  pose="pred", pose_scale="oracle",
               label="oracle-scale base"),
    "O1": dict(depth_scale="oracle",  resid="shape", pose="pred", pose_scale="oracle",
               label="oracle scale + shape residual"),
    "G0": dict(depth_scale="oracle",  resid="none",  pose="gt",   pose_scale="none",
               label="oracle scale + GT poses"),
    "G1": dict(depth_scale="oracle",  resid="shape", pose="gt",   pose_scale="none",
               label="oracle scale + shape residual + GT poses"),
}

CONTRASTS = [
    ("C0", "C1", "original reported improvement"),
    ("C0", "C2", "scalar correction, depth only"),
    ("C0", "C3", "scalar correction, correctly coupled"),
    ("C0", "C4", "shape only under the deployable constant"),
    ("C3", "C5", "added shape contribution under the learned scale"),
    ("C1", "C5", "effect of correcting the pose-scale coupling"),
    ("O0", "O1", "pure shape contribution with oracle scale"),
    ("G0", "G1", "pure shape contribution with oracle scale and GT poses"),
]

REPRO_TARGETS = {"C0": 0.0573, "C1": 0.0775, "O0": 0.0769, "G0": 0.0782}
REPRO_TOL = 5e-4


# --------------------------------------------------------------------------- #
# Pose scaling — applied to camera centres relative to the clip anchor, BEFORE any
# world / velodyne alignment.  See assertions in `check_pose_scaling`.
# --------------------------------------------------------------------------- #
def scaled_relative_pose(pose_c2w: np.ndarray, f: int, anchor: int, s: float) -> np.ndarray:
    """Transform camera ``f`` -> camera ``anchor`` after scaling camera centres by ``s``.

        c_scaled[t] = c_anchor + s * (c_canonical[t] - c_anchor)
        R_scaled[t] = R[t]

    The anchor is a fixed point of that map, so ``s`` never moves it, and the scaling is
    done in the clip's own canonical frame; the SemanticKITTI world/velodyne transform is
    applied only afterwards, to the fused points.
    """
    Pa, Pf = pose_c2w[anchor], pose_c2w[f]
    ca = Pa[:3, 3]
    Pf_s = Pf.copy()
    Pf_s[:3, 3] = ca + s * (Pf[:3, 3] - ca)
    return np.linalg.inv(Pa) @ Pf_s


def check_pose_scaling(pose_c2w: np.ndarray, anchor: int, s: float) -> None:
    """Assert the four required properties of the pose-scaling implementation."""
    ident = scaled_relative_pose(pose_c2w, anchor, anchor, s)
    assert np.allclose(ident, np.eye(4), atol=1e-9), "anchor position/orientation moved"
    for f in range(len(pose_c2w)):
        r1 = scaled_relative_pose(pose_c2w, f, anchor, 1.0)
        rs = scaled_relative_pose(pose_c2w, f, anchor, s)
        assert np.allclose(rs[:3, :3], r1[:3, :3], atol=1e-12), "rotation was scaled"
        n1, ns = np.linalg.norm(r1[:3, 3]), np.linalg.norm(rs[:3, 3])
        if n1 > 1e-9:                       # relative translation scales by exactly s
            assert abs(ns / n1 - s) < 1e-8, f"translation ratio {ns/n1} != {s}"
        # equivalence with the frozen Gate-0/1 convention: no second application of s
        assert np.allclose(rs, relative_metric(pose_c2w, f, anchor, s), atol=1e-9), \
            "scaled_relative_pose disagrees with the Gate-1 relative_metric convention"


def as4x4(pose: np.ndarray) -> np.ndarray:
    p = pose.astype(np.float64)
    if p.shape[-2] == 3:
        p4 = np.tile(np.eye(4), (len(p), 1, 1)); p4[:, :3, :4] = p; p = p4
    return p


def fuse(depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray, pose_c2w: np.ndarray,
         s_pose: float) -> np.ndarray:
    """Unproject already-metric per-frame depth and fuse into the anchor camera."""
    T, H, W = depth_m.shape
    anchor = T - 1
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    chunks = []
    for f in range(T):
        m = mask[f]
        if not m.any():
            continue
        d = depth_m[f][m]
        p = np.stack([(u[m] - K[f][0, 2]) * d / K[f][0, 0],
                      (v[m] - K[f][1, 2]) * d / K[f][1, 1], d], axis=-1)
        if f != anchor:
            p = apply_transform(scaled_relative_pose(pose_c2w, f, anchor, s_pose), p)
        chunks.append(p)
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def spearman(x, y) -> float:
    rank = lambda z: np.argsort(np.argsort(np.asarray(z, float))).astype(float)
    return pearson(rank(x), rank(y))


def q(v, ps):
    return {f"p{int(p*100)}": float(np.quantile(v, p)) for p in ps}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/depth_gate/refine.yaml")
    ap.add_argument("--run", default="depth_cnn")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="artifacts/depth_gate/residual_decomposition")
    a = ap.parse_args()

    cfg = load_config(a.config)
    scfg = load_config(cfg.data.scale_gate_config)
    scfg["voxel"]["pixel_stride"] = 1                       # frozen Gate-1/2 setting
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    cache = os.path.join(REPO_ROOT, scfg.cache.root)
    rgb_dir = os.path.join(REPO_ROOT, cfg.data.rgb_cache)
    sg = os.path.join(REPO_ROOT, scfg.experiment.output_dir)
    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    out_dir = os.path.join(REPO_ROOT, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- frozen head, exactly as reported ---------------------------------- #
    ck_path = os.path.join(run_dir, "best.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["in_ch"], ck["base_channels"], ck["max_log_residual"])
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    stats, use_rgb = ck["stats"], ck["use_rgb"]

    # s0 comes from the resolved run configuration; no second constant is introduced.
    s0 = float(cfg.scale.constant)
    assert abs(float(ck["metric_scale"]) - s0) < 1e-9, (
        f"checkpoint metric_scale {ck['metric_scale']} != resolved config constant {s0}")

    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)

    tgt = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(sg, "scale_targets_val.csv")))}
    recs = read_manifest(os.path.join(sg, "manifests", "val.jsonl"))
    if a.limit:
        recs = recs[: a.limit]

    specs = {}
    occ_rows, scale_rows = [], []
    dep_acc = {n: [] for n in CONFIGS}          # per-config depth stacks for pooled metrics
    gt_acc, val_acc, clip_acc = [], [], []
    pose_checked = 0

    for i, rec in enumerate(recs):
        cid, seq = rec["clip_id"], rec["sequence"]
        lp = os.path.join(cache, "lingbot", f"{cid}.npz")
        dp = os.path.join(cache, "lidar_depth", f"{cid}.npz")
        rp = os.path.join(rgb_dir, f"{cid}.npz")
        if not all(os.path.exists(x) for x in (lp, dp, rp)) or cid not in tgt:
            continue
        try:
            s_oracle = float(tgt[cid]["s_joint"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s_oracle) or s_oracle <= 0:
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        target, valid_mask = spec.target(int(rec["frame_ids"][-1]))
        if target is None:
            continue

        L = np.load(lp, allow_pickle=False)
        D = np.load(dp, allow_pickle=False)
        rgb = np.load(rp, allow_pickle=False)["rgb"].astype(np.float32) / 255.0
        dep = L["pred_depth"].astype(np.float32)
        # float64 once: `s0` is a weak Python float, so `s0 * dep` would silently stay
        # float32 under NEP 50 and make the C1/C5 equality assertion fail at 1e-7.
        depf = dep.astype(np.float64)
        conf = L["pred_depth_conf"].astype(np.float32)
        K = L["pred_K"].astype(np.float64)
        pose_pred = as4x4(L["pred_pose_c2w"])
        pose_gt = as4x4(L["gt_pose_c2w"])
        T = dep.shape[0]

        # ---- residual, computed ONCE on the reported deployable input ------- #
        base0 = depf * s0
        b = {"rgb": torch.from_numpy(rgb).to(dev),
             "base_depth": torch.from_numpy(base0.astype(np.float32)).unsqueeze(1).to(dev),
             "lingbot_confidence": torch.from_numpy(conf).unsqueeze(1).to(dev),
             "valid_lingbot_mask": torch.from_numpy(
                 np.isfinite(base0) & (base0 > 0)).unsqueeze(1).to(dev)}
        with torch.no_grad():
            r = model(build_inputs(b, stats, use_rgb))[:, 0].double().cpu().numpy()

        # ---- unchanged fusion support (the C0 mask): no LiDAR, no GT -------- #
        support = ((conf >= conf_thr) & np.isfinite(base0) & (base0 > dmin) & (base0 < dmax))
        if support.sum() < 1:
            continue
        a_clip = float(np.median(r[support]))
        r_shape = r - a_clip
        assert abs(float(np.median(r_shape[support]))) < 1e-9, \
            f"{cid}: median(r_shape) on the clip support is {np.median(r_shape[support])}, not 0"
        s_learned = s0 * float(np.exp(a_clip))

        # C5 depth must equal C1 depth: s0*exp(a)*exp(r-a) == s0*exp(r)
        assert np.allclose(s_learned * depf * np.exp(r_shape), s0 * depf * np.exp(r),
                           rtol=1e-9, atol=1e-9), f"{cid}: C5 depth != C1 depth"

        if pose_checked < 5:                     # pose-scaling assertions on real poses
            check_pose_scaling(pose_pred, T - 1, s_learned)
            check_pose_scaling(pose_pred, T - 1, s_oracle)
            pose_checked += 1

        per_frame_med = [float(np.median(r[f][support[f]])) if support[f].any() else float("nan")
                         for f in range(T)]
        rs_sup, r_sup = r_shape[support], r[support]
        scale_rows.append({
            "clip_id": cid, "sequence": seq, "a_clip": a_clip, "s_learned": s_learned,
            "s_oracle": s_oracle, "s0": s0,
            "ratio_learned_over_oracle": s_learned / s_oracle,
            "abs_log_scale_err_learned": abs(np.log(s_learned) - np.log(s_oracle)),
            "abs_log_scale_err_const": abs(np.log(s0) - np.log(s_oracle)),
            "rel_scale_err_learned": abs(s_learned / s_oracle - 1.0),
            "rel_scale_err_const": abs(s0 / s_oracle - 1.0),
            "frame_median_std": float(np.nanstd(per_frame_med)),
            "frame_median_range": float(np.nanmax(per_frame_med) - np.nanmin(per_frame_med)),
            "median_abs_r_shape": float(np.median(np.abs(rs_sup))),
            "rms_r_shape": float(np.sqrt(np.mean(rs_sup ** 2))),
            "rms_r": float(np.sqrt(np.mean(r_sup ** 2))),
            "scalar_energy_fraction": float(a_clip ** 2 / max(np.mean(r_sup ** 2), 1e-30)),
            "n_support_px": int(support.sum()),
            **{f"frame_median_{f}": per_frame_med[f] for f in range(T)},
        })

        # ---- depth-metric valid mask, identical to the Gate-2 dataset ------- #
        vmask = D["valid"].astype(bool) & np.isfinite(base0) & (base0 > 0) & (D["depth"] > 0)
        gt_acc.append(torch.from_numpy(D["depth"].astype(np.float64)))
        val_acc.append(torch.from_numpy(vmask))
        clip_acc += [cid] * T

        # ---- every configuration ------------------------------------------- #
        SC = {"const": s0, "learned": s_learned, "oracle": s_oracle, "none": 1.0}
        RR = {"none": np.zeros_like(r), "full": r, "shape": r_shape}
        for name, c in CONFIGS.items():
            sd = SC[c["depth_scale"]]
            depth_m = sd * depf * np.exp(RR[c["resid"]])
            dep_acc[name].append(torch.from_numpy(depth_m))
            m = (conf >= conf_thr) & np.isfinite(depth_m) & (depth_m > dmin) & (depth_m < dmax)
            pose = pose_gt if c["pose"] == "gt" else pose_pred
            pts = fuse(depth_m, m, K, pose, SC[c["pose_scale"]])
            pg = apply_transform(spec.cam_to_velo, pts) if len(pts) else pts
            fin, fout = grid_fractions(pg)
            sc = binary_occupancy_scores(
                occupancy_from_points(pg, G, scfg.voxel.min_points_per_voxel),
                target, G, valid=valid_mask)
            occ_rows.append({
                "clip_id": cid, "config": name, "depth_scale_kind": c["depth_scale"],
                "resid": c["resid"], "pose": c["pose"], "pose_scale_kind": c["pose_scale"],
                "s_depth": sd, "s_pose": SC[c["pose_scale"]],
                "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                "n_pred_occupied": int(sc["n_pred_occupied"]),
                "n_points": int(pts.shape[0]),
                "in_grid_fraction": fin, "out_of_grid_fraction": fout,
                "n_masked_px": int(m.sum()),
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} clips", flush=True)

    n_clips = len({r["clip_id"] for r in occ_rows})
    print(f"\n{n_clips} clips evaluated, pose-scaling assertions passed on {pose_checked} clips")

    # ---- depth metrics ------------------------------------------------------ #
    GT, VA = torch.cat(gt_acc), torch.cat(val_acc)
    bins = [tuple(x) for x in cfg.eval.distance_bins_m]
    clips = np.asarray(clip_acc)
    depth_summary, depth_per_clip = {}, {}
    for name in CONFIGS:
        P = torch.cat(dep_acc[name])
        depth_summary[name] = {"overall": depth_metrics(P, GT, VA),
                               "by_distance": binned_metrics(P, GT, VA, bins)}
        per = {}
        for cid in sorted(set(clip_acc)):
            idx = np.flatnonzero(clips == cid)
            per[cid] = depth_metrics(P[idx], GT[idx], VA[idx])
        depth_per_clip[name] = per
        del P

    # ---- occupancy aggregation and paired bootstrap -------------------------- #
    by = {}
    for row in occ_rows:
        by.setdefault(row["config"], {})[row["clip_id"]] = row
    mean = lambda n, k: float(np.mean([v[k] for v in by[n].values()]))
    occ_summary = {n: {"label": CONFIGS[n]["label"],
                       **{k: mean(n, k) for k in
                          ("iou", "precision", "recall", "n_pred_occupied", "n_points",
                           "in_grid_fraction", "out_of_grid_fraction", "n_masked_px")},
                       "n_clips": len(by[n])} for n in CONFIGS}

    repro = {}
    for n, t in REPRO_TARGETS.items():
        got = occ_summary[n]["iou"]
        repro[n] = {"target": t, "reproduced": got, "abs_diff": abs(got - t),
                    "within_tolerance": bool(abs(got - t) <= REPRO_TOL)}
    if not all(v["within_tolerance"] for v in repro.values()):
        for n, v in repro.items():
            print(f"  {n}: target {v['target']:.4f} got {v['reproduced']:.4f} "
                  f"diff {v['abs_diff']:.5f} {'OK' if v['within_tolerance'] else 'FAIL'}")
        write_json(os.path.join(out_dir, "reproduction_failure.json"), repro)
        print("\nPIPELINE_INVALID: the reported Gate-2 baselines were not reproduced.")
        return 2

    def paired(frm: str, to: str, key: str, src):
        ids = sorted(set(src[frm]) & set(src[to]))
        d = [src[to][c][key] - src[frm][c][key] for c in ids]
        ci = bootstrap_ci(d, cfg.eval.bootstrap_n, cfg.eval.bootstrap_seed)
        ci["improved_clip_fraction"] = float(np.mean([x > 0 for x in d]))
        ci["n_pairs"] = len(d)
        return ci

    contrasts = {}
    for frm, to, why in CONTRASTS:
        ci = paired(frm, to, "iou", by)
        ci_d = paired(frm, to, "abs_rel", depth_per_clip)
        ci_d["improved_clip_fraction"] = 1.0 - ci_d["improved_clip_fraction"]   # lower is better
        contrasts[f"{frm}->{to}"] = {"why": why, "iou": ci, "abs_rel": ci_d}

    g_c1 = contrasts["C0->C1"]["iou"]["mean"]
    frac = {k: (contrasts[k]["iou"]["mean"] / g_c1 if abs(g_c1) > 1e-12 else float("nan"))
            for k in ("C0->C2", "C0->C3", "C0->C4")}

    # ---- scale diagnostics --------------------------------------------------- #
    sl = np.array([r["s_learned"] for r in scale_rows])
    so = np.array([r["s_oracle"] for r in scale_rows])
    ac = np.array([r["a_clip"] for r in scale_rows])
    ratio = sl / so
    scale_diag = {
        "n_clips": len(scale_rows),
        "median_abs_log_scale_error_learned": float(np.median(
            [r["abs_log_scale_err_learned"] for r in scale_rows])),
        "median_abs_log_scale_error_constant": float(np.median(
            [r["abs_log_scale_err_const"] for r in scale_rows])),
        "median_relative_scale_error_learned": float(np.median(
            [r["rel_scale_err_learned"] for r in scale_rows])),
        "median_relative_scale_error_constant": float(np.median(
            [r["rel_scale_err_const"] for r in scale_rows])),
        "pearson_s_learned_vs_oracle": pearson(sl, so),
        "spearman_s_learned_vs_oracle": spearman(sl, so),
        "pearson_log_s_learned_vs_log_oracle": pearson(np.log(sl), np.log(so)),
        "spearman_log_s_learned_vs_log_oracle": spearman(np.log(sl), np.log(so)),
        "ratio_learned_over_oracle": {"median": float(np.median(ratio)),
                                      "mean": float(ratio.mean()),
                                      **q(ratio, [0.05, 0.25, 0.5, 0.75, 0.95]),
                                      "min": float(ratio.min()), "max": float(ratio.max())},
        "a_clip": {"median": float(np.median(ac)), "mean": float(ac.mean()),
                   **q(ac, [0.05, 0.25, 0.5, 0.75, 0.95]),
                   "min": float(ac.min()), "max": float(ac.max())},
        "s_learned": {"median": float(np.median(sl)), **q(sl, [0.05, 0.95])},
        "s_oracle": {"median": float(np.median(so)), **q(so, [0.05, 0.95])},
        "frame_median_std_within_clip": {
            "median": float(np.median([r["frame_median_std"] for r in scale_rows])),
            **q([r["frame_median_std"] for r in scale_rows], [0.05, 0.95])},
        "frame_median_range_within_clip": {
            "median": float(np.median([r["frame_median_range"] for r in scale_rows])),
            **q([r["frame_median_range"] for r in scale_rows], [0.05, 0.95])},
        "median_abs_r_shape": float(np.median([r["median_abs_r_shape"] for r in scale_rows])),
        "rms_r_shape": float(np.median([r["rms_r_shape"] for r in scale_rows])),
        "rms_r": float(np.median([r["rms_r"] for r in scale_rows])),
        "scalar_energy_fraction": {
            "median": float(np.median([r["scalar_energy_fraction"] for r in scale_rows])),
            **q([r["scalar_energy_fraction"] for r in scale_rows], [0.05, 0.25, 0.75, 0.95])},
        "note": ("In-domain SemanticKITTI diagnostic on sequence 08 only. This does not "
                 "establish generalisable metric-scale recovery."),
    }

    # ---- write --------------------------------------------------------------- #
    def dump(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    dump(os.path.join(out_dir, "per_clip_occupancy.csv"), occ_rows)
    dump(os.path.join(out_dir, "per_clip_scale.csv"), scale_rows)
    dump(os.path.join(out_dir, "per_clip_depth.csv"),
         [{"clip_id": c, "config": n, **depth_per_clip[n][c]}
          for n in CONFIGS for c in sorted(depth_per_clip[n])])

    write_json(os.path.join(out_dir, "summary.json"), {
        "provenance": collect_provenance(scfg, "depth_gate.decompose_residual"),
        "run": a.run, "checkpoint": os.path.relpath(ck_path, REPO_ROOT),
        "checkpoint_sha256": sha256(ck_path),
        "lingbot_checkpoint": scfg.lingbot.checkpoint,
        "constant_scale_s0": s0, "n_clips": n_clips, "pixel_stride": 1,
        "confidence_threshold": conf_thr, "depth_range_m": [dmin, dmax],
        "arch": ck["arch"], "n_params": int(sum(p.numel() for p in ck["state_dict"].values())),
        "reproduction": repro, "configs": {n: dict(CONFIGS[n]) for n in CONFIGS},
        "occupancy": occ_summary, "depth": depth_summary,
        "contrasts": contrasts, "fraction_of_C0_C1_gain": frac,
        "scale_diagnostics": scale_diag,
    })

    # ---- console ------------------------------------------------------------- #
    print(f"\n{'cfg':4s} {'IoU':>8} {'P':>7} {'R':>7} {'occupied':>9} {'points':>9} "
          f"{'in-grid':>8} {'AbsRel':>8} {'d1':>7}")
    for n in CONFIGS:
        s, d = occ_summary[n], depth_summary[n]["overall"]
        print(f"{n:4s} {s['iou']:8.4f} {s['precision']:7.3f} {s['recall']:7.3f} "
              f"{s['n_pred_occupied']:9.0f} {s['n_points']:9.0f} {s['in_grid_fraction']:8.3f} "
              f"{d['abs_rel']:8.4f} {d['delta1']:7.4f}")
    print(f"\n{'contrast':12s} {'dIoU':>9}  95% CI              improved  {'dAbsRel':>9}")
    for k, v in contrasts.items():
        c, dd = v["iou"], v["abs_rel"]
        print(f"{k:12s} {c['mean']:+9.4f}  [{c['lo']:+.4f},{c['hi']:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '} {c['improved_clip_fraction']:8.1%}  "
              f"{dd['mean']:+9.4f}{'*' if dd['excludes_zero'] else ' '}")
    print("\nfraction of the C0->C1 IoU gain reproduced:")
    for k, v in frac.items():
        print(f"  {k}: {v:.1%}")
    print(f"\nwrote {os.path.relpath(out_dir, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
