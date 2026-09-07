#!/usr/bin/env python
"""Gate 1 — factorize LingBot geometry error into depth, pose and coverage.

    python tools/geometry_gate/factorize.py --config configs/scale_gate/semantickitti.yaml

Six configurations on the **identical** Gate-0 protocol (SemanticKITTI seq 08, 5-frame
clips at stride 5, same calibration, manifests, caches, voxeliser and evaluation mask).
Only the factor under test changes between rows:

    A  GT LiDAR depth            + GT poses            -- five-frame visible-reconstruction ceiling
    B  GT LiDAR depth            + LingBot poses (s*)  -- A + pose error
    C  LingBot depth on LiDAR support + GT poses       -- A + depth error, matched support
    D  LingBot depth, full       + GT poses            -- C + coverage beyond LiDAR support
    E  LingBot depth, full       + LingBot poses (s*)  -- D + pose error
    F  LingBot depth, full       + LingBot poses (27.35) -- E + scale-estimation error (deployable)

A and B use metric LiDAR depth directly. C-F scale LingBot depth *and* the pose
translations by the same scalar, never one without the other.

**Support discipline.** C is a *sparse common-support* evaluation: predicted depth is read
only at pixels where LiDAR returned. D-F are *full predicted-depth* evaluations. The two
are never averaged together, and the distinction is carried in every output row.
"""
from __future__ import annotations

import argparse, csv, itertools, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)

CONFIGS = {
    "A_gtdepth_gtpose":   dict(depth="lidar", pose="gt",   support="lidar", scale=None),
    # A1: identical pixel set to C, so A_common -> C isolates depth *shape* alone.
    "A_common_gtdepth":   dict(depth="lidar", pose="gt",   support="common", scale=None),
    "B_gtdepth_predpose": dict(depth="lidar", pose="pred", support="lidar", scale="oracle"),
    "C_preddepth_common": dict(depth="pred",  pose="gt",   support="lidar", scale="oracle"),
    "D_preddepth_full":   dict(depth="pred",  pose="gt",   support="full",  scale="oracle"),
    "E_preddepth_predpose": dict(depth="pred", pose="pred", support="full", scale="oracle"),
    "F_deployable":       dict(depth="pred",  pose="pred", support="full",  scale="const"),
}
DEPLOYABLE_S = 27.3498          # global_median_train, fitted on source sequences only
SUPPORT_KIND = {"lidar": "sparse_lidar_support", "common": "controlled_common_support",
                "full": "full_predicted_depth"}


def frame_masks(cfg, L, D, support: str, depth_source: str, s: float) -> np.ndarray:
    """Boolean ``[T, H, W]`` pixel mask for one configuration.

    ``common`` reproduces exactly the mask configuration C applies -- valid LiDAR, the
    LingBot confidence threshold, and the metric depth range evaluated on *predicted*
    depth -- so ``A_common`` and ``C`` select the identical pixels and differ only in the
    depth value read there. That is what makes their contrast a pure depth-shape effect.
    """
    valid = D["valid"]
    conf = L["pred_depth_conf"].astype(np.float64)
    pred = L["pred_depth"].astype(np.float64)
    T = valid.shape[0]
    out = np.zeros_like(valid)
    for f in range(T):
        if support == "full":
            m = np.ones_like(valid[f], bool)
        else:                                        # "lidar" or "common"
            m = valid[f].copy()
        if support == "common" or depth_source == "pred":
            # C's own admission rules, evaluated on predicted depth in canonical units.
            lo = cfg.voxel.min_depth_m / max(s, 1e-9)
            hi = cfg.voxel.max_depth_m / max(s, 1e-9)
            m &= (conf[f] >= cfg.lingbot.confidence_threshold)
            m &= np.isfinite(pred[f]) & (pred[f] > lo) & (pred[f] < hi)
        out[f] = m
    return out


def unproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``[H, W]`` z-depth + boolean mask -> ``[N, 3]`` camera-frame points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    m = mask & np.isfinite(depth) & (depth > 0)
    if not m.any():
        return np.zeros((0, 3))
    d = depth[m]
    return np.stack([(u[m] - K[0, 2]) * d / K[0, 0],
                     (v[m] - K[1, 2]) * d / K[1, 1], d], axis=-1)


def relative_metric(pose_c2w: np.ndarray, f: int, anchor: int, s: float) -> np.ndarray:
    """Transform from camera ``f`` into camera ``anchor``, with translation scaled by ``s``.

    Rotation is never scaled. For GT (already metric) poses ``s`` is 1.
    """
    rel = np.linalg.inv(pose_c2w[anchor]) @ pose_c2w[f]
    out = rel.copy()
    out[:3, 3] *= s
    return out


def build_points(cfg, L, D, spec, name: str, s_star: float):
    """Fused metric points in the anchor camera for one configuration."""
    c = CONFIGS[name]
    T = len(D["depth"])
    anchor = T - 1
    s = 1.0 if c["scale"] is None else (s_star if c["scale"] == "oracle" else DEPLOYABLE_S)

    if c["depth"] == "lidar":
        dep = D["depth"].astype(np.float64)                  # already metric
        K = np.tile(D["K_processed"].astype(np.float64), (T, 1, 1))
        depth_scale = 1.0
    else:
        dep = L["pred_depth"].astype(np.float64)             # canonical units
        K = L["pred_K"].astype(np.float64)
        depth_scale = s

    pose = (L["gt_pose_c2w"] if c["pose"] == "gt" else L["pred_pose_c2w"]).astype(np.float64)
    if pose.shape[-2] == 3:
        p4 = np.tile(np.eye(4), (T, 1, 1)); p4[:, :3, :4] = pose; pose = p4
    pose_scale = 1.0 if c["pose"] == "gt" else s             # GT poses are metric already

    # The admission rules are evaluated on *predicted* depth in canonical units, so the
    # mask must use C's scale. A_common carries metric LiDAR values but must inherit
    # exactly C's pixel set, hence s_star rather than its own s.
    s_mask = s_star if c["support"] == "common" else s
    masks = frame_masks(cfg, L, D, c["support"], c["depth"], s_mask)
    n_masked = int(masks.sum())
    chunks = []
    for f in range(T):
        p = unproject(dep[f] * depth_scale, masks[f], K[f])
        if p.shape[0] == 0:
            continue
        if f != anchor:
            p = apply_transform(relative_metric(pose, f, anchor, pose_scale), p)
        chunks.append(p)
    pts = np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))
    return pts, n_masked


def grid_fractions(pts_grid: np.ndarray):
    """Fraction of fused points falling inside vs outside the voxel grid."""
    if pts_grid.shape[0] == 0:
        return 0.0, 0.0
    idx = np.floor((pts_grid - np.array(G.origin)) / G.voxel_size)
    inside = ((idx >= 0) & (idx < np.array(G.dims))).all(axis=1)
    return float(inside.mean()), float(1.0 - inside.mean())


def pose_errors(L, s_star: float):
    """Rotation (deg) and translation (m) error of scaled LingBot relative poses."""
    gp = L["gt_pose_c2w"].astype(np.float64)
    pp = L["pred_pose_c2w"].astype(np.float64)
    if pp.shape[-2] == 3:
        p4 = np.tile(np.eye(4), (len(pp), 1, 1)); p4[:, :3, :4] = pp; pp = p4
    if gp.shape[-2] == 3:
        g4 = np.tile(np.eye(4), (len(gp), 1, 1)); g4[:, :3, :4] = gp; gp = g4
    T = len(pp); a = T - 1
    rot, tra = [], []
    for f in range(T - 1):
        rg = np.linalg.inv(gp[a]) @ gp[f]
        rp = relative_metric(pp, f, a, s_star)
        dR = rg[:3, :3].T @ rp[:3, :3]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        tra.append(float(np.linalg.norm(rp[:3, 3] - rg[:3, 3])))
    return float(np.mean(rot)) if rot else 0.0, float(np.mean(tra)) if tra else 0.0


def sparse_absrel(L, D, s: float) -> float:
    """AbsRel of scaled predicted depth against LiDAR, on LiDAR-valid pixels only."""
    dep = L["pred_depth"].astype(np.float64) * s
    gt = D["depth"].astype(np.float64)
    m = D["valid"] & (gt > 0) & (dep > 0) & np.isfinite(dep)
    return float(np.mean(np.abs(dep[m] - gt[m]) / gt[m])) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/scale_gate/semantickitti.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="artifacts/geometry_gate")
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache = os.path.join(REPO_ROOT, cfg.cache.root)
    sg = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    out_dir = os.path.join(REPO_ROOT, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tgt = {r["clip_id"]: r for r in
           csv.DictReader(open(os.path.join(sg, f"scale_targets_{a.split}.csv")))}
    recs = read_manifest(os.path.join(sg, "manifests", f"{a.split}.jsonl"))
    if a.limit:
        recs = recs[: a.limit]

    specs, rows = {}, []
    for i, rec in enumerate(recs):
        cid, seq = rec["clip_id"], rec["sequence"]
        lp = os.path.join(cache, "lingbot", f"{cid}.npz")
        dp = os.path.join(cache, "lidar_depth", f"{cid}.npz")
        if not (os.path.exists(lp) and os.path.exists(dp)) or cid not in tgt:
            continue
        try:
            s_star = float(tgt[cid]["s_joint"])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(s_star) or s_star <= 0:
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        target, valid_mask = spec.target(int(rec["frame_ids"][-1]))
        if target is None:
            continue
        L = np.load(lp, allow_pickle=False)
        D = np.load(dp, allow_pickle=False)
        rot_err, tra_err = pose_errors(L, s_star)

        counts = {}
        for name in CONFIGS:
            c = CONFIGS[name]
            s_used = (1.0 if c["scale"] is None
                      else s_star if c["scale"] == "oracle" else DEPLOYABLE_S)
            pts, n_masked = build_points(cfg, L, D, spec, name, s_star)
            counts[name] = n_masked
            pg = apply_transform(spec.cam_to_velo, pts) if pts.shape[0] else pts
            fin, fout = grid_fractions(pg)
            sc = binary_occupancy_scores(
                occupancy_from_points(pg, G, cfg.voxel.min_points_per_voxel),
                target, G, valid=valid_mask)
            rows.append({
                "clip_id": cid, "sequence": seq, "config": name,
                "support": SUPPORT_KIND[c["support"]],
                "depth_source": c["depth"], "pose_source": c["pose"], "scale_used": s_used,
                "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                "n_pred_occupied": int(sc["n_pred_occupied"]),
                "n_gt_occupied": int(sc["n_gt_occupied"]),
                "n_points": int(pts.shape[0]),
                "in_grid_fraction": fin, "out_of_grid_fraction": fout,
                "sparse_absrel": (sparse_absrel(L, D, s_used) if c["depth"] == "pred"
                                  else float("nan")),
                "pose_rot_err_deg": (rot_err if c["pose"] == "pred" else 0.0),
                "pose_trans_err_m": (tra_err if c["pose"] == "pred" else 0.0),
                "n_masked_pixels": n_masked,
            })
        # A1 requirement: the controlled pair must select identical pixels, or stop.
        if counts["A_common_gtdepth"] != counts["C_preddepth_common"]:
            raise SystemExit(
                f"{cid}: controlled-support mismatch -- A_common has "
                f"{counts['A_common_gtdepth']} masked pixels, C has "
                f"{counts['C_preddepth_common']}. The two must select the identical "
                "pixel set for A_common -> C to be a pure depth-shape contrast.")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} clips", flush=True)

    p_csv = os.path.join(out_dir, "factorization_v2_per_clip.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    agg, m = {}, lambda rs, k: float(np.nanmean([x[k] for x in rs]))
    for name, rs in by.items():
        agg[name] = {
            "support": rs[0]["support"], "depth_source": rs[0]["depth_source"],
            "pose_source": rs[0]["pose_source"], "n_clips": len(rs),
            "iou": m(rs, "iou"), "precision": m(rs, "precision"), "recall": m(rs, "recall"),
            "n_pred_occupied": m(rs, "n_pred_occupied"), "n_points": m(rs, "n_points"),
            "in_grid_fraction": m(rs, "in_grid_fraction"),
            "out_of_grid_fraction": m(rs, "out_of_grid_fraction"),
            "sparse_absrel": m(rs, "sparse_absrel"),
            "n_masked_pixels": m(rs, "n_masked_pixels"),
            "pose_rot_err_deg": m(rs, "pose_rot_err_deg"),
            "pose_trans_err_m": m(rs, "pose_trans_err_m"),
        }

    # Pairwise clip-level bootstrap for every ordered pair, plus the factor chain.
    iou = {n: {r["clip_id"]: r["iou"] for r in rs} for n, rs in by.items()}
    pairs = {}
    for x, y in itertools.combinations(sorted(CONFIGS), 2):
        ids = sorted(set(iou[x]) & set(iou[y]))
        d = [iou[y][k] - iou[x][k] for k in ids]
        pairs[f"{y} - {x}"] = bootstrap_ci(d, cfg.bootstrap.n_boot, cfg.bootstrap.seed,
                                           cfg.bootstrap.alpha)
    def contrast(frm: str, to: str):
        """Bootstrap CI for IoU(to) - IoU(frm) over the clips both evaluated."""
        ids = sorted(set(iou[frm]) & set(iou[to]))
        return bootstrap_ci([iou[to][k] - iou[frm][k] for k in ids],
                            cfg.bootstrap.n_boot, cfg.bootstrap.seed, cfg.bootstrap.alpha)

    CHAIN = [
        ("pose only (A -> B)", "A_gtdepth_gtpose", "B_gtdepth_predpose"),
        ("support restriction (A -> A_common)", "A_gtdepth_gtpose", "A_common_gtdepth"),
        ("DEPTH SHAPE, controlled (A_common -> C)", "A_common_gtdepth", "C_preddepth_common"),
        ("depth+support, uncontrolled (A -> C)", "A_gtdepth_gtpose", "C_preddepth_common"),
        ("coverage (C -> D)", "C_preddepth_common", "D_preddepth_full"),
        ("pose given predicted depth (D -> E)", "D_preddepth_full", "E_preddepth_predpose"),
        ("scale estimation (E -> F)", "E_preddepth_predpose", "F_deployable"),
    ]
    chain = {label: {"from": a, "to": b, **contrast(a, b)} for label, a, b in CHAIN}

    ceiling = agg["A_gtdepth_gtpose"]["iou"]
    print(f"\n{'config':24s} {'support':22s} {'IoU':>8} {'%ceil':>7} {'P':>7} {'R':>7} "
          f"{'masked px':>10} {'in-grid':>8} {'AbsRel':>8} {'rot°':>7} {'trans m':>8}")
    for n in sorted(CONFIGS):
        s = agg[n]
        print(f"{n:24s} {s['support']:22s} {s['iou']:8.4f} {100*s['iou']/ceiling:6.1f}% "
              f"{s['precision']:7.3f} {s['recall']:7.3f} {s['n_masked_pixels']:10.0f} "
              f"{s['in_grid_fraction']:8.3f} "
              f"{s['sparse_absrel']:8.4f} {s['pose_rot_err_deg']:7.3f} {s['pose_trans_err_m']:8.3f}")
    print(f"\n{'factor':42s} {'mean dIoU':>10}  95% CI          %ceil")
    for label, a_, b_ in CHAIN:
        c = chain[label]
        print(f"{label:42s} {c['mean']:+10.4f}  [{c['lo']:+.4f},{c['hi']:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '} {100*c['mean']/ceiling:+7.1f}%")

    write_json(os.path.join(out_dir, "factorization_v2_summary.json"),
               {"provenance": collect_provenance(cfg, "geometry_gate.factorize"),
                "split": a.split, "n_clips": len(by[next(iter(by))]),
                "deployable_scale": DEPLOYABLE_S,
                "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size},
                "five_frame_visible_reconstruction_ceiling": ceiling,
                "configs": agg, "pairwise_bootstrap": pairs, "factor_chain": chain,
                "per_clip_csv": os.path.relpath(p_csv, REPO_ROOT)})
    print(f"\nwrote {os.path.relpath(p_csv, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
