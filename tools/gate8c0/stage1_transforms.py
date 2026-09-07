#!/usr/bin/env python
"""Gate 8C-0 Stage 1: round-trip every transform pair, then draw the chain.

The round trips are exact-inverse checks with a reported tolerance; they catch a
transposed rotation, a wrong pose direction, a dropped rectification and an origin or
sign error. The figures put the official target, the current sweep, the frustum, the
trajectory, the causal map and the future observations on identical axes, so a residual
timing or frame error is visible rather than argued about.

    python tools/gate8c0/stage1_transforms.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg                                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import targets as G6T                                                 # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402

#: 1 mm = 1/200 of a voxel. Justified rather than aspirational: the shipped KITTI-360
#: matrices carry six decimals, so an exactly-rigid inverse of them is only good to ~1e-4 m.
TOL_M = 1e-3


def audit_anchors(c) -> list:
    """Three anchors per partition, spread evenly across each drive."""
    out = []
    n = int(c["n_audit_anchors_per_partition"])
    for part, drives in (("train", c["train_drives"]), ("source_validation", [c["val_drive"]]),
                         ("heldout", [c["heldout_drive"]])):
        # spread n picks over the partition's drives, round-robin, at even quantiles
        per = {d: [] for d in drives}
        for i in range(n):
            per[drives[i % len(drives)]].append(i)
        picked = []
        for d, slots in per.items():
            ldir = os.path.join(c["sscbench_root"], "preprocess", "labels", d)
            anchors = sorted(int(f.split("_")[0]) for f in os.listdir(ldir)
                             if f.endswith("_1_1.npy"))
            qs = np.linspace(0.2, 0.8, max(len(slots), 1))
            for q in qs[:len(slots)]:
                picked.append((d, int(anchors[int(q * (len(anchors) - 1))])))
        out += [{"partition": part, "drive": d, "anchor": a} for d, a in picked]
    return out


def round_trips(g: TF.DriveGeometry, anchor_native: int, src_native: int, rng) -> dict:
    """Every transform pair, forward then inverse, on 4096 sampled points."""
    p = rng.uniform(-40, 40, size=(4096, 3))
    res = {}

    def rt(name, T):
        q = TF.apply(TF.inv(T), TF.apply(T, p))
        e = np.abs(q - p).max()
        res[name] = {"max_abs_error_m": float(e), "within_tol": bool(e < TOL_M)}

    rt("velo_to_world", g.velo_to_world(src_native))
    rt("rect_cam_to_velo", g.rect_cam_to_velo)
    rt("rect_cam_to_world", g.rect_cam_to_world(src_native))
    rt("velo_src_to_velo_anchor", g.velo_to_velo(src_native, anchor_native))
    rt("cam0_unrect_to_velo", np.asarray(g.calib.cam0_to_velo, np.float64))
    # the rigid-inverse control: how much of any residual is the shipped data's rounding
    res["rigid_inverse_residual"] = {}
    for nm, T in (("R_rect_00", np.asarray(g.calib.R_rect_00, np.float64)),
                  ("cam0_to_velo", np.asarray(g.calib.cam0_to_velo, np.float64)),
                  ("velo_to_world", g.velo_to_world(src_native))):
        R = np.asarray(T, np.float64)[:3, :3]
        res["rigid_inverse_residual"][nm] = {
            "det": float(np.linalg.det(R)),
            "max_abs_RtR_minus_I": float(np.abs(R.T @ R - np.eye(3)).max()),
            "roundtrip_with_R_transpose_m": float(np.abs(
                TF.apply(TF.inv_rigid(T), TF.apply(T, p)) - p).max()),
            "roundtrip_with_true_inverse_m": float(np.abs(
                TF.apply(TF.inv(T), TF.apply(T, p)) - p).max())}
    # composition consistency: velo(src) -> world -> velo(anchor) equals the direct compose
    A = g.velo_to_velo(src_native, anchor_native)
    B = TF.inv(g.velo_to_world(anchor_native)) @ g.velo_to_world(src_native)
    res["composition_matches_direct"] = {"max_abs_error_m": float(np.abs(A - B).max()),
                                         "within_tol": bool(np.abs(A - B).max() < TOL_M)}
    # rectification is not the identity (a dropped R_rect would be a silent ~0.4 deg error)
    R = np.asarray(g.calib.R_rect_00, np.float64)[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    res["R_rect_00_rotation_deg"] = float(ang)
    res["rect_matters"] = bool(ang > 0.05)
    # the shipped cam0_to_velo is not exactly orthonormal; report how far off
    Rc = np.asarray(g.calib.cam0_to_velo, np.float64)[:3, :3]
    res["cam0_to_velo_orthonormality"] = {
        "det": float(np.linalg.det(Rc)),
        "max_abs_RtR_minus_I": float(np.abs(Rc.T @ Rc - np.eye(3)).max())}
    # grid index <-> metric centre round trip (floor convention)
    dims = TF.DIMS
    idx = np.stack([rng.integers(0, d, 4096) for d in dims], -1)
    back, keep = TF.voxelize(TF.centres(idx), TF.GRID)
    res["grid_index_centre_roundtrip"] = {
        "n": int(len(idx)), "n_kept": int(keep.sum()),
        "exact": bool(keep.all() and np.array_equal(back, idx))}
    return res


def figure(c, row, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    d, anchor = row["drive"], row["anchor"]
    g = TF.DriveGeometry(d, c["sscbench_root"], c["kitti360_root"])
    native = g.native(anchor)
    target, keep = G6T.semantic_target("kitti360", {"anchor": anchor, "sequence": d}, REPO_ROOT)
    occ = OR.occupied_from_target(target, keep)
    lid = g.read_velodyne(native)
    lid_vol = OR.volume_from_points(lid)                    # already in the grid frame
    # future observations: the next 20 anchors' sweeps, mapped into this anchor's frame
    fut = np.zeros_like(lid_vol)
    for k in range(1, c["future_frames"] + 1):
        nf = native + k
        if not os.path.exists(g.velodyne_path(nf)) or nf not in g.cam0_to_world:
            continue
        p = TF.apply(g.velo_to_velo(nf, native), g.read_velodyne(nf))
        fut |= OR.volume_from_points(p)
    # trajectory of +-40 native frames, in the anchor's velodyne frame
    traj = []
    for nf in range(native - 40, native + 41):
        if nf in g.cam0_to_world:
            traj.append(TF.inv(g.velo_to_world(native))[:3, :3] @ g.velo_to_world(nf)[:3, 3]
                        + TF.inv(g.velo_to_world(native))[:3, 3])
    traj = np.asarray(traj) if traj else np.zeros((0, 3))
    # camera in the grid frame
    T_velo_from_cam = g.rect_cam_to_velo
    cam_o = T_velo_from_cam[:3, 3]
    cam_fwd = T_velo_from_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
    K = np.asarray(g.calib.K, np.float64); H, W = g.calib.native_hw
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    rays = np.stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1],
                     np.ones(4)], -1)
    frus = np.stack([TF.apply(T_velo_from_cam, r[None] * z) for z in (1.0, 50.0)
                     for r in rays]).reshape(-1, 3)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, (ia, ib, an, bn) in zip(axes, [(0, 1, "x forward (m)", "y left (m)"),
                                           (0, 2, "x forward (m)", "z up (m)")]):
        oi = np.argwhere(occ); li = np.argwhere(lid_vol); fi = np.argwhere(fut)
        for pts, col, lab, sz in ((oi, "#1f77b4", "official occupied target", 1.2),
                                  (fi, "#2ca02c", "future obs (t+1..t+20 GT LiDAR)", 0.5),
                                  (li, "#d62728", "current-frame GT LiDAR", 1.2)):
            if len(pts):
                m = TF.centres(pts)
                ax.scatter(m[:, ia], m[:, ib], s=sz, c=col, label=lab, alpha=.55, linewidths=0)
        if len(traj):
            ax.plot(traj[:, ia], traj[:, ib], "-", color="k", lw=1.2, label="GT trajectory ±40 frames")
        ax.plot([cam_o[ia]], [cam_o[ib]], marker="*", ms=13, c="orange", mec="k",
                zorder=6, label="camera origin", ls="none")
        ax.annotate("", xy=(cam_o[ia] + 6 * cam_fwd[ia], cam_o[ib] + 6 * cam_fwd[ib]),
                    xytext=(cam_o[ia], cam_o[ib]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="orange", lw=1.8))
        ax.add_patch(Polygon(np.stack([frus[4:8, ia], frus[4:8, ib]], -1), closed=True,
                             fill=False, ec="orange", ls="--", lw=1.1, label="frustum @ 50 m"))
        lo = TF.ORIGIN; hi = TF.ORIGIN + np.asarray(TF.DIMS) * TF.VOXEL
        ax.set_xlim(lo[ia], hi[ia]); ax.set_ylim(lo[ib], hi[ib])
        ax.set_xlabel(an); ax.set_ylabel(bn); ax.grid(alpha=.25)
        if ib == 1:
            ax.set_aspect("equal")
    axes[0].set_title("bird's-eye view"); axes[1].set_title("side view")
    axes[0].legend(fontsize=7, loc="lower right", markerscale=6, framealpha=.92)
    fig.suptitle(f"Gate 8C-0 · {row['partition']} · {d} · sscbench {anchor} (native {native}) "
                 f"· grid = velodyne of the anchor, 256×256×32 @ 0.2 m, origin (0, −25.6, −2)")
    fig.tight_layout(); fig.savefig(out_png, dpi=125); plt.close(fig)
    return {"n_target_occupied": int(occ.sum()), "n_valid": int(keep.sum()),
            "n_current_lidar_voxels": int(lid_vol.sum()), "n_future_lidar_voxels": int(fut.sum()),
            "camera_origin_in_grid_frame_m": [round(float(x), 4) for x in cam_o],
            "camera_forward_in_grid_frame": [round(float(x), 4) for x in cam_fwd]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    c = cfg(); rng = np.random.default_rng(c["seed"])
    t0 = time.time()
    rows = audit_anchors(c)
    out = {"tolerance_m": TOL_M, "audit_anchors": rows, "round_trips": {}, "figures": {},
           "chain": TF.__doc__}
    for r in rows:
        g = TF.DriveGeometry(r["drive"], c["sscbench_root"], c["kitti360_root"])
        na = g.native(r["anchor"])
        src = g.native(r["anchor"] - 4 * 5) if r["anchor"] >= 20 else na
        key = f"{r['drive']}_{r['anchor']:06d}"
        out["round_trips"][key] = round_trips(g, na, src, rng)
        if not a.no_figures:
            p = os.path.join(ART, f"fig_chain_{r['partition']}_{r['drive'][-9:-5]}_{r['anchor']:06d}.png")
            out["figures"][key] = dict(figure(c, r, p), png=os.path.relpath(p, REPO_ROOT))
            print("  wrote", os.path.basename(p), flush=True)
    bad = {k: {n: v for n, v in rt.items() if isinstance(v, dict) and v.get("within_tol") is False}
           for k, rt in out["round_trips"].items()}
    out["failures"] = {k: v for k, v in bad.items() if v}
    out["all_round_trips_pass"] = not out["failures"]
    out["seconds"] = time.time() - t0
    write_json(os.path.join(ART, "stage1_transforms.json"), out)
    e = max(v["max_abs_error_m"] for rt in out["round_trips"].values()
            for v in rt.values() if isinstance(v, dict) and "max_abs_error_m" in v)
    print(f"stage 1: {len(rows)} anchors, all round trips pass = {out['all_round_trips_pass']}, "
          f"worst error {e:.2e} m (tol {TOL_M:g})")
    r0 = out["round_trips"][next(iter(out["round_trips"]))]
    print(f"   R_rect_00 rotation {r0['R_rect_00_rotation_deg']:.4f} deg (matters: {r0['rect_matters']})")
    print(f"   cam0_to_velo det {r0['cam0_to_velo_orthonormality']['det']:.9f}, "
          f"|RtR-I|max {r0['cam0_to_velo_orthonormality']['max_abs_RtR_minus_I']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
