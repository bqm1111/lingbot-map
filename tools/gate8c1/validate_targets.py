#!/usr/bin/env python
"""Gate 8C-1: validate the rebuilt KITTI-360 supervision before any training starts.

Gate 8C-0 disqualified the published SSCBench label with five measurements. This runs the
same five against the rebuilt target, so the replacement is held to the standard that
caught the original:

1. **endpoint agreement by construction** -- every *supervised* voxel a raw return landed
   in must be OCCUPIED, and none may be FREE;
2. **transform round trips** -- the chain Gate 8C-0 verified, re-asserted on these drives;
3. **current-to-future alignment** -- future sweeps, transformed into the anchor frame with
   ground-truth poses, must land on the anchor's own surfaces;
4. **recall grows with future sweeps** -- more observation must mean more supervised
   occupancy, or the temporal aggregation is not doing anything;
5. **no systematic one-voxel offset** -- the +-1 voxel shift scan that SSCBench failed
   (+99 % IoU at +1 z) must be won by the identity.

SemanticKITTI and Occ3D are not touched: the firewall is still up.

    python tools/gate8c1/validate_targets.py --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402
from gates.gate8c1 import rawtarget as RT, sources as SRC                              # noqa: E402
from tools.gate8c1.build_targets import load as load_target                       # noqa: E402

SWEEP_COUNTS = (1, 5, 10, 20)
TOL_M = 1e-3


def pick(repo_root, per_drive=3):
    out = []
    for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]:
        ancs = [a for a in SRC.anchors(d, repo_root)
                if os.path.exists(SRC.target_path(d, a.stream_index)) and a.n_future >= 20]
        if not ancs:
            continue
        for q in np.linspace(0.2, 0.8, per_drive):
            out.append(ancs[int(q * (len(ancs) - 1))])
    return out


def figure(geo, anc, tgt, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dims = TF.DIMS
    occ = tgt["occupied"].reshape(dims)
    val = tgt["valid"].reshape(dims)
    cur = OR.volume_from_points(geo.read_velodyne(anc.native_frame))
    fut = np.zeros(dims, bool)
    for nf in anc.future_natives:
        if nf in geo.cam0_to_world:
            fut |= OR.volume_from_points(TF.apply(geo.velo_to_velo(nf, anc.native_frame),
                                                  geo.read_velodyne(nf)))
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, (ia, ib, an, bn) in zip(axes, [(0, 1, "x forward (m)", "y left (m)"),
                                           (0, 2, "x forward (m)", "z up (m)")]):
        for vol, col, lab, sz in ((occ, "#1f77b4", "rebuilt target: OCCUPIED", 1.0),
                                  (fut & ~cur, "#2ca02c", "future sweeps t+1..t+20", 0.4),
                                  (cur, "#d62728", "anchor sweep (t)", 1.0)):
            p = np.argwhere(vol)
            if len(p):
                m = TF.centres(p)
                ax.scatter(m[:, ia], m[:, ib], s=sz, c=col, label=lab, alpha=.55, linewidths=0)
        lo = TF.ORIGIN; hi = TF.ORIGIN + np.asarray(dims) * TF.VOXEL
        ax.set_xlim(lo[ia], hi[ia]); ax.set_ylim(lo[ib], hi[ib])
        ax.set_xlabel(an); ax.set_ylabel(bn); ax.grid(alpha=.25)
        if ib == 1:
            ax.set_aspect("equal")
    axes[0].set_title("bird's-eye view"); axes[1].set_title("side view")
    axes[0].legend(fontsize=7, loc="lower right", markerscale=6, framealpha=.92)
    fig.suptitle(f"Gate 8C-1 rebuilt supervision · {anc.drive} · stream {anc.stream_index} "
                 f"(native {anc.native_frame}) · valid {val.sum() / val.size:.1%} of grid, "
                 f"prevalence {occ.sum() / max(val.sum(), 1):.1%}")
    fig.tight_layout(); fig.savefig(out_png, dpi=125); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--per-drive", type=int, default=3)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    t0 = time.time()
    ancs = pick(REPO_ROOT, a.per_drive)
    geos = {}
    per, rt_fail = [], []
    growth = {n: [0, 0] for n in SWEEP_COUNTS}
    shift_tot = {}
    rng = np.random.default_rng(0)
    for anc in ancs:
        g = geos.setdefault(anc.drive, TF.DriveGeometry(anc.drive, SRC.SSCBENCH_ROOT,
                                                        SRC.KITTI360_ROOT))
        tgt = load_target(SRC.target_path(anc.drive, anc.stream_index))
        occ = tgt["occupied"].reshape(TF.DIMS)
        val = tgt["valid"].reshape(TF.DIMS)
        # -- 1. endpoint agreement by construction ------------------------------------
        pts = g.read_velodyne(anc.native_frame)
        idx, _ = TF.voxelize(pts)
        flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), TF.DIMS)
        sup = tgt["valid"][flat]
        is_occ = tgt["occupied"][flat]
        rec = {"drive": anc.drive, "partition": anc.partition,
               "stream_index": anc.stream_index, "native_frame": anc.native_frame,
               "n_endpoint_voxels": int(len(np.unique(flat))),
               "frac_endpoints_supervised": float(sup.mean()),
               "frac_supervised_endpoints_occupied": float(is_occ[sup].mean()) if sup.any() else float("nan"),
               "frac_endpoints_free": float((~is_occ & sup).mean()),
               "valid_fraction": float(val.mean()),
               "prevalence_in_valid": float(occ.sum() / max(val.sum(), 1))}
        # -- 2. transform round trips --------------------------------------------------
        p = rng.uniform(-40, 40, size=(2048, 3))
        errs = {}
        for nm, T in (("velo_to_world", g.velo_to_world(anc.native_frame)),
                      ("rect_cam_to_velo", g.rect_cam_to_velo),
                      ("velo_future_to_anchor",
                       g.velo_to_velo(anc.future_natives[-1], anc.native_frame))):
            e = float(np.abs(TF.apply(TF.inv(T), TF.apply(T, p)) - p).max())
            errs[nm] = e
            if e >= TOL_M:
                rt_fail.append({"anchor": rec["stream_index"], "pair": nm, "err": e})
        rec["round_trip_max_err_m"] = errs
        # -- 3. current-to-future alignment -------------------------------------------
        cur = OR.volume_from_points(pts)
        futv = np.zeros(TF.DIMS, bool)
        for nf in anc.future_natives:
            if nf in g.cam0_to_world:
                futv |= OR.volume_from_points(TF.apply(g.velo_to_velo(nf, anc.native_frame),
                                                       g.read_velodyne(nf)))
        d = OR.distance_stats(futv & val, cur)
        rec["future_to_current_surface_distance"] = d
        # -- 4. recall grows with the number of future sweeps -------------------------
        for n in SWEEP_COUNTS:
            sub = [anc.native_frame] + list(anc.future_natives[:n - 1])
            t = RT.build(g, anc.native_frame, sub, dev)
            growth[n][0] += int(t.occupied.sum()); growth[n][1] += 1
            rec.setdefault("sweep_growth", {})[str(n)] = {
                "n_occupied": int(t.occupied.sum()), "n_valid": int(t.valid.sum()),
                "recall_of_full_target": float((t.occupied & tgt["occupied"]).sum()
                                               / max(tgt["occupied"].sum(), 1))}
        # -- 5. no systematic one-voxel offset ----------------------------------------
        sc = OR.shift_scan(cur, occ, val)
        for k, v in sc.items():
            s = shift_tot.setdefault(k, [0, 0, 0])
            for i, f in enumerate(("tp", "fp", "fn")):
                s[i] += v[f]
        rec["shift_scan"] = {k: {"precision": v["precision"], "iou": v["iou"]}
                             for k, v in sc.items()}
        per.append(rec)
        print(f"  {anc.drive[-9:-5]} {anc.partition:17s} s{anc.stream_index:4d}  "
              f"endpoints supervised {rec['frac_endpoints_supervised']:.3f} -> occupied "
              f"{rec['frac_supervised_endpoints_occupied']:.4f} (free {rec['frac_endpoints_free']:.4f})  "
              f"fut->cur median {d['median_m']:.3f} m", flush=True)
    sh = {k: {"precision": v[0] / max(v[0] + v[1], 1), "iou": v[0] / max(sum(v), 1)}
          for k, v in shift_tot.items()}
    best = max(sh.items(), key=lambda kv: kv[1]["iou"])
    gr = {str(n): growth[n][0] / max(growth[n][1], 1) for n in SWEEP_COUNTS}
    out = {"n_anchors": len(per), "tolerance_m": TOL_M, "sweep_counts": list(SWEEP_COUNTS),
           "per_anchor": per, "aggregate_shift_scan": sh,
           "mean_occupied_by_sweep_count": gr,
           "checks": {
               "endpoints_never_labelled_free": bool(all(r["frac_endpoints_free"] == 0.0 for r in per)),
               "supervised_endpoints_all_occupied": bool(all(
                   r["frac_supervised_endpoints_occupied"] >= 1.0 - 1e-12 for r in per)),
               "all_round_trips_within_tolerance": bool(not rt_fail),
               "occupancy_grows_with_future_sweeps": bool(all(
                   gr[str(b)] >= gr[str(a_)] for a_, b in zip(SWEEP_COUNTS, SWEEP_COUNTS[1:]))),
               "identity_wins_shift_scan": bool(best[0] == "(0, 0, 0)"),
               "identity_iou": sh["(0, 0, 0)"]["iou"], "best_shift": best[0],
               "best_shift_iou": best[1]["iou"],
               "shift_relative_gain": (best[1]["iou"] - sh["(0, 0, 0)"]["iou"])
               / max(sh["(0, 0, 0)"]["iou"], 1e-9),
               "future_to_current_median_distance_m": float(np.median(
                   [r["future_to_current_surface_distance"]["median_m"] for r in per]))},
           "round_trip_failures": rt_fail, "seconds": time.time() - t0}
    out["all_checks_pass"] = bool(all(v for k, v in out["checks"].items()
                                      if isinstance(v, bool)))
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, "target_validation.json"), out)
    if not a.no_figures:
        for anc in ancs[::max(1, len(ancs) // 4)]:
            g = geos[anc.drive]
            tgt = load_target(SRC.target_path(anc.drive, anc.stream_index))
            p = os.path.join(ART, f"fig_target_{anc.drive[-9:-5]}_{anc.stream_index:05d}.png")
            figure(g, anc, tgt, p)
            print("  wrote", os.path.basename(p), flush=True)
    c = out["checks"]
    print(f"\nvalidation over {len(per)} anchors: all checks pass = {out['all_checks_pass']}")
    for k, v in c.items():
        print(f"   {k}: {v if not isinstance(v, float) else round(v, 6)}")
    print(f"   mean occupied voxels by sweep count: {gr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
