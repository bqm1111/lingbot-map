#!/usr/bin/env python
"""Gate 8C-0 Stage 3: oracle temporal reconstruction from ground-truth LiDAR and poses.

No learned completion, no predicted depth, no predicted pose. Ground-truth velodyne
sweeps of the causal history are transformed into the anchor's grid with ground-truth
poses and integrated exactly once each. This bounds what *any* method reading this map
could achieve: if a 20-frame ground-truth oracle cannot recall the KITTI-360 target, the
target is not reachable from causal observation and no completion head is at fault.

History lengths count **stream frames**, which sit one SSCBench anchor apart, i.e. five
native frames -- the same cadence the Gate 8B causal map used. "All available past" walks
back until the sensor has left the 51.2 m box (capped at 200 native frames), since a
sweep taken further back contributes no voxel to this grid.

The ``future`` row is the same construction over frames t+1..t+20. It is a
**target-quality diagnostic**: it measures how much of the privileged completion target
the teacher's own future views can even see. It is never a deployable baseline.

    python tools/gate8c0/stage3_temporal.py
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg                                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import targets as G6T                                                 # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402
from tools.gate8c0.stage2_alignment import one as stage2_one                      # noqa: E402

STREAM_STRIDE = 5            # native frames between consecutive stream frames
MAX_PAST_NATIVE = 200        # a sweep further back than this has left the 51.2 m box


def oracle_volume(g: TF.DriveGeometry, native: int, offsets) -> tuple:
    """Union of the sweeps at ``native + o`` for o in ``offsets``, in the anchor's grid."""
    vol = np.zeros(TF.DIMS, bool)
    used = []
    for o in offsets:
        nf = int(native + o)
        if nf not in g.cam0_to_world or not os.path.exists(g.velodyne_path(nf)):
            continue
        p = g.read_velodyne(nf)
        if o != 0:
            p = TF.apply(g.velo_to_velo(nf, native), p)
        idx, _ = TF.voxelize(p)
        if len(idx):
            vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        used.append(nf)
    return vol, used


def history_offsets(kind, g: TF.DriveGeometry, native: int):
    if kind == "future":
        return list(range(1, 21))
    if kind == "all_past":
        out, diag = [0], np.linalg.norm(np.asarray(TF.DIMS) * TF.VOXEL)
        for k in range(1, MAX_PAST_NATIVE + 1):
            nf = native - k * STREAM_STRIDE
            if nf < 0 or nf not in g.cam0_to_world:
                break
            t = g.velo_to_velo(nf, native)[:3, 3]
            if np.linalg.norm(t) > diag:
                break
            out.append(-k * STREAM_STRIDE)
        return out
    return [-k * STREAM_STRIDE for k in range(int(kind))][::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-drive", type=int, default=6)
    a = ap.parse_args()
    c = cfg(); t0 = time.time()
    part_of = {}
    for part, ds in (("train", c["train_drives"]), ("source_validation", [c["val_drive"]]),
                     ("heldout", [c["heldout_drive"]])):
        for d in ds:
            part_of[d] = part
    rng = np.random.default_rng(c["seed"])
    kinds = ["1", "5", "20", "all_past", "future"]
    per, store = [], {}
    for d, part in part_of.items():
        g = TF.DriveGeometry(d, c["sscbench_root"], c["kitti360_root"])
        ldir = os.path.join(c["sscbench_root"], "preprocess", "labels", d)
        anchors = sorted(int(f.split("_")[0]) for f in os.listdir(ldir) if f.endswith("_1_1.npy"))
        elig = [x for x in anchors if x >= 20 * 5 and x <= anchors[-1] - 25]
        pick = sorted(int(x) for x in rng.choice(elig, size=min(a.per_drive, len(elig)),
                                                 replace=False))
        for anc in pick:
            native = g.native(anc)
            target, keep = G6T.semantic_target("kitti360", {"anchor": anc, "sequence": d},
                                               REPO_ROOT)
            occ = OR.occupied_from_target(target, keep)
            T_grid_to_cam = TF.inv(g.rect_cam_to_velo)
            fr = OR.frustum_mask(g.calib.K, g.calib.native_hw, T_grid_to_cam)
            rec = {"partition": part, "drive": d, "anchor": anc, "native_frame": native,
                   "n_target_occupied": int(occ.sum()), "n_valid": int(keep.sum()),
                   "target_prevalence": float(occ.sum() / max(keep.sum(), 1)),
                   "valid_fraction_of_grid": float(keep.sum() / np.prod(TF.DIMS)),
                   "histories": {}}
            for kind in kinds:
                offs = history_offsets(kind, g, native)
                vol, used = oracle_volume(g, native, offs)
                cnt_full = OR.counts(vol, occ, keep)
                cnt_fr = OR.counts(vol, occ, keep & fr)
                dist = OR.distance_stats(vol & keep, occ)
                rec["histories"][kind] = {
                    "n_frames_requested": len(offs), "n_frames_used": len(used),
                    "native_span": [min(used), max(used)] if used else None,
                    "n_voxels": int((vol & keep).sum()),
                    "full_grid": cnt_full, "frustum": cnt_fr, "distance": dist,
                    "target_coverage": cnt_full["recall"],
                    "valid_grid_coverage": float((vol & keep).sum() / max(keep.sum(), 1))}
                for reg, cc in (("full_grid", cnt_full), ("frustum", cnt_fr)):
                    k = (part, kind, reg); store.setdefault(k, [0, 0, 0])
                    for i, f in enumerate(("tp", "fp", "fn")):
                        store[k][i] += cc[f]
            per.append(rec)
            h = rec["histories"]
            print(f"  {part:17s} {d[-9:-5]} {anc:6d} prev {rec['target_prevalence']:.3f} | "
                  + " ".join(f"{k}:R{h[k]['full_grid']['recall']:.3f}/P{h[k]['full_grid']['precision']:.3f}"
                             for k in kinds), flush=True)
    agg = {}
    for (part, kind, reg), (tp, fp, fn) in store.items():
        agg.setdefault(part, {}).setdefault(kind, {})[reg] = {
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}
    for part in agg:
        rs = [x for x in per if x["partition"] == part]
        for kind in kinds:
            agg[part][kind]["median_distance_m"] = float(np.median(
                [x["histories"][kind]["distance"]["median_m"] for x in rs]))
            agg[part][kind]["mean_valid_grid_coverage"] = float(np.mean(
                [x["histories"][kind]["valid_grid_coverage"] for x in rs]))
            agg[part][kind]["mean_n_frames_used"] = float(np.mean(
                [x["histories"][kind]["n_frames_used"] for x in rs]))
        agg[part]["target_prevalence"] = float(np.mean([x["target_prevalence"] for x in rs]))
        agg[part]["valid_fraction_of_grid"] = float(np.mean([x["valid_fraction_of_grid"] for x in rs]))
    out = {"stream_stride_native_frames": STREAM_STRIDE, "max_past_native": MAX_PAST_NATIVE,
           "history_kinds": kinds, "n_samples": len(per), "aggregate": agg, "per_sample": per,
           "note": "'future' is a target-quality diagnostic over t+1..t+20, never a baseline",
           "seconds": time.time() - t0}
    # does recall improve monotonically with history?
    mono = {}
    for part, v in agg.items():
        r = [v[k]["full_grid"]["recall"] for k in ("1", "5", "20", "all_past")]
        mono[part] = {"recalls": r, "monotone_non_decreasing": all(b >= a - 1e-9 for a, b in zip(r, r[1:])),
                      "future_target_coverage": v["future"]["full_grid"]["recall"],
                      "precision_at_1_frame": v["1"]["full_grid"]["precision"]}
    out["sanity"] = mono
    write_json(os.path.join(ART, "stage3_temporal.json"), out)
    print(f"\nstage 3 aggregate ({len(per)} samples)")
    for part, v in agg.items():
        print(f"  {part} (target prevalence {v['target_prevalence']:.3f}, "
              f"valid {v['valid_fraction_of_grid']:.3f} of grid)")
        for kind in kinds:
            f_, fr = v[kind]["full_grid"], v[kind]["frustum"]
            print(f"     {kind:9s} frames {v[kind]['mean_n_frames_used']:5.1f}  full P {f_['precision']:.4f} "
                  f"R {f_['recall']:.4f} IoU {f_['iou']:.4f} | frustum R {fr['recall']:.4f} "
                  f"| cover {v[kind]['mean_valid_grid_coverage']:.4f} "
                  f"| med dist {v[kind]['median_distance_m']:.3f} m")
        print(f"     recall monotone with history: {mono[part]['monotone_non_decreasing']} "
              f"{[round(x,4) for x in mono[part]['recalls']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
