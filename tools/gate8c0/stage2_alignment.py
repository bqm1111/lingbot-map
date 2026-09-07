#!/usr/bin/env python
"""Gate 8C-0 Stage 2: does the anchor frame's own GT LiDAR land on the official target?

The one question that separates a pipeline defect from everything else. Ground-truth
velodyne of the anchor frame is already *in* the grid frame (the grid is the anchor's
velodyne frame), so this measures the adapter's origin, resolution, axis order and binning
with no pose, no depth model and no learned component in the path.

Recall is expected to be low -- one sweep cannot cover a densified completion target -- so
the test is **precision and surface distance**. The +-1 voxel shift scan and the
neighbouring-anchor scan are reported as diagnosis: if either beats the identity by a wide
margin, the pipeline is off by that amount.

    python tools/gate8c0/stage2_alignment.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg                                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import targets as G6T                                                 # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402
from tools.gate8c0.stage1_transforms import audit_anchors                        # noqa: E402

NEIGHBOUR_ANCHORS = (-2, -1, 0, 1, 2)      # in SSCBench index units (5 native frames each)


def one(c, drive, anchor, g=None, n_neighbour=True) -> dict:
    g = g or TF.DriveGeometry(drive, c["sscbench_root"], c["kitti360_root"])
    native = g.native(anchor)
    target, keep = G6T.semantic_target("kitti360", {"anchor": anchor, "sequence": drive},
                                       REPO_ROOT)
    occ = OR.occupied_from_target(target, keep)
    pts = g.read_velodyne(native)
    vol = OR.volume_from_points(pts)
    idx_all, kept = TF.voxelize(pts)
    T_grid_to_cam = TF.inv(g.rect_cam_to_velo)
    fr = OR.frustum_mask(g.calib.K, g.calib.native_hw, T_grid_to_cam)
    out = {"drive": drive, "anchor": anchor, "native_frame": native,
           "n_lidar_points": int(len(pts)), "n_lidar_points_in_grid": int(kept.sum()),
           "n_lidar_voxels": int(vol.sum()),
           "full_grid": OR.counts(vol, occ, keep),
           "frustum": OR.counts(vol, occ, keep & fr),
           "n_frustum_valid": int((keep & fr).sum()),
           "distance_full": OR.distance_stats(vol & keep, occ),
           "distance_frustum": OR.distance_stats(vol & keep & fr, occ),
           "shift_scan": OR.shift_scan(vol, occ, keep)}
    # in-grid LiDAR voxels overlapping official occupied, as the brief phrases it
    ig = vol & keep
    out["fraction_of_in_grid_lidar_voxels_on_official_occupied"] = \
        float((ig & occ).sum() / max(ig.sum(), 1))
    if n_neighbour:
        nb = {}
        for k in NEIGHBOUR_ANCHORS:
            a2 = anchor + k * 5
            p2 = os.path.join(c["sscbench_root"], "preprocess", "labels", drive,
                              f"{a2:06d}_1_1.npy")
            if not os.path.exists(p2):
                continue
            t2, k2 = G6T.semantic_target("kitti360", {"anchor": a2, "sequence": drive}, REPO_ROOT)
            o2 = OR.occupied_from_target(t2, k2)
            nb[str(k)] = OR.counts(vol, o2, k2)
        out["neighbour_anchor_scan"] = nb
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra-per-drive", type=int, default=7,
                    help="additional anchors per drive beyond the Stage-1 audit subset")
    a = ap.parse_args()
    c = cfg(); t0 = time.time()
    rows = list(audit_anchors(c))
    seen = {(r["drive"], r["anchor"]) for r in rows}
    part_of = {}
    for part, ds in (("train", c["train_drives"]), ("source_validation", [c["val_drive"]]),
                     ("heldout", [c["heldout_drive"]])):
        for d in ds:
            part_of[d] = part
    rng = np.random.default_rng(c["seed"])
    for d, part in part_of.items():
        ldir = os.path.join(c["sscbench_root"], "preprocess", "labels", d)
        anchors = sorted(int(f.split("_")[0]) for f in os.listdir(ldir) if f.endswith("_1_1.npy"))
        pick = rng.choice(anchors[2:-2], size=min(a.extra_per_drive, len(anchors) - 4),
                          replace=False)
        for x in sorted(int(v) for v in pick):
            if (d, x) not in seen:
                rows.append({"partition": part, "drive": d, "anchor": x}); seen.add((d, x))
    per, geo = [], {}
    for r in rows:
        geo.setdefault(r["drive"], TF.DriveGeometry(r["drive"], c["sscbench_root"],
                                                    c["kitti360_root"]))
        rec = one(c, r["drive"], r["anchor"], geo[r["drive"]],
                  n_neighbour=(r["drive"], r["anchor"]) in
                  {(x["drive"], x["anchor"]) for x in audit_anchors(c)})
        rec["partition"] = r["partition"]
        per.append(rec)
        print(f"  {r['partition']:17s} {r['drive'][-9:-5]} {r['anchor']:6d}  "
              f"P {rec['full_grid']['precision']:.4f} R {rec['full_grid']['recall']:.4f} "
              f"IoU {rec['full_grid']['iou']:.4f} | frustum P {rec['frustum']['precision']:.4f} "
              f"| median dist {rec['distance_full']['median_m']:.3f} m "
              f"p95 {rec['distance_full']['p95_m']:.3f} m", flush=True)

    def agg(rs, key):
        tp = sum(x[key]["tp"] for x in rs); fp = sum(x[key]["fp"] for x in rs)
        fn = sum(x[key]["fn"] for x in rs)
        return {"n_samples": len(rs), "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1)}

    out = {"n_samples": len(per), "neighbour_anchor_offsets_in_sscbench_index": list(NEIGHBOUR_ANCHORS),
           "aggregate": {}, "per_partition": {}, "per_sample": per, "seconds": time.time() - t0}
    for reg in ("full_grid", "frustum"):
        out["aggregate"][reg] = agg(per, reg)
        for part in ("train", "source_validation", "heldout"):
            rs = [x for x in per if x["partition"] == part]
            if rs:
                out["per_partition"].setdefault(part, {})[reg] = agg(rs, reg)
    for part in ("train", "source_validation", "heldout"):
        rs = [x for x in per if x["partition"] == part]
        if rs:
            out["per_partition"][part]["distance_full_median_m"] = \
                float(np.median([x["distance_full"]["median_m"] for x in rs]))
            out["per_partition"][part]["distance_full_p95_m"] = \
                float(np.median([x["distance_full"]["p95_m"] for x in rs]))
            out["per_partition"][part]["frac_lidar_voxels_on_official_occupied"] = \
                float(np.mean([x["fraction_of_in_grid_lidar_voxels_on_official_occupied"] for x in rs]))
    # aggregate shift scan: does any integer shift beat the identity?
    sh = {}
    for k in per[0]["shift_scan"]:
        tp = sum(x["shift_scan"][k]["tp"] for x in per)
        fp = sum(x["shift_scan"][k]["fp"] for x in per)
        fn = sum(x["shift_scan"][k]["fn"] for x in per)
        sh[k] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                 "iou": tp / max(tp + fp + fn, 1)}
    out["aggregate_shift_scan"] = sh
    base = sh["(0, 0, 0)"]["iou"]
    best = max(sh.items(), key=lambda kv: kv[1]["iou"])
    out["shift_verdict"] = {"identity_iou": base, "best_shift": best[0], "best_iou": best[1]["iou"],
                            "relative_gain": (best[1]["iou"] - base) / max(base, 1e-9),
                            "identity_is_best": best[0] == "(0, 0, 0)"}
    nb = {}
    withnb = [x for x in per if "neighbour_anchor_scan" in x]
    for k in ("-2", "-1", "0", "1", "2"):
        rs = [x["neighbour_anchor_scan"][k] for x in withnb if k in x["neighbour_anchor_scan"]]
        if rs:
            tp = sum(r["tp"] for r in rs); fp = sum(r["fp"] for r in rs); fn = sum(r["fn"] for r in rs)
            nb[k] = {"n": len(rs), "precision": tp / max(tp + fp, 1), "iou": tp / max(tp + fp + fn, 1)}
    out["aggregate_neighbour_anchor_scan"] = nb
    out["timing_verdict"] = {"best_offset": max(nb.items(), key=lambda kv: kv[1]["iou"])[0] if nb else None,
                             "zero_is_best": (max(nb.items(), key=lambda kv: kv[1]["iou"])[0] == "0") if nb else None}
    write_json(os.path.join(ART, "stage2_alignment.json"), out)
    A = out["aggregate"]
    print(f"\nstage 2 aggregate over {len(per)} samples")
    print(f"   full grid : P {A['full_grid']['precision']:.4f} R {A['full_grid']['recall']:.4f} "
          f"IoU {A['full_grid']['iou']:.4f}")
    print(f"   frustum   : P {A['frustum']['precision']:.4f} R {A['frustum']['recall']:.4f} "
          f"IoU {A['frustum']['iou']:.4f}")
    print(f"   shift scan: identity IoU {base:.4f}, best {best[0]} IoU {best[1]['iou']:.4f} "
          f"({100*out['shift_verdict']['relative_gain']:+.1f}%)  identity best = "
          f"{out['shift_verdict']['identity_is_best']}")
    print(f"   anchor timing: best offset {out['timing_verdict']['best_offset']} "
          f"(0 is best = {out['timing_verdict']['zero_is_best']})  " +
          " ".join(f"{k}:{v['iou']:.4f}" for k, v in nb.items()))
    for part, v in out["per_partition"].items():
        print(f"   {part:17s} P {v['full_grid']['precision']:.4f} "
              f"median dist {v['distance_full_median_m']:.3f} m  "
              f"frac on official occupied {v['frac_lidar_voxels_on_official_occupied']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
