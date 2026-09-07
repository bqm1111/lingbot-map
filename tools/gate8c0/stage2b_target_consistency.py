#!/usr/bin/env python
"""Gate 8C-0 Stage 2b: is the +1 voxel z offset ours, or the published dataset's?

Stage 2 found that shifting our voxelized LiDAR up by one voxel nearly doubles agreement
with the KITTI-360 target. The brief forbids adopting a better-scoring alignment without
confirming it against official metadata, so this tool runs the three controls that decide
where the offset lives:

1. **our voxelization vs SSCBench's own ``.bin``** -- their published single-frame voxel
   input for the same anchor, in the same grid. If our transform chain were wrong this
   would fail at zero shift.
2. **SSCBench's own ``.bin`` vs its own ``.label``/``_1_1.npy``** -- both official files,
   ours nowhere in the path. If the offset appears here it is internal to the release.
3. **the same test on SemanticKITTI**, whose voxelizer is the reference implementation, to
   show the offset is not a general property of SSC completion targets.

It also re-verifies the adapter's elementwise 255/0/occupied rule against the raw
``.label`` + ``.invalid`` files.

    python tools/gate8c0/stage2b_target_consistency.py
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg                                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G, targets as G6T, vocab             # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as SKG                 # noqa: E402

SHIFTS = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 0, 2)]
DZ = np.round(np.arange(-0.30, 0.61, 0.02), 3)


def _acc(store, key, pred, gt, valid, shifts=SHIFTS):
    for sh in shifts:
        c = OR.counts(OR.shift_volume(pred, sh), gt, valid)
        k = (key, str(sh)); store.setdefault(k, [0, 0, 0])
        for i, f in enumerate(("tp", "fp", "fn")):
            store[k][i] += c[f]


def _rates(store):
    out = {}
    for (key, sh), (tp, fp, fn) in store.items():
        out.setdefault(key, {})[sh] = {"precision": tp / max(tp + fp, 1),
                                       "recall": tp / max(tp + fn, 1),
                                       "iou": tp / max(tp + fp + fn, 1),
                                       "tp": tp, "fp": fp, "fn": fn}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k360-anchors", type=int, default=8)
    ap.add_argument("--sk-clips", type=int, default=8)
    a = ap.parse_args()
    c = cfg(); t0 = time.time()
    d = c["heldout_drive"]
    g = TF.DriveGeometry(d, c["sscbench_root"], c["kitti360_root"])
    VD = os.path.join(c["sscbench_root"], "data_2d_raw", d, "voxels")
    have = sorted(int(f[:-4]) for f in os.listdir(VD) if f.endswith(".bin")) if os.path.isdir(VD) else []
    rng = np.random.default_rng(c["seed"])
    anchors = sorted(int(x) for x in rng.choice(have[2:-2], size=min(a.k360_anchors, len(have) - 4),
                                                replace=False)) if have else []
    store, rule_ok, dz_scan = {}, [], {float(x): [0, 0, 0] for x in DZ}
    vk = vocab.load("kitti360"); names_k = dict(zip(vk.labels, vk.names))
    cls = {str(s): {} for s in ((0, 0, 0), (0, 0, 1))}
    for anc in anchors:
        native = g.native(anc)
        bin_ = np.unpackbits(np.fromfile(f"{VD}/{anc:06d}.bin", np.uint8)).reshape(TF.DIMS).astype(bool)
        lab = np.fromfile(f"{VD}/{anc:06d}.label", np.uint16).reshape(TF.DIMS)
        inv = np.fromfile(f"{VD}/{anc:06d}.invalid", np.uint8).reshape(TF.DIMS).astype(bool)
        npy, keep = G6T.semantic_target("kitti360", {"anchor": anc, "sequence": d}, REPO_ROOT)
        occ_npy = OR.occupied_from_target(npy, keep)
        rule = np.where(inv & (lab == 0), 255, np.where(lab == 0, 0, -1))
        rule_ok.append({"anchor": anc,
                        "ignore_255_exact": bool(((npy == 255) == (rule == 255)).all()),
                        "free_0_exact": bool(((npy == 0) == (rule == 0)).all()),
                        "occupied_set_equals_label_gt_0": bool((occ_npy == (lab > 0)).all())})
        pts = g.read_velodyne(native)
        mine = OR.volume_from_points(pts)
        ones = np.ones_like(keep)
        _acc(store, "ours_vs_official_bin", mine, bin_, ones)
        _acc(store, "official_bin_vs_official_label", bin_, occ_npy, keep)
        _acc(store, "ours_vs_official_label", mine, occ_npy, keep)
        for sh in ((0, 0, 0), (0, 0, 1)):
            l = npy[OR.shift_volume(mine, sh) & keep]
            u, cn = np.unique(l, return_counts=True)
            for x, n in zip(u, cn):
                cls[str(sh)][int(x)] = cls[str(sh)].get(int(x), 0) + int(n)
        for dz in DZ:
            p = pts.copy(); p[:, 2] += dz
            cc = OR.counts(OR.volume_from_points(p), occ_npy, keep)
            for i, f in enumerate(("tp", "fp", "fn")):
                dz_scan[float(dz)][i] += cc[f]
    # ---- SemanticKITTI control -------------------------------------------------------
    sk = {}
    root = os.path.join(REPO_ROOT, "data/kitti/dataset")
    recs = [r.raw for r in G6F.read_manifest("semantickitti", REPO_ROOT)]
    sel = recs[10:10 + a.sk_clips * 6:6]
    vs = vocab.load("semantickitti"); names_s = dict(zip(vs.labels, vs.names))
    cls_sk = {str(s): {} for s in ((0, 0, 0), (0, 0, 1))}
    for rec in sel:
        f = int(rec["frame_ids"][-1])
        pts = np.fromfile(os.path.join(root, "sequences", "08", "velodyne", f"{f:06d}.bin"),
                          np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        t, keep = G6T.semantic_target("semantickitti", rec, REPO_ROOT)
        occ = (t != SKG.empty_class) & keep
        idx, _ = G6G.voxelize(pts, SKG)
        vol = np.zeros(tuple(SKG.dims), bool); vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        _acc(sk, "sweep_vs_official_label", vol, occ, keep)
        for sh in ((0, 0, 0), (0, 0, 1)):
            l = t[OR.shift_volume(vol, sh) & keep]
            u, cn = np.unique(l, return_counts=True)
            for x, n in zip(u, cn):
                cls_sk[str(sh)][int(x)] = cls_sk[str(sh)].get(int(x), 0) + int(n)
    R = _rates(store); RS = _rates(sk)
    dz_rates = {str(k): {"precision": v[0] / max(v[0] + v[1], 1),
                         "iou": v[0] / max(sum(v), 1)} for k, v in dz_scan.items()}
    best_dz = max(dz_rates.items(), key=lambda kv: kv[1]["iou"])
    out = {"n_kitti360_anchors": len(anchors), "kitti360_anchors": anchors,
           "n_semantickitti_clips": len(sel),
           "adapter_rule_verification": rule_ok,
           "adapter_rule_holds_everywhere": all(all(v for k, v in r.items() if k != "anchor")
                                                for r in rule_ok),
           "kitti360": R, "semantickitti": RS,
           "kitti360_continuous_dz_scan": dz_rates,
           "kitti360_best_dz_m": float(best_dz[0]), "kitti360_best_dz_iou": best_dz[1]["iou"],
           "kitti360_label_of_voxel_hit": {sh: {("EMPTY" if k == 0 else "IGNORE" if k == 255
                                                 else names_k.get(k, str(k))): n
                                                for k, n in sorted(v.items(), key=lambda kv: -kv[1])[:12]}
                                           for sh, v in cls.items()},
           "semantickitti_label_of_voxel_hit": {sh: {("EMPTY" if k == 0 else "IGNORE" if k == 255
                                                      else names_s.get(k, str(k))): n
                                                     for k, n in sorted(v.items(), key=lambda kv: -kv[1])[:12]}
                                                for sh, v in cls_sk.items()},
           "seconds": time.time() - t0}
    z0 = R["official_bin_vs_official_label"]["(0, 0, 0)"]["precision"]
    z1 = R["official_bin_vs_official_label"]["(0, 0, 1)"]["precision"]
    out["verdict"] = {
        "our_chain_matches_official_input": bool(
            R["ours_vs_official_bin"]["(0, 0, 0)"]["recall"] > 0.95
            and R["ours_vs_official_bin"]["(0, 0, 0)"]["iou"]
            > max(R["ours_vs_official_bin"][s]["iou"] for s in ("(0, 0, 1)", "(0, 0, -1)"))),
        "offset_is_internal_to_the_published_release": bool(z1 > 1.4 * z0),
        "official_bin_vs_label_precision_at_0": z0,
        "official_bin_vs_label_precision_at_plus1z": z1,
        "semantickitti_is_aligned_at_zero_shift": bool(
            RS["sweep_vs_official_label"]["(0, 0, 0)"]["precision"] > 0.95
            and RS["sweep_vs_official_label"]["(0, 0, 0)"]["precision"]
            > RS["sweep_vs_official_label"]["(0, 0, 1)"]["precision"])}
    write_json(os.path.join(ART, "stage2b_target_consistency.json"), out)
    print(f"stage 2b ({len(anchors)} KITTI-360 anchors, {len(sel)} SemanticKITTI clips)")
    for key in ("ours_vs_official_bin", "official_bin_vs_official_label", "ours_vs_official_label"):
        print(f"  KITTI-360 {key}")
        for sh in ("(0, 0, 0)", "(0, 0, 1)", "(0, 0, -1)"):
            v = R[key][sh]
            print(f"     {sh:11s} P {v['precision']:.4f} R {v['recall']:.4f} IoU {v['iou']:.4f}")
    print("  SemanticKITTI sweep_vs_official_label")
    for sh in ("(0, 0, 0)", "(0, 0, 1)", "(0, 0, -1)"):
        v = RS["sweep_vs_official_label"][sh]
        print(f"     {sh:11s} P {v['precision']:.4f} R {v['recall']:.4f} IoU {v['iou']:.4f}")
    print(f"  best continuous dz on KITTI-360: {out['kitti360_best_dz_m']:+.2f} m")
    for k, v in out["verdict"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
