#!/usr/bin/env python
"""Pre-training validation for the ONCE arm: brief §6, checks 1-9, measured not asserted.

Each check is a number against the KITTI-360 arm's own cache measured the same way, so
"correct" means "in family with the arm this recipe was built on", not "passes a threshold I
chose". Check 10 (the frozen KITTI-360 evaluation is unchanged) is the source freeze, run
separately.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/validate.py --device cuda:3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from tools.method_vs_data.once_arm import anchors as AN, config as A, geometry as G   # noqa: E402
from tools.method_vs_data.once_reader import OnceSequence                             # noqa: E402

K360_INPUTS = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs"
K360_TARGETS = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets"
N = 256 * 256 * 32
un = lambda x: np.unpackbits(x)[:N].astype(bool)                               # noqa: E731


def map_vs_target(ipath: str, tpath: str) -> dict | None:
    if not (os.path.exists(ipath) and os.path.exists(tpath)):
        return None
    zi, zt = np.load(ipath), np.load(tpath)
    S, F = un(zi["S_H"]), un(zi["F_H"])
    occ, val = un(zt["occ_packed"]), un(zt["valid_packed"])
    tp = int((S & occ & val).sum()); fp = int((S & ~occ & val).sum())
    fn = int((~S & occ & val).sum())
    return {"precision": 100 * tp / max(tp + fp, 1), "recall": 100 * tp / max(tp + fn, 1),
            "iou": 100 * tp / max(tp + fp + fn, 1), "n_S": int(S.sum()), "n_F": int(F.sum()),
            "n_gt_occ": int(occ.sum()), "n_valid": int(val.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-align", type=int, default=400, help="anchors for the alignment stats")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = A.Paths()
    man = AN.load_or_build(p)
    R: dict = {"provenance": A.provenance(), "manifest_digest": man["digest"], "checks": {}}

    # ---- 2. five-frame causal ordering, 7. train/val separation ---------------
    bad_order, bad_future, spans = 0, 0, []
    for r in man["anchors"]:
        idx = r["input_frames"]
        if idx != sorted(idx) or idx[-1] != r["anchor"]:
            bad_order += 1
        if max(idx) > r["anchor"]:                      # no future camera frame may enter
            bad_future += 1
        spans.append(r["input_span_s"])
    sp = A.split(A.sequences())
    R["checks"]["2_causal_ordering"] = {
        "n_anchors": len(man["anchors"]), "anchors_misordered": bad_order,
        "anchors_using_a_future_camera_frame": bad_future,
        "input_span_s_p50": float(np.median(spans)), "pass": bad_order == 0 and bad_future == 0}
    R["checks"]["7_train_val_sequence_disjoint"] = {
        "train": sp["train"], "val": sp["val"],
        "overlap": sorted(set(sp["train"]) & set(sp["val"])),
        "pass": not (set(sp["train"]) & set(sp["val"]))}

    # ---- 3. poses and scale --------------------------------------------------
    scales, logs_ok, rope = [], [], []
    for f in sorted(glob.glob(os.path.join(p.clips, "*", "*.npz")))[:a.n_align]:
        z = np.load(f, allow_pickle=True)
        scales.append(float(z["scale"]))
        logs_ok.append(int(np.isfinite(z["log_s_frames"]).sum()))
        rope.append(int(z["rope_slots_used"]))
    sc = np.array(scales)
    R["checks"]["3_scale_and_poses"] = {
        "n_clips_checked": len(sc), "scale_nonfinite": int((~np.isfinite(sc)).sum()),
        "scale_m_per_unit": {"p10": float(np.percentile(sc, 10)),
                             "p50": float(np.median(sc)), "p90": float(np.percentile(sc, 90))},
        "moge_candidates_per_clip_p50": float(np.median(logs_ok)),
        "rope_slots_used_max": int(max(rope)),
        "pass": bool((~np.isfinite(sc)).sum() == 0 and min(logs_ok) > 0)}

    # ---- 1, 5, 6. camera/LiDAR alignment, tensor contract, target agreement ---
    once_rows = []
    for f in sorted(glob.glob(os.path.join(p.inputs, "*", "*.npz")))[:a.n_align]:
        s = os.path.basename(os.path.dirname(f)); t = int(os.path.basename(f)[:-4])
        m = map_vs_target(f, p.target_path(s, t))
        if m:
            once_rows.append(m)
    k_rows = []
    for d in sorted(os.listdir(K360_INPUTS))[:8]:
        for f in sorted(glob.glob(os.path.join(K360_INPUTS, d, "*.npz")))[:40]:
            m = map_vs_target(f, os.path.join(K360_TARGETS, d, os.path.basename(f)))
            if m:
                k_rows.append(m)

    def agg(rows):
        if not rows:
            return None
        g = lambda k: np.array([r[k] for r in rows], float)                    # noqa: E731
        return {k: {"mean": float(g(k).mean()), "p50": float(np.median(g(k)))}
                for k in ("precision", "recall", "iou", "n_S", "n_F", "n_gt_occ")} | \
               {"n_anchors": len(rows)}
    ao, ak = agg(once_rows), agg(k_rows)
    R["checks"]["1_6_alignment_vs_kitti360"] = {
        "once": ao, "kitti360_reference": ak,
        "note": ("map-vs-target precision/recall/IoU through one function on both caches. "
                 "KITTI-360 is the arm this recipe was built on, so 'in family' is the "
                 "criterion -- not a threshold."),
        "once_iou_over_k360": (ao["iou"]["mean"] / ak["iou"]["mean"]) if (ao and ak) else None}
    R["checks"]["5_tensor_contract"] = {
        "channels": "[S, 0, F] float, [1,3,256,256,32]", "dims": list(A.GRID_DIMS),
        "input_keys": ["S_H", "F_H"], "target_keys": ["valid_packed", "occ_packed"],
        "pass": bool(ao and ao["n_anchors"] > 0)}

    # ---- 4. representative sample for inspection ----------------------------
    R["checks"]["4_representative_samples"] = once_rows[:5]

    out = a.out or os.path.join(p.logs, "validate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(R, open(out, "w"), indent=2, default=float)
    print(f"[validate] causal ordering: {R['checks']['2_causal_ordering']}")
    print(f"[validate] train/val disjoint: {R['checks']['7_train_val_sequence_disjoint']['pass']} "
          f"train={sp['train']} val={sp['val']}")
    print(f"[validate] scale p50 {R['checks']['3_scale_and_poses']['scale_m_per_unit']['p50']:.2f} "
          f"m/unit, nonfinite {R['checks']['3_scale_and_poses']['scale_nonfinite']}")
    print(f"[validate] map-vs-target  ONCE      : precision {ao['precision']['mean']:5.2f}%  "
          f"recall {ao['recall']['mean']:5.2f}%  IoU {ao['iou']['mean']:5.2f}%  "
          f"(n={ao['n_anchors']})")
    print(f"[validate] map-vs-target  KITTI-360 : precision {ak['precision']['mean']:5.2f}%  "
          f"recall {ak['recall']['mean']:5.2f}%  IoU {ak['iou']['mean']:5.2f}%  "
          f"(n={ak['n_anchors']})")
    print(f"[validate] ONCE IoU / KITTI-360 IoU = {R['checks']['1_6_alignment_vs_kitti360']['once_iou_over_k360']:.3f}")
    print(f"[validate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
