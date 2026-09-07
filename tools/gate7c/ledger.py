#!/usr/bin/env python
"""Task 1 — the denominator ledger: every population every Gate-7A/7B number stands on.

Reproduces, from the saved per-clip count blocks, the Gate-7A B-D miss population and
the Gate-7B temporal-recoverability population, and shows exactly where they diverge.

    python tools/gate7c/ledger.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G                              # noqa: E402
from gates.gate7b import streams as ST                                          # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7c")
DS = ("semantickitti", "occ3d", "kitti360")
STRIDE_7B = 4


def main() -> int:
    ledger, rows_csv = {}, []
    for ds in DS:
        g6 = json.load(open(f"{REPO_ROOT}/artifacts/gate6/summary_{ds}.json"))
        a7 = json.load(open(f"{REPO_ROOT}/artifacts/gate7a/summary_{ds}.json"))
        b7 = json.load(open(f"{REPO_ROOT}/artifacts/gate7b/recoverability_{ds}.json"))
        man = json.load(open(f"{REPO_ROOT}/artifacts/gate6/prediction_manifest_{ds}.json"))
        # per-clip B-D count blocks of Gate 6 (identical to Gate 7A's base)
        z = np.load(f"{REPO_ROOT}/artifacts/gate6/counts_{ds}_B-D.npz", allow_pickle=False)
        cid = [str(c) for c in z["clip_id"]]
        binary = z["binary"]                     # btp bfp bfn n_valid n_gt_occ n_pred
        bfn = {c: int(b[2]) for c, b in zip(cid, binary)}
        n_valid = int(binary[:, 3].sum()); n_gt = int(binary[:, 4].sum())
        # the Gate-7B recoverability anchor subset: every STRIDE-th anchor per segment,
        # anchors ordered by stream index, exactly as tools/gate7b/recoverability.py does
        segs = ST.build(ds, REPO_ROOT)
        sub, n_skipped_no_miss = [], 0
        for seg in segs:
            anchors = sorted(seg.anchors.items(), key=lambda kv: kv[1])[::STRIDE_7B]
            for c, _t in anchors:
                if c in bfn:
                    if bfn[c] == 0:
                        n_skipped_no_miss += 1      # recoverability skips clips with no miss
                    else:
                        sub.append(c)
        miss_sub = int(sum(bfn[c] for c in sub))
        G = G6G.EVAL_GRID[ds]
        d = {
            "official_evaluation_clips": g6["n_clips"],
            "official_prediction_files": man["n_files"],
            "evaluation_grid": G.name, "voxels_per_clip": int(np.prod(G.dims)),
            "valid_voxels_after_official_mask": n_valid,
            "valid_gt_occupied_voxels": n_gt,
            "gate6_B_D_coverage_miss": int(g6["conditions"]["B-D"]["decomposition"]["coverage_miss"]),
            "gate7a_B_D_miss_population": int(a7["miss_distance"]["B-D"]["all"]["n_miss"]),
            "gate7a_anchors": a7["n_clips"],
            "gate7b_recoverability_anchor_stride": b7["anchor_stride"],
            "gate7b_recoverability_anchors_reported": b7["n_anchors"],
            "gate7b_anchors_reconstructed_here": len(sub),
            "gate7b_anchors_skipped_for_zero_misses": n_skipped_no_miss,
            "gate7b_miss_population_reported": b7["n_b_d_coverage_misses"],
            "gate7b_miss_population_reconstructed_from_gate6_counts": miss_sub,
            "reconstruction_exact": miss_sub == b7["n_b_d_coverage_misses"] and len(sub) == b7["n_anchors"],
            "gate7b_classes": {k: int(v) for k, v in
                               ((cn, b7["counts"][cn]["all"]) for cn in b7["classes"])},
            "explanation": ("Gate 7A scored every official anchor; Gate 7B's "
                            "temporal-recoverability tool was run with --stride 4 (every "
                            "fourth anchor per segment, in stream order) and skips anchors "
                            "with zero B-D misses. The Gate-7B population is therefore a "
                            "deterministic ~1/4 subsample of the Gate-7A population, not a "
                            "different filtering of it."),
        }
        d["fraction_of_gate7a_population"] = miss_sub / d["gate7a_B_D_miss_population"]
        ledger[ds] = d
        for k, v in d.items():
            if not isinstance(v, (dict, str, bool)):
                rows_csv.append({"dataset": ds, "stage": k, "count": v})
        print(f"{ds:14s} 7A misses {d['gate7a_B_D_miss_population']:>12,} | 7B reported "
              f"{d['gate7b_miss_population_reported']:>12,} reconstructed {miss_sub:>12,} "
              f"anchors {len(sub)}/{b7['n_anchors']} exact={d['reconstruction_exact']}")
    write_json(os.path.join(ART, "denominator_ledger.json"), ledger)
    with open(os.path.join(ART, "denominator_ledger.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "stage", "count"]); w.writeheader()
        w.writerows(rows_csv)
    ok = all(v["reconstruction_exact"] for v in ledger.values())
    print("ALL EXACT" if ok else "MISMATCH — reachability conclusion must be marked invalid")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
