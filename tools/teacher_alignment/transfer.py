#!/usr/bin/env python
"""One frozen transfer evaluation per checkpoint and dataset, plus a sparse record of the
per-voxel semantic prediction so R and S can be compared on a *common* set afterwards.

The gate's evaluator runs unchanged. The record is taken by wrapping the metrics function
it calls per anchor -- the wrapper calls the original and returns its result untouched, and
only remembers the "completion" label map it was handed. Nothing the evaluator computes or
writes changes; ``verify`` in the report compares its JSON against a plain run.

    python tools/teacher_alignment/transfer.py --condition R --seed 0 --dataset semantickitti
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402

EXPECTED_CLIPS = {"semantickitti": 163, "occ3d": 1182}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(ST.CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, choices=list(EXPECTED_CLIPS))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    man = ST.manifest_path(a.condition)
    if not os.path.exists(man):
        raise SystemExit(f"refusing to open {a.dataset}: {man} is not frozen yet")
    os.environ["G8C1_MANIFEST"] = man
    from core.run import eval_target as EV                                        # noqa: E402
    EV.MANIFEST = man
    art = os.path.join(ST.ART, "transfer", a.condition)
    os.makedirs(art, exist_ok=True)
    # ---- record the completion label map per anchor, sparsely, without touching results
    rec_dir = os.path.join(art, f"semlab_{a.dataset}_seed{a.seed}")
    os.makedirs(rec_dir, exist_ok=True)
    original = EV.G6M.clip_counts

    def recording(pred_labels, a1, target, keep, labels, empty_label, dist, height, cid, grp, name):
        if name == "completion":
            pl = np.asarray(pred_labels).reshape(-1)
            occ = pl != empty_label
            np.savez_compressed(os.path.join(rec_dir, f"{cid}.npz"),
                                idx=np.flatnonzero(occ).astype(np.int32),
                                label=pl[occ].astype(np.uint8),
                                gt_label=np.asarray(target).reshape(-1)[occ].astype(np.uint8),
                                gt_occ_idx=np.flatnonzero((np.asarray(target).reshape(-1) != empty_label)
                                                          & np.asarray(keep).reshape(-1)).astype(np.int32),
                                gt_occ_label=np.asarray(target).reshape(-1)[
                                    (np.asarray(target).reshape(-1) != empty_label)
                                    & np.asarray(keep).reshape(-1)].astype(np.uint8))
        return original(pred_labels, a1, target, keep, labels, empty_label, dist, height, cid, grp, name)

    EV.G6M.clip_counts = recording
    try:
        out = EV.run(a.dataset, "past5", a.seed, a.device, None, art=art)
    finally:
        EV.G6M.clip_counts = original
    n, want = out["n_clips"], EXPECTED_CLIPS[a.dataset]
    if n != want:
        raise SystemExit(f"population mismatch on {a.dataset}: scored {n}, manifest says {want}")
    print(f"[{a.condition} seed{a.seed} {a.dataset}] population verified: {n} anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
