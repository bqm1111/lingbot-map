#!/usr/bin/env python
"""The map-observability partition on KITTI-360 drive 0006, frozen before any prediction.

The primary measurement of this study is completion *where the causal map had nothing*, so
the partition has to separate three populations and it has to do so from measurement, not
from a heuristic:

* **map-observed** -- the causal input map at the anchor reached the voxel;
* **map-unobserved / later-measured** -- the causal map did not, and a future sweep in the
  declared window later resolved it: a return landed in it (occupied) **or** a ray of that
  same window traversed it (free). Both halves count: scoring only the endpoints would
  reward a model that fills everything;
* **still unknown** -- everything else. Ignored.

Two exclusions come free with the rebuilt target and are kept:

* a voxel with an occupied return from one sweep and a free traversal from another stays
  UNKNOWN and is in neither population. That disagreement is mostly moving traffic and
  occlusion boundaries, and this study does not adjudicate it either;
* the open-sky rule is a *geometric inference*, not a measurement, so sky-free voxels are
  excluded from "later-measured". They remain in the occupancy training target, where they
  belong; they must not become later-confirmed free space in the measurement.

Only the **measurement** stage of ``rawtarget.build`` is used here -- endpoint counts and
free-traversal counts, which are a function of the raw sweeps and the poses alone. The sky
stage is not used, and the approved occupancy targets are not rebuilt. That distinction is
load-bearing on this machine: the adopted ``..._sky`` targets were written on 2026-09-04,
three days before the open-sky rule was last changed (commit 647bcf3, which added the
neighbourhood roof), so today's ``rawtarget.py`` frees substantially more sky than the
version those targets -- and the checkpoints trained on them -- actually used. The
measurement stage does reproduce: the rebuilt occupied set matches the stored one voxel for
voxel, which is what ``matches_frozen_occupancy`` asserts per anchor. The divergence in the
inferred-free set is recorded (``n_sky_rule_drift``) and otherwise left alone.

Limitation worth stating plainly: "map-unobserved" is a statement about the mapper's causal
state, not about visibility. A voxel the mapper missed may have been perfectly visible in
RGB; a voxel later measured free may have been occupied at anchor time and vacated by a
moving object. Neither is corrected here.

    python tools/feature_study/build_eval_masks.py --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.datasets import kitti360_transforms as TF                               # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.supervision import rawtarget as RT                                      # noqa: E402
from core.run.build_targets import load as load_target                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default=SRC.VAL_DRIVE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    SRC.assert_no_target_access([a.drive])

    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        ancs = [x for x in SRC.anchors(a.drive, REPO_ROOT)
                if os.path.exists(SRC.sample_path(a.drive, x.stream_index))]
        ancs = ancs[a.shard::a.shards]
        if a.limit:
            ancs = ancs[: a.limit]
        rows, t0, nb, n_skip, n_mismatch = [], time.time(), 0, 0, 0
        for anc in ancs:
            dst = ST.eval_mask_path(a.drive, anc.stream_index)
            if os.path.exists(dst):
                n_skip += 1
                continue
            tgt = RT.build(geo, anc.native_frame,
                           [anc.native_frame] + list(anc.future_natives), dev)
            # measurement only: endpoint counts and free traversals, both a function of the
            # raw sweeps and the ground-truth poses. tgt.state is NOT consulted -- it carries
            # today's sky rule, which is not the one the approved targets were built with.
            occ_any = tgt.n_obs > 0
            free_any = tgt.n_free > 0
            measured_occ = occ_any & ~free_any          # cross-sweep conflicts excluded
            measured_free = free_any & ~occ_any
            conflict = occ_any & free_any
            ref = load_target(SRC.target_path(a.drive, anc.stream_index))
            # the approved target's own inferred-free set: valid, but neither measured
            sky_free = ref["valid"] & ~measured_occ & ~measured_free
            ok = bool(np.array_equal(ref["occupied"], measured_occ)
                      and not (measured_free & ~ref["valid"]).any()
                      and not (measured_occ & ~ref["valid"]).any())
            n_mismatch += (not ok)
            drift = int((tgt.valid != ref["valid"]).sum())
            with np.load(SRC.sample_path(a.drive, anc.stream_index)) as z:
                n = int(np.prod([int(x) for x in z["dims"]]))
                obs = np.zeros(n, bool)
                obs[z["rows"].astype(np.int64)] = z["observed"]
            later = (~obs) & (measured_occ | measured_free)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp.npz"
            np.savez(tmp, measured_occ=np.packbits(measured_occ),
                     measured_free=np.packbits(measured_free),
                     sky_free=np.packbits(sky_free), conflict=np.packbits(conflict),
                     map_observed=np.packbits(obs), later_measured=np.packbits(later),
                     dims=np.asarray(TF.DIMS, np.int32),
                     stream_index=np.int32(anc.stream_index),
                     native_frame=np.int32(anc.native_frame),
                     matches_frozen_occupancy=np.bool_(ok),
                     n_sky_rule_drift=np.int64(drift))
            os.replace(tmp, dst)
            nb += os.path.getsize(dst)
            rows.append({"stream_index": anc.stream_index,
                         "n_map_observed": int(obs.sum()),
                         "n_later_measured": int(later.sum()),
                         "n_later_measured_occupied": int((later & measured_occ).sum()),
                         "n_later_measured_free": int((later & measured_free).sum()),
                         "n_sky_free_excluded": int(sky_free.sum()),
                         "n_conflict_excluded": int(conflict.sum()),
                         "n_still_unknown": int((~obs & ~later).sum()),
                         "n_sky_rule_drift": drift,
                         "matches_frozen_occupancy": ok})
            if len(rows) % 100 == 0:
                print(f"  [{a.drive[17:21]} s{a.shard}] {len(rows)}/{len(ancs)} "
                      f"{(time.time()-t0)/len(rows):.2f}s each", flush=True)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "stream_index"} \
        if rows else {}
    ST.write_json(os.path.join(ST.ART, "eval_masks", f"{a.drive[17:21]}_s{a.shard}.json"),
                  {"drive": a.drive, "shard": a.shard, "shards": a.shards,
                   "n_selected": len(ancs), "n_built": len(rows),
                   "n_already_present": n_skip,
                   "n_disagreeing_with_frozen_occupancy": n_mismatch, "bytes": nb,
                   "provenance": "measurement stage of rawtarget.build only; the "
                                 "approved occupancy targets are read, never rebuilt. "
                                 "n_sky_rule_drift counts voxels where today's sky rule "
                                 "would differ from the approved target's valid mask.",
                   "definition": {
                       "map_observed": "causal input map hit the voxel at the anchor",
                       "later_measured": "not map-observed, and a future-window sweep either "
                                         "put a return in it or traversed it",
                       "excluded": ["the approved target's inferred-free set "
                                    "(valid but never measured) -- the sky rule",
                                    "occupied/free conflict across sweeps",
                                    "never resolved by any sweep"]},
                   "aggregate": agg, "seconds": time.time() - t0,
                   "firewall": dict(audit.summary()), "rows": rows[:100],
                   "env": ST.environment()})
    print(f"[{a.drive[17:21]} s{a.shard}] {len(rows)} built (+{n_skip} present), "
          f"{n_mismatch} disagree with the frozen occupancy, {(time.time()-t0)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
