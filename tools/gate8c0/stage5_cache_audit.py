#!/usr/bin/env python
"""Gate 8C-0 Stage 5: run every causal-separation and cache invariant over real samples.

The predicates live in ``gate8c0.checks`` so that ``tests/gate8c0`` can inject a one-frame
offset, a reversed pose, a drive-ID collision, a future-frame leak and a one-voxel origin
shift and prove each check fires with a clear diagnostic.

    python tools/gate8c0/stage5_cache_audit.py --device cuda:1
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg, default_device                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G                                                   # noqa: E402
from gates.gate8 import sources as S, targets as TG                                    # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.mapper import IncrementalMapper                                       # noqa: E402
from gates.gate8b.sources import G8B_ROOT, K360_TRAIN_DRIVES                           # noqa: E402
from gates.gate8c0 import checks as CK, transforms as TF                               # noqa: E402

EXPECT_GRID = ((256, 256, 32), 0.2, (0.0, -25.6, -2.0), "velodyne_of_anchor_frame")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--n-samples", type=int, default=40)
    a = ap.parse_args()
    c = cfg(); dev = torch.device(a.device); t0 = time.time()
    res = {"per_sample": [], "global": {}, "expect_grid": list(map(list, [EXPECT_GRID[0]]))
           + [EXPECT_GRID[1], list(EXPECT_GRID[2]), EXPECT_GRID[3]]}
    # ---- global checks --------------------------------------------------------------
    g = G6G.PREDICTION_GRID["kitti360"]
    ok, det = CK.grid_declaration(g.dims, g.voxel_size, g.origin, g.frame, EXPECT_GRID)
    res["global"]["grid_declaration"] = {"ok": ok, **det}
    ok, det = CK.partitions_disjoint(c["train_drives"], [c["val_drive"]], [c["heldout_drive"]])
    res["global"]["drive_partitions_disjoint"] = {"ok": ok, **det}
    keys_by_drive = {}
    for src, segs in (("k360_train", K360_TRAIN_DRIVES), ("kitti360", (c["heldout_drive"],))):
        for seg in S.segments(src, REPO_ROOT):
            if seg.name in segs:
                keys_by_drive[seg.name] = [f.key for f in seg.frames]
    ok, det = CK.cache_keys_cannot_collide(keys_by_drive)
    res["global"]["cache_keys_cannot_collide"] = {"ok": ok, **det}
    res["global"]["trident_path_is_drive_scoped"] = {
        "k360_train": S.trident_path("k360_train", "0003_0000000004"),
        "kitti360": S.trident_path("kitti360", "0000000004"),
        "ok": S.trident_path("k360_train", "x") != S.trident_path("kitti360", "x")}
    # ---- per-sample checks ---------------------------------------------------------
    files = []
    for src in ("k360_train", "kitti360"):
        files += [(src, p) for p in sorted(glob.glob(f"{G8B_ROOT}/samples/{src}/*.npz"))]
    rng = np.random.default_rng(c["seed"])
    files = [files[i] for i in rng.choice(len(files), size=min(a.n_samples, len(files)),
                                          replace=False)]
    segcache = {}
    fails = {}
    for src, p in files:
        stem = os.path.basename(p)[:-4]; seg_name, ti = stem.rsplit("_", 1); ti = int(ti)
        if (src, seg_name) not in segcache:
            segcache[(src, seg_name)] = [s for s in S.segments(src, REPO_ROOT)
                                         if s.name == seg_name][0]
        seg = segcache[(src, seg_name)]
        with np.load(p) as z:
            t = int(z["t"]); inp = z["input_frames"]; fut = z["target_frames"]
            gt_occ = np.unpackbits(z["gt_occ"])[:np.prod(TF.DIMS)].astype(bool)
            gt_valid = np.unpackbits(z["gt_valid"])[:np.prod(TF.DIMS)].astype(bool)
            dims = tuple(int(x) for x in z["dims"])
        row = {"source": src, "segment": seg_name, "t": t, "sample": os.path.basename(p)}
        for nm, (ok, det) in (
                ("causal_input_only", CK.causal_input_only(t, inp)),
                ("future_target_window", CK.future_target_window(t, fut, len(seg))),
                ("no_future_leak", CK.no_future_leak(inp, fut)),
                ("dims_match_declaration", (dims == EXPECT_GRID[0], {"dims": list(dims)})),
                ("unknown_not_supervised",
                 CK.unknown_handled_consistently(gt_occ, gt_valid, gt_valid))):
            row[nm] = {"ok": bool(ok), **det}
            if not ok:
                fails.setdefault(nm, []).append(row["sample"])
        f = seg.frames[t]
        gm = TF.DriveGeometry(seg_name, c["sscbench_root"], c["kitti360_root"])
        ok, det = CK.frame_identity_consistent(
            f.key, os.path.basename(S.trident_path(src, f.key))[:-4],
            int(f.order), int(f.gt_ref["anchor"]), gm.native(int(f.gt_ref["anchor"])))
        row["frame_identity_consistent"] = {"ok": bool(ok), **det}
        if not ok:
            fails.setdefault("frame_identity_consistent", []).append(row["sample"])
        res["per_sample"].append(row)
    # ---- live mapper checks (integrated once, scale window) ------------------------
    live = []
    for src, seg_name in [("k360_train", K360_TRAIN_DRIVES[0]), ("kitti360", c["heldout_drive"])]:
        seg = segcache.get((src, seg_name)) or [s for s in S.segments(src, REPO_ROOT)
                                                if s.name == seg_name][0]
        # semantics are irrelevant to the causal checks and the table would need
        # n_teacher columns to accept them
        feed = CachedFeed(seg, dev, with_semantics=False)
        m = IncrementalMapper(dev, n_teacher=0)
        seen = []
        for i in range(12):
            before = m.scale_state.frozen
            m.step(feed.frame(i))
            if not before:
                seen.append(i)
        ok1, d1 = CK.integrated_once(m.n_integrated, 12)
        ok2, d2 = CK.scale_anchor_window(seen)
        live.append({"source": src, "segment": seg_name,
                     "integrated_once": {"ok": ok1, **d1},
                     "scale_anchor_window": {"ok": ok2, **d2}})
        if not (ok1 and ok2):
            fails.setdefault("live_mapper", []).append(seg_name)
    res["live_mapper"] = live
    res["n_samples"] = len(res["per_sample"])
    res["failures"] = {k: v[:10] for k, v in fails.items()}
    res["all_checks_pass"] = not fails
    res["seconds"] = time.time() - t0
    write_json(os.path.join(ART, "stage5_cache_audit.json"), res)
    print(f"stage 5: {res['n_samples']} samples, all checks pass = {res['all_checks_pass']}")
    for k, v in res["global"].items():
        print(f"   global {k}: {v.get('ok')}")
    for r in live:
        print(f"   live {r['source']:11s} integrated_once {r['integrated_once']['ok']} "
              f"({r['integrated_once']['n_integrated']}/12)  scale window "
              f"{r['scale_anchor_window']['ok']} {r['scale_anchor_window']['indices']}")
    if fails:
        for k, v in res["failures"].items():
            print(f"   FAIL {k}: {len(v)} e.g. {v[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
