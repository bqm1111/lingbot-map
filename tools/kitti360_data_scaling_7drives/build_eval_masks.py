#!/usr/bin/env python
"""Frozen evaluation masks for the development drive 0006 (590 anchors, stride 3).

Everything comes from raw LiDAR and ground-truth poses; no prediction and no LingBot state:

* ``measured_past``  -- a voxel measured (return landed in it, or a ray traversed it) by any
                        of the sweeps at stream frames t-19 .. t -- the causal window the
                        input stream also covers at its most recent end;
* ``measured_occ_f`` / ``measured_free_f`` -- resolved by the future sweeps t+1 .. t+20 alone
                        (return-only / traversal-only; cross-sweep conflicts excluded);
* ``unobserved_later`` (PRIMARY) = target valid & ~measured_past & (measured_occ_f | measured_free_f);
* ``observed``       = target valid & measured_past.

Semantic ground truth (evaluation only): SSCBench-KITTI-360 human voxel labels, read at the
voxel of each LiDAR-occupied centre after the declared frame conversion (the label grid is
anchored at the rectified-camera origin: index = floor((p - t_cam0_to_velo - origin)/0.2));
free voxels are the empty class; occupied voxels without a class label are ignored for
semantics. The label volume's own occupancy is never used.

    python tools/kitti360_data_scaling_7drives/build_eval_masks.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC               # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402
from core.run.build_targets import load as load_target                                    # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

LABELS = "/media/SSD1/MINH_DATASETS/sscbench_kitti360/preprocess/labels"
PAST_WINDOW = 20
DIMS = (256, 256, 32); ORIGIN = np.array([0.0, -25.6, -2.0]); VS = 0.2
OUT = os.path.join(drives7.DATA_ROOT, "eval_masks", drives7.VAL_DRIVE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = drives7.VAL_DRIVE
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); pf = geo.pose_frames
    t_c2v = np.asarray(geo.calib.cam0_to_velo, np.float64)[:3, 3]
    anchors = SRC.anchors(d, REPO_ROOT)[::3]; natives_of = {}
    from core.datasets import sources as S
    seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; natives = [int(f.order) for f in seg.frames]
    os.makedirs(OUT, exist_ok=True); rows = []; t0 = time.time(); n = int(np.prod(DIMS))
    for anc in anchors[a.shard::a.shards]:
        op = os.path.join(OUT, f"{anc.stream_index:05d}.npz")
        if os.path.exists(op):
            continue
        tgt = load_target(SRC.target_path(d, anc.stream_index)); valid, occ = tgt["valid"], tgt["occupied"]
        past = [natives[j] for j in range(max(0, anc.stream_index - PAST_WINDOW + 1), anc.stream_index + 1)]
        rp = RT.build(geo, anc.native_frame, past, dev)
        rf = RT.build(geo, anc.native_frame, list(anc.future_natives), dev)
        m_past = np.asarray(rp.n_obs > 0) | np.asarray(rp.n_free > 0)
        mo, mf = np.asarray(rf.n_obs > 0), np.asarray(rf.n_free > 0)
        later = valid & ~m_past & ((mo & ~mf) | (mf & ~mo)); observed = valid & m_past
        # semantic ground truth
        k = np.flatnonzero(pf[1:] == anc.native_frame)
        sem = np.full(n, 255, np.uint8)                                         # 255 = ignore
        sem[valid & ~occ] = 0
        lp = os.path.join(LABELS, d, f"{int(k[0]):06d}_1_1.npy") if len(k) else ""
        has_label = os.path.exists(lp)
        if has_label:
            L = np.load(lp).astype(np.int32); oi = np.flatnonzero(valid & occ)
            ctr = (np.stack(np.unravel_index(oi, DIMS), 1) + 0.5) * VS + ORIGIN
            li = np.floor((ctr - t_c2v - ORIGIN) / VS).astype(int); ins = (li >= 0).all(1) & (li < np.array(DIMS)).all(1)
            lab = np.zeros(len(oi), np.int32); lab[ins] = L[tuple(li[ins].T)]
            good = (lab >= 1) & (lab <= 18); sem[oi[good]] = lab[good].astype(np.uint8)
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, unobserved_later=np.packbits(later), observed=np.packbits(observed), measured_past=np.packbits(m_past),
                            valid=np.packbits(valid), occupied=np.packbits(occ), sem_gt=sem, has_label=np.bool_(has_label),
                            stream_index=np.int32(anc.stream_index), native_frame=np.int32(anc.native_frame), past_natives=np.asarray(past, np.int32),
                            future_natives=np.asarray(anc.future_natives, np.int32))
        os.replace(tmp, op)
        rows.append({"t": anc.stream_index, "valid": int(valid.sum()), "unobserved_later": int(later.sum()), "unobserved_later_occ": int((later & occ).sum()),
                     "observed": int(observed.sum()), "occ_with_label": int(((sem >= 1) & (sem <= 18)).sum()), "occ": int((valid & occ).sum()),
                     "unobserved_later_labelled_occ": int((later & (sem >= 1) & (sem <= 18)).sum()), "has_label": has_label})
        if len(rows) % 50 == 0:
            print(f"  [eval masks s{a.shard}] {len(rows)} anchors, {(time.time()-t0)/len(rows):.1f}s each", flush=True)
    agg = {k: int(sum(r[k] for r in rows)) for k in rows[0] if k != "t"} if rows else {}
    C0.write_json(os.path.join(drives7.ART, "eval", f"masks_s{a.shard}.json"),
                  {"n_built": len(rows), "totals": agg, "past_window_stream_frames": PAST_WINDOW, "future_frames": SRC.FUTURE_FRAMES,
                   "label_frame_conversion": "index = floor((p - t_cam0_to_velo - origin) / 0.2); verified: fraction of LiDAR-occupied voxels on a labelled voxel 0.196 -> 0.464 (59 anchors)",
                   "rows": rows, "seconds": time.time() - t0})
    print(f"eval masks s{a.shard}: {len(rows)} built, totals {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
