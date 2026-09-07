#!/usr/bin/env python
"""Per-frame metric-gauge candidates for every stream frame, cached once.

Every scale policy consumes the same per-frame candidate, so it is computed once here and
the policies differ only in how they aggregate it causally. Uses the streamed LingBot
depth and confidence and the calibrated-FOV (G51-B) MoGe depth; no target, LiDAR or
oracle is read.

    python tools/gate7b/scale_candidates.py --dataset kitti360
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate7b import config as C, depth as D7, scale as SC, streams as ST          # noqa: E402
from tools.gate7b.stream_lingbot import seg_path                                 # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")
OUT_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate7b/scale"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    a = ap.parse_args()
    ds = a.dataset
    os.makedirs(os.path.join(OUT_ROOT, ds), exist_ok=True)
    index = D7.moge_frame_index(ds, REPO_ROOT)
    segs = ST.build(ds, REPO_ROOT)
    t0, n_frames, n_abstain = time.time(), 0, 0
    for si, seg in enumerate(segs):
        out_p = os.path.join(OUT_ROOT, ds, f"{seg.name}.npz")
        sp = seg_path(ds, seg.name)
        if not os.path.exists(sp):
            raise SystemExit(f"missing stream for {ds}/{seg.name}; run stream_lingbot.py")
        with np.load(sp) as z:
            dep = z["pred_depth"].astype(np.float32)
            cnf = z["pred_depth_conf"].astype(np.float32)
            keys = [str(k) for k in z["keys"]]
        logs, nval, mads = [], [], []
        for i, k in enumerate(keys):
            got = D7.load_moge_frame(ds, k, index)
            if got is None:
                logs.append(np.nan); nval.append(0); mads.append(np.nan)
                n_abstain += 1
                continue
            md, mm, _fov = got
            c = SC.frame_candidate(md, mm, dep[i], cnf[i])
            logs.append(c["log_s"]); nval.append(c["n_valid"]); mads.append(c["mad"])
            if not np.isfinite(c["log_s"]):
                n_abstain += 1
        np.savez_compressed(out_p, keys=np.asarray(keys),
                            log_s=np.asarray(logs, np.float64),
                            n_valid=np.asarray(nval, np.int64),
                            mad=np.asarray(mads, np.float64))
        n_frames += len(keys)
        if (si + 1) % 25 == 0 or len(segs) < 25:
            print(f"  [{ds}] {si+1}/{len(segs)} segments, {n_frames} frames", flush=True)
    meta = {"dataset": ds, "n_segments": len(segs), "n_frames": n_frames,
            "n_abstained": n_abstain, "seconds": time.time() - t0,
            "estimator": "G51-B per frame, 3xMAD clip, >=1000 valid pixels",
            "root": os.path.join(OUT_ROOT, ds)}
    write_json(os.path.join(ART, f"scale_candidates_{ds}.json"), meta)
    print(f"[{ds}] {n_frames} frames, {n_abstain} abstained, "
          f"{meta['seconds']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
