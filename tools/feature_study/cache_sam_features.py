#!/usr/bin/env python
"""Cache the frozen SAM-H dense embedding for every KITTI-360 stream frame.

One ``[256, 64, 64]`` fp16 tensor per frame, in the encoder's own space: the 64-d PCA
projection is fitted afterwards and applied at lift time, so the cache survives a refit and
the projection's provenance stays separable from the teacher's.

Only KITTI-360 is read. Drive 0006's frames are cached too -- it is the source-validation
drive, inside the boundary; the PCA is nevertheless fitted on the three training drives
alone (``fit_pca.py``).

    python tools/feature_study/cache_sam_features.py --drive 2013_05_28_drive_0003_sync \
        --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.datasets import sources as S, kitti360_splits as SRC                    # noqa: E402
from tools.feature_study.sam_teacher import SamFeatureTeacher                     # noqa: E402

DRIVES = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True, choices=DRIVES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()

    src = SRC.source_of(a.drive)
    SRC.assert_no_target_access([a.drive, src])
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    frames = list(seg.frames)[a.shard::a.shards]
    if a.limit:
        frames = frames[: a.limit]
    out_dir = os.path.dirname(ST.sam_feature_path(src, frames[0].key))
    os.makedirs(out_dir, exist_ok=True)

    teacher = SamFeatureTeacher(a.device)
    t0, done, skipped = time.time(), 0, 0
    for f in frames:
        dst = ST.sam_feature_path(src, f.key)
        if os.path.exists(dst):
            skipped += 1
            continue
        r = teacher.encode(f.path)
        tmp = dst + f".tmp{a.shard}.npz"
        np.savez(tmp, feat=r["feat"].cpu().numpy().astype(np.float16),
                 native_hw=np.asarray(r["native_hw"], np.int32),
                 input_hw=np.asarray(r["input_hw"], np.int32))
        os.replace(tmp, dst)
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  [{a.drive[17:21]} s{a.shard}] {done}/{len(frames)-skipped} "
                  f"{el/done:.2f}s/frame", flush=True)
    ST.write_json(os.path.join(ST.ART, "cache_sam", f"{a.drive[17:21]}_s{a.shard}.json"),
                  {"drive": a.drive, "source": src, "shard": a.shard, "shards": a.shards,
                   "n_frames_in_shard": len(frames), "n_written": done,
                   "n_already_present": skipped,
                   "sam_checkpoint": ST.SAM_CKPT,
                   "sam_checkpoint_sha256": ST.sha256_file(ST.SAM_CKPT),
                   "extraction_point": "segment_anything.Sam.image_encoder(preprocess(x))",
                   "native_dim": ST.SAM_NATIVE_DIM, "grid": ST.SAM_GRID,
                   "seconds": time.time() - t0, "env": ST.environment()})
    print(f"[{a.drive[17:21]} s{a.shard}] {done} written (+{skipped} present) in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
