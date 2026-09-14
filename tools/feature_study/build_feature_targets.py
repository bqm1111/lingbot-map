#!/usr/bin/env python
"""Lift frozen SAM-H features into voxel targets over the privileged future window.

Per anchor: a throwaway mapper holding stream frames ``t+1 .. t+20`` only, fed the SAM
feature map in place of the class map, with the **same frozen metric scale the causal input
was exported under** -- read straight out of the cached sample, so the two volumes cannot
drift apart. Queried on the same 256x256x32 grid through the same camera-to-grid transform.

Only KITTI-360 is opened, and only the adopted ceiling-fixed sample set is read (for the
scale, the grid and the existing class-probability support).

    python tools/feature_study/build_feature_targets.py \
        --drive 2013_05_28_drive_0003_sync --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.datasets import grids as G6G                                            # noqa: E402
from core.datasets import sources as S, kitti360_splits as SRC                    # noqa: E402
from core.datasets import kitti360_transforms as TF                               # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.mapping.mapper import IncrementalMapper                                 # noqa: E402
from tools.feature_study import feature_targets as FT                             # noqa: E402

DRIVES = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]


def future_feature_volume(feed, t: int, scale: float, device, n_future: int):
    """Stream frames ``t+1 .. t+n_future``, fused exactly as the class target is."""
    m = IncrementalMapper(device, sem_into=None, n_teacher=ST.FEATURE_DIM)
    m.scale_state.force(scale)
    used = []
    for j in range(t + 1, min(t + 1 + n_future, len(feed))):
        m.step(feed.frame(j))
        used.append(j)
    return m, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True, choices=DRIVES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive)
    SRC.assert_no_target_access([a.drive, src])
    MAP = G6G.PREDICTION_GRID["kitti360"]
    n_vox = int(np.prod(MAP.dims))

    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
        geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        T_cam_to_grid = np.asarray(geo.rect_cam_to_velo, np.float64)
        proj = FT.Projection(dev)
        feed = FT.FeatureFeed(seg, dev, src, proj)
        idxs = [x.stream_index for x in SRC.anchors(a.drive, REPO_ROOT)
                if os.path.exists(SRC.sample_path(a.drive, x.stream_index))]
        idxs = idxs[a.shard::a.shards]
        if a.limit:
            idxs = idxs[: a.limit]
        rows_log, t0, nb, n_skip = [], time.time(), 0, 0
        for i in idxs:
            dst = ST.feature_target_path(a.drive, i)
            if os.path.exists(dst):
                n_skip += 1
                continue
            with np.load(SRC.sample_path(a.drive, i)) as z:
                scale = float(z["scale"])
                cls_rows = z["fut_rows"].astype(np.int64)
            P = feed.pose[i].copy(); P[:3, 3] *= scale
            Tgw = P @ np.linalg.inv(T_cam_to_grid)
            m, used = future_feature_volume(feed, i, scale, dev, SRC.FUTURE_FRAMES)
            q = m.query(MAP, Tgw)
            fw = q["sem_w"]
            feat_sup = fw >= ST.SEM_W_MIN
            cls_sup = torch.zeros(n_vox, dtype=torch.bool, device=dev)
            cls_sup[torch.from_numpy(cls_rows).to(dev)] = True
            common = feat_sup & cls_sup
            sel = common.nonzero(as_tuple=True)[0]
            v = q["sem"][sel] / fw[sel].clamp_min(1e-9).unsqueeze(1)
            nrm = v.norm(dim=1, keepdim=True)
            v = v / nrm.clamp_min(1e-9)
            nb += FT.save(dst, sel.cpu().numpy(), v.cpu().numpy(),
                          fw[sel].cpu().numpy(),
                          {"class": int(cls_sup.sum()), "feature": int(feat_sup.sum()),
                           "common": int(common.sum())}, MAP.dims, i)
            rows_log.append({"stream_index": i, "n_future_frames": len(used),
                             "n_class_support": int(cls_sup.sum()),
                             "n_feature_support": int(feat_sup.sum()),
                             "n_common_support": int(common.sum()),
                             "mean_fused_weight": float(fw[sel].mean()) if len(sel) else 0.0,
                             "mean_raw_norm": float(nrm.mean()) if len(sel) else 0.0})
            del m, q
            if len(rows_log) % 50 == 0:
                torch.cuda.empty_cache()
                print(f"  [{a.drive[17:21]} s{a.shard}] {len(rows_log)}/{len(idxs)} "
                      f"{(time.time()-t0)/len(rows_log):.2f}s each", flush=True)

    agg = {}
    if rows_log:
        for k in ("n_class_support", "n_feature_support", "n_common_support",
                  "mean_fused_weight", "mean_raw_norm"):
            agg[k] = float(np.mean([r[k] for r in rows_log]))
        agg["common_fraction_of_class_support"] = float(
            np.sum([r["n_common_support"] for r in rows_log])
            / max(np.sum([r["n_class_support"] for r in rows_log]), 1))
    ST.write_json(os.path.join(ST.ART, "feature_targets",
                               f"{a.drive[17:21]}_s{a.shard}.json"),
                  {"drive": a.drive, "source": src, "shard": a.shard, "shards": a.shards,
                   "partition": SRC.partition_of(a.drive),
                   "n_selected": len(idxs), "n_built": len(rows_log),
                   "n_already_present": n_skip, "bytes": nb,
                   "future_frames": SRC.FUTURE_FRAMES, "sem_w_min": ST.SEM_W_MIN,
                   "pca_sha256": proj.sha256, "pca_path": proj.path,
                   "n_missing_sam_cache": feed.n_missing_feat,
                   "support_rule": "fused feature weight >= SEM_W_MIN, intersected with the "
                                   "gate's class-probability support (weight >= SEM_W_MIN, "
                                   "geometrically supported, future-window halves agreeing)",
                   "aggregate": agg, "seconds": time.time() - t0,
                   "firewall": dict(audit.summary()),
                   "rows": rows_log[:100], "env": ST.environment()})
    print(f"[{a.drive[17:21]} s{a.shard}] {len(rows_log)} built (+{n_skip} present), "
          f"{nb/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
