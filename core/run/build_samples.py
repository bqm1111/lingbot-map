# extracted from tools/gate8c1/build_samples.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Gate 8C-1: cache (causal input, rebuilt target) training pairs for one KITTI-360 drive.

The **input** is Gate 8's frozen 32-channel causal export: one incremental mapper per
drive, stepped frame by frame, scale fixed from the first five stream frames and never
touched again, every frame integrated exactly once, queried at the anchor. Not one line of
that path changes here.

The **geometry target** is the rebuilt raw-LiDAR occupancy of
``tools/gate8c1/build_targets.py`` -- never SSCBench's ``_1_1.npy``.

The **semantic target** is frozen Trident-H fused over the future window through the same
frozen mapper, kept only where it is geometrically supported and where the two halves of
the window agree. Building the window as two halves costs nothing extra -- the mapper's
semantic accumulators are additive, so their sum *is* the full-window target -- and it
yields the teacher-consistency signal the semantic threshold is later selected on, with no
human label anywhere.

    python tools/gate8c1/build_samples.py --drive 2013_05_28_drive_0007_sync --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.run._common import ART, REPO_ROOT, default_device                               # noqa: E402
from core.datasets.config import write_json                                         # noqa: E402
from core.datasets import grids as G6G, vocab                                            # noqa: E402
from core.datasets import sources as S, union_vocab as V8                                      # noqa: E402
from core.mapping.feed import CachedFeed                                                # noqa: E402
from core.mapping.mapper import IncrementalMapper                                       # noqa: E402
from core.datasets import kitti360_transforms as TF                                             # noqa: E402
from core.datasets import kitti360_splits as SRC                                               # noqa: E402
from core.run.validate_targets import load_target                           # noqa: E402

#: A voxel needs at least this much fused teacher weight to carry a semantic target.
SEM_W_MIN = 1.0
#: ...and the two halves of the future window must agree on the top-1 class.
REQUIRE_HALF_AGREEMENT = True


def half_volume(feed, t, scale, into, device, lo, hi, n_teacher):
    """A throwaway mapper holding future stream frames ``t+lo .. t+hi`` only."""
    m = IncrementalMapper(device, sem_into=into, n_teacher=n_teacher)
    m.scale_state.force(scale)
    used = []
    for j in range(t + lo, min(t + hi + 1, len(feed))):
        m.step(feed.frame(j))
        used.append(j)
    return m, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive)
    ds = "kitti360"
    MAP = G6G.PREDICTION_GRID[ds]
    C = len(vocab.load(ds)); into = V8.into_matrix(ds)
    n = int(np.prod(MAP.dims))
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    ancs = {x.stream_index: x for x in SRC.anchors(a.drive, REPO_ROOT)[::a.stride]
            if os.path.exists(SRC.target_path(a.drive, x.stream_index))}
    out_dir = os.path.dirname(SRC.sample_path(a.drive, 0)); os.makedirs(out_dir, exist_ok=True)
    SRC.assert_no_target_access([out_dir, S.stream_path(src, a.drive)])
    # For KITTI-360 the camera->grid transform is a per-drive constant from calibration.
    # Taking it from the per-frame record would fail on stream frames that are not official
    # SSCBench anchors -- and Gate 8C-1 deliberately anchors on all eligible raw frames.
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    T_cam_to_grid = np.asarray(geo.rect_cam_to_velo, np.float64)
    feed = CachedFeed(seg, dev)
    m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
    rows, t0, nw = [], time.time(), 0
    for i in range(len(seg)):
        m.step(feed.frame(i))
        if i not in ancs or not m.scale_state.frozen:
            continue
        p = SRC.sample_path(a.drive, i)
        if os.path.exists(p):
            nw += 1; continue
        anc = ancs[i]
        f = seg.frames[i]
        P = feed.pose[i].copy(); P[:3, 3] *= m.scale_state.scale
        Tgw = P @ np.linalg.inv(T_cam_to_grid)
        q = m.query(MAP, Tgw)
        # ---- future teacher, in two halves so consistency comes for free ---------------
        h = SRC.FUTURE_FRAMES // 2
        m1, u1 = half_volume(feed, i, m.scale_state.scale, into, dev, 1, h, C)
        m2, u2 = half_volume(feed, i, m.scale_state.scale, into, dev, h + 1,
                             SRC.FUTURE_FRAMES, C)
        q1, q2 = m1.query(MAP, Tgw), m2.query(MAP, Tgw)
        sw = q1["sem_w"] + q2["sem_w"]
        sm = q1["sem"] + q2["sem"]
        agree = torch.ones(n, dtype=torch.bool, device=dev)
        if REQUIRE_HALF_AGREEMENT:
            both = (q1["sem_w"] > 0) & (q2["sem_w"] > 0)
            a1 = q1["sem"].argmax(1); a2 = q2["sem"].argmax(1)
            agree = (~both) | (a1 == a2)          # only *contradiction* disqualifies
        tgt = load_target(SRC.target_path(a.drive, i))
        gt_occ = tgt["occupied"]; gt_valid = tgt["valid"]
        geo_sup = torch.from_numpy(gt_valid).to(dev)
        keep_sem = (sw >= SEM_W_MIN) & geo_sup & agree
        fut_rows = keep_sem.nonzero(as_tuple=True)[0]
        fut_p = sm[fut_rows] / sw[fut_rows].clamp_min(1e-9).unsqueeze(1)
        conf = fut_p.max(1).values
        # ---- causal input state (frozen Gate 8 layout) --------------------------------
        obs = q["observed"].cpu().numpy()
        fut_obs = ((q1["observed"] | q2["observed"]).cpu().numpy())
        rws = np.flatnonzero(obs | fut_obs | (gt_occ & gt_valid))
        sem_rows = np.flatnonzero(obs & (q["sem_w"].cpu().numpy() > 0))
        semv = q["sem"].cpu().numpy(); semw = q["sem_w"].cpu().numpy()
        u8 = lambda x: np.clip(np.rint(np.asarray(x) * 255.0), 0, 255).astype(np.uint8)
        age = np.where(q["last_time"].cpu().numpy() >= 0, i - q["last_time"].cpu().numpy(), -1)
        smp = {
            "t": np.int64(i), "input_frames": np.arange(0, i + 1, dtype=np.int32),
            "target_frames": np.asarray(u1 + u2, np.int32),
            "scale": np.float64(m.scale_state.scale),
            "dims": np.asarray(MAP.dims, np.int32), "rows": rws.astype(np.int32),
            "logodds": q["logodds"].cpu().numpy()[rws].astype(np.float16),
            "w_occ": q["w_occ"].cpu().numpy()[rws].astype(np.float16),
            "w_free": q["w_free"].cpu().numpy()[rws].astype(np.float16),
            "n_obs": np.clip(q["n_obs"].cpu().numpy()[rws], 0, 255).astype(np.uint8),
            "n_lb": np.clip(q["n_lb"].cpu().numpy()[rws], 0, 255).astype(np.uint8),
            "age": np.clip(age[rws], -1, 254).astype(np.int16), "observed": obs[rws],
            "sem_rows": sem_rows.astype(np.int32),
            "sem_p": u8(semv[sem_rows] / np.maximum(semw[sem_rows, None], 1e-9)),
            "sem_w": semw[sem_rows].astype(np.float16),
            "gt_occ": np.packbits(gt_occ), "gt_valid": np.packbits(gt_valid),
            "fut_observed": np.packbits(fut_obs),
            "fut_rows": fut_rows.cpu().numpy().astype(np.int32),
            "fut_p": u8(fut_p.cpu().numpy()),
            "fut_w": sw[fut_rows].cpu().numpy().astype(np.float16),
            "fut_conf": u8(conf.cpu().numpy()),
            "fut_half_agree": np.packbits(agree.cpu().numpy()),
            "n_obs_target": tgt["n_obs"], "n_frames_occ_target": tgt["n_frames_occ"],
            "occ_rows_target": tgt["occ_rows"],
            "native_frame": np.int32(anc.native_frame)}
        tmp = p + ".tmp.npz"; np.savez_compressed(tmp, **smp); os.replace(tmp, p)
        rows.append({"stream_index": i, "native_frame": anc.native_frame,
                     "input_frames": [0, i], "target_frames": [u1[0], u2[-1]] if u2 else [u1[0], u1[-1]],
                     "n_gt_occ": int(gt_occ.sum()), "n_gt_valid": int(gt_valid.sum()),
                     "n_sem_rows": int(len(fut_rows)),
                     "sem_kept_fraction_of_supported": float(
                         len(fut_rows) / max(int(((sw >= SEM_W_MIN) & geo_sup).sum()), 1)),
                     "bytes": os.path.getsize(p)})
        nw += 1
        if nw % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  [{a.drive[17:21]}] {nw}/{len(ancs)} {(time.time()-t0)/max(len(rows),1):.2f}s each",
                  flush=True)
    if feed.n_missing_sem:
        print(f"  WARNING {a.drive}: {feed.n_missing_sem} frames without a Trident cache")
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"samples_{a.drive[17:21]}.json"),
               {"drive": a.drive, "partition": SRC.partition_of(a.drive),
                "n_anchors": len(ancs), "n_written": len(rows), "n_present": nw,
                "future_frames": SRC.FUTURE_FRAMES, "sem_w_min": SEM_W_MIN,
                "require_half_agreement": REQUIRE_HALF_AGREEMENT,
                "n_missing_trident": feed.n_missing_sem,
                "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                "total_bytes": int(sum(r["bytes"] for r in rows)),
                "audit": rows[:50]})
    print(f"[{a.drive[17:21]}] {len(rows)} written, {nw} present, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
