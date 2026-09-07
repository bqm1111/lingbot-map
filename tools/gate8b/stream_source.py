#!/usr/bin/env python
"""Frozen-model passes for the Gate 8B training source ``k360_train``: LingBot native
streaming, calibrated-FOV MoGe-B depth per frame, and the per-frame metric-gauge
candidate. Line for line the Gate 8 tool (``tools/gate8/stream_sources.py``) pointed at
the Gate 8B source and artifact root; nothing opens a target.

    python tools/gate8b/stream_source.py --shard 0 --shards 3 --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import load_config, write_json                            # noqa: E402
from moge_gauge.calibrated import CalibratedMoGe                                 # noqa: E402
from moge_gauge.calibration import calibrated_fov_x_deg                          # noqa: E402
from gates.gate7b import replay, scale as SC, depth as D7                              # noqa: E402
from gates.gate8 import sources as S                                                   # noqa: E402
from gates.gate8b.sources import SOURCE, G8B_ROOT                                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    for sub in ("stream", "moge_b", "scale"):
        os.makedirs(f"{G8B_ROOT}/{sub}/{a.source}", exist_ok=True)
    segs = S.segments(a.source, REPO_ROOT)[a.shard::a.shards]
    model, _ = replay.build_model(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    t0, rows, nfr = time.time(), [], 0
    for seg in segs:
        sp = S.stream_path(a.source, seg.name)
        if os.path.exists(sp) and os.path.exists(S.scale_path(a.source, seg.name)):
            continue
        imgs = replay.load_images([f.path for f in seg.frames])
        from PIL import Image
        with Image.open(seg.frames[0].path) as im:
            native_hw = (im.height, im.width)
        k = replay.keyframe_interval_for(len(seg))
        ts = time.time()
        out = replay.replay_segment(model, imgs, k, dev)
        dt = time.time() - ts
        np.savez_compressed(sp, keys=np.asarray([f.key for f in seg.frames]),
                            order=np.asarray([f.order for f in seg.frames], np.int64),
                            pred_depth=out["pred_depth"], pred_depth_conf=out["pred_depth_conf"],
                            pred_pose_c2w=out["pred_pose_c2w"], pred_K=out["pred_K"],
                            pose_enc=out["pose_enc"], proc_hw=out["proc_hw"],
                            keyframe_interval=np.int64(k), native_hw=np.asarray(native_hw))
        pw = int(out["proc_hw"][1]); logs, nval = [], []
        dep = out["pred_depth"].astype(np.float32); cnf = out["pred_depth_conf"].astype(np.float32)
        for i, f in enumerate(seg.frames):
            mp = S.moge_path(a.source, f.key)
            fov = calibrated_fov_x_deg(float(f.K_native[0, 0]) * (pw / native_hw[1]), pw)
            if not os.path.exists(mp):
                o = moge.infer_calibrated(imgs[i].numpy(), fov_x_deg=fov)
                mk = np.asarray(o.mask, bool)
                np.savez_compressed(mp, moge_depth=np.asarray(o.depth_z, np.float16),
                                    moge_mask=np.packbits(mk.reshape(-1)),
                                    mask_shape=np.asarray(mk.shape, np.int32),
                                    fov_x_deg=np.float32(fov), fov_out_deg=np.float32(o.fov_x_deg))
            with np.load(mp) as z:
                md = z["moge_depth"].astype(np.float32)
                mm = D7.unpack_mask(z["moge_mask"], z["mask_shape"])
            c = SC.frame_candidate(md, mm, dep[i], cnf[i])
            logs.append(c["log_s"]); nval.append(c["n_valid"])
        np.savez_compressed(S.scale_path(a.source, seg.name),
                            keys=np.asarray([f.key for f in seg.frames]),
                            log_s=np.asarray(logs, np.float64), n_valid=np.asarray(nval, np.int64))
        nfr += len(seg)
        rows.append({"segment": seg.name, "n_frames": len(seg), "keyframe_interval": k,
                     "lingbot_seconds": dt, "fps": len(seg) / max(dt, 1e-9)})
        print(f"  [{a.source} s{a.shard}] {seg.name} {len(seg)} frames k={k} {len(seg)/dt:.1f} FPS "
              f"total {nfr} in {(time.time()-t0)/60:.1f} min", flush=True)
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"stream_{a.source}_s{a.shard}.json"),
               {"source": a.source, "shard": a.shard, "segments": rows, "n_frames": nfr,
                "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
