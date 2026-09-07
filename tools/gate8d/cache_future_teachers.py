#!/usr/bin/env python
"""Gate 8D Phase 2.2/2.3: cache frozen-teacher evidence on future KITTI-360 frames.

Two privileged teachers, both frozen, both KITTI-360 only:

* **MoGe-2** metric depth, for dense near-field free space. Stored strided by
  ``MOGE_PIXEL_STRIDE`` with its validity mask; the safety margin is applied later, at
  fusion time, from ``artifacts/gate8d/moge_margin.json``.
* **Trident-H** sky probability, for rays that leave the scene entirely. The project's
  KITTI-360 phrase list has no sky class, so the teacher is run with that list extended by
  the protocol's single fixed ``SKY_PROMPT``; only the sky channel is kept.

Neither cache is reachable from inference: ``tests/gate8d`` asserts it.

    python tools/gate8d/cache_future_teachers.py --teacher moge --drive <d> --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import vocab                                                          # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from gates.gate8d import protocol as P, sources as SRC                                 # noqa: E402


def frames_needed(drive):
    """Every native frame an anchor in this drive may look at, at the image stride."""
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    ancs = SRC.anchors(drive, REPO_ROOT)
    want = set()
    for a in ancs:
        lo = int(a.native_frame)
        hi = max([int(f) for f in a.future_natives] + [lo])
        want.update(range(lo, hi + 1, P.IMAGE_EVIDENCE_NATIVE_STRIDE))
    have = set(geo.cam0_to_world)
    return geo, sorted(want & have)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, choices=["moge", "sky"])
    ap.add_argument("--drive", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    geo, frames = frames_needed(a.drive)
    frames = frames[a.shard::a.shards]
    K0 = np.asarray(geo.calib.K, np.float64); W0 = int(geo.calib.native_hw[1])
    fov_x = float(np.degrees(2 * np.arctan(W0 / (2 * K0[0, 0]))))
    from PIL import Image
    out_dir = (SRC.moge_future_path(a.drive, 0) if a.teacher == "moge"
               else SRC.sky_future_path(a.drive, 0)).rsplit("/", 1)[0]
    os.makedirs(out_dir, exist_ok=True)
    if a.teacher == "moge":
        from moge_gauge.calibrated import CalibratedMoGe
        from gates.scale_gate.config import load_config
        g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
        model = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo)
        st = P.MOGE_PIXEL_STRIDE
    else:
        from gates.gate6.trident_adapter import TridentTeacher
        from tools.gate6.cache_semantics import SAM_CKPT
        phrases = tuple(vocab.load("kitti360").phrases) + (P.SKY_PROMPT,)
        model = TridentTeacher(phrases, sam_ckpt=SAM_CKPT, device=a.device)
        st = P.SKY_CACHE_STRIDE
    n, t0, skipped = 0, time.time(), 0
    for nf in frames:
        dst = (SRC.moge_future_path(a.drive, nf) if a.teacher == "moge"
               else SRC.sky_future_path(a.drive, nf))
        if os.path.exists(dst):
            skipped += 1; continue
        ip = SRC.image_path(a.drive, nf)
        if not os.path.exists(ip):
            continue
        if a.teacher == "moge":
            rgb = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
            o = model.infer_calibrated(rgb.transpose(2, 0, 1), fov_x_deg=fov_x)
            d = np.asarray(o.depth_z, np.float32)[::st, ::st]
            m = np.asarray(o.mask, bool)[::st, ::st]
            np.savez_compressed(dst, depth_z=d.astype(np.float16), mask=np.packbits(m),
                                shape=np.asarray(d.shape, np.int32),
                                stride=np.int32(st), fov_x_deg=np.float32(fov_x))
        else:
            # the sky prompt is the LAST phrase, so its channel is the last row
            out = model.predict_frame(ip)
            sky = np.asarray(out.probs[-1], np.float32)[::st, ::st]
            np.savez_compressed(dst, sky=sky.astype(np.float16),
                                shape=np.asarray(sky.shape, np.int32),
                                stride=np.int32(st), prompt=str(P.SKY_PROMPT))
        n += 1
        if n % 200 == 0:
            print(f"  {a.drive[17:21]} {a.teacher} {n}/{len(frames)} "
                  f"({(time.time()-t0)/n:.2f}s each)", flush=True)
    print(f"[{a.drive[17:21]} {a.teacher}] wrote {n}, skipped {skipped}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
