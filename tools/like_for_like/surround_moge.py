"""MoGe-2 metric-scale depth for the five non-front surround cameras.

The clip scale is the frozen rule: the per-frame MoGe candidate, median over the clip's frames. Our cache only ever held
``CAM_FRONT``, so the other five cameras are computed here with the SAME recipe as
``tools/occany_comparison/moge_extra.py`` (same model revision, same processed lattice, same calibrated-FOV rule
``fx_native * W_proc / W_native``), written into the same extra root that ``resolve_moge_path`` already consults.
Per-camera intrinsics come from the nuScenes calibration; nothing here feeds the reconstruction.

Before writing anything, the recipe is re-run on frames the frozen cache already holds and compared.

    PYTHONPATH=$PWD python -m tools.like_for_like.surround_moge --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets import sources as S
from gates.gate7b import replay
from tools.like_for_like import config as LC, surround as SR
from tools.occany_comparison import config as OC
from tools.occany_comparison.moge_extra import MOGE_REVISION, NATIVE_WIDTH

OUT_DIR = os.path.join(OC.MOGE_EXTRA_ROOT, "occ3d")


def _fov_deg(fx_native: float, proc_width: int) -> float:
    from moge_gauge.calibration import calibrated_fov_x_deg
    return calibrated_fov_x_deg(fx_native * proc_width / NATIVE_WIDTH["occ3d"], proc_width)


def infer_frame(moge, image_path: str, fx_native: float) -> Dict[str, np.ndarray]:
    rgb = replay.load_images([image_path])[0].numpy()
    fov = _fov_deg(fx_native, rgb.shape[-1])
    o = moge.infer_calibrated(rgb, fov_x_deg=fov)
    mk = np.asarray(o.mask, bool)
    return {"moge_depth": np.asarray(o.depth_z, np.float32).astype(np.float16), "moge_mask": np.packbits(mk.reshape(-1)),
            "mask_shape": np.asarray(mk.shape, np.int32), "fov_x_deg": np.float32(fov), "fov_out_deg": np.float32(o.fov_x_deg)}


def load_moge(device: str):
    from gates.scale_gate.config import load_config
    from moge_gauge.calibrated import CalibratedMoGe
    dev = torch.device(device); torch.cuda.set_device(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo, revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == MOGE_REVISION, f"MoGe revision drifted: {prov['hf_revision']}"
    return moge, prov


def reproduction_check(moge, n: int = 3) -> list:
    """The recipe must reproduce frames the frozen cache already holds (CAM_FRONT of the first samples)."""
    nusc = SR.nuscenes(); out = []
    for s in SR.samples()[:n]:
        v = SR.ring_views(SR.sample_token(s))[0]
        frozen = S.moge_path("occ3d", v["key"])
        if not os.path.exists(frozen):
            continue
        sd = nusc.get("sample_data", nusc.get("sample", SR.sample_token(s))["data"]["CAM_FRONT"])
        fx = float(np.asarray(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"], float)[0, 0])
        with np.load(frozen) as z:
            ref = {k: z[k] for k in ("moge_depth", "moge_mask", "fov_x_deg")}
        new = infer_frame(moge, v["path"], fx)
        d_ref, d_new = ref["moge_depth"].astype(np.float32), new["moge_depth"].astype(np.float32)
        fin = np.isfinite(d_ref) & np.isfinite(d_new)
        out.append({"key": v["key"], "fov_abs_diff": float(abs(ref["fov_x_deg"] - new["fov_x_deg"])),
                    "mask_equal": bool(np.array_equal(ref["moge_mask"], new["moge_mask"])),
                    "depth_max_rel_diff": float(np.max(np.abs(d_ref[fin] - d_new[fin]) / np.maximum(d_ref[fin], 1e-3)))})
    return out


def run(device: str, shard: int, shards: int, limit: int | None = None) -> Dict:
    nusc = SR.nuscenes()
    todo: Dict[str, tuple] = {}
    for s in SR.samples():
        sample = nusc.get("sample", SR.sample_token(s))
        for v in SR.ring_views(SR.sample_token(s))[1:]:                     # CAM_FRONT is cached
            if v["key"] in todo or os.path.exists(S.moge_path("occ3d", v["key"])) or os.path.exists(os.path.join(OUT_DIR, f"{v['key']}.npz")):
                continue
            sd = nusc.get("sample_data", sample["data"][v["camera"]])
            fx = float(np.asarray(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"], float)[0, 0])
            todo[v["key"]] = (v["path"], fx)
    keys = sorted(todo)[shard::shards][:limit]
    moge, prov = load_moge(device)
    rec = {"shard": shard, "shards": shards, "n_todo": len(keys), "root": OUT_DIR, "moge_provenance": prov}
    if shard == 0:
        rec["reproduction_on_frozen_frames"] = reproduction_check(moge)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0, written = time.time(), 0
    for i, key in enumerate(keys):
        op = os.path.join(OUT_DIR, f"{key}.npz")
        if os.path.exists(op):
            continue
        path, fx = todo[key]
        tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **infer_frame(moge, path, fx)); os.replace(tmp, op); written += 1
        if (i + 1) % 200 == 0:
            print(f"  [moge surround s{shard}] {i + 1}/{len(keys)} {(time.time() - t0) / max(written, 1):.2f}s each", flush=True)
    rec.update({"n_written": written, "seconds": time.time() - t0})
    os.makedirs(LC.LOG_DIR, exist_ok=True)
    json.dump(rec, open(os.path.join(LC.LOG_DIR, f"moge_surround_s{shard}of{shards}.json"), "w"), indent=1, default=str)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    r = run(a.device, a.shard, a.shards, a.limit)
    print(json.dumps({k: v for k, v in r.items() if k != "moge_provenance"}, indent=1, default=str))
if __name__ == "__main__":
    main()
