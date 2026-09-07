#!/usr/bin/env python
"""Run the frozen Trident-H teacher once per unique RGB frame and cache the result.

For each frame the cache stores the **complete per-class probability tensor** resampled
onto the LingBot processed lattice -- the lattice the lifting will actually sample. That
is where the tensor is consumed, so storing it there is lossless with respect to the
pipeline and turns a 250 GB native-resolution cache into a 38 GB one.

Determinism: frames are processed in sorted order, sharding is a pure stride, and each
frame is independent, so any shard layout produces byte-identical files.

No target, LiDAR sweep or oracle table is opened. Runs in the Trident environment.

    python tools/gate6/cache_semantics.py --dataset kitti360 --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, vocab                                      # noqa: E402
from gates.gate6.trident_adapter import (TridentTeacher, resample_to_lattice,   # noqa: E402
                                   sha256_file)

SAM_CKPT = os.environ.get(
    "GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def proc_hw_of(dataset: str) -> tuple:
    """The LingBot processed lattice, read from the frozen geometry cache itself."""
    recs = F.read_manifest(dataset, REPO_ROOT)
    for rec in recs:
        p = F.lingbot_cache_path(dataset, rec.clip_id, REPO_ROOT)
        if os.path.exists(p):
            with np.load(p) as d:
                return tuple(int(x) for x in d["proc_hw"])
    raise FileNotFoundError(f"no LingBot cache for {dataset}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(vocab.DATASETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fp32-check-every", type=int, default=257,
                    help="also compute float32 on every Nth frame and record the error")
    a = ap.parse_args()

    ds = a.dataset
    v = vocab.load(ds)
    hp, wp = proc_hw_of(ds)
    out_dir = F.semantic_cache_dir(ds)
    os.makedirs(out_dir, exist_ok=True)

    allf = F.unique_frames(ds, REPO_ROOT)
    mine = F.shard(allf, a.shard, a.num_shards)
    if a.limit:
        mine = mine[:a.limit]

    T = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(a.device)

    stamp = {"dataset": ds, "n_classes": len(v), "phrases": list(v.phrases),
             "labels": list(v.labels), "proc_hw": [hp, wp],
             "teacher": "Trident-H", "config": "official cfg_city_scapes"}
    stamp_s = json.dumps(stamp, sort_keys=True)

    t0, done, skipped, fp32 = time.time(), 0, 0, []
    for i, (key, path) in enumerate(mine):
        dst = F.semantic_cache_path(ds, key)
        if os.path.exists(dst) and not a.overwrite:
            skipped += 1
            continue
        out = T.predict_frame(path)
        probs = resample_to_lattice(out.probs, (hp, wp))          # [C, hp, wp] float32
        if i % a.fp32_check_every == 0:
            lab32 = probs.argmax(0)
            p16 = probs.half().float()
            p16 = p16 / p16.sum(0, keepdim=True).clamp_min(1e-12)
            fp32.append({"frame": key,
                         "max_abs_prob_err": float((p16 - probs).abs().max()),
                         "argmax_disagree": int((p16.argmax(0) != lab32).sum()),
                         "pixels": int(lab32.numel())})
        tmp = dst + ".tmp.npz"
        np.savez(tmp, probs=probs.numpy().astype(np.float16),
                 label=probs.argmax(0).numpy().astype(np.uint8),
                 native_hw=np.asarray(out.lattice_hw, np.int32),
                 proc_hw=np.asarray([hp, wp], np.int32),
                 ties=np.int64(out.ties), stamp=np.asarray(stamp_s))
        os.replace(tmp, dst)
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"[{ds} shard {a.shard}] {done}/{len(mine) - skipped} "
                  f"{el / max(done,1):.2f}s/frame eta {(len(mine)-skipped-done)*el/max(done,1)/60:.1f}min",
                  flush=True)

    rep = {"dataset": ds, "shard": a.shard, "num_shards": a.num_shards,
           "n_assigned": len(mine), "n_written": done, "n_skipped_existing": skipped,
           "seconds": time.time() - t0,
           "seconds_per_frame": (time.time() - t0) / max(done, 1),
           "peak_gpu_gib": (torch.cuda.max_memory_allocated(a.device) / 2**30
                            if a.device.startswith("cuda") else None),
           "fp16_checks": fp32, "proc_hw": [hp, wp], "n_classes": len(v),
           "sam_ckpt_sha256": sha256_file(SAM_CKPT)[:16], "targets_opened": "none"}
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6",
                            f"cache_semantics_{ds}_shard{a.shard}.json"), rep)
    print(json.dumps({k: rep[k] for k in
                      ("dataset", "shard", "n_assigned", "n_written", "n_skipped_existing",
                       "seconds_per_frame", "peak_gpu_gib")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
