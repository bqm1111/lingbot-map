#!/usr/bin/env python
"""Frozen Trident-H on every frame of ``k360_train``. Runs in the Trident env.

Mirrors ``tools/gate8/cache_trident.py`` (itself a mirror of Gate 6's cache) with the
KITTI-360 vocabulary and the KITTI-360 Trident lattice (140 x 518, read from the Gate 6
kitti360 cache stamp so it cannot drift from the validation drive's).

    PYTHONPATH=$REPO:$TRI python tools/gate8b/cache_trident.py --shard 0 --num-shards 2
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import vocab                                                          # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, resample_to_lattice            # noqa: E402
from gates.gate8 import sources as S                                                   # noqa: E402
from gates.gate8b.sources import SOURCE                                                # noqa: E402
from tools.gate6.cache_semantics import proc_hw_of                               # noqa: E402

SAM_CKPT = os.environ.get("GATE6_SAM_CKPT",
                          "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    ds = S.DATASET_OF[a.source]
    v = vocab.load(ds)
    hp, wp = proc_hw_of(ds)
    out_dir = os.path.dirname(S.trident_path(a.source, "x"))
    os.makedirs(out_dir, exist_ok=True)
    frames = [f for seg in S.segments(a.source, REPO_ROOT) for f in seg.frames]
    frames = sorted(frames, key=lambda f: f.key)[a.shard::a.num_shards]
    T = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)
    stamp = json.dumps({"dataset": ds, "source": a.source, "n_classes": len(v),
                        "phrases": list(v.phrases), "labels": list(v.labels),
                        "proc_hw": [hp, wp], "teacher": "Trident-H",
                        "config": "official cfg_city_scapes"}, sort_keys=True)
    t0, done, skipped = time.time(), 0, 0
    for f in frames:
        dst = S.trident_path(a.source, f.key)
        if os.path.exists(dst):
            skipped += 1; continue
        out = T.predict_frame(f.path)
        probs = resample_to_lattice(out.probs, (hp, wp))
        tmp = dst + ".tmp.npz"
        np.savez(tmp, probs=probs.numpy().astype(np.float16),
                 label=probs.argmax(0).numpy().astype(np.uint8),
                 native_hw=np.asarray(out.lattice_hw, np.int32),
                 proc_hw=np.asarray([hp, wp], np.int32), ties=np.int64(out.ties),
                 stamp=np.asarray(stamp), image_path=np.asarray(f.path))
        os.replace(tmp, dst)
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"[{a.source} s{a.shard}] {done}/{len(frames)-skipped} {el/done:.2f}s/frame "
                  f"eta {(len(frames)-skipped-done)*el/done/60:.1f} min", flush=True)
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"trident_{a.source}_s{a.shard}.json"),
               {"source": a.source, "shard": a.shard, "n_written": done, "n_skipped": skipped,
                "proc_hw": [hp, wp], "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(a.device) / 2 ** 30})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
