#!/usr/bin/env python
"""Frozen Trident-H on every frame of a Gate-8 training source. Runs in the Trident env.

Mirrors ``tools/gate6/cache_semantics.py`` exactly (same teacher, config, vocabulary,
lattice resampling and stamp), for the new sources only. No target is opened.

    PYTHONPATH=$REPO:$TRI python tools/gate8/cache_trident.py --source sk_train --shard 0 --num-shards 4
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import vocab                                                   # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, resample_to_lattice     # noqa: E402
from gates.gate8 import sources as S                                            # noqa: E402

SAM_CKPT = os.environ.get("GATE6_SAM_CKPT",
                          "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")
PROC_HW = {"sk_train": (154, 518), "occ3d_train": (294, 518)}


def _default_device() -> str:
    """First GPU of the Gate-8 pool. ``GATE8_GPUS`` (default "1 2 3") reserves
    GPU 0 for other users; ``GATE8_DEVICE`` overrides outright."""
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=list(S.TRAIN_SOURCES))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=_default_device())
    a = ap.parse_args()
    ds = S.DATASET_OF[a.source]
    v = vocab.load(ds)
    hp, wp = PROC_HW[a.source]
    out_dir = f"{S.G8_ROOT}/semantics/{a.source}"
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
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate8",
                            f"trident_{a.source}_s{a.shard}.json"),
               {"source": a.source, "shard": a.shard, "n_written": done,
                "n_skipped": skipped, "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(a.device) / 2 ** 30})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
