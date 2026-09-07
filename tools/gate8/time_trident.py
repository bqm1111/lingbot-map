#!/usr/bin/env python
"""Time frozen Trident-H online on a representative subset (Trident environment)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import vocab                                                   # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, resample_to_lattice     # noqa: E402
from gates.gate8 import sources as S                                            # noqa: E402
SAM_CKPT = os.environ.get("GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def _default_device() -> str:
    """First GPU of the Gate-8 pool. ``GATE8_GPUS`` (default "1 2 3") reserves
    GPU 0 for other users; ``GATE8_DEVICE`` overrides outright."""
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="kitti360")
    ap.add_argument("--frames", type=int, default=30); ap.add_argument("--device", default=_default_device())
    a = ap.parse_args()
    v = vocab.load(S.DATASET_OF[a.source])
    fr = S.segments(a.source, REPO_ROOT)[0].frames[:a.frames]
    T = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)
    T.predict_frame(fr[0].path)
    ts = []
    for f in fr:
        torch.cuda.synchronize(); t = time.time()
        out = T.predict_frame(f.path); resample_to_lattice(out.probs, (140, 518))
        torch.cuda.synchronize(); ts.append(time.time() - t)
    res = {"source": a.source, "n": len(ts), "median_ms": 1e3 * float(np.median(ts)),
           "p95_ms": 1e3 * float(np.percentile(ts, 95)),
           "peak_gpu_gib": torch.cuda.max_memory_allocated(a.device) / 2 ** 30}
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate8", "trident_online_timing.json"), res)
    print(res); return 0


if __name__ == "__main__":
    raise SystemExit(main())
