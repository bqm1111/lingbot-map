#!/usr/bin/env python
"""Does float16 caching change anything downstream?

The brief allows float16 caching only after proving that class predictions **and fused
results** agree with float32 to negligible tolerance. Pixel-level agreement is not enough:
a voxel averages many pixels, so errors could in principle accumulate (or, more likely,
cancel). This recomputes the teacher in float32 for a handful of whole clips, runs the
identical lifting, and compares the fused voxel labels against the ones the float16 cache
produces.

Runs in the Trident environment.

    python tools/gate6/fp16_fidelity.py --dataset kitti360 --n 5
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, grids, lifting, pipelines, vocab           # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, resample_to_lattice     # noqa: E402

SAM_CKPT = os.environ.get(
    "GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def fused_labels(ds, clip, sem, dev):
    p = pipelines.predict_clip(ds, clip, sem, dev)
    return p.raw_flat, p.raw_channel, p.dil_flat, p.dil_channel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kitti360", choices=list(vocab.DATASETS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ds, dev = a.dataset, torch.device(a.device)
    v = vocab.load(ds)
    T = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)

    clips = [c for c in pipelines.iter_clips(ds, REPO_ROOT) if c.scale is not None]
    sel = [clips[(i * len(clips)) // a.n] for i in range(a.n)]     # deterministic stride
    root = F.image_root(ds, REPO_ROOT)
    rows = []
    for clip in sel:
        rec = {r.clip_id: r for r in F.read_manifest(ds, REPO_ROOT)}[clip.clip_id]
        hp, wp = None, None
        f32 = []
        for rel in rec.rel_images:
            out = T.predict_frame(os.path.join(root, rel))
            if hp is None:
                with np.load(F.semantic_cache_path(ds, rec.keys[0])) as z:
                    hp, wp = [int(x) for x in z["proc_hw"]]
            f32.append(resample_to_lattice(out.probs, (hp, wp)))
        sem32 = torch.stack(f32, 0).to(dev)
        sem16 = pipelines.load_semantics(ds, clip.frame_keys, dev)
        r32 = fused_labels(ds, clip, sem32, dev)
        r16 = fused_labels(ds, clip, sem16, dev)
        assert np.array_equal(r32[0], r16[0]) and np.array_equal(r32[2], r16[2]), \
            "occupancy must not depend on the teacher at all"
        raw_d = int((r32[1] != r16[1]).sum())
        dil_d = int((r32[3] != r16[3]).sum())
        rows.append({"clip_id": clip.clip_id, "n_raw": int(len(r32[0])),
                     "raw_label_disagreements": raw_d,
                     "raw_disagreement_fraction": raw_d / max(len(r32[0]), 1),
                     "n_dil": int(len(r32[2])), "dil_label_disagreements": dil_d,
                     "dil_disagreement_fraction": dil_d / max(len(r32[2]), 1),
                     "max_abs_prob_err": float((sem32 - sem16).abs().max())})
        print(rows[-1], flush=True)
        del sem32, sem16

    summ = {"dataset": ds, "n_clips": len(rows),
            "total_raw_voxels": sum(r["n_raw"] for r in rows),
            "total_raw_disagreements": sum(r["raw_label_disagreements"] for r in rows),
            "total_dil_voxels": sum(r["n_dil"] for r in rows),
            "total_dil_disagreements": sum(r["dil_label_disagreements"] for r in rows),
            "max_abs_prob_err": max(r["max_abs_prob_err"] for r in rows),
            "occupancy_identical": True, "rows": rows}
    summ["raw_disagreement_fraction"] = (summ["total_raw_disagreements"]
                                         / max(summ["total_raw_voxels"], 1))
    summ["dil_disagreement_fraction"] = (summ["total_dil_disagreements"]
                                         / max(summ["total_dil_voxels"], 1))
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6", f"fp16_fidelity_{ds}.json"),
               summ)
    print(json.dumps({k: summ[k] for k in summ if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
