#!/usr/bin/env python
"""Consistency check: does the trained source change the CompletionUNet's inference cost?

The three source heads are the same class at the same width on the same grid, so they cannot
differ in cost -- but that is an argument, and the brief asks for a check. This times the
forward of each arm's seed-0 checkpoint on identical inputs, so the claim is measured. The
inputs are the hard-map channels of the benchmark anchors, which the arms share exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                           # noqa: E402
from core.datasets import grids as CG                                                  # noqa: E402
from core.model.net import apply_residual, load_checkpoint                             # noqa: E402
from tools.method_vs_data.runtime import config as RC                                  # noqa: E402
from tools.occany_comparison import config as OC                                       # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels    # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.prediction.grid import GridSpec                               # noqa: E402

CKPT = {"k360": "artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt",
        "waymo": "artifacts/waymo_arm/checkpoints/geo_seed0_step06000.pt",
        "once": "artifacts/method_vs_data/once_arm/checkpoints/geo_seed0_step06000.pt"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="semantickitti")
    ap.add_argument("--device", default=RC.BENCH_DEVICE)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RC.ARTIFACT_DIR, "unet_arm_consistency.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    fine = GridSpec.of(CG.PREDICTION_GRID[a.dataset])
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(a.dataset)}
    anchors = [x for x in RC.read_anchors(a.dataset) if x in by_sid][:a.n]

    xs = []
    with torch.no_grad():
        for sid in anchors:
            s = by_sid[sid]
            clip = WindowClip.load(OC.clip_path(a.dataset, "causal", sid))
            m = build_mapper(clip, dev)
            S, F_ = hard_channels(m, CG.PREDICTION_GRID[a.dataset],
                                  np.asarray(s.T_cam_to_grid_ours, np.float64))
            del m
            xs.append(torch.stack([S.reshape(fine.dims),
                                   torch.zeros(fine.dims, dtype=torch.bool, device=dev),
                                   F_.reshape(fine.dims)]).float()[None])

    rows = {}
    for arm, rel in CKPT.items():
        p = os.path.join(ZC.REPO_ROOT, rel)
        if not os.path.exists(p):
            rows[arm] = {"checkpoint": rel, "status": "absent"}
            continue
        net = load_checkpoint(p, dev).net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        lat = []
        with torch.no_grad():
            for _ in range(2):                                  # warm-up
                for x in xs[:3]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        net(x)
            torch.cuda.synchronize()
            for _ in range(a.repeats):
                for x in xs:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        res, _ = net(x)
                    r = res[:, 0].float()
                    _ = apply_residual(torch.zeros_like(r), r).reshape(1, -1) > 0
                    torch.cuda.synchronize()
                    lat.append(time.perf_counter() - t)
        rows[arm] = {"checkpoint": rel, "n_params": int(net.n_params()),
                     **RC.summarise(lat)}
        del net
        torch.cuda.empty_cache()
    out = {"dataset": a.dataset, "grid": fine.name, "dims": list(fine.dims),
           "n_anchors": len(anchors), "repeats": a.repeats, "anchors": anchors,
           "device": a.device, "arms": rows,
           "note": "CompletionUNet forward plus the residual decision only, on identical "
                   "hard-map inputs; this is the stage a different trained source could in "
                   "principle change, and the rest of the pipeline is arm-independent by "
                   "construction (the hard map is bit-identical across arms)."}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    for k, v in rows.items():
        print(f"  {k:7s} {v.get('n_params', 0):>9,}  mean {v.get('mean_ms', float('nan')):7.3f} ms"
              f"  median {v.get('median_ms', float('nan')):7.3f}")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
