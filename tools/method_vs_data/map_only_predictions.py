#!/usr/bin/env python
"""The map-only arm: the hard map's occupied channel, with no learned completion, on the eval grid.

Why it exists. Every "ours" arm in the canonical comparison is ``hard map -> CompletionUNet``.
To attribute a regional gain to the *learned* part rather than to accumulated reconstruction we
need the same pipeline with the completion head removed, scored by the same scorer on the same
anchors. That is exactly the hard map's S channel.

What this writes is therefore not a new method and not a new evaluator: it is the intermediate
tensor the frozen pipeline already computes, read out on the evaluation grid by the same
``reduce_any`` rule the pipeline uses for its own occupancy.

    ours        = reduce_any(CompletionUNet(stack(S, 0, F)) > 0, fine, ratio)
    map_only    = reduce_any(S,                                  fine, ratio)

The lines below are copied from ``tools/zero_shot_eval/prediction/pipeline.py::predict_sample``
up to and including ``hard_channels``; nothing after the mapper query is reproduced. The clips
are the frozen causal clips the canonical predictions themselves were produced from, so the
anchor transform, the clip scale and the five input frames are identical by construction -- which
is asserted per sample against the stored canonical prediction.

    PYTHONPATH=$PWD python tools/method_vs_data/map_only_predictions.py \\
        --dataset semantickitti --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch                                                                           # noqa: E402
from core.datasets import grids as CG                                                  # noqa: E402
from tools.occany_comparison import config as OC                                       # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels    # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any                   # noqa: E402

CACHE_ROOT = os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data",
                          "observed_hidden", "map_only")


def cache_path(dataset: str, sample_id: str) -> str:
    return os.path.join(CACHE_ROOT, dataset, f"{sample_id}.npz")


def load_map_only(dataset: str, sample_id: str):
    """``(occupied [N] bool, T_cam_to_grid [4,4])`` for one anchor."""
    with np.load(cache_path(dataset, sample_id)) as z:
        dims = tuple(int(d) for d in z["dims"])
        n = int(np.prod(dims))
        return np.unpackbits(z["occupied"])[:n].astype(bool), z["T_cam_to_grid"]


@torch.no_grad()
def build(dataset: str, device: str, limit=None) -> dict:
    fine = GridSpec.of(CG.PREDICTION_GRID[dataset])
    native = GridSpec.of(CG.EVAL_GRID[dataset])
    ratio = int(round(native.voxel_size / fine.voxel_size))
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    samples = ZS.evaluated_samples(dataset)[:limit]
    os.makedirs(os.path.join(CACHE_ROOT, dataset), exist_ok=True)
    written = present = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        out = cache_path(dataset, s.sample_id)
        if os.path.exists(out):
            present += 1
            continue
        T = np.asarray(s.T_cam_to_grid_ours, np.float64)
        clip = WindowClip.load(OC.clip_path(dataset, "causal", s.sample_id))
        assert clip.anchor_slot == 4, s.sample_id
        mapper = build_mapper(clip, dev)
        S, F_ = hard_channels(mapper, CG.PREDICTION_GRID[dataset], T)
        del mapper
        occ_n = reduce_any(S.reshape(fine.dims).reshape(-1), fine, ratio).cpu().numpy()
        # the canonical prediction of the SAME anchor must agree on placement and on the
        # hard-map support, or this arm is not the same pipeline minus the head
        ref = CanonicalPrediction.load(ZC.prediction_path(dataset, s.sample_id))
        assert np.allclose(ref.T_cam_to_grid, T, atol=1e-6), s.sample_id
        known = reduce_any((S | F_).reshape(fine.dims).reshape(-1), fine, ratio).cpu().numpy()
        assert np.array_equal(known, ref.observed), \
            f"{s.sample_id}: recomputed hard-map support differs from the stored one"
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, occupied=np.packbits(occ_n),
                            dims=np.asarray(native.dims, np.int64),
                            T_cam_to_grid=T, sample_id=s.sample_id, dataset=dataset,
                            clip_scale=np.float64(clip.scale_m_per_unit))
        os.replace(tmp, out)
        written += 1
        if written % 100 == 0:
            print(f"  {i + 1}/{len(samples)}  {(time.time() - t0) / written:.2f}s each",
                  flush=True)
    return {"dataset": dataset, "samples": len(samples), "written": written,
            "present": present, "cache_root": CACHE_ROOT,
            "rule": "map_only = reduce_any(hard-map S, prediction grid, ratio)",
            "seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(build(a.dataset, a.device, a.limit), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
