#!/usr/bin/env python
"""Phase A — occupancy on the evaluation grid for each one-frame / five-frame arm.

Every arm goes through the SAME frozen chain: clip -> ``build_mapper`` -> ``hard_channels``
-> ``network_input`` -> frozen ``CompletionUNet`` -> ``occ_to_eval_grid``. The only things
that differ are which clip is loaded and whether the network runs at all.

Arms (``A0`` is not produced here -- it is read from the existing canonical predictions, so
the ablation is anchored to the already-published 26.396948):

  one_frame_unet        A1   one-frame clip, frozen CompletionUNet
  one_frame_map         A2   one-frame clip, hard map only (the ``S`` channel)
  five_frame_map             five-frame clip, hard map only -- the published map-only floor
  one_frame_unet_scale5 DIAG one-frame clip, CompletionUNet, but the metric scale forced to
                             the five-frame clip's value. This is deliberate leakage and is
                             NOT a protocol arm: it exists only to split a one-frame
                             regression into its metric-scale part and its coverage part,
                             which the five-frame/one-frame scale ratio of ~0.72-0.78 shows
                             is a real confound.

Stored per sample: ``occupied`` and ``observed`` (= S|F reduced), bit-packed on the dataset's
official evaluation grid, so the scorer can treat every arm identically.

    PYTHONPATH=$PWD python tools/method_vs_data/one_frame_predict.py \
        --dataset semantickitti --arms one_frame_unet,one_frame_map,five_frame_map
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import grids as CG                                                   # noqa: E402
from core.evaluation.regions import occ_to_eval_grid                                    # noqa: E402
from tools.baseline_reconstruction.current.geometry import network_input                # noqa: E402
from tools.method_vs_data import one_frame as OF                                        # noqa: E402
from tools.occany_comparison import config as OC                                        # noqa: E402
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels     # noqa: E402
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, occupancy_and_trunk  # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402

PRED_ROOT = os.environ.get("ONE_FRAME_PRED_ROOT",
                           "/media/minh/MINH1/lingbot_one_frame/predictions")
#: arm -> (which clip, run the network?, force the five-frame scale?)
ARMS = {
    "one_frame_unet":        ("one", True, False),
    "one_frame_map":         ("one", False, False),
    "five_frame_map":        ("five", False, False),
    "one_frame_unet_scale5": ("one", True, True),
}


def out_path(arm: str, dataset: str, sid: str) -> str:
    return os.path.join(PRED_ROOT, arm, dataset, f"{sid}.npz")


def clip_for(which: str, dataset: str, sid: str) -> str:
    return (OF.clip_path(dataset, sid) if which == "one"
            else OC.clip_path(dataset, "causal", sid))


@torch.no_grad()
def predict(arm: str, dataset: str, net, device, limit=None) -> dict:
    which, use_net, force5 = ARMS[arm]
    pgrid = CG.PREDICTION_GRID[dataset]
    samples = ZS.evaluated_samples(dataset)[:limit]
    n_written = n_skipped = 0
    no_scale: list = []
    t0 = time.time()
    for i, s in enumerate(samples):
        op = out_path(arm, dataset, s.sample_id)
        if os.path.exists(op):
            n_skipped += 1
            continue
        cp = clip_for(which, dataset, s.sample_id)
        if not os.path.exists(cp):
            raise FileNotFoundError(f"{arm}: missing clip {cp}")
        # A one-frame clip has a single MoGe-B candidate, so there is no median to fall back
        # on: one non-finite candidate means no metric scale at all. The frozen loader
        # refuses such a clip, which is correct. Record it and skip -- the scorer then drops
        # the sample from EVERY arm so the comparison stays paired.
        with np.load(cp) as _z:
            if not np.isfinite(float(_z["scale"])) or float(_z["scale"]) <= 0:
                no_scale.append(s.sample_id)
                continue
        clip = WindowClip.load(cp)
        if which == "one":
            assert clip.pose_c2w_canonical.shape[0] == 1 and clip.anchor_slot == 0, \
                f"{s.sample_id}: one-frame clip has {clip.pose_c2w_canonical.shape[0]} slots"
            if force5:                                 # the labelled diagnostic arm only
                five = WindowClip.load(OC.clip_path(dataset, "causal", s.sample_id))
                clip = WindowClip(clip.anchor_slot, clip.depth_canonical, clip.conf,
                                  clip.pose_c2w_canonical, clip.K, five.scale_m_per_unit)
        else:
            assert clip.anchor_slot == 4 and clip.pose_c2w_canonical.shape[0] == 5

        T = np.asarray(s.T_cam_to_grid_ours, np.float64)
        mapper = build_mapper(clip, device)
        S, F_ = hard_channels(mapper, pgrid, T)
        del mapper
        if use_net:
            occ_fine, _ = occupancy_and_trunk(net, network_input(S, F_, pgrid.dims))
            occ_fine = occ_fine[0]
        else:
            occ_fine = S
        occupied = occ_to_eval_grid(occ_fine, dataset).cpu().numpy()
        observed = occ_to_eval_grid(S | F_, dataset).cpu().numpy()
        os.makedirs(os.path.dirname(op), exist_ok=True)
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, occupied=np.packbits(occupied),
                            observed=np.packbits(observed),
                            n=np.int64(occupied.size), arm=arm,
                            scale=np.float64(clip.scale_m_per_unit),
                            n_input_frames=np.int64(clip.pose_c2w_canonical.shape[0]))
        os.replace(tmp, op)
        n_written += 1
        if n_written % 200 == 0:
            print(f"  [{arm} {dataset}] {i + 1}/{len(samples)} "
                  f"{(time.time() - t0) / n_written:.2f}s each", flush=True)
    return {"arm": arm, "dataset": dataset, "n_samples": len(samples),
            "n_written": n_written, "n_skipped": n_skipped,
            "n_no_scale": len(no_scale), "no_scale_samples": no_scale,
            "minutes": (time.time() - t0) / 60, "root": PRED_ROOT}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--arms", default="one_frame_unet,one_frame_map,five_frame_map")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    net = load_frozen_geometry(ZC.GEOMETRY_SEED, dev)
    recs = []
    for arm in a.arms.split(","):
        assert arm in ARMS, f"unknown arm {arm}; choose from {sorted(ARMS)}"
        r = predict(arm, a.dataset, net, dev, a.limit)
        recs.append(r)
        print(f"[{arm} {a.dataset}] {r['n_written']} written (+{r['n_skipped']} present) "
              f"of {r['n_samples']}, {r['n_no_scale']} without a one-frame scale, "
              f"{r['minutes']:.1f} min", flush=True)
    d = os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data", "logs")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"one_frame_predict_{a.dataset}.json")
    json.dump(recs, open(p, "w"), indent=1, default=float)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
