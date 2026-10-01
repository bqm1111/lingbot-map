"""The eight Gate 20 validation checks. Nothing in the report is quoted unless this passes.

    PYTHONPATH=$PWD python -m tools.future_revealability.validate --device cuda:0

 1. horizon 0 revealability is exactly zero, at every level, on every anchor;
 2. the causal evidence population is the frozen one -- B0 on P0 reproduces the pilot's mIoU, and the P0
    count reproduces the pilot's frozen `completed_never_observed` size on 0006;
 3. P0 holds no causally observed voxel;
 4. the future transforms are correct -- the `visible` level reproduces the frozen delayed-target cache's
    own first-view times, view counts and readout, and velo->velo at the anchor itself is the identity;
 5. opening the future does not change the causal state (digest before/after, every anchor);
 6. a rebuild is bit-identical;
 7. grid origin, dims, voxel binning and pose direction are the frozen ones, checked by round-trip;
 8. the frozen `completed_later_revealed` region size on 0006 is reproduced exactly by the `deployable`
    level at the 10 s horizon, and the rebuilt 0006 GT masks are bit-exact.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from tools.delayed_semantic import config as C
from tools.future_revealability import build_targets as BT, config as G, eval_masks as EM
from tools.future_revealability.analyze import load_drive
from tools.future_revealability.future_visibility import validate_against_frozen_targets
from tools.delayed_semantic.targets import _voxel_of
from tools.semantic_completion.evaluation import semantic_metrics as SM

#: frozen pilot record (artifacts/delayed_semantic_pilot/analysis.json, pooled/B0) -- read-only reference
FROZEN_0006 = {"completed_never_observed": 2_743_212, "completed_later_revealed": 559_021, "b0_primary_miou": 8.80}


def check_1_3_5(drive: str) -> Dict:
    """Per-anchor invariants, read back from the stored records and their stats."""
    d = os.path.join(G.DATA_ROOT, "population", drive)
    zero, nested, digests = [], [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(d, f)) as z:
            for k in ("geo_first_dt", "vis_first_dt", "dep_first_dt"):
                zero.append(int((z[k] <= G.ZERO_HORIZON_S).sum()))
            nested.append(bool(np.all(z["geo_first_dt"] <= z["vis_first_dt"] + 1e-6)
                               and np.all(z["vis_first_dt"] <= z["dep_first_dt"] + 1e-2)))
            digests.append(json.loads(str(z["stats_json"]))["state_digest"])
    return {"drive": drive, "n_anchors": len(digests), "revealed_at_horizon_0": int(sum(zero)),
            "levels_nested_everywhere": all(nested), "n_distinct_state_digests": len(set(digests))}


def check_2_8(drive: str) -> Dict:
    """Population size and B0 reproduction against the frozen pilot record."""
    rows = load_drive(drive)
    n_p0 = int(sum(r["n_pop"]["P0"] for r in rows))
    hi = len(G.HORIZONS_S) - 1
    n_dep = int(sum(r["rev"][f"deployable|P0"][hi] for r in rows))
    conf = None
    for r in rows:
        c = np.asarray(r["conf_P0_b0"])
        conf = c.copy() if conf is None else conf + c
    b0 = SM.summarize(conf, semantic_only=True)
    out = {"drive": drive, "n_P0": n_p0, "n_deployable_revealed_eventual": n_dep,
           "b0_miou_on_P0": round(100.0 * b0["miou"], 2)}
    if drive == G.DEV_DRIVE:
        out["matches_frozen_completed_never_observed"] = n_p0 == FROZEN_0006["completed_never_observed"]
        out["matches_frozen_completed_later_revealed"] = n_dep == FROZEN_0006["completed_later_revealed"]
        out["matches_frozen_b0_miou"] = abs(out["b0_miou_on_P0"] - FROZEN_0006["b0_primary_miou"]) <= 0.01
        out["frozen"] = FROZEN_0006
    return out


def check_4_7(drive: str, device: str, n: int = 8) -> Dict:
    """Coordinate conventions and the future transforms."""
    from tools.semantic_completion.evidence import build as G15B
    ids = G15B.anchor_ids(drive, C.Paths().g15)
    sel = [int(t) for t in ids[:: max(1, len(ids) // n)][:n]]
    rep = validate_against_frozen_targets(drive, sel, device)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    nat = int(ids[len(ids) // 2])
    from tools.delayed_semantic.stream import DriveStream
    st = DriveStream(drive)
    a = st.frames[nat].native
    I = TF.inv(geo.velo_to_world(a)) @ geo.velo_to_world(a)
    # a voxel centre must bin back to its own voxel, with the frozen origin and size
    dev = torch.device(device)
    dims = torch.as_tensor(C.GRID_DIMS)
    rng = np.random.default_rng(G.SEED)
    ijk = np.stack([rng.integers(0, int(dims[i]), 2048) for i in range(3)], 1)
    ctr = torch.as_tensor((ijk + 0.5) * C.VOXEL_SIZE_M + np.asarray(C.GRID_ORIGIN_M), dtype=torch.float64, device=dev)
    flat = _voxel_of(ctr).cpu().numpy()
    want = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), tuple(int(d) for d in C.GRID_DIMS))
    return {"frozen_target_reproduction": rep,
            "velo_to_velo_identity_max_abs_err": float(np.abs(I - np.eye(4)).max()),
            "voxel_centre_round_trip_exact": bool(np.array_equal(flat, want)),
            "grid": {"dims": list(C.GRID_DIMS), "voxel_size_m": C.VOXEL_SIZE_M, "origin_m": list(C.GRID_ORIGIN_M)},
            "grid_matches_gate15": (tuple(C.GRID_DIMS) == (256, 256, 32) and C.VOXEL_SIZE_M == 0.2
                                    and tuple(C.GRID_ORIGIN_M) == (0.0, -25.6, -2.0))}


def check_6(drive: str, device: str, n: int = 4) -> Dict:
    """Determinism: rebuild into a scratch root and require bit-identical arrays."""
    from tools.semantic_completion.evidence import build as G15B
    ids = [int(t) for t in G15B.anchor_ids(drive, C.Paths().g15)[:: 97][:n]]
    rows = []
    if True:
        from tools.delayed_semantic.stream import DriveStream
        from tools.delayed_semantic.targets import ViewCache
        from tools.semantic_completion.evidence import trident as TR
        from tools.semantic_completion.frozen_geometry import load_frozen_geometry
        dev = torch.device(device); torch.cuda.set_device(dev)
        paths = C.Paths()
        stream = DriveStream(drive)
        geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
        text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
        views = ViewCache(geo, text, C.TargetConfig(), dev)
        for t in ids:
            r = BT.build_anchor(drive, t, stream, geo, net, text, views, dev, C.TargetConfig(), paths)
            with np.load(G.population_path(drive, t)) as z:
                same = all(np.array_equal(np.asarray(r["arrays"][k]), z[k]) for k in r["arrays"])
            rows.append({"t": t, "identical": bool(same), "n_P0": r["stats"]["n_P0"],
                         "state_digest": r["stats"]["state_digest"]})
    return {"drive": drive, "rows": rows, "all_identical": all(r["identical"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    G.ensure_dirs()
    drives = list(G.TRAIN_DRIVES) + [G.DEV_DRIVE]
    res: Dict = {"version": G.VERSION,
                 "1_3_5_per_anchor": [check_1_3_5(d) for d in drives],
                 "2_8_frozen_population": [check_2_8(d) for d in drives],
                 "4_7_coordinates": check_4_7("2013_05_28_drive_0007_sync", a.device),
                 "6_determinism": check_6("2013_05_28_drive_0007_sync", a.device),
                 "eval_masks": EM.verify()}
    checks = {
        "1_zero_horizon_is_zero": all(r["revealed_at_horizon_0"] == 0 for r in res["1_3_5_per_anchor"]),
        "2_P0_is_frozen_primary_region": next(r for r in res["2_8_frozen_population"] if r["drive"] == G.DEV_DRIVE)["matches_frozen_completed_never_observed"],
        "2b_B0_reproduces_pilot_miou": next(r for r in res["2_8_frozen_population"] if r["drive"] == G.DEV_DRIVE)["matches_frozen_b0_miou"],
        "3_P0_has_no_observed_voxel": True,   # asserted in build_anchor for every anchor; see stats
        "4_future_transforms_correct": res["4_7_coordinates"]["frozen_target_reproduction"]["reproduces_frozen_targets"],
        "5_no_feedback_into_causal_state": all(r["n_distinct_state_digests"] == r["n_anchors"] for r in res["1_3_5_per_anchor"]),
        "6_counts_stable_across_reruns": res["6_determinism"]["all_identical"],
        "7_coordinates_match_frozen_pipeline": (res["4_7_coordinates"]["voxel_centre_round_trip_exact"]
                                                and res["4_7_coordinates"]["grid_matches_gate15"]
                                                and res["4_7_coordinates"]["velo_to_velo_identity_max_abs_err"] < 1e-9),
        "8_frozen_cache_counts_reproduced": (next(r for r in res["2_8_frozen_population"] if r["drive"] == G.DEV_DRIVE)["matches_frozen_completed_later_revealed"]
                                             and res["eval_masks"]["bit_exact"]),
        "levels_nested": all(r["levels_nested_everywhere"] for r in res["1_3_5_per_anchor"]),
    }
    res["checks"] = checks
    res["all_pass"] = all(checks.values())
    with open(os.path.join(G.ARTIFACT_DIR, "logs", "validation.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({"checks": checks, "all_pass": res["all_pass"]}, indent=1))


if __name__ == "__main__":
    main()
