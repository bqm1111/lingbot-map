#!/usr/bin/env python
"""Gate 8D Phase 3: audit the dense supervision on KITTI-360 before any weight is trained.

The decisive number is the **pseudo-free collision rate**: a voxel that MoGe or the sky
teacher called free, which a *held-out* later LiDAR sweep -- one that was NOT used to build
that target -- reports as an endpoint. That is a direct measurement of how often the
pseudo-teachers erase real geometry, and the protocol pre-registered the ceiling
(``TARGET_ACCEPTANCE``) before any target file existed.

Also reports, per evidence source, the supervision the new target adds over the Gate 8C-1
baselines, by height and by range, and the occupied-recall check that stops the run if the
dense target has destroyed geometry the sparse one had.

Runs on KITTI-360 only, inside the file-access audit.

    python tools/gate8d/audit_targets.py --n 120 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from gates.gate8d import densetarget as DT, futureframes as FF, protocol as P          # noqa: E402
from gates.gate8d import sources as SRC                                                # noqa: E402
from sscbench_kitti360.audit import FileAudit                                    # noqa: E402

#: the Gate 8C-1 target sets this audit compares against, all KITTI-360
BASELINES = {
    "gate8c1_raw_lidar": "/media/SSD1/MINH_DATASETS/lingbot_gate8c1/targets",
    "gate8c1_plain_column": "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky/targets",
    "gate8c1_neighbourhood_rejected": "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky2/targets",
}


def load_baseline(root, drive, idx):
    p = f"{root}/{drive}/{idx:05d}.npz"
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        n = int(np.prod([int(x) for x in z["dims"]]))
        return {"valid": np.unpackbits(z["valid_packed"])[:n].astype(bool),
                "occupied": np.unpackbits(z["occ_packed"])[:n].astype(bool)}


def heldout_endpoints(geo, anchor_native, used_frames, grid, device, n_extra=40):
    """Voxels that LiDAR sweeps *after* the target's horizon report as occupied.

    These sweeps were not available to the target builder, so agreeing with them is
    evidence the pseudo-free rule did not erase real geometry.
    """
    dims = tuple(int(d) for d in grid.dims)
    n = int(np.prod(dims))
    vs = float(grid.voxel_size)
    origin = torch.as_tensor(np.asarray(grid.origin, np.float64), device=device)
    hi = max(used_frames) if len(used_frames) else int(anchor_native)
    later = [f for f in range(hi + 1, hi + 1 + n_extra) if f in geo.cam0_to_world]
    occ = torch.zeros(n, dtype=torch.bool, device=device)
    for nf in later:
        try:
            pts_np = geo.read_velodyne(nf)
        except (FileNotFoundError, OSError):
            continue
        T = torch.as_tensor(geo.velo_to_velo(nf, int(anchor_native)), device=device)
        pts = torch.as_tensor(pts_np[:, :3].astype(np.float64), device=device)
        pts = pts @ T[:3, :3].T + T[:3, 3]
        idx, _ = DT._voxelize(pts, origin, vs, dims)
        if len(idx):
            occ[DT._flat(idx, dims)] = True
    return occ.cpu().numpy(), len(later)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="anchors per drive")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    grid = TF.GRID
    dims = tuple(int(d) for d in grid.dims)
    vs = float(grid.voxel_size)
    zc = (np.arange(dims[2]) + 0.5) * vs + float(grid.origin[2])

    acc = {"n_anchors": 0, "collision_num": 0, "collision_den": 0,
           "lidar_free_collision_num": 0, "lidar_free_collision_den": 0,
           "occ_recall_num": 0, "occ_recall_den": 0}
    sup_z = np.zeros(dims[2]); occ_z = np.zeros(dims[2]); tot_z = np.zeros(dims[2])
    base_sup_z = {k: np.zeros(dims[2]) for k in BASELINES}
    frac = {"occupied": [], "free": [], "unknown": [], "free_lidar": [], "free_pseudo": []}
    gain = []
    rng_sup = np.zeros(9); rng_tot = np.zeros(9)

    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for drive in SRC.ALL_DRIVES:
            geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
            ancs = SRC.anchors(drive, REPO_ROOT)
            step = max(1, len(ancs) // a.n)
            for anc in ancs[::step][:a.n]:
                p = SRC.target_path(drive, anc.stream_index)
                if not os.path.exists(p):
                    continue
                t = DT.load(p)
                st_valid, st_occ, st_free = t["valid"], t["occupied"], t["free"]
                pseudo_free = st_free & (t["w_pseudo_free"] > 0) & ~st_occ
                acc["n_anchors"] += 1
                n = st_valid.size
                frac["occupied"].append(st_occ.mean())
                frac["free"].append(st_free.mean())
                frac["unknown"].append(1 - st_valid.mean())
                frac["free_pseudo"].append(pseudo_free.mean())
                frac["free_lidar"].append((st_free & ~pseudo_free).mean())
                cov = FF.coverage(anc.native_frame, anc.future_natives, geo.cam0_to_world)
                gain.append(cov["gain"])

                v3 = st_valid.reshape(dims); o3 = st_occ.reshape(dims)
                sup_z += v3.sum(axis=(0, 1)); occ_z += o3.sum(axis=(0, 1))
                tot_z += v3[:, :, 0].size
                ii, jj = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
                x = (ii + .5) * vs + float(grid.origin[0])
                y = (jj + .5) * vs + float(grid.origin[1])
                rb = np.minimum((np.hypot(x, y) // 6).astype(int), 8)
                for b in range(9):
                    m = rb == b
                    rng_sup[b] += v3[m].sum(); rng_tot[b] += m.sum() * dims[2]

                # --- the acceptance gate -------------------------------------------
                held, n_later = heldout_endpoints(geo, anc.native_frame,
                                                  t["frames_used"], grid, dev)
                if n_later:
                    acc["collision_num"] += int((pseudo_free & held).sum())
                    acc["collision_den"] += int(pseudo_free.sum())
                    lf = st_free & ~pseudo_free
                    acc["lidar_free_collision_num"] += int((lf & held).sum())
                    acc["lidar_free_collision_den"] += int(lf.sum())

                for k, root in BASELINES.items():
                    b = load_baseline(root, drive, anc.stream_index)
                    if b is None:
                        continue
                    base_sup_z[k] += b["valid"].reshape(dims).sum(axis=(0, 1))
                    if k == "gate8c1_plain_column":
                        acc["occ_recall_num"] += int((b["occupied"] & st_occ).sum())
                        acc["occ_recall_den"] += int(b["occupied"].sum())

    coll = acc["collision_num"] / max(acc["collision_den"], 1)
    lid_coll = acc["lidar_free_collision_num"] / max(acc["lidar_free_collision_den"], 1)
    recall = acc["occ_recall_num"] / max(acc["occ_recall_den"], 1)
    ok_coll = coll <= P.TARGET_ACCEPTANCE["max_pseudo_free_collision_rate"]
    ok_rec = recall >= P.TARGET_ACCEPTANCE["min_occupied_recall_vs_gate8c1"]

    out = {"n_anchors": acc["n_anchors"], "protocol_sha256": P.source_digest(),
           "acceptance": dict(P.TARGET_ACCEPTANCE),
           "pseudo_free_collision_rate": coll,
           "lidar_free_collision_rate_reference": lid_coll,
           "occupied_recall_vs_gate8c1_plain_column": recall,
           "accepted": bool(ok_coll and ok_rec),
           "native_sweep_gain_mean": float(np.mean(gain)) if gain else 0.0,
           "fractions_mean": {k: float(np.mean(v)) for k, v in frac.items() if v},
           "supervised_pct_by_z": (100 * sup_z / np.maximum(tot_z, 1)).tolist(),
           "occupied_pct_by_z": (100 * occ_z / np.maximum(tot_z, 1)).tolist(),
           "baseline_supervised_pct_by_z": {k: (100 * v / np.maximum(tot_z, 1)).tolist()
                                            for k, v in base_sup_z.items()},
           "supervised_pct_by_range": (100 * rng_sup / np.maximum(rng_tot, 1)).tolist(),
           "z_centres_m": zc.tolist(), "firewall": audit.summary()}
    write_json(os.path.join(ART, "target_audit.json"), out)

    print(f"anchors audited: {acc['n_anchors']}   native-sweep gain "
          f"{out['native_sweep_gain_mean']:.2f}x")
    print(f"\nstate fractions: occupied {100*out['fractions_mean']['occupied']:.2f}%  "
          f"free {100*out['fractions_mean']['free']:.2f}%  "
          f"unknown {100*out['fractions_mean']['unknown']:.2f}%")
    print(f"  free from LiDAR {100*out['fractions_mean']['free_lidar']:.2f}%  "
          f"from pseudo-teachers {100*out['fractions_mean']['free_pseudo']:.2f}%")
    print(f"\n{'z (m)':>7} {'8D':>7} {'raw':>7} {'column':>8} {'nbhd':>7}  supervised %")
    b = out["baseline_supervised_pct_by_z"]
    for k in range(dims[2] - 1, -1, -3):
        print(f"{zc[k]:+7.1f} {out['supervised_pct_by_z'][k]:7.1f} "
              f"{b['gate8c1_raw_lidar'][k]:7.1f} {b['gate8c1_plain_column'][k]:8.1f} "
              f"{b['gate8c1_neighbourhood_rejected'][k]:7.1f}")
    print(f"\n--- acceptance gate (pre-registered) ---")
    print(f"  pseudo-free collision rate      {100*coll:6.3f}%   "
          f"limit {100*P.TARGET_ACCEPTANCE['max_pseudo_free_collision_rate']:.1f}%   "
          f"{'PASS' if ok_coll else 'FAIL'}")
    print(f"  (LiDAR-only free, for reference) {100*lid_coll:6.3f}%")
    print(f"  occupied recall vs plain column  {100*recall:6.2f}%   "
          f"limit {100*P.TARGET_ACCEPTANCE['min_occupied_recall_vs_gate8c1']:.0f}%   "
          f"{'PASS' if ok_rec else 'FAIL'}")
    print(f"\nACCEPTED: {out['accepted']}   firewall violations "
          f"{len(out['firewall']['violations'])}")
    return 0 if out["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
