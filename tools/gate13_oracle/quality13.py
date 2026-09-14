#!/usr/bin/env python
"""Input-quality sanity check on 0006 before training: observed-map IoU / precision / recall /
occupied-in-LiDAR-free of M (S_H) and O (S_O) against the Gate 9 reference -- the same five sweeps'
LiDAR measurement (occupied = return and never traversed, free = traversed and never a return),
inside the union of the five camera frusta (1-60 m). Reference and mask code copied from Gate 9.

    python tools/gate13_oracle/quality13.py --device cuda:0 [--stride 1]

Writes input_quality.json (+ per-anchor counts). O is also compared with Gate 9's stored LD metric
vector and LD map for the same clip (they must be identical).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
sys.path.insert(0, g13.G9_DIR); sys.path.insert(0, g13.REPO_ROOT)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.supervision import rawtarget as RT                                                 # noqa: E402
from core.mapping import depth as D7                                                         # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402

G9_CLIPS = "/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy/clips/2013_05_28_drive_0006_sync"
DIMS = TF.GRID.dims; NV = int(np.prod(DIMS)); LH, LW = 140, 518


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--stride", type=int, default=1); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = g13.VAL_DRIVE
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    Tgw = np.linalg.inv(np.asarray(geo.rect_cam_to_velo, np.float64)); ORIGIN = np.asarray(TF.GRID.origin); VS = float(TF.GRID.voxel_size)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64); ctr_c = torch.from_numpy(((gi + 0.5) * VS + ORIGIN) @ Tgw[:3, :3].T + Tgw[:3, 3]).to(dev)
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(g13.o_path(d, 0))))[::a.stride]
    rec = np.zeros((len(ids), 2, 5), np.int64)                                              # [anchor, M/O, tp fp fn occ_in_mask occ_on_free]
    g9_same = g9_map_same = 0; t0 = time.time()
    for k, t in enumerate(ids):
        zo = np.load(g13.o_path(d, t)); zm = np.load(g13.m_path(d, t)); TG = zo["T_j_to_t"]; Kl = zo["K_lattice"]
        ref = RT.build(geo, nat[t], [nat[j] for j in range(t - 4, t + 1)], dev)
        n_obs = np.asarray(ref.n_obs) > 0; n_free = np.asarray(ref.n_free) > 0; r_occ = n_obs & ~n_free; r_free = n_free & ~n_obs
        fr_ = torch.zeros(NV, dtype=torch.bool, device=dev)
        for T in TG:
            Ti = torch.as_tensor(np.linalg.inv(T), device=dev); pc = ctr_c @ Ti[:3, :3].T + Ti[:3, 3]; zc = pc[:, 2]
            uu = Kl[0, 0] * pc[:, 0] / zc.clamp_min(1e-6) + Kl[0, 2]; vv = Kl[1, 1] * pc[:, 1] / zc.clamp_min(1e-6) + Kl[1, 2]
            fr_ |= (zc > D7.MIN_DEPTH_M) & (zc < D7.MAX_DEPTH_M) & (uu >= -0.5) & (uu < LW - 0.5) & (vv >= -0.5) & (vv < LH - 0.5)
        E = (r_occ | r_free) & fr_.cpu().numpy()
        for c, occ in enumerate((np.unpackbits(zm["S_H"])[:NV].astype(bool), np.unpackbits(zo["S_O"])[:NV].astype(bool))):
            rec[k, c] = [int((occ & r_occ & E).sum()), int((occ & ~r_occ & E).sum()), int((~occ & r_occ & E).sum()), int((occ & E).sum()), int((occ & r_free & E).sum())]
        p9 = os.path.join(G9_CLIPS, f"{t:05d}.npz")
        if os.path.exists(p9):
            g = np.load(p9); SO = np.unpackbits(zo["S_O"])[:NV].astype(bool)
            g9_same += int(np.array_equal(rec[k, 1, :3], g["m_LD"][:3])); g9_map_same += int(np.array_equal(np.flatnonzero(SO).astype(np.int32), g["occ_LD"]))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    s = rec.sum(0)
    summ = {n: {"iou": s[c, 0] / max(s[c, :3].sum(), 1), "precision": s[c, 0] / max(s[c, 0] + s[c, 1], 1), "recall": s[c, 0] / max(s[c, 0] + s[c, 2], 1),
                "occupied_in_lidar_free": s[c, 4] / max(s[c, 3], 1), "occupied_voxels_in_mask_per_anchor": s[c, 3] / len(ids)} for c, n in enumerate(("M", "O"))}
    out = {"n_anchors": len(ids), "reference": "Gate 9: five-sweep LiDAR measurement inside the five camera frusta", "summary": summ,
           "O_equals_gate9_LD_metrics": [g9_same, len(ids)], "O_equals_gate9_LD_map": [g9_map_same, len(ids)], "seconds": time.time() - t0}
    os.makedirs(g13.ART, exist_ok=True); json.dump(out, open(os.path.join(g13.ART, "input_quality.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(g13.ART, "input_quality_counts.npz"), ids=np.asarray(ids), counts=rec)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
