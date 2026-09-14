#!/usr/bin/env python
"""Oracle input O: causal LiDAR camera depth through the same hard mapper (Gate 9 LD path).

    python tools/gate13_oracle/build_oracle.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n] [--limit k] [--out dir]

For every Gate 12 anchor t: sweeps of stream frames t-4..t only (never a later frame), each projected
into its own rectified camera with the drive calibration, nearest-depth z-buffer on the lattice camera
(Gate 9 lattice_K / zbuffer), depth kept in (1, 60) m; KITTI-360 GT relative poses T_j->t; frozen
mapper VariantMapper(0.2, 0.2, carve) with conf = SUPPORT_CONF on supported pixels and 0 elsewhere,
scale forced to 1; queried on the anchor SSC grid. Output S_O = occupied, F_O = free. No target,
evaluation mask, region, label or future sweep is read (file audit + recorded natives).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
sys.path.insert(0, g13.G9_DIR); sys.path.insert(0, g13.REPO_ROOT)
import common as G9                                                                          # noqa: E402  (Gate 9)
from audit import lattice_K, zbuffer                                                         # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper                                                     # noqa: E402  (Gate 9)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.datasets.audit import FileAudit                                                    # noqa: E402
from core.mapping.mapper import FrameInput                                                   # noqa: E402
from core.mapping import depth as D7                                                         # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402

AUDIT_PATTERNS = tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("lingbot_k360_7drives/targets", "eval_masks", "regions", "semantickitti", "occ3d", "nuscenes", "trident", "semantics")
DIMS = TF.GRID.dims; NV = int(np.prod(DIMS))


def build_one(dev, geo, nat, t, Kl, v2c, Tgw):
    idx = [t - 4, t - 3, t - 2, t - 1, t]; Gt = geo.rect_cam_to_world(nat[t])
    TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
    mp = VariantMapper(dev, 0.2, 0.2, True); mp.scale_state.force(1.0); nsup = []
    for q, j in enumerate(idx):
        pc = geo.read_velodyne(nat[j]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl)
        sup = (zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M); nsup.append(int(sup.sum()))
        mp.step(FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(zb), zb, 0.0).astype(np.float32)).to(dev),
                           conf=torch.from_numpy(np.where(sup, G9.SUPPORT_CONF, 0.0).astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q], log_scale_candidate=0.0))
    qq = mp.query(TF.GRID, Tgw)
    return qq["occupied"].cpu().numpy(), qq["free"].cpu().numpy(), {"natives": [nat[j] for j in idx], "stream_indices": idx, "n_support_pixels": nsup}, np.stack(TG)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args(); d = a.drive
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(g13.m_path(d, 0))))[a.shard::a.shards][:a.limit]      # the Gate 12 anchors
    out_dir = a.out or os.path.dirname(g13.o_path(d, 0)); os.makedirs(out_dir, exist_ok=True); t0 = time.time(); nw = 0
    with FileAudit(patterns=AUDIT_PATTERNS, strict=True) as audit:
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        Kl = lattice_K(np.asarray(geo.calib.K, np.float64)); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64)
        Tgw = np.linalg.inv(np.asarray(geo.rect_cam_to_velo, np.float64))
        for t in ids:
            op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            S_O, F_O, meta, TG = build_one(dev, geo, nat, t, Kl, v2c, Tgw)
            meta.update({"n_S_O": int(S_O.sum()), "n_F_O": int(F_O.sum()), "overlap": int((S_O & F_O).sum())})
            tmp = op + ".tmp.npz"
            np.savez_compressed(tmp, t=np.int64(t), S_O=np.packbits(S_O), F_O=np.packbits(F_O), natives=np.asarray(meta["natives"], np.int64), stream_indices=np.asarray(meta["stream_indices"], np.int64),
                                T_j_to_t=TG, K_lattice=Kl, meta=json.dumps(meta))
            os.replace(tmp, op); nw += 1
            if nw % 50 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(ids)} {(time.time()-t0)/nw:.2f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} oracle inputs, {(time.time()-t0)/60:.1f} min, files opened {len(audit.opened)}, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
