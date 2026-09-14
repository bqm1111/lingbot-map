#!/usr/bin/env python
"""GT-registered semantic targets for one drive of the seven-drive experiment.

Reads the samples written by the unchanged ``core/run/build_samples.py`` (causal inputs
+ raw-LiDAR occupancy under the current ``rawtarget.py``) from ``<DATA_ROOT>/samples`` and
writes ``<DATA_ROOT>/samples_gtsem`` with every input array and the occupancy target
copied byte for byte and the semantic target rebuilt WITHOUT the LingBot future mapper:

    LiDAR-occupied voxel centre (anchor velodyne frame)
      --velo_to_world(anchor) . inv(rect_cam_to_world(j))--> rectified camera j   (GT poses + calibration)
      --K--> native pixel (u, v); 1.5 m < z < 60 m; 2-px border
      visible in j iff frame j's own sweep (i) has returns in the 7x7-px window whose depth
      range brackets z within tol = 0.3 m + 1 % z  (surface present), and (ii) its minimum
      depth in the 3x3-px window is not closer than z - tol  (no occluder in front)
      --bilinear sample of the cached Trident-H lattice probabilities of frame j (18 -> 25 union)
      accumulated with weight 1 per visible view over future stream frames t+1 .. t+20

A voxel carries a target with >= 1 visible view, unless the two halves of the window
contradict on the class (the adopted rule). Moving objects: a voxel occupied at the anchor
but vacated later fails (i) and receives nothing from that frame; no other motion model.

    python tools/kitti360_data_scaling_7drives/build_semantic_targets.py --drive 2013_05_28_drive_0003_sync --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from collections import OrderedDict
import numpy as np, torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, union_vocab as V8  # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = 376, 1408
GRID_DIMS = (256, 256, 32); ORIGIN = np.array([0.0, -25.6, -2.0]); VS = 0.2
FUTURE_FRAMES = 20; DEPTH_MIN, DEPTH_MAX = 1.5, 60.0; BORDER_PX = 2
VIS_TOL = (0.3, 0.01); WIN_PRESENT = 7; WIN_OCCLUDE = 3; SEM_W_MIN = 1.0
INPUT_KEYS = ("t", "input_frames", "scale", "dims", "rows", "logodds", "w_occ", "w_free", "n_obs", "n_lb", "age", "observed",
              "sem_rows", "sem_p", "sem_w", "gt_occ", "gt_valid", "n_obs_target", "n_frames_occ_target", "occ_rows_target", "native_frame")
RULE = {"visibility": "present: sweep depth range in the 7x7 window brackets z within tol; not occluded: z - min(3x3 window) < tol; tol = 0.3 m + 1 % z",
        "depth_m": [DEPTH_MIN, DEPTH_MAX], "border_px": BORDER_PX, "future_frames": FUTURE_FRAMES, "sem_w_min": SEM_W_MIN,
        "half_agreement": "a voxel with evidence in both halves of the window is dropped if the halves' argmax differ",
        "weight": "1 per visible future view", "probabilities": "cached Trident-H lattice 140x518, bilinear at the native pixel, align_corners=False, renormalised, 18 -> 25 via into_matrix",
        "mapper_used": False}


def project(K, p):
    z = p[:, 2]; zz = np.where(z > 0, z, 1.0)
    return K[0, 0] * p[:, 0] / zz + K[0, 2], K[1, 1] * p[:, 1] / zz + K[1, 2], z


class FrameCache:
    def __init__(self, geo, src, key_of, K, dev, cap=48):
        self.geo, self.src, self.key_of, self.K, self.dev, self.cap = geo, src, key_of, K, dev, cap
        self.v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); self.d = OrderedDict()

    def get(self, native):
        if native in self.d:
            self.d.move_to_end(native); return self.d[native]
        pts = self.geo.read_velodyne(native); pc = pts @ self.v2c[:3, :3].T + self.v2c[:3, 3]
        u, v, z = project(self.K, pc); ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vi, ui = v[ok].astype(int), u[ok].astype(int)
        mn = np.full((H, W), np.inf); np.minimum.at(mn, (vi, ui), z[ok])
        mx = np.full((H, W), -np.inf); np.maximum.at(mx, (vi, ui), z[ok])
        zb = {"min_p": ndimage.minimum_filter(mn, WIN_PRESENT, mode="nearest"), "max_p": ndimage.maximum_filter(mx, WIN_PRESENT, mode="nearest"),
              "min_o": ndimage.minimum_filter(mn, WIN_OCCLUDE, mode="nearest")}
        with np.load(S.trident_path(self.src, self.key_of[native])) as zf:
            pr = torch.from_numpy(zf["probs"].astype(np.float32)).to(self.dev)[None]
        self.d[native] = (zb, pr)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return self.d[native]


def visible(zb, u, v, z):
    """-> (inside, no_reference, occluded, visible) boolean arrays."""
    inside = (z > DEPTH_MIN) & (z < DEPTH_MAX) & (u >= BORDER_PX) & (u < W - BORDER_PX) & (v >= BORDER_PX) & (v < H - BORDER_PX)
    n = len(u); mnp = np.full(n, np.inf); mxp = np.full(n, -np.inf); mno = np.full(n, np.inf)
    vi, ui = v[inside].astype(int), u[inside].astype(int)
    mnp[inside] = zb["min_p"][vi, ui]; mxp[inside] = zb["max_p"][vi, ui]; mno[inside] = zb["min_o"][vi, ui]
    tol = VIS_TOL[0] + VIS_TOL[1] * z
    noref = inside & ~np.isfinite(mnp)
    occl = inside & np.isfinite(mno) & (z - mno > tol)
    present = inside & np.isfinite(mnp) & (z >= mnp - tol) & (z <= mxp + tol)
    return inside, noref, occl, present & ~occl


def sample_probs(pr, u, v, dev):
    g = torch.stack([(torch.from_numpy(u).float().to(dev) + 0.5) / W * 2 - 1, (torch.from_numpy(v).float().to(dev) + 0.5) / H * 2 - 1], -1)[None, None]
    out = F.grid_sample(pr, g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    return out / out.sum(1, keepdim=True).clamp_min(1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    natives = [int(f.key.split("_")[-1]) for f in seg.frames]; key_of = {n: f.key for n, f in zip(natives, seg.frames)}
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K, np.float64)
    into = torch.from_numpy(V8.into_matrix("kitti360")).to(dev)
    cache = FrameCache(geo, src, key_of, K, dev)
    files = sorted(glob.glob(os.path.join(drives7.DATA_ROOT, "samples", a.drive, "*.npz")))[:a.limit]
    out_dir = os.path.join(drives7.DATA_ROOT, "samples_gtsem", a.drive); os.makedirs(out_dir, exist_ok=True)
    n = int(np.prod(GRID_DIMS)); t0 = time.time(); nw = nskip = 0
    st = {"n_anchors": 0, "occupied": 0, "candidates": 0, "outside_or_range": 0, "no_reference": 0, "occluded": 0, "not_present": 0, "visible": 0,
          "target_voxels": 0, "contradictions": 0, "mapper_target_voxels": 0, "views_per_voxel_hist": np.zeros(21, np.int64)}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Z = {k: z[k] for k in z.files}
            t = int(Z["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                nskip += 1; continue
            nat = natives[t]; fut = list(range(t + 1, min(t + 1 + FUTURE_FRAMES, len(natives))))
            assert sorted(fut) == sorted(Z["target_frames"].tolist()), "future window differs from the sample builder's"
            occ = (np.unpackbits(Z["gt_occ"])[:n] & np.unpackbits(Z["gt_valid"])[:n]).astype(bool)
            idx = np.flatnonzero(occ); ctr = (np.stack(np.unravel_index(idx, GRID_DIMS), 1) + 0.5) * VS + ORIGIN
            Tvw = geo.velo_to_world(nat); m = len(idx)
            acc = torch.zeros(m, V8.U, device=dev); wsum = torch.zeros(m, device=dev)
            half = [torch.zeros(m, V8.U, device=dev), torch.zeros(m, V8.U, device=dev)]; nfr = np.zeros(m, np.int32); vis_any = np.zeros(m, bool)
            st["n_anchors"] += 1; st["occupied"] += m
            for q, j in enumerate(fut):
                nj = natives[j]
                if nj not in geo.cam0_to_world:
                    continue
                T = TF.inv(geo.rect_cam_to_world(nj)) @ Tvw; u, v, zc = project(K, ctr @ T[:3, :3].T + T[:3, 3])
                zb, pr = cache.get(nj); inside, noref, occl, vis = visible(zb, u, v, zc)
                st["candidates"] += m; st["outside_or_range"] += int((~inside).sum()); st["no_reference"] += int(noref.sum())
                st["occluded"] += int(occl.sum()); st["not_present"] += int((inside & ~noref & ~occl & ~vis).sum()); st["visible"] += int(vis.sum())
                if not vis.any():
                    continue
                p25 = sample_probs(pr, u[vis], v[vis], dev) @ into; vi = torch.from_numpy(np.flatnonzero(vis)).to(dev)
                acc.index_add_(0, vi, p25); wsum.index_add_(0, vi, torch.ones(len(vi), device=dev)); half[min(q // (FUTURE_FRAMES // 2), 1)].index_add_(0, vi, p25)
                nfr[vis] += 1; vis_any |= vis
            both = (half[0].sum(1) > 0) & (half[1].sum(1) > 0); agree = (~both) | (half[0].argmax(1) == half[1].argmax(1))
            keep = (wsum >= SEM_W_MIN) & agree; kr = keep.cpu().numpy()
            st["contradictions"] += int((both & ~agree).sum()); st["target_voxels"] += int(kr.sum()); st["mapper_target_voxels"] += int(len(Z["fut_rows"]))
            st["views_per_voxel_hist"] += np.bincount(np.clip(nfr, 0, 20), minlength=21)
            fut_p = (acc[keep] / wsum[keep].unsqueeze(1)).cpu().numpy()
            fut_obs = np.zeros(n, bool); fut_obs[idx[vis_any]] = True
            ha = np.ones(n, bool); ha[idx[(both & ~agree).cpu().numpy()]] = False
            u8 = lambda x: np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
            rec = {k: Z[k] for k in INPUT_KEYS if k in Z}
            rec.update({"target_frames": np.asarray(fut, np.int32), "fut_observed": np.packbits(fut_obs), "fut_rows": idx[kr].astype(np.int32),
                        "fut_p": u8(fut_p), "fut_w": wsum[keep].cpu().numpy().astype(np.float16), "fut_conf": u8(fut_p.max(1)),
                        "fut_half_agree": np.packbits(ha), "fut_n_views": nfr[kr].astype(np.uint8),
                        "target_builder": np.asarray("gtsem-v2: GT poses + calibration, LiDAR presence+occlusion visibility, direct Trident sampling; no mapper")})
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 200 == 0:
                print(f"  [{a.drive[17:21]}] {nw}/{len(files)} anchors, {(time.time()-t0)/nw:.2f}s each; targets/occupied {st['target_voxels']/max(st['occupied'],1):.3f} (mapper {st['mapper_target_voxels']/max(st['occupied'],1):.3f})", flush=True)
    stj = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    drives7_json = os.path.join(drives7.ART, "targets", f"semantic_{a.drive[17:21]}.json")
    from tools.kitti360_data_scaling import common as C0
    C0.write_json(drives7_json, {"drive": a.drive, "n_written": nw, "n_skipped": nskip, "stats": stj, "rule": RULE, "calibration": geo.calib.to_dict(),
                                 "seconds": time.time() - t0, "firewall": audit.summary(), "env": C0.environment()})
    print(f"[{a.drive}] semantic targets written {nw}, skipped {nskip}, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
