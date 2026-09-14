#!/usr/bin/env python
"""Regenerate the training samples of one drive with GT-registered semantic targets.

For every existing sample (anchor t): the input arrays are copied unchanged; the occupancy
target (raw LiDAR, GT poses) is kept; the semantic target is rebuilt without the mapper:

    occupied voxel centre (anchor velodyne frame)
      --velo_to_world(anchor) . inv(rect_cam_to_world(j))--> camera j   (GT poses + calibration)
      --K--> pixel (u, v); depth 1.5-60 m, inside the image
      visible iff frame j's own sweep has a return in the 7x7 window whose depth range
      brackets z within tol = 0.3 m + 1 % z (surface present) AND the 3x3 minimum sweep
      depth is not closer than z - tol (no occluder in front)
      --bilinear sample of the cached Trident-H lattice probabilities of frame j-->
      accumulate p (18 -> 25 union) with weight 1 per visible view

over the same 20 future stream frames the cached target used. ``fut_observed`` now means
"LiDAR-occupied and visible in at least one future camera". The two-halves contradiction
rule of the adopted builder is kept. Moving objects: a voxel occupied at the anchor but
vacated later fails the z-buffer test (background farther than the projected depth) and
receives no evidence from that frame; objects still present pass. No further motion model.

    python tools/kitti360_data_scaling/build_targets_gtsem.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from collections import OrderedDict
import numpy as np, torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.kitti360_data_scaling import common as C                                       # noqa: E402
os.environ.setdefault("G8C1_DATA_ROOT", C.SOURCE_ROOT)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, union_vocab as V8, vocab  # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = C.NATIVE_HW; LH, LW = C.LATTICE_HW
GRID_DIMS = (256, 256, 32); ORIGIN = np.array([0.0, -25.6, -2.0]); VS = 0.2
INPUT_KEYS = ("t", "input_frames", "scale", "dims", "rows", "logodds", "w_occ", "w_free", "n_obs", "n_lb", "age", "observed",
              "sem_rows", "sem_p", "sem_w", "gt_occ", "gt_valid", "n_obs_target", "n_frames_occ_target", "occ_rows_target", "native_frame")


def project(K, p):
    z = p[:, 2]; zz = np.where(z > 0, z, 1.0)
    return K[0, 0] * p[:, 0] / zz + K[0, 2], K[1, 1] * p[:, 1] / zz + K[1, 2], z


class FrameCache:
    def __init__(self, geo, src, key_of, K, dev, cap=48):
        self.geo, self.src, self.key_of, self.K, self.dev, self.cap = geo, src, key_of, K, dev, cap; self.d = OrderedDict()
        self.v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64)

    def get(self, native):
        if native in self.d:
            self.d.move_to_end(native); return self.d[native]
        pts = self.geo.read_velodyne(native); pc = pts @ self.v2c[:3, :3].T + self.v2c[:3, 3]
        u, v, z = project(self.K, pc); ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img = np.full((H, W), np.inf); np.minimum.at(img, (v[ok].astype(int), u[ok].astype(int)), z[ok])
        imx = np.full((H, W), -np.inf); np.maximum.at(imx, (v[ok].astype(int), u[ok].astype(int)), z[ok])
        zb = {"min3": ndimage.minimum_filter(img, size=3, mode="nearest"), "min7": ndimage.minimum_filter(img, size=C.ZBUF_WINDOW, mode="nearest"),
              "max7": ndimage.maximum_filter(imx, size=C.ZBUF_WINDOW, mode="nearest")}
        with np.load(S.trident_path(self.src, self.key_of[native])) as zf:
            pr = torch.from_numpy(zf["probs"].astype(np.float32)).to(self.dev)[None]           # [1, 18, 140, 518]
        self.d[native] = (zb, pr)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return self.d[native]


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
    into = torch.from_numpy(V8.into_matrix("kitti360")).to(dev)                                    # [18, 25]
    cache = FrameCache(geo, src, key_of, K, dev)
    files = sorted(glob.glob(os.path.join(C.SOURCE_ROOT, "samples", a.drive, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    out_dir = os.path.join(C.DATA_ROOT, "samples", a.drive); os.makedirs(out_dir, exist_ok=True)
    n = int(np.prod(GRID_DIMS)); t0 = time.time(); nw = nskip = 0
    stats = {"n_anchors": 0, "occupied": 0, "candidates": 0, "rej_range": 0, "rej_outside": 0, "rej_no_ref": 0, "rej_occluded": 0, "rej_in_front": 0,
             "visible": 0, "fut_rows": 0, "contradictions": 0, "old_fut_rows": 0, "frames_per_voxel_hist": np.zeros(21, np.int64)}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Z = {k: z[k] for k in z.files}
            t = int(Z["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                nskip += 1; continue
            nat = natives[t]; fut_stream = [j for j in range(t + 1, min(t + 1 + C.FUTURE_FRAMES, len(natives)))]
            assert sorted(fut_stream) == sorted(Z["target_frames"].tolist()), "future window differs from the cached one"
            occ = (np.unpackbits(Z["gt_occ"])[:n] & np.unpackbits(Z["gt_valid"])[:n]).astype(bool)
            idx = np.flatnonzero(occ); ctr = (np.stack(np.unravel_index(idx, GRID_DIMS), 1) + 0.5) * VS + ORIGIN
            Tvw = geo.velo_to_world(nat)
            acc = torch.zeros(len(idx), V8.U, device=dev); wsum = torch.zeros(len(idx), device=dev)
            half = [torch.zeros(len(idx), V8.U, device=dev), torch.zeros(len(idx), V8.U, device=dev)]
            nfr = np.zeros(len(idx), np.int32); vis_any = np.zeros(len(idx), bool)
            stats["n_anchors"] += 1; stats["occupied"] += len(idx)
            for q, j in enumerate(fut_stream):
                nj = natives[j]
                if nj not in geo.cam0_to_world:
                    continue
                T = TF.inv(geo.rect_cam_to_world(nj)) @ Tvw; pc = ctr @ T[:3, :3].T + T[:3, 3]
                u, v, zc = project(K, pc); stats["candidates"] += len(idx)
                r1 = ~((zc > C.DEPTH_MIN) & (zc < C.DEPTH_MAX)); stats["rej_range"] += int(r1.sum())
                inside = ~r1 & (u >= C.BORDER_PX) & (u < W - C.BORDER_PX) & (v >= C.BORDER_PX) & (v < H - C.BORDER_PX); stats["rej_outside"] += int((~r1 & ~inside).sum())
                zb, pr = cache.get(nj)
                vi_, ui_ = v[inside].astype(int), u[inside].astype(int)
                mn7 = np.full(len(idx), np.inf); mx7 = np.full(len(idx), -np.inf); mn3 = np.full(len(idx), np.inf)
                mn7[inside] = zb["min7"][vi_, ui_]; mx7[inside] = zb["max7"][vi_, ui_]; mn3[inside] = zb["min3"][vi_, ui_]
                noref = inside & ~np.isfinite(mn7); stats["rej_no_ref"] += int(noref.sum())
                cand = inside & np.isfinite(mn7); tol = C.VIS_TOL[0] + C.VIS_TOL[1] * zc
                occl = cand & np.isfinite(mn3) & (zc - mn3 > tol); present = cand & (zc >= mn7 - tol) & (zc <= mx7 + tol)
                stats["rej_occluded"] += int(occl.sum()); stats["rej_in_front"] += int((cand & ~occl & (zc < mn7 - tol)).sum())
                vis = present & ~occl; stats["visible"] += int(vis.sum())
                if not vis.any():
                    continue
                p18 = sample_probs(pr, u[vis], v[vis], dev); p25 = p18 @ into
                vi = torch.from_numpy(np.flatnonzero(vis)).to(dev)
                acc.index_add_(0, vi, p25); wsum.index_add_(0, vi, torch.ones(len(vi), device=dev)); half[q // (C.FUTURE_FRAMES // 2)].index_add_(0, vi, p25)
                nfr[vis] += 1; vis_any |= vis
            stats["frames_per_voxel_hist"] += np.bincount(np.clip(nfr, 0, 20), minlength=21)
            both = (half[0].sum(1) > 0) & (half[1].sum(1) > 0)
            agree = (~both) | (half[0].argmax(1) == half[1].argmax(1))
            keep = (wsum >= C.SEM_W_MIN) & (agree if C.REQUIRE_HALF_AGREEMENT else True)
            stats["contradictions"] += int((both & ~agree).sum())
            kr = keep.cpu().numpy(); fut_rows = idx[kr]; stats["fut_rows"] += len(fut_rows); stats["old_fut_rows"] += len(Z["fut_rows"])
            fut_p = (acc[keep] / wsum[keep].unsqueeze(1)).cpu().numpy()
            fut_obs = np.zeros(n, bool); fut_obs[idx[vis_any]] = True
            ha = np.ones(n, bool); ha[idx[(both & ~agree).cpu().numpy()]] = False
            u8 = lambda x: np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
            rec = {k: Z[k] for k in INPUT_KEYS if k in Z}
            rec.update({"target_frames": np.asarray(fut_stream, np.int32), "fut_observed": np.packbits(fut_obs),
                        "fut_rows": fut_rows.astype(np.int32), "fut_p": u8(fut_p), "fut_w": wsum[keep].cpu().numpy().astype(np.float16),
                        "fut_conf": u8(fut_p.max(1)), "fut_half_agree": np.packbits(ha), "fut_n_frames": nfr[kr].astype(np.uint8),
                        "target_builder": np.asarray("gtsem: GT poses + calibration, LiDAR z-buffer visibility, direct Trident sampling; no mapper")})
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 50 == 0:
                print(f"  [{a.drive[17:21]}] {nw} anchors, {(time.time()-t0)/nw:.2f}s each; fut_rows/occupied {stats['fut_rows']/max(stats['occupied'],1):.3f} (old {stats['old_fut_rows']/max(stats['occupied'],1):.3f})", flush=True)
    st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}
    C.write_json(os.path.join(C.ART, "targets", f"{a.drive}.json"),
                 {"drive": a.drive, "n_written": nw, "n_skipped": nskip, "stats": st, "rule": {"visibility": "present: sweep depth range in the 7x7 window brackets z within tol; not occluded: z - min(3x3 window) < tol; tol = a + b z", "tol_a_b": C.VIS_TOL, "window_px": C.ZBUF_WINDOW, "depth_m": [C.DEPTH_MIN, C.DEPTH_MAX],
                  "future_frames": C.FUTURE_FRAMES, "sem_w_min": C.SEM_W_MIN, "half_agreement": C.REQUIRE_HALF_AGREEMENT, "weight": "1 per visible view",
                  "probabilities": "cached Trident lattice (140x518), bilinear at the projected native pixel, 18 -> 25 union via into_matrix"},
                  "calibration": geo.calib.to_dict(), "seconds": time.time() - t0, "firewall": audit.summary(), "env": C.environment()})
    print(f"[{a.drive}] wrote {nw}, skipped {nskip}, {(time.time()-t0)/60:.1f} min; {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
