#!/usr/bin/env python
"""Replay the frozen future mapper one future frame at a time for every anchor of one
drive, and record per eligible voxel the semantic evidence each future frame contributes.

Eligible voxels are the previous experiment's population *without* the teacher-present
requirement: map-unobserved at ``t``, observed by the full future window, occupied in the
raw-LiDAR target. Everything about the mapper, the scale, the grid and the transform is the
gate's sample builder (``core/run/build_samples.py``) verbatim; the sole difference is that
each of the 20 future frames is integrated into its own throwaway mapper. The mapper's
semantic accumulators are additive, so any frame-disjoint grouping of the window can be
formed offline *exactly* from these records, and the sum over all 20 must reproduce the
cached full-window target -- checked per anchor and written to the file.

Per anchor: ``elig`` (flat voxel ids); sparse ``(row, frame offset, w, s, n_obs)`` for every
(voxel, frame) pair with teacher weight ``w > 0`` (``p = s / w``); a 20-bit observed mask per
voxel; the frames integrated.

    python tools/future_semantic_completion/target_consistency/build_evidence.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from core.datasets import grids as G6G, vocab, sources as S, union_vocab as V8            # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC              # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping.feed import CachedFeed                                                  # noqa: E402
from core.mapping.mapper import IncrementalMapper                                         # noqa: E402


def sub_volume(feed, t, scale, into, device, lo, hi, n_teacher):
    """A throwaway mapper over stream frames ``t+lo .. t+hi`` -- build_samples.half_volume."""
    m = IncrementalMapper(device, sem_into=into, n_teacher=n_teacher)
    m.scale_state.force(scale)
    used = []
    for j in range(t + lo, min(t + hi + 1, len(feed))):
        m.step(feed.frame(j)); used.append(j)
    return m, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--population", default="eligible", choices=("eligible", "all_future_observed"),
                    help="eligible: unobserved at t; all_future_observed: every LiDAR-occupied voxel the future window observes")
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); ds = "kitti360"
    MAP = G6G.PREDICTION_GRID[ds]; n = int(np.prod(MAP.dims))
    C = len(vocab.load(ds)); into = V8.into_matrix(ds)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    T_cam_to_grid = np.asarray(geo.rect_cam_to_velo, np.float64)
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(a.drive, 0)) + "/*.npz"))
    if a.limit:
        files = files[:a.limit]
    ev_root = TC.EVIDENCE_DIR if a.population == "eligible" else TC.EVIDENCE_DIR + "_allocc"
    out_dir = os.path.join(ev_root, a.drive); os.makedirs(out_dir, exist_ok=True)
    SRC.assert_no_target_access([out_dir, S.stream_path(src, a.drive)])
    feed = CachedFeed(seg, dev)
    t0, nw, nskip, devs = time.time(), 0, 0, []
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            i = int(os.path.basename(f)[:-4])
            op = os.path.join(out_dir, f"{i:05d}.npz")
            if os.path.exists(op):
                nskip += 1; continue
            with np.load(f) as z:
                scale = float(z["scale"]); t = int(z["t"]); assert t == i
                assert tuple(int(d) for d in z["dims"]) == tuple(int(d) for d in MAP.dims)
                target_frames = z["target_frames"].astype(np.int64)
                rows = z["rows"].astype(np.int64); obs_rows = z["observed"].astype(bool)
                gt_occ = np.unpackbits(z["gt_occ"])[:n].astype(bool)
                gt_valid = np.unpackbits(z["gt_valid"])[:n].astype(bool)
                fut_obs = np.unpackbits(z["fut_observed"])[:n].astype(bool)
                fut_rows = z["fut_rows"].astype(np.int64); fut_p = z["fut_p"].astype(np.float32) / 255.0
                fut_w = z["fut_w"].astype(np.float32)
            observed = np.zeros(n, bool); observed[rows] = obs_rows
            elig = np.flatnonzero((~observed if a.population == "eligible" else True) & fut_obs & gt_occ & gt_valid)
            P = feed.pose[i].copy(); P[:3, 3] *= scale
            Tgw = P @ np.linalg.inv(T_cam_to_grid)
            rec = {"t": np.int64(i), "scale": np.float64(scale), "dims": np.asarray(MAP.dims, np.int32),
                   "T_grid_to_world": Tgw, "elig": elig.astype(np.int32),
                   "n_eligible": np.int64(len(elig)), "cached_target_frames": target_frames}
            el = torch.from_numpy(elig).to(dev)
            tot_s = torch.zeros(len(elig), V8.U, device=dev); tot_w = torch.zeros(len(elig), device=dev)
            obs_mask = np.zeros(len(elig), np.uint32)
            pr, pf, pw, ps, pn, used = [], [], [], [], [], []
            for off in range(1, SRC.FUTURE_FRAMES + 1):
                j = i + off
                if j >= len(feed):
                    break
                m, u = sub_volume(feed, i, scale, into, dev, off, off, C)
                assert u == [j]
                q = m.query(MAP, Tgw); used.append(j)
                w = q["sem_w"][el]; s = q["sem"][el]
                hit = w > 0
                if hit.any():
                    r = hit.nonzero(as_tuple=True)[0]
                    pr.append(r.cpu().numpy().astype(np.int32)); pf.append(np.full(len(r), off, np.int8))
                    pw.append(w[r].cpu().numpy().astype(np.float16)); ps.append(s[r].cpu().numpy().astype(np.float16))
                    pn.append(q["n_obs"][el][r].clamp(max=255).cpu().numpy().astype(np.uint8))
                obs_mask |= (q["observed"][el].cpu().numpy().astype(np.uint32) << (off - 1))
                tot_s += s; tot_w += w
                del m, q
            assert sorted(used) == sorted(target_frames.tolist()), "per-frame replay != cached window"
            cat = lambda xs, dt, shape=(): (np.concatenate(xs) if xs else np.zeros((0,) + shape, dt))
            rec.update({"frames": np.asarray(used, np.int32), "pair_row": cat(pr, np.int32),
                        "pair_frame": cat(pf, np.int8), "pair_w": cat(pw, np.float16),
                        "pair_s": cat(ps, np.float16, (V8.U,)), "pair_nobs": cat(pn, np.uint8),
                        "obs_mask": obs_mask})
            # additivity: the 20 single-frame accumulators reproduce the cached fusion on target rows
            fv = np.zeros(n, bool); fv[fut_rows] = True
            fp = np.zeros((n, V8.U), np.float32); fp[fut_rows] = fut_p
            fw = np.zeros(n, np.float32); fw[fut_rows] = fut_w
            sel = fv[elig]
            if sel.any():
                p_here = (tot_s / tot_w.clamp_min(1e-9).unsqueeze(1)).cpu().numpy()[sel]
                dp = np.abs(p_here - fp[elig][sel]).max()
                dw = np.abs(tot_w.cpu().numpy()[sel] - fw[elig][sel]).max()
            else:
                dp = dw = 0.0
            rec["reproduces_cached_target_max_abs_dp"] = np.float64(dp)
            rec["reproduces_cached_target_max_abs_dw"] = np.float64(dw)
            rec["existing_target_present"] = sel
            devs.append((dp, dw))
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op)
            nw += 1
            if nw % 50 == 0:
                print(f"  [{a.drive[17:21]}] {nw} anchors, {(time.time()-t0)/nw:.2f}s each, "
                      f"max |dp| {max(d[0] for d in devs):.4f} max |dw| {max(d[1] for d in devs):.3f}", flush=True)
    ST.write_json(os.path.join(TC.ART, "evidence_build", f"{a.drive}{'' if a.population == 'eligible' else '_allocc'}.json"),
                  {"drive": a.drive, "population": a.population, "n_written": nw, "n_skipped_existing": nskip, "n_files": len(files),
                   "per_frame": True, "future_frames": SRC.FUTURE_FRAMES, "seconds": time.time() - t0,
                   "max_abs_dp": float(max([d[0] for d in devs], default=0.0)),
                   "max_abs_dw": float(max([d[1] for d in devs], default=0.0)),
                   "firewall": audit.summary(), "env": ST.environment()})
    print(f"[{a.drive}] wrote {nw}, skipped {nskip}, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
