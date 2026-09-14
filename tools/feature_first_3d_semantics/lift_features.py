#!/usr/bin/env python
"""Lift the cached dense feature into 3D with the frozen mapper, one future frame at a
time, at exactly the voxels and weights of the target-consistency evidence cache.

The mapper's semantic accumulator is linear (``sem += w * p`` at every band voxel of every
accepted ray), so feeding it the 1024-d feature instead of the 18-d probability yields the
evidence-weighted feature sum per voxel per frame under the identical rays, weights and
voxel correspondence. Alongside the feature we lift the *per-view coarse classification*
of the same feature (softmax(40 * f_pix . text)) -- the classify-then-fuse baseline that
differs from the feature path only in where the semantic decision is taken -- and check
per anchor that the weights reproduce the cached probability-path weights.

Per frame the token grid is resampled exactly as the official code resamples its logits:
bilinear to native resolution, then bilinear to the LingBot lattice (gate-6 cache path).

    python tools/feature_first_3d_semantics/lift_features.py --drive 2013_05_28_drive_0006_sync --population eligible --device cuda:0
"""
from __future__ import annotations
import argparse, dataclasses, glob, os, sys, time
from collections import OrderedDict
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from core.datasets import grids as G6G, vocab, sources as S, kitti360_splits as SRC       # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping.feed import CachedFeed                                                  # noqa: E402
from core.mapping.mapper import IncrementalMapper, pack                                   # noqa: E402

LATTICE_HW = (140, 518)


class LatticeCache:
    """frame index -> [H, W, 1024 + C] float32 on the device (feature, then coarse probs)."""

    def __init__(self, seg, src, text, dev, capacity=24):
        self.seg, self.src, self.text, self.dev, self.cap = seg, src, text, dev, capacity
        self.d = OrderedDict(); self.stats = []

    def get(self, j):
        if j in self.d:
            self.d.move_to_end(j); return self.d[j]
        key = self.seg.frames[j].key
        with np.load(FF.token_path(self.src, key)) as z:
            tok = torch.from_numpy(z["tokens"].astype(np.float32)).to(self.dev)
            h, w = (int(x) for x in z["grid_hw"]); H, W = (int(x) for x in z["native_hw"])
        f = tok.reshape(h, w, -1).permute(2, 0, 1)                                    # [1024, h, w]
        out = torch.empty(FF.FEAT_DIM, *LATTICE_HW, device=self.dev)
        for c0 in range(0, FF.FEAT_DIM, 128):                                        # two-step official resampling, chunked
            nat = F.interpolate(f[None, c0:c0 + 128], size=(H, W), mode="bilinear")
            out[c0:c0 + 128] = F.interpolate(nat, size=LATTICE_HW, mode="bilinear", align_corners=False)[0]
        p = (torch.einsum("chw,kc->khw", out, self.text) * FF.LOGIT_SCALE).softmax(0)  # [C, H, W] per-view coarse classification
        # fidelity vs the cached (SAM-refined) teacher at the lattice
        with np.load(S.trident_path(self.src, key)) as z:
            cached = torch.from_numpy(z["probs"].astype(np.float32)).to(self.dev)
        self.stats.append((float((p.argmax(0) == cached.argmax(0)).float().mean()), float((p - cached).abs().max())))
        lat = torch.cat([out, p], 0).permute(1, 2, 0).contiguous()
        self.d[j] = lat
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return lat


def subset_query(m, grid, Tgw, elig, dev):
    """The mapper's ``query`` restricted to the given flat voxel ids: (sem [n, C], sem_w [n])."""
    X, Y, Z = (int(d) for d in grid.dims)
    idx = torch.from_numpy(np.stack(np.unravel_index(elig, (X, Y, Z)), 1)).to(dev)
    c = (idx.double() + 0.5) * float(grid.voxel_size) + torch.as_tensor(np.asarray(grid.origin, np.float64), device=dev)
    T = torch.as_tensor(np.asarray(Tgw, np.float64), device=dev)
    wpt = c @ T[:3, :3].T + T[:3, 3]
    keys = pack(torch.floor(wpt / m.vs).to(torch.int64))
    row = m.table.lookup(keys); hit = row >= 0
    sem = torch.zeros(len(elig), m.table.n_sem, device=dev); semw = torch.zeros(len(elig), device=dev)
    if hit.any():
        sem[hit] = m.table.sem[row[hit]]; semw[hit] = m.table.sem_w[row[hit]]
    return sem, semw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--population", default="eligible", choices=list(FF.EVIDENCE_DIRS))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    FF.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); MAP = G6G.PREDICTION_GRID["kitti360"]; C = len(vocab.load("kitti360"))
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    with np.load(FF.TEXT_PATH) as z:
        text = torch.from_numpy(z["text"]).to(dev); assert text.shape == (C, FF.FEAT_DIM)
    feed = CachedFeed(seg, dev, with_semantics=False)
    cache = LatticeCache(seg, src, text, dev)
    files = sorted(glob.glob(os.path.join(FF.EVIDENCE_DIRS[a.population], a.drive, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    out_dir = os.path.join(FF.LIFT_DIR, a.population, a.drive); os.makedirs(out_dir, exist_ok=True)
    SRC.assert_no_target_access([out_dir])
    t0, nw, nskip, wdev = time.time(), 0, 0, []
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                ev = {k: z[k] for k in z.files}
            i = int(ev["t"]); op = os.path.join(out_dir, f"{i:05d}.npz")
            if os.path.exists(op):
                nskip += 1; continue
            elig = ev["elig"].astype(np.int64); Tgw = ev["T_grid_to_world"]; scale = float(ev["scale"])
            ref = {}                                                    # (row, frame) -> cached probability-path weight
            for r, fr_, w_ in zip(ev["pair_row"].tolist(), ev["pair_frame"].tolist(), ev["pair_w"].astype(np.float32).tolist()):
                ref[(r, fr_)] = w_
            pr, pf, pw, pfeat, ppc = [], [], [], [], []
            for off in range(1, SRC.FUTURE_FRAMES + 1):
                j = i + off
                if j >= len(feed):
                    break
                fr = dataclasses.replace(feed.frame(j), sem_probs=cache.get(j))
                m = IncrementalMapper(dev, sem_into=None, n_teacher=FF.FEAT_DIM + C)
                m.scale_state.force(scale); m.step(fr)
                sem, semw = subset_query(m, MAP, Tgw, elig, dev)
                hit = semw > 0
                if hit.any():
                    r = hit.nonzero(as_tuple=True)[0]
                    pr.append(r.cpu().numpy().astype(np.int32)); pf.append(np.full(len(r), off, np.int8))
                    pw.append(semw[r].cpu().numpy().astype(np.float32))
                    pfeat.append(sem[r, :FF.FEAT_DIM].cpu().numpy().astype(np.float16))
                    ppc.append(sem[r, FF.FEAT_DIM:].cpu().numpy().astype(np.float16))
                del m
            cat = lambda xs, dt, shape=(): (np.concatenate(xs) if xs else np.zeros((0,) + shape, dt))
            rows, frs, ws = cat(pr, np.int32), cat(pf, np.int8), cat(pw, np.float32)
            # correspondence check against the probability-path evidence cache
            got = {(int(r), int(fr_)): float(w_) for r, fr_, w_ in zip(rows, frs, ws)}
            same_keys = set(got) == set(ref)
            if same_keys and got:
                wd = max(abs(got[k] - ref[k]) / max(abs(ref[k]), 1e-6) for k in got)
            else:
                wd = float("inf") if (got or ref) else 0.0
            wdev.append(wd)
            rec = {"t": np.int64(i), "elig": elig.astype(np.int32), "n_eligible": np.int64(len(elig)),
                   "pair_row": rows, "pair_frame": frs, "pair_w": ws, "pair_feat": cat(pfeat, np.float16, (FF.FEAT_DIM,)),
                   "pair_pcoarse": cat(ppc, np.float16, (C,)), "matches_probability_path_pairs": np.bool_(same_keys),
                   "max_rel_weight_dev": np.float64(wd), "frames": ev["frames"]}
            tmp = op + ".tmp.npz"; np.savez(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 25 == 0:
                st = np.asarray(cache.stats)
                print(f"  [{a.drive[17:21]} {a.population}] {nw} anchors, {(time.time()-t0)/nw:.2f}s each; pairs matched on all: "
                      f"{all(np.isfinite(wdev))}, max rel |dw| {max(wdev):.2e}; per-frame coarse-vs-cached argmax agree "
                      f"{st[:, 0].mean():.3f}", flush=True)
    st = np.asarray(cache.stats).reshape(-1, 2)
    FF.write_json(os.path.join(FF.ART, "lifting", f"{a.drive}_{a.population}.json"),
                  {"drive": a.drive, "population": a.population, "n_written": nw, "n_skipped": nskip, "lattice_hw": LATTICE_HW,
                   "channels": {"feature": FF.FEAT_DIM, "coarse_probs": C}, "resampling": "token grid -> native (bilinear) -> LingBot lattice (bilinear, align_corners=False), as the official logits are resampled",
                   "all_pairs_match_probability_path": bool(all(np.isfinite(wdev))), "max_rel_weight_dev": float(max(wdev) if wdev else 0.0),
                   "per_frame_coarse_vs_cached_refined": {"argmax_agree_mean": float(st[:, 0].mean()) if len(st) else None,
                                                          "argmax_agree_min": float(st[:, 0].min()) if len(st) else None,
                                                          "n_frames": int(len(st))},
                   "seconds": time.time() - t0, "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                   "firewall": audit.summary(), "env": FF.environment()})
    print(f"[{a.drive} {a.population}] wrote {nw}, skipped {nskip}, {(time.time()-t0)/60:.1f} min, all pairs matched {all(np.isfinite(wdev))}, max rel |dw| {max(wdev):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
