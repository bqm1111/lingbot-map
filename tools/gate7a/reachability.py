#!/usr/bin/env python
"""Gate 7A — the reachability and envelope analysis, one dataset, one shard.

Target-dependent **oracle analysis**. It writes no prediction and consumes the Gate-6
predictions read-only, re-hashing every one of them and reproducing its B-R/B-D geometry
voxel-for-voxel before anything is measured.

    python tools/gate7a/reachability.py --dataset kitti360 --shard 0 --shards 4 \
        --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, grids as G6G, metrics as G6M, pipelines as G6P, \
    targets as G6T, vocab                                                 # noqa: E402
from gates.gate7a import config as C, distance as D, envelopes as E, frustum as FR, \
    pipeline as P7, transport as TR                                       # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7a")
NB = int(round(C.MISS_HIST_MAX_M / C.MISS_HIST_BIN_M)) + 1
#: distance-to-base bands for the frustum decomposition, derived from the pinned radii
DIST_TO_BASE_EDGES = tuple(C.RADII_M[1:]) + (float("inf"),)
#: Signed ray-depth residual histogram: +-40 m at 0.25 m, with saturating end bins.
RESID_BIN_M, RESID_MAX_M = 0.25, 40.0
RESID_NB = int(round(2 * RESID_MAX_M / RESID_BIN_M)) + 1


def band_masks(grid):
    """Range-band masks on the evaluation grid, using Gate 6's own definition."""
    dist, _h = G6M.band_masks(grid, dist_bands=C.RANGE_BANDS, height_bands=())
    return dist


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    ds, dev = a.dataset, a.device
    torch.use_deterministic_algorithms(True, warn_only=True)
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(dev)

    pin = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate7a",
                                      "precommit_pin.json")))
    if sha256_file(os.path.join(REPO_ROOT, pin["path"])) != pin["sha256"]:
        raise SystemExit("gate7a precommit no longer matches its pin; refusing to run")

    man = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate6",
                                      f"prediction_manifest_{ds}.json")))
    proot = man["prediction_root"]
    v = vocab.load(ds)
    C_n = len(v)
    G = G6G.EVAL_GRID[ds]
    nvox = int(np.prod(G.dims))
    vs = float(G.voxel_size)
    diag = float(np.sqrt(3.0) * vs)
    R = len(C.RADII_M)
    I = len(C.TRANSPORT_INTERVALS_M)
    NBands = len(C.RANGE_BANDS)
    d2max = D.d2_threshold(C.max_radius_m(), vs)

    lab = np.asarray(v.labels, np.int32)
    chan_of = np.full(int(max(lab.max(), v.empty_label)) + 2, -1, np.int32)
    chan_of[lab] = np.arange(C_n, dtype=np.int32)

    bands = band_masks(G)
    recs = {r.clip_id: r for r in F.read_manifest(ds, REPO_ROOT)}
    names = sorted(man["per_file_sha256"])
    names = F.shard(names, a.shard, a.shards)
    if a.limit:
        names = names[:a.limit]
    geo = {c.clip_id: c for c in G6P.iter_clips(ds, REPO_ROOT)}

    # ---------------------------------------------------------------- accumulators
    n = len(names)
    clip_ids, groups = [], []
    binary = np.zeros((n, 2, 2, R, 3), np.int64)
    perclass = np.zeros((n, 2, 2, R, C_n, 3), np.int32)
    volume = np.zeros((n, 2, 2, R, 4), np.int64)          # n_base n_added n_tp n_fp
    base_binary = np.zeros((n, 2, 3), np.int64)
    base_perclass = np.zeros((n, 2, C_n, 3), np.int32)
    trans = np.zeros((n, 2, I, C_n, 2), np.int32)         # gt-row, correct
    trans_conf = np.zeros((n, 2, I, 3), np.float64)       # count, sum max_prob, sum entropy
    miss_hist = np.zeros((2, 1 + NBands, NB), np.int64)
    miss_in_r = np.zeros((2, 1 + NBands, R), np.int64)
    band_gt = np.zeros((2, 1 + NBands), np.int64)
    band_tp = np.zeros((2, 1 + NBands), np.int64)
    frust = np.zeros((2, 1 + NBands, 1 + len(DIST_TO_BASE_EDGES), 2), np.int64)
    resid = np.zeros((2, 1 + NBands, len(FR.RESIDUAL_CLASSES)), np.int64)
    resid_hist = np.zeros((2, 1 + NBands, RESID_NB), np.int64)
    tie_hist = np.zeros((2, 3), np.int64)                 # count, sum n_tied, n_tied>1
    empty_base = {c: [] for c in C.CONDITIONS}
    rt_worst = 0.0
    gap_worst = 0.0
    chan_mismatch = np.zeros(4, np.int64)
    verified = 0

    t0 = time.time()
    for ci, name in enumerate(names):
        cid = name[:-4]
        path = os.path.join(proot, name)
        clip = geo.get(cid)
        rec = recs.get(cid)
        if clip is None or rec is None:
            raise SystemExit(f"{cid}: frozen geometry or manifest record missing")
        sem = G6P.load_semantics(ds, rec.keys, dev)
        if sem is None:
            raise SystemExit(f"{cid}: frozen Trident cache missing -- refusing to re-run "
                             "the teacher")
        fc = P7.rebuild(ds, clip, sem, dev)
        vr = P7.check_against_pinned(fc, path, man["per_file_sha256"][name])
        verified += 1
        chan_mismatch += (vr["n_raw_channel_mismatch"], vr["n_dil_channel_mismatch"],
                          vr["n_raw"], vr["n_dil"])
        gap_worst = max(gap_worst, vr["max_top2_gap_at_mismatch"])
        rt = FR.verify_roundtrip(fc.points_anchor, fc.frame, fc.v_pix, fc.u_pix,
                                 clip.pose, clip.K, clip.scale, n_check=4000)
        rt_worst = max(rt_worst, rt["max_pixel_error"])
        del sem

        target, keep = G6T.semantic_target(ds, rec.raw, REPO_ROOT)
        if target is None:
            raise SystemExit(f"{cid}: official target missing")
        k = keep.reshape(-1)
        t = target.reshape(-1)[k]
        t_ch = np.where(t != v.empty_label, chan_of[np.clip(t, 0, len(chan_of) - 1)],
                        -1).astype(np.int32)
        gt_occ_v = t_ch >= 0
        bmask = [np.ones(int(k.sum()), bool)] + [m[k] for _nm, m in bands]

        clip_ids.append(cid)
        groups.append(clip.group)

        # the PINNED Gate-6 channel is the base label; the rebuilt probability vectors
        # are used only to propagate semantics to voxels Gate 6 never labelled
        sets = {"B-R": (fc.raw_flat, fc.raw_probs, fc.pinned_raw_channel),
                "B-D": (fc.dil_flat, fc.dil_probs, fc.pinned_dil_channel)}
        for ic, cond in enumerate(C.CONDITIONS):
            flat, probs, pinned_ch = sets[cond]
            occ_full = np.zeros(nvox, bool)
            ch_full = np.full(nvox, -1, np.int32)
            if len(flat):
                occ_full[flat] = True
                ch_full[flat] = pinned_ch
            if not len(flat):
                empty_base[cond].append(cid)

            d, d2 = D.edt(occ_full.reshape(G.dims))
            d2f, dfv = d2.reshape(-1), d.reshape(-1)

            prop_ch = np.full(nvox, -1, np.int32)
            prop_mx = np.zeros(nvox, np.float32)
            prop_en = np.zeros(nvox, np.float32)
            sel = np.flatnonzero((d2f > 0) & (d2f <= d2max))
            if len(sel) and len(flat):
                st = TR.propagate_stats(G.dims, flat, probs, sel, d2f[sel], dev)
                if not st["assigned"].all():
                    raise AssertionError(f"{cid}/{cond}: a voxel inside the maximum radius "
                                         "found no source; the distance transform and the "
                                         "shell search disagree")
                prop_ch[sel] = st["channel"].astype(np.int32)
                prop_mx[sel] = st["max_prob"]
                prop_en[sel] = st["entropy"]
                tie_hist[ic] += (len(sel), int(st["n_tied"].sum()),
                                 int((st["n_tied"] > 1).sum()))

            base_occ_v = occ_full[k]
            base_ch_v = ch_full[k]
            prop_ch_v = prop_ch[k]
            d2_v = d2f[k]
            d_v = dfv[k] * vs

            base_blk = E.count(t_ch, np.where(base_occ_v, base_ch_v, -1).astype(np.int32),
                               C_n)
            base_binary[ci, ic] = (base_blk.btp, base_blk.bfp, base_blk.bfn)
            base_perclass[ci, ic] = np.stack([base_blk.tp, base_blk.fp, base_blk.fn], -1)

            blocks = E.evaluate(base_occ_v, base_ch_v, prop_ch_v, d2_v, t_ch, vs,
                                C.RADII_M, C_n)
            E.assert_invariants(blocks, base_blk, C.RADII_M)
            for ik, constr in enumerate(C.CONSTRUCTIONS):
                for ir, r in enumerate(C.RADII_M):
                    b = blocks[constr][float(r)]
                    binary[ci, ic, ik, ir] = (b.btp, b.bfp, b.bfn)
                    perclass[ci, ic, ik, ir] = np.stack([b.tp, b.fp, b.fn], -1)
                    volume[ci, ic, ik, ir] = (b.n_base, b.n_added, b.n_added_tp,
                                              b.n_added_fp)

            # ---- miss-distance distribution, overall and by range band
            miss_v = gt_occ_v & ~base_occ_v
            for ib, bm in enumerate(bmask):
                mm = miss_v & bm
                band_gt[ic, ib] += int((gt_occ_v & bm).sum())
                band_tp[ic, ib] += int((gt_occ_v & base_occ_v & bm).sum())
                dm = d_v[mm]
                miss_hist[ic, ib] += D.histogram(dm, C.MISS_HIST_BIN_M, C.MISS_HIST_MAX_M)
                for ir, r in enumerate(C.RADII_M):
                    miss_in_r[ic, ib, ir] += int(D.within_radius(d2_v[mm], float(r),
                                                                 vs).sum())

            # ---- semantic transport on the added true positives, by distance interval
            for ii, (lo, hi) in enumerate(C.TRANSPORT_INTERVALS_M):
                sel_i = (miss_v & D.within_radius(d2_v, hi, vs)
                         & ~D.within_radius(d2_v, lo, vs))
                if not sel_i.any():
                    continue
                tt = t_ch[sel_i]
                pp = prop_ch_v[sel_i]
                trans[ci, ic, ii, :, 0] += np.bincount(tt, minlength=C_n).astype(np.int32)
                ok = tt == pp
                trans[ci, ic, ii, :, 1] += np.bincount(tt[ok],
                                                       minlength=C_n).astype(np.int32)
                idx = np.flatnonzero(sel_i)
                kk = np.flatnonzero(k)[idx]
                trans_conf[ci, ic, ii] += (len(idx), float(prop_mx[kk].sum()),
                                           float(prop_en[kk].sum()))

            # ---- frustum. The brief asks for the primary condition; B-R is computed
            # too because with a tolerance of one voxel diagonal (<= 0.4 m) the
            # 'near the predicted surface' class is empty for B-D *by construction* --
            # B-D is exactly a 0.4 m dilation, so it has already absorbed that band.
            if miss_v.any():
                flat_miss = np.flatnonzero(k)[np.flatnonzero(miss_v)]
                centres = FR.voxel_centres(flat_miss, G)
                fa = FR.analyse(centres, clip.T_anchor_to_grid, clip.pose, clip.K,
                                clip.dep, clip.conf, clip.scale, G6P.CONF_THRESHOLD,
                                G6P.MIN_DEPTH_M, G6P.MAX_DEPTH_M, diag)
                dmiss = d_v[miss_v]
                dband = np.searchsorted(np.asarray(DIST_TO_BASE_EDGES), dmiss,
                                        side="left")
                for ib, bm in enumerate(bmask):
                    sub = bm[miss_v]
                    if not sub.any():
                        continue
                    inb = fa["in_any"][sub]
                    db = dband[sub]
                    np.add.at(frust[ic, ib], (db[inb], 0), 1)
                    np.add.at(frust[ic, ib], (db[~inb], 1), 1)
                    np.add.at(resid[ic, ib], fa["klass"][sub], 1)
                    rr = fa["residual"][sub]
                    rr = rr[np.isfinite(rr)]
                    if rr.size:
                        rb = np.clip(np.floor((rr + RESID_MAX_M) / RESID_BIN_M),
                                     0, RESID_NB - 1).astype(np.int64)
                        resid_hist[ic, ib] += np.bincount(rb,
                                                          minlength=RESID_NB).astype(np.int64)

        if (ci + 1) % 50 == 0:
            print(f"[{ds} s{a.shard}] {ci+1}/{n} {(time.time()-t0)/(ci+1):.2f}s/clip",
                  flush=True)

    out = os.path.join(ART, f"shard_{ds}_{a.shard}of{a.shards}.npz")
    np.savez_compressed(
        out, clip_id=np.asarray(clip_ids), group=np.asarray(groups),
        binary=binary[:len(clip_ids)], perclass=perclass[:len(clip_ids)],
        volume=volume[:len(clip_ids)], base_binary=base_binary[:len(clip_ids)],
        base_perclass=base_perclass[:len(clip_ids)], trans=trans[:len(clip_ids)],
        trans_conf=trans_conf[:len(clip_ids)], miss_hist=miss_hist,
        miss_in_r=miss_in_r, band_gt=band_gt, band_tp=band_tp, frust=frust, resid=resid,
        resid_hist=resid_hist, resid_bin_m=np.float64(RESID_BIN_M),
        resid_max_m=np.float64(RESID_MAX_M),
        dist_to_base_edges=np.asarray(DIST_TO_BASE_EDGES),
        tie=tie_hist, labels=np.asarray(v.labels, np.int32), names=np.asarray(v.names),
        radii=np.asarray(C.RADII_M), conditions=np.asarray(C.CONDITIONS),
        constructions=np.asarray(C.CONSTRUCTIONS))
    meta = {"dataset": ds, "shard": a.shard, "shards": a.shards, "n_clips": len(clip_ids),
            "n_verified_against_pinned": verified,
            "max_projection_roundtrip_pixel_error": rt_worst,
            "pinned_channel_agreement": {
                "n_raw_voxels": int(chan_mismatch[2]),
                "n_raw_channel_mismatch": int(chan_mismatch[0]),
                "n_dil_voxels": int(chan_mismatch[3]),
                "n_dil_channel_mismatch": int(chan_mismatch[1]),
                "max_top1_top2_gap_at_mismatch": gap_worst,
                "note": ("geometry is bit-exact on every clip; these are argmax flips on "
                         "voxels whose top two fused probabilities differ by less than the "
                         "CUDA atomic accumulation error of index_add_. The PINNED Gate-6 "
                         "channel is what this gate uses as the base label.")},
            "empty_base_clips": {c: empty_base[c] for c in C.CONDITIONS},
            "voxel_size_m": vs, "voxel_diagonal_m": diag,
            "seconds": time.time() - t0,
            "peak_gpu_gib": (torch.cuda.max_memory_allocated(dev) / 2**30
                             if dev.startswith("cuda") else 0.0),
            "gate6_rollup_sha256": man["rollup_sha256"],
            "precommit_sha256": pin["sha256"]}
    write_json(os.path.join(ART, f"shard_{ds}_{a.shard}of{a.shards}.json"), meta)
    print(f"[{ds} s{a.shard}] {len(clip_ids)} clips, {meta['seconds']:.0f}s, "
          f"roundtrip<= {rt_worst:.2e}px -> {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
