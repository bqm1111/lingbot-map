#!/usr/bin/env python
"""Voxel-level vs region-level independent cross-view semantic repeatability.

Regions (geometry only, deterministic): the raw-LiDAR occupied grid of the anchor is cut
into cubic blocks of side ``s`` voxels; inside each block the 26-connected components of
occupied voxels are the regions. Four scales, ``s`` = 3 / 5 / 9 / 15 (0.6 / 1.0 / 1.8 / 3.0 m).
No semantic quantity enters the region definition.

Evidence: the per-frame future teacher observations at the eligible voxels (unobserved at
``t``, future-observed, LiDAR-occupied) from the target-consistency cache, split into the
interleaved frame-disjoint groups A / B / C. A region's group vector is the evidence-weighted
mean of the observations its voxels received from that group; a voxel's group vector is its
own. Nothing from group C touches the A or B vectors.

Repeatability is scored for every pair of groups on the units that carry both; three-way
consistency on the units that carry all three. Controls: R0 pooled class prior, R1
within-anchor shuffle stratified by observation count, R2 local shuffle within (z-layer,
xy-cell) strata. Matched comparison: on the voxels that carry their own A and B evidence,
the voxel's own agreement vs the agreement of the region it belongs to.

    python tools/region_semantic_consistency/analyze.py --drive 2013_05_28_drive_0006_sync
    python tools/region_semantic_consistency/analyze.py --merge
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.region_semantic_consistency import common as RC                                # noqa: E402
from tools.future_semantic_completion.target_consistency.analyze import GROUPINGS, cosine, js  # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8                       # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402

U = V8.U
EVIDENCE_DIR = RC.EVIDENCE_DIR; OUT_TAG = ""
STRUCT = np.ones((3, 3, 3), bool)
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))


def probs(w, s):
    return s / np.maximum(w, 1e-9)[:, None]


# ------------------------------------------------------------------------------ regions
def label_regions(occ, xyz_elig, s):
    """Region id (int64, -1 = none) for every eligible voxel, plus per-region stats.

    Blocks are ``xyz // s``; only blocks holding an eligible voxel are labelled. Region ids
    are dense 0..R-1 in block-then-label order, so the assignment is deterministic.
    """
    X, Y, Z = occ.shape
    b = xyz_elig // s
    bkey = (b[:, 0] * 10_000 + b[:, 1]) * 10_000 + b[:, 2]
    ub, inv = np.unique(bkey, return_inverse=True)
    rid = np.full(len(xyz_elig), -1, np.int64)
    stats = []                      # (n_voxels, extent_max, cz, cx, cy)
    next_id = 0
    for k, key in enumerate(ub):
        bz = key % 10_000; by = (key // 10_000) % 10_000; bx = key // 100_000_000
        x0, y0, z0 = bx * s, by * s, bz * s
        sub = occ[x0:x0 + s, y0:y0 + s, z0:z0 + s]
        lab, n = ndimage.label(sub, structure=STRUCT)
        if n == 0:
            continue
        sel = np.flatnonzero(inv == k)
        loc = xyz_elig[sel] - np.array([x0, y0, z0])
        l = lab[loc[:, 0], loc[:, 1], loc[:, 2]]
        assert (l > 0).all(), "eligible voxel not occupied"
        sizes = np.bincount(lab.reshape(-1), minlength=n + 1)
        objs = ndimage.find_objects(lab)
        rid[sel] = next_id + l - 1
        for j in range(n):
            sl = objs[j]
            ext = max(sl[a].stop - sl[a].start for a in range(3))
            stats.append((int(sizes[j + 1]), int(ext), z0 + 0.5 * (sl[2].start + sl[2].stop),
                          x0 + 0.5 * (sl[0].start + sl[0].stop), y0 + 0.5 * (sl[1].start + sl[1].stop)))
        next_id += n
    return rid, np.asarray(stats, np.float64).reshape(-1, 5)


# ------------------------------------------------------------------------ accumulators
class Acc:
    def __init__(self):
        self.pair = {p: np.zeros(4) for p in PAIRS}          # n, top1, cos, js   (unit-weighted)
        self.pair_v = {p: np.zeros(4) for p in PAIRS}        # voxel-weighted (eligible voxels in unit)
        self.three = np.zeros(2); self.three_v = np.zeros(2)
        self.shuf = {"R1": np.zeros(3), "R2": np.zeros(3), "R1_v": np.zeros(3), "R2_v": np.zeros(3)}  # n, top1, cos
        self.cls_q = np.zeros(U); self.cls_r = np.zeros(U); self.cls_ok = np.zeros(U)      # unit-weighted, A vs B
        self.cls_q_v = np.zeros(U); self.cls_ok_v = np.zeros(U)
        self.size = []                                        # per validated unit: n_vox, extent, n_obs_AB, n_frames_AB, n_elig, ok_AB, cos_AB
        self.n_units = 0; self.n_units_any = 0
        self.cov = np.zeros(4)                                # eligible: total, any-evidence, AB, ABC (through the unit)
        self.matched = np.zeros(5)                            # n, own top1, own cos, region top1, region cos
        self.anchors = []                                     # per anchor: n_AB units, top1 sum (for block CI)


def agree_stats(pa, pb):
    ok = (pa.argmax(1) == pb.argmax(1)).astype(np.float64)
    return ok, cosine(pa, pb), js(pa, pb)


def shuffle_within(strata, rng):
    """Permutation index that permutes only inside strata with >= 2 members; returns
    (perm, valid) where valid marks units that were actually permuted."""
    perm = np.arange(len(strata)); valid = np.zeros(len(strata), bool)
    for u_ in np.unique(strata):
        ix = np.flatnonzero(strata == u_)
        if len(ix) >= 2:
            perm[ix] = ix[rng.permutation(len(ix))]; valid[ix] = True
    return perm, valid


def score_units(acc, G, n_elig_per_unit, stats, rng, cx=None, cy=None, cz=None):
    """G: group -> (w [R], s [R,U], n_obs [R], n_frames [R]). Units = regions or voxels."""
    w = {g: G[g][0] for g in "ABC"}; p = {g: probs(G[g][0], G[g][1]) for g in "ABC"}
    has = {g: w[g] > 0 for g in "ABC"}
    ne = n_elig_per_unit.astype(np.float64)
    acc.n_units += len(ne); acc.n_units_any += int((has["A"] | has["B"] | has["C"]).sum())
    acc.cov += (ne.sum(), ne[has["A"] | has["B"] | has["C"]].sum(), ne[has["A"] & has["B"]].sum(),
                ne[has["A"] & has["B"] & has["C"]].sum())
    for a, b in PAIRS:
        m = has[a] & has[b]
        if not m.any():
            continue
        ok, cs, jd = agree_stats(p[a][m], p[b][m])
        acc.pair[(a, b)] += (m.sum(), ok.sum(), cs.sum(), jd.sum())
        acc.pair_v[(a, b)] += (ne[m].sum(), (ok * ne[m]).sum(), (cs * ne[m]).sum(), (jd * ne[m]).sum())
    m3 = has["A"] & has["B"] & has["C"]
    if m3.any():
        ok3 = ((p["A"][m3].argmax(1) == p["B"][m3].argmax(1)) & (p["B"][m3].argmax(1) == p["C"][m3].argmax(1))).astype(float)
        acc.three += (m3.sum(), ok3.sum()); acc.three_v += (ne[m3].sum(), (ok3 * ne[m3]).sum())
    m = has["A"] & has["B"]
    if not m.any():
        acc.anchors.append((0, 0.0)); return
    pa, pb, nem = p["A"][m], p["B"][m], ne[m]
    qa, qb = pa.argmax(1), pb.argmax(1); ok = (qa == qb).astype(float)
    acc.anchors.append((int(m.sum()), float(ok.sum())))
    acc.cls_q += np.bincount(qa, minlength=U); acc.cls_r += np.bincount(qb, minlength=U); acc.cls_ok += np.bincount(qa[ok > 0], minlength=U)
    acc.cls_q_v += np.bincount(qa, weights=nem, minlength=U); acc.cls_ok_v += np.bincount(qa[ok > 0], weights=nem[ok > 0], minlength=U)
    nobs = (G["A"][2] + G["B"][2])[m]; nfr = (G["A"][3] + G["B"][3])[m]
    cs_ab = cosine(pa, pb)
    if stats is not None:
        acc.size += [(stats[i, 0], stats[i, 1], nobs[k], nfr[k], nem[k], ok[k], cs_ab[k]) for k, i in enumerate(np.flatnonzero(m))]
    else:
        acc.size += [(1, 1, nobs[k], nfr[k], 1, ok[k], cs_ab[k]) for k in range(len(ok))]
    # R1: within anchor, stratified by A+B observation count
    strata = np.clip(np.searchsorted(np.asarray(RC.SIZE_EDGES), nobs, side="right") - 1, 0, len(RC.SIZE_EDGES) - 1)
    perm, valid = shuffle_within(strata, rng)
    if valid.any():
        oks, css, _ = agree_stats(pa[valid], pb[perm][valid])
        acc.shuf["R1"] += (valid.sum(), oks.sum(), css.sum())
        acc.shuf["R1_v"] += (nem[valid].sum(), (oks * nem[valid]).sum(), (css * nem[valid]).sum())
    # R2: local, within (z-layer bin, xy cell)
    if cx is not None:
        zc = (cz[m] // RC.LOCAL_Z_BIN).astype(np.int64); xc = (cx[m] // RC.LOCAL_XY_CELL).astype(np.int64); yc = (cy[m] // RC.LOCAL_XY_CELL).astype(np.int64)
        strata2 = (zc * 1000 + xc) * 1000 + yc
        perm, valid = shuffle_within(strata2, rng)
        if valid.any():
            oks, css, _ = agree_stats(pa[valid], pb[perm][valid])
            acc.shuf["R2"] += (valid.sum(), oks.sum(), css.sum())
            acc.shuf["R2_v"] += (nem[valid].sum(), (oks * nem[valid]).sum(), (css * nem[valid]).sum())


def group_units(unit_of_pair, n_units, off, w, s, grouping):
    G = {}
    for g, offs in grouping.items():
        m = np.isin(off, offs)
        wg = np.zeros(n_units); sg = np.zeros((n_units, U)); ng = np.zeros(n_units)
        np.add.at(wg, unit_of_pair[m], w[m]); np.add.at(sg, unit_of_pair[m], s[m]); np.add.at(ng, unit_of_pair[m], 1)
        uf = np.unique(unit_of_pair[m] * 100 + off[m])            # (unit, frame) pairs
        nf = np.bincount(uf // 100, minlength=n_units).astype(np.float64)
        G[g] = (wg, sg, ng, nf)
    return G


def summarize(acc, n_elig_total):
    def pr(v):
        n = max(v[0], 1); return {"n": int(v[0]), "top1": float(v[1] / n), "cosine": float(v[2] / n), "js": float(v[3] / n) if len(v) > 3 else None}
    q = acc.cls_q / max(acc.cls_q.sum(), 1); r = acc.cls_r / max(acc.cls_r.sum(), 1)
    qv = acc.cls_q_v / max(acc.cls_q_v.sum(), 1)
    ent = lambda d: float(-(d[d > 0] * np.log(d[d > 0])).sum())
    arr = np.asarray(acc.anchors, np.float64); ci = None
    if len(arr) and arr[:, 0].sum() > 0:
        blocks = [arr[i:i + 20] for i in range(0, len(arr), 20)]; rng = np.random.default_rng(0); vals = []
        for _ in range(1000):
            pick = rng.integers(0, len(blocks), len(blocks)); sm = np.concatenate([blocks[k] for k in pick])
            vals.append(sm[:, 1].sum() / max(sm[:, 0].sum(), 1))
        ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    sz = np.asarray(acc.size, np.float64).reshape(-1, 7)
    out = {"n_units": int(acc.n_units), "n_units_any_evidence": int(acc.n_units_any),
           "pairs": {f"{a}{b}": pr(acc.pair[(a, b)]) for a, b in PAIRS},
           "pairs_voxel_weighted": {f"{a}{b}": pr(acc.pair_v[(a, b)]) for a, b in PAIRS},
           "AB_top1_ci95_blocks": ci,
           "three_way": {"n": int(acc.three[0]), "consistent": float(acc.three[1] / max(acc.three[0], 1))},
           "three_way_voxel_weighted": {"n": int(acc.three_v[0]), "consistent": float(acc.three_v[1] / max(acc.three_v[0], 1))},
           "controls": {k: {"n": int(v[0]), "top1": float(v[1] / max(v[0], 1)), "cosine": float(v[2] / max(v[0], 1))} for k, v in acc.shuf.items()},
           "R0_class_prior_top1": float((q * r).sum()), "majority_rate_B": float(r.max()) if r.sum() else None,
           "coverage_of_eligible": {"any_evidence": float(acc.cov[1] / max(n_elig_total, 1)), "AB": float(acc.cov[2] / max(n_elig_total, 1)),
                                    "ABC": float(acc.cov[3] / max(n_elig_total, 1))},
           "class_distribution_A_units": {V8.UNION[u]: float(q[u]) for u in range(U) if q[u] > 0},
           "class_distribution_A_voxel_weighted": {V8.UNION[u]: float(qv[u]) for u in range(U) if qv[u] > 0},
           "entropy_nats": ent(q), "entropy_nats_voxel_weighted": ent(qv),
           "dominant_class": V8.UNION[int(q.argmax())] if q.sum() else None, "dominant_share": float(q.max()) if q.sum() else None,
           "dominant_share_voxel_weighted": float(qv.max()) if qv.sum() else None,
           "n_classes_support_ge_200": int((acc.cls_q >= 200).sum()),
           "per_class": {V8.UNION[u]: {"n": int(acc.cls_q[u]), "top1": float(acc.cls_ok[u] / acc.cls_q[u]),
                                       "n_voxels": float(acc.cls_q_v[u]), "top1_voxel_weighted": float(acc.cls_ok_v[u] / max(acc.cls_q_v[u], 1e-9))}
                         for u in range(U) if acc.cls_q[u] >= 200},
           "matched_voxels_with_own_AB": {"n": int(acc.matched[0]), "own_top1": float(acc.matched[1] / max(acc.matched[0], 1)),
                                          "own_cosine": float(acc.matched[2] / max(acc.matched[0], 1)),
                                          "region_top1": float(acc.matched[3] / max(acc.matched[0], 1)),
                                          "region_cosine": float(acc.matched[4] / max(acc.matched[0], 1))}}
    if len(sz):
        out["size"] = {"n_validated_units": int(len(sz)), "median_occupied_voxels": float(np.median(sz[:, 0])),
                       "p90_occupied_voxels": float(np.percentile(sz[:, 0], 90)), "median_extent_vox": float(np.median(sz[:, 1])),
                       "median_extent_m": float(0.2 * np.median(sz[:, 1])), "median_observations_AB": float(np.median(sz[:, 2])),
                       "mean_observations_AB": float(sz[:, 2].mean()), "median_frames_AB": float(np.median(sz[:, 3])),
                       "mean_frames_AB": float(sz[:, 3].mean()), "median_eligible_voxels": float(np.median(sz[:, 4])),
                       "hist_frames_AB": np.bincount(np.clip(sz[:, 3].astype(int), 0, 12), minlength=13).tolist()}
        def binned(col, edges, names):
            b = np.clip(np.searchsorted(np.asarray(edges), sz[:, col], side="right") - 1, 0, len(names) - 1)
            return {names[k]: {"n": int((b == k).sum()), "top1": float(sz[b == k, 5].mean()), "cosine": float(sz[b == k, 6].mean()),
                               "voxels": float(sz[b == k, 4].sum())} for k in range(len(names)) if (b == k).any()}
        out["by_occupied_voxels"] = binned(0, (1, 2, 4, 8, 16, 32, 64, 10 ** 9), ["1", "2-3", "4-7", "8-15", "16-31", "32-63", ">=64"])
        out["by_observations_AB"] = binned(2, (2, 3, 5, 9, 17, 33, 10 ** 9), ["2", "3-4", "5-8", "9-16", "17-32", ">32"])
        out["by_frames_AB"] = binned(3, (2, 3, 4, 6, 10 ** 9), ["2", "3", "4-5", ">=6"])
    return out


def run_drive(drive, stride):
    grouping = GROUPINGS[RC.GROUPING]
    files = sorted(glob.glob(os.path.join(EVIDENCE_DIR, drive, "*.npz")))[::stride]
    acc = {"voxel": Acc(), **{sc: Acc() for sc in RC.SCALES}}
    rng = np.random.default_rng(20260913); n_elig_total = 0; t0 = time.time(); n_done = 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                z = {k: z[k] for k in z.files}
            idx = int(z["t"]); dims = tuple(int(x) for x in z["dims"]); n = int(np.prod(dims))
            with np.load(SRC.sample_path(drive, idx)) as smp:
                occ = (np.unpackbits(smp["gt_occ"])[:n] & np.unpackbits(smp["gt_valid"])[:n]).astype(bool).reshape(dims)
            elig = z["elig"].astype(np.int64); ne = len(elig); n_elig_total += ne
            xyz = np.stack(np.unravel_index(elig, dims), 1).astype(np.int64)
            row = z["pair_row"].astype(np.int64); off = z["pair_frame"].astype(np.int64)
            w = z["pair_w"].astype(np.float64); s = z["pair_s"].astype(np.float64)
            # ---- voxel level: units = eligible voxels
            Gv = group_units(row, ne, off, w, s, grouping)
            score_units(acc["voxel"], Gv, np.ones(ne), None, rng, cx=xyz[:, 0], cy=xyz[:, 1], cz=xyz[:, 2])
            own_ab = (Gv["A"][0] > 0) & (Gv["B"][0] > 0)
            pa_own, pb_own = probs(Gv["A"][0], Gv["A"][1]), probs(Gv["B"][0], Gv["B"][1])
            # ---- region level
            for sc, side in RC.SCALES.items():
                rid, st = label_regions(occ, xyz, side)
                assert (rid >= 0).all()
                R = len(st)
                Gr = group_units(rid[row], R, off, w, s, grouping)
                n_elig_per_region = np.bincount(rid, minlength=R).astype(np.float64)
                score_units(acc[sc], Gr, n_elig_per_region, st, rng, cx=st[:, 3], cy=st[:, 4], cz=st[:, 2])
                # matched: voxels with own A&B -> their region's A/B vectors
                if own_ab.any():
                    r_ = rid[own_ab]
                    pra, prb = probs(Gr["A"][0], Gr["A"][1])[r_], probs(Gr["B"][0], Gr["B"][1])[r_]
                    ok_o, cs_o, _ = agree_stats(pa_own[own_ab], pb_own[own_ab]); ok_r, cs_r, _ = agree_stats(pra, prb)
                    acc[sc].matched += (own_ab.sum(), ok_o.sum(), cs_o.sum(), ok_r.sum(), cs_r.sum())
            n_done += 1
            if n_done % 50 == 0:
                print(f"  [{drive[17:21]}] {n_done}/{len(files)} anchors, {(time.time()-t0)/n_done:.2f}s each", flush=True)
    out = {"drive": drive, "n_anchors": n_done, "n_eligible": int(n_elig_total), "stride": stride,
           "results": {k: summarize(a, n_elig_total) for k, a in acc.items()},
           "seconds": time.time() - t0, "firewall": audit.summary()}
    os.makedirs(os.path.join(RC.ART, "per_drive"), exist_ok=True)
    RC.write_json(os.path.join(RC.ART, "per_drive", f"{drive}{OUT_TAG}.json"), out)
    # raw accumulators for pooling
    np.savez_compressed(os.path.join(RC.ART, "per_drive", f"{drive}{OUT_TAG}_acc.npz"),
                        **{f"{k}__{name}": getattr(a, name) for k, a in acc.items() for name in ("three", "three_v", "cls_q", "cls_r", "cls_ok", "cls_q_v", "cls_ok_v", "cov", "matched")},
                        **{f"{k}__pair_{a_}{b_}": acc[k].pair[(a_, b_)] for k in acc for a_, b_ in PAIRS},
                        **{f"{k}__pairv_{a_}{b_}": acc[k].pair_v[(a_, b_)] for k in acc for a_, b_ in PAIRS},
                        **{f"{k}__shuf_{nm}": acc[k].shuf[nm] for k in acc for nm in acc[k].shuf},
                        **{f"{k}__size": np.asarray(acc[k].size, np.float64).reshape(-1, 7) for k in acc},
                        **{f"{k}__anchors": np.asarray(acc[k].anchors, np.float64).reshape(-1, 2) for k in acc},
                        n_eligible=np.int64(n_elig_total))
    r = out["results"]
    print(f"[{drive[17:21]}] eligible {n_elig_total:,}; AB top-1: voxel {r['voxel']['pairs']['AB']['top1']:.3f} (n {r['voxel']['pairs']['AB']['n']}) "
          + " ".join(f"{sc} {r[sc]['pairs']['AB']['top1']:.3f} (n {r[sc]['pairs']['AB']['n']}, cov {100*r[sc]['coverage_of_eligible']['AB']:.1f}%)" for sc in RC.SCALES), flush=True)
    return out


def merge():
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    per = {d: json.load(open(os.path.join(RC.ART, "per_drive", f"{d}{OUT_TAG}.json"))) for d in drives}
    pooled = {}
    keys = ["voxel"] + list(RC.SCALES)
    n_el = 0
    accs = {k: Acc() for k in keys}
    for d in SRC.TRAIN_DRIVES:
        with np.load(os.path.join(RC.ART, "per_drive", f"{d}{OUT_TAG}_acc.npz")) as z:
            n_el += int(z["n_eligible"])
            for k in keys:
                a = accs[k]
                for name in ("three", "three_v", "cls_q", "cls_r", "cls_ok", "cls_q_v", "cls_ok_v", "cov", "matched"):
                    setattr(a, name, getattr(a, name) + z[f"{k}__{name}"])
                for a_, b_ in PAIRS:
                    a.pair[(a_, b_)] += z[f"{k}__pair_{a_}{b_}"]; a.pair_v[(a_, b_)] += z[f"{k}__pairv_{a_}{b_}"]
                for nm in a.shuf:
                    a.shuf[nm] += z[f"{k}__shuf_{nm}"]
                a.size += [tuple(x) for x in z[f"{k}__size"]]; a.anchors += [tuple(x) for x in z[f"{k}__anchors"]]
    pooled = {k: summarize(accs[k], n_el) for k in keys}
    out = {"grouping": RC.GROUPING, "groups": GROUPINGS[RC.GROUPING], "scales_block_side_vox": RC.SCALES,
           "local_shuffle": {"z_bin_vox": RC.LOCAL_Z_BIN, "xy_cell_vox": RC.LOCAL_XY_CELL}, "size_strata_edges": RC.SIZE_EDGES,
           "drives": {d: per[d] for d in drives}, "train_pooled": {"n_eligible": n_el, "results": pooled},
           "firewall_violations": sum(len(per[d]["firewall"]["violations"]) for d in drives), "env": RC.environment()}
    out["population"] = "eligible (unobserved at t)" if not OUT_TAG else "all future-observed LiDAR-occupied voxels"
    RC.write_json(os.path.join(RC.ART, f"analysis{OUT_TAG}.json"), out)
    for d in drives + ["train_pooled"]:
        r = per[d]["results"] if d != "train_pooled" else pooled
        print(f"\n== {d[17:21] if d.startswith('2013') else d}")
        print("  unit      AB-n    top1   cos   R1(shuf)  R2(local)  3way   cov-AB  cls>=200 dom   H    | matched own->region top1")
        for k in keys:
            x = r[k]; m = x["matched_voxels_with_own_AB"]
            print(f"  {k:8s} {x['pairs']['AB']['n']:8d} {x['pairs']['AB']['top1']:.3f} {x['pairs']['AB']['cosine']:.3f} "
                  f"{x['controls']['R1']['top1']:.3f}     {x['controls']['R2']['top1']:.3f}     {x['three_way']['consistent']:.3f}  "
                  f"{100*x['coverage_of_eligible']['AB']:5.1f}%  {x['n_classes_support_ge_200']:3d}   {x['dominant_share']:.2f} {x['entropy_nats']:.2f} | "
                  f"{m['own_top1']:.3f} -> {m['region_top1']:.3f} (n {m['n']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--population", default="eligible", choices=("eligible", "all_future_observed"))
    a = ap.parse_args()
    RC.require_data_root()
    global EVIDENCE_DIR, OUT_TAG
    EVIDENCE_DIR = RC.EVIDENCE_DIR if a.population == "eligible" else RC.EVIDENCE_DIR + "_allocc"
    OUT_TAG = "" if a.population == "eligible" else "_allocc"
    if a.merge:
        return merge()
    run_drive(a.drive, a.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
