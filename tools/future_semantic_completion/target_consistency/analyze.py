#!/usr/bin/env python
"""Candidate targets from future groups A+B, validated against independent group C.

Two frame-disjoint groupings of the 20-frame horizon are evaluated:

* ``contiguous``  -- A = t+1..t+7, B = t+8..t+14, C = t+15..t+20 (maximal temporal
                     separation);
* ``interleaved`` -- offset mod 3: A = {1,4,...,19}, B = {2,5,...,20}, C = {3,6,...,18}
                     (every frame in one group only; consecutive frames in different groups).

For each grouping and each filter the retained set is decided from A and B alone, the
candidate vector is the evidence-weighted fusion of A and B alone, and repeatability is the
agreement of that candidate with C on the retained voxels that have C evidence. Group C and
the causal map never enter the retention rule (``verify.py`` proves it by perturbation).
Thresholds are inspected on drive 0006 only; the training drives are reported afterwards.

    python tools/future_semantic_completion/target_consistency/analyze.py
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8                       # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402

U = V8.U
GROUPINGS = {
    "contiguous": {"A": [o for o in range(1, 8)], "B": [o for o in range(8, 15)], "C": [o for o in range(15, 21)]},
    "interleaved": {"A": [o for o in range(1, 21) if o % 3 == 1], "B": [o for o in range(1, 21) if o % 3 == 2],
                    "C": [o for o in range(1, 21) if o % 3 == 0]},
}
CONF_EDGES = np.asarray([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
MIN_CLASS_SUPPORT = 500


def filters():
    """name -> (family, tau, gamma). Ordered from loosest to strictest."""
    out = {"any_AB": ("any", 0.0, 0.0), "F0_existing_rule": ("F0", 0.0, 0.0), "F1_agree": ("F1", 0.0, 0.0)}
    for t in TC.TAUS[1:]:
        out[f"F2_tau{t:.2f}"] = ("F2", t, 0.0)
    for g in TC.GAMMAS[1:]:
        out[f"F3_gam{g:.2f}"] = ("F3", 0.0, g)
    for t in TC.TAUS[1:]:
        for g in TC.GAMMAS[1:]:
            out[f"F4_tau{t:.2f}_gam{g:.2f}"] = ("F4", t, g)
    return out


def group_sums(z, grouping):
    """Per group: w [n], s [n,U], n_frames [n] over the eligible rows, from the sparse pairs."""
    n = int(z["n_eligible"]); row = z["pair_row"].astype(np.int64); off = z["pair_frame"].astype(np.int64)
    w = z["pair_w"].astype(np.float32); s = z["pair_s"].astype(np.float32)
    out = {}
    for g, offs in grouping.items():
        m = np.isin(off, offs)
        wg = np.zeros(n, np.float32); sg = np.zeros((n, U), np.float32); ng = np.zeros(n, np.int32)
        np.add.at(wg, row[m], w[m]); np.add.at(sg, row[m], s[m]); np.add.at(ng, row[m], 1)
        out[g] = (wg, sg, ng)
    return out


def probs(w, s):
    return s / np.maximum(w, 1e-9)[:, None]


def cosine(p, q):
    return (p * q).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1), 1e-9)


def js(p, q):
    p = np.clip(p, 1e-6, None); q = np.clip(q, 1e-6, None); m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(1) + 0.5 * (q * np.log(q / m)).sum(1)


def retain_mask(family, tau, gamma, wA, sA, wB, sB):
    """The retention rule. Reads A and B only."""
    both = (wA > 0) & (wB > 0)
    pA, pB = probs(wA, sA), probs(wB, sB)
    aA, aB = pA.argmax(1), pB.argmax(1)
    if family == "any":
        return (wA + wB) > 0
    if family == "F0":
        return ((wA + wB) >= TC.SEM_W_MIN) & ~(both & (aA != aB))
    keep = both & (aA == aB)
    if family in ("F2", "F4"):
        keep &= (pA.max(1) >= tau) & (pB.max(1) >= tau)
    if family in ("F3", "F4"):
        keep &= cosine(pA, pB) >= gamma
    return keep


class Acc:
    """Sums for one (grouping, filter, drive)."""

    def __init__(self):
        self.n_ret = 0; self.n_val = 0; self.top1 = 0.0; self.cos = 0.0; self.js = 0.0
        self.shuf_top1 = 0.0; self.shuf_cos = 0.0
        self.cls_ret = np.zeros(U, np.int64)            # argmax of candidate over retained set
        self.cls_val_q = np.zeros(U, np.int64); self.cls_val_r = np.zeros(U, np.int64)
        self.cls_ok = np.zeros(U, np.int64)             # per candidate class, agreements with C
        self.conf = np.zeros((len(CONF_EDGES) - 1, 2), np.float64)
        self.anchors = []                               # (n_val, top1_sum) for block bootstrap

    def add(self, keep, pAB, pC, hasC, rng):
        self.n_ret += int(keep.sum())
        if keep.any():
            self.cls_ret += np.bincount(pAB[keep].argmax(1), minlength=U)
        v = keep & hasC
        nv = int(v.sum())
        if nv == 0:
            self.anchors.append((0, 0.0)); return
        a, c = pAB[v], pC[v]
        qa, qc = a.argmax(1), c.argmax(1)
        ok = (qa == qc).astype(np.float64)
        self.n_val += nv; self.top1 += ok.sum(); self.cos += cosine(a, c).sum(); self.js += js(a, c).sum()
        perm = rng.permutation(nv)
        self.shuf_top1 += float((qa == qc[perm]).sum()); self.shuf_cos += float(cosine(a, c[perm]).sum())
        self.cls_val_q += np.bincount(qa, minlength=U); self.cls_val_r += np.bincount(qc, minlength=U)
        self.cls_ok += np.bincount(qa[ok > 0], minlength=U)
        b = np.clip(np.searchsorted(CONF_EDGES, a.max(1), side="right") - 1, 0, len(CONF_EDGES) - 2)
        np.add.at(self.conf, b, np.stack([np.ones(nv), ok], 1))
        self.anchors.append((nv, float(ok.sum())))

    def merge(self, o):
        for k in ("n_ret", "n_val", "top1", "cos", "js", "shuf_top1", "shuf_cos"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for k in ("cls_ret", "cls_val_q", "cls_val_r", "cls_ok", "conf"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.anchors += o.anchors
        return self

    def summary(self, n_elig):
        nv = max(self.n_val, 1)
        dist = self.cls_ret / max(self.cls_ret.sum(), 1)
        pd = dist[dist > 0]
        q = self.cls_val_q / nv; r = self.cls_val_r / nv
        # block bootstrap over anchors (20-anchor blocks) for the top-1 repeatability
        arr = np.asarray(self.anchors, np.float64)
        ci = None
        if len(arr) and arr[:, 0].sum() > 0:
            blocks = [arr[i:i + 20] for i in range(0, len(arr), 20)]
            rng = np.random.default_rng(0); vals = []
            for _ in range(1000):
                pick = rng.integers(0, len(blocks), len(blocks))
                s = np.concatenate([blocks[k] for k in pick])
                vals.append(s[:, 1].sum() / max(s[:, 0].sum(), 1))
            ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        return {"n_retained": int(self.n_ret), "coverage_of_eligible": self.n_ret / max(n_elig, 1),
                "n_validated": int(self.n_val), "validated_fraction_of_retained": self.n_val / max(self.n_ret, 1),
                "top1_repeatability": self.top1 / nv, "top1_ci95_blocks": ci,
                "cosine": self.cos / nv, "js_divergence": self.js / nv,
                "shuffled_top1": self.shuf_top1 / nv, "shuffled_cosine": self.shuf_cos / nv,
                "class_prior_top1": float((q * r).sum()), "majority_rate_C": float(r.max()) if nv > 1 else None,
                "n_classes_present": int((self.cls_ret > 0).sum()),
                "n_classes_support_ge_500": int((self.cls_ret >= MIN_CLASS_SUPPORT).sum()),
                "top1_class": V8.UNION[int(dist.argmax())] if dist.sum() else None,
                "top1_class_share": float(dist.max()) if dist.sum() else None,
                "top2_class_share": float(np.sort(dist)[-2:].sum()) if dist.sum() else None,
                "entropy_nats": float(-(pd * np.log(pd)).sum()) if len(pd) else None,
                "class_distribution": {V8.UNION[u]: float(dist[u]) for u in range(U) if dist[u] > 0},
                "per_class_repeatability": {V8.UNION[u]: {"n": int(self.cls_val_q[u]),
                                                          "top1": float(self.cls_ok[u] / self.cls_val_q[u])}
                                            for u in range(U) if self.cls_val_q[u] >= MIN_CLASS_SUPPORT},
                "by_confidence": {f"[{CONF_EDGES[k]:.1f},{CONF_EDGES[k+1]:.1f})": {"n": int(self.conf[k, 0]),
                                                                                    "top1": float(self.conf[k, 1] / max(self.conf[k, 0], 1))}
                                  for k in range(len(CONF_EDGES) - 1) if self.conf[k, 0] > 0}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    ST.require_data_root()
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    F = filters()
    acc = {(g, f, d): Acc() for g in GROUPINGS for f in F for d in drives}
    support = {d: {"n_anchors": 0, "n_eligible": 0, "n_any_evidence": 0, "n_existing_target": 0,
                   "frames_per_voxel_hist": np.zeros(21, np.int64), "pairs_by_offset": np.zeros(21, np.int64),
                   **{f"{g}_{k}": 0 for g in GROUPINGS for k in ("A", "B", "C", "AB", "AC", "BC", "ABC")}}
               for d in drives}
    pair = {(g, d, k): np.zeros(3) for g in GROUPINGS for d in drives for k in ("AB", "AC", "BC")}
    rng = np.random.default_rng(20260913); t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for d in drives:
            files = sorted(glob.glob(os.path.join(TC.EVIDENCE_DIR, d, "*.npz")))[::a.stride]
            for f in files:
                with np.load(f) as z:
                    z = {k: z[k] for k in z.files}
                n = int(z["n_eligible"]); S = support[d]
                S["n_anchors"] += 1; S["n_eligible"] += n; S["n_existing_target"] += int(z["existing_target_present"].sum())
                nf = np.bincount(z["pair_row"].astype(np.int64), minlength=n)
                S["n_any_evidence"] += int((nf > 0).sum())
                S["frames_per_voxel_hist"] += np.bincount(np.clip(nf[nf > 0], 0, 20), minlength=21)
                S["pairs_by_offset"] += np.bincount(z["pair_frame"].astype(np.int64), minlength=21)
                for g, grouping in GROUPINGS.items():
                    G = group_sums(z, grouping)
                    (wA, sA, _), (wB, sB, _), (wC, sC, _) = G["A"], G["B"], G["C"]
                    hA, hB, hC = wA > 0, wB > 0, wC > 0
                    for k, m in (("A", hA), ("B", hB), ("C", hC), ("AB", hA & hB), ("AC", hA & hC),
                                 ("BC", hB & hC), ("ABC", hA & hB & hC)):
                        S[f"{g}_{k}"] += int(m.sum())
                    pA, pB, pC = probs(wA, sA), probs(wB, sB), probs(wC, sC)
                    for k, (m, p1, p2) in (("AB", (hA & hB, pA, pB)), ("AC", (hA & hC, pA, pC)), ("BC", (hB & hC, pB, pC))):
                        if m.any():
                            pair[(g, d, k)] += (m.sum(), (p1[m].argmax(1) == p2[m].argmax(1)).sum(), cosine(p1[m], p2[m]).sum())
                    pAB = probs(wA + wB, sA + sB)
                    for fname, (fam, tau, gam) in F.items():
                        keep = retain_mask(fam, tau, gam, wA, sA, wB, sB)
                        acc[(g, fname, d)].add(keep, pAB, pC, hC, rng)
            print(f"  {d[17:21]}: {support[d]['n_anchors']} anchors, eligible {support[d]['n_eligible']:,} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    # ---- assemble -------------------------------------------------------------------------
    out = {"groupings": GROUPINGS, "filters": {k: {"family": v[0], "tau": v[1], "gamma": v[2]} for k, v in F.items()},
           "sem_w_min": TC.SEM_W_MIN, "min_class_support": MIN_CLASS_SUPPORT, "stride": a.stride,
           "support": {}, "pairwise_raw_agreement": {}, "results": {}}
    for d in drives:
        S = support[d]
        out["support"][d] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in S.items()}
    pooled = {k: 0 for k in support[drives[0]] if not isinstance(support[drives[0]][k], np.ndarray)}
    for d in SRC.TRAIN_DRIVES:
        for k in pooled:
            pooled[k] += support[d][k]
    out["support"]["train_pooled"] = pooled
    for g in GROUPINGS:
        for k in ("AB", "AC", "BC"):
            for d in drives + ["train_pooled"]:
                v = (sum(pair[(g, dd, k)] for dd in SRC.TRAIN_DRIVES) if d == "train_pooled" else pair[(g, d, k)])
                out["pairwise_raw_agreement"][f"{g}:{k}:{d}"] = {"n": int(v[0]), "top1": float(v[1] / max(v[0], 1)),
                                                                 "cosine": float(v[2] / max(v[0], 1))}
    for g in GROUPINGS:
        out["results"][g] = {}
        for fname in F:
            row = {}
            for d in drives:
                row[d] = acc[(g, fname, d)].summary(support[d]["n_eligible"])
            m = Acc()
            for d in SRC.TRAIN_DRIVES:
                m.merge(acc[(g, fname, d)])
            row["train_pooled"] = m.summary(pooled["n_eligible"])
            out["results"][g][fname] = row
    out["previous_reference"] = {"causal_vs_future_top1": 0.456, "causal_vs_future_cosine": 0.51,
                                 "shuffled_top1": 0.35, "shuffled_cosine": 0.40,
                                 "teacher_coverage_of_key_population": 0.228}
    out["seconds"] = time.time() - t0; out["firewall"] = audit.summary(); out["env"] = ST.environment()
    os.makedirs(TC.ART, exist_ok=True)
    ST.write_json(os.path.join(TC.ART, "analysis.json"), out)
    # ---- print the 0006 table for the primary grouping ----------------------------------
    for g in GROUPINGS:
        print(f"\n== {g}: drive 0006 | filter | retained | %elig | validated | top1 | cos | shuf | prior | ncls>=500 | top1 share | H")
        for fname in F:
            r = out["results"][g][fname][SRC.VAL_DRIVE]
            if r["n_validated"] == 0:
                print(f"  {fname:24s} {r['n_retained']:9d} {100*r['coverage_of_eligible']:6.2f}  (no C support)"); continue
            print(f"  {fname:24s} {r['n_retained']:9d} {100*r['coverage_of_eligible']:6.2f} {r['n_validated']:8d} "
                  f"{r['top1_repeatability']:.3f} {r['cosine']:.3f} {r['shuffled_top1']:.3f} {r['class_prior_top1']:.3f} "
                  f"{r['n_classes_support_ge_500']:3d} {r['top1_class_share']:.3f} {r['entropy_nats']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
