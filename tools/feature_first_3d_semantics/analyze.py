#!/usr/bin/env python
"""Dense-feature vs class-probability view consistency, and classify-then-fuse vs
feature-fuse-then-classify, on matched voxels with independent future views.

Per voxel and frame-disjoint group g in {A, B, C} (interleaved grouping) three matched
representations are formed from the same rays and weights:

* ``P_ref``    -- the existing pipeline: SAM-refined per-view probabilities, lifted and
                  evidence-averaged (the cached target path);
* ``P_coarse`` -- per-view classification of the dense feature (softmax(40 f_pix . text)),
                  lifted and evidence-averaged: classify-then-fuse;
* ``F``        -- the dense feature itself, lifted and evidence-averaged, unit-normalised:
                  fuse-then-classify (softmax(40 f . text)).

Same-surface similarity is compared against C0 (random within drive), C1 (within anchor),
C2 (local: same anchor, z band, xy cell) and C3 (same class, different voxel) controls.
Held-out validation: the fused A+B prediction of each pipeline against the independent
group C (and the two rotations B+C -> A, A+C -> B), plus feature cosine f_AB . f_C.

    python tools/feature_first_3d_semantics/analyze.py --population eligible
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from tools.future_semantic_completion.target_consistency.analyze import GROUPINGS          # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8, vocab                # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402

GRP = GROUPINGS[FF.GROUPING]
Z_BIN, XY_CELL = 4, 20
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
ROT = (("A", "B", "C"), ("B", "C", "A"), ("A", "C", "B"))       # (fuse, fuse, held-out)


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cos_rows(a, b):
    return (l2n(a) * l2n(b)).sum(1)


def js(p, q):
    p = np.clip(p, 1e-6, None); q = np.clip(q, 1e-6, None); m = 0.5 * (p + q)
    return 0.5 * (p * np.log(p / m)).sum(1) + 0.5 * (q * np.log(q / m)).sum(1)


def classify(f, text):
    """unit feature -> 18-class probabilities, the official coarse rule."""
    lg = l2n(f) @ text.T * FF.LOGIT_SCALE
    lg = lg - lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def load_units(drive, population, text, k360_of_union):
    """Collect per-voxel per-group sums for every voxel with evidence in >= 2 groups."""
    ev_dir = os.path.join(FF.EVIDENCE_DIRS[population], drive); lf_dir = os.path.join(FF.LIFT_DIR, population, drive)
    U = {g: {"w": [], "p_ref": [], "p_co": [], "f": [], "nf": []} for g in "ABC"}
    meta = {"anchor": [], "xyz": [], "n_frames_total": []}
    n_elig = 0; n_any = 0; n_mismatch = 0
    for f in sorted(glob.glob(os.path.join(lf_dir, "*.npz"))):
        with np.load(f) as z:
            L = {k: z[k] for k in z.files}
        with np.load(os.path.join(ev_dir, os.path.basename(f))) as z:
            E = {k: z[k] for k in z.files}
        assert int(L["t"]) == int(E["t"]) and np.array_equal(L["elig"], E["elig"])
        n = int(L["n_eligible"]); n_elig += n
        # align pair order: (row, frame) keys
        kl = L["pair_row"].astype(np.int64) * 100 + L["pair_frame"].astype(np.int64)
        ke = E["pair_row"].astype(np.int64) * 100 + E["pair_frame"].astype(np.int64)
        ol, oe = np.argsort(kl, kind="stable"), np.argsort(ke, kind="stable")
        if not np.array_equal(kl[ol], ke[oe]):
            n_mismatch += 1; continue
        row = L["pair_row"].astype(np.int64)[ol]; off = L["pair_frame"].astype(np.int64)[ol]; w = L["pair_w"][ol].astype(np.float64)
        feat = L["pair_feat"][ol].astype(np.float32); pco = L["pair_pcoarse"][ol].astype(np.float32)
        s_ref = E["pair_s"][oe].astype(np.float32)[:, k360_of_union]           # union 25 -> kitti360 18 channels
        nf_all = np.bincount(row, minlength=n); n_any += int((nf_all > 0).sum())
        sums = {}
        for g, offs in GRP.items():
            m = np.isin(off, offs)
            wg = np.zeros(n); fg = np.zeros((n, FF.FEAT_DIM), np.float32); pg = np.zeros((n, pco.shape[1]), np.float32)
            rg = np.zeros((n, pco.shape[1]), np.float32); ng = np.zeros(n)
            np.add.at(wg, row[m], w[m]); np.add.at(fg, row[m], feat[m]); np.add.at(pg, row[m], pco[m]); np.add.at(rg, row[m], s_ref[m]); np.add.at(ng, row[m], 1)
            sums[g] = (wg, fg, pg, rg, ng)
        has = np.stack([sums[g][0] > 0 for g in "ABC"], 1)
        keep = has.sum(1) >= 2
        if not keep.any():
            continue
        dims = (256, 256, 32); xyz = np.stack(np.unravel_index(L["elig"].astype(np.int64)[keep], dims), 1)
        for g in "ABC":
            wg, fg, pg, rg, ng = sums[g]
            U[g]["w"].append(wg[keep]); U[g]["f"].append(fg[keep]); U[g]["p_co"].append(pg[keep]); U[g]["p_ref"].append(rg[keep]); U[g]["nf"].append(ng[keep])
        meta["anchor"].append(np.full(int(keep.sum()), int(L["t"]))); meta["xyz"].append(xyz); meta["n_frames_total"].append(nf_all[keep])
    out = {g: {k: np.concatenate(v) for k, v in d.items()} for g, d in U.items()}
    for g in "ABC":
        wg = out[g]["w"]; pos = wg > 0
        for k in ("f", "p_co", "p_ref"):
            out[g][k][pos] /= wg[pos, None]                       # evidence-weighted means
    out["meta"] = {k: np.concatenate(v) for k, v in meta.items()}
    out["support"] = {"n_eligible": n_elig, "n_any_evidence": n_any, "n_units_ge2_groups": int(len(out["meta"]["anchor"])), "n_anchor_pair_mismatch": n_mismatch}
    return out


def shuffle_within(strata, rng, exclude_self=True):
    perm = np.arange(len(strata)); valid = np.zeros(len(strata), bool)
    for u_ in np.unique(strata):
        ix = np.flatnonzero(strata == u_)
        if len(ix) >= 2:
            p = rng.permutation(len(ix))
            if exclude_self and len(ix) > 2:
                fixed = np.flatnonzero(p == np.arange(len(ix)))
                if len(fixed):
                    p[fixed] = np.roll(p[fixed], 1) if len(fixed) > 1 else p[fixed]
            perm[ix] = ix[p]; valid[ix] = True
    return perm, valid


def auc(true, ctrl):
    """P(true pair scores higher than a random control pair): rank-based, scale-free."""
    from scipy.stats import rankdata
    x = np.concatenate([true, ctrl]); r = rankdata(x)
    n1, n2 = len(true), len(ctrl)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n2, 1))


def pair_metrics(D, a, b, text, rng, xyz, anchor, cls):
    m = (D[a]["w"] > 0) & (D[b]["w"] > 0)
    if m.sum() < 10:
        return None
    fa, fb = D[a]["f"][m], D[b]["f"][m]; pa, pb = D[a]["p_co"][m], D[b]["p_co"][m]; ra, rb = D[a]["p_ref"][m], D[b]["p_ref"][m]
    n = int(m.sum())
    res = {"n": n,
           "P_ref": {"top1": float((ra.argmax(1) == rb.argmax(1)).mean()), "cosine": float(cos_rows(ra, rb).mean()), "js": float(js(ra, rb).mean())},
           "P_coarse": {"top1": float((pa.argmax(1) == pb.argmax(1)).mean()), "cosine": float(cos_rows(pa, pb).mean()), "js": float(js(pa, pb).mean())},
           "F": {"cosine": float(cos_rows(fa, fb).mean()), "euclid_unit": float(np.linalg.norm(l2n(fa) - l2n(fb), axis=1).mean()),
                 "top1_classify_each": float((classify(fa, text).argmax(1) == classify(fb, text).argmax(1)).mean())}}
    true = {"F": cos_rows(fa, fb), "P_coarse": cos_rows(pa, pb), "P_ref": cos_rows(ra, rb),
            "P_coarse_top1": (pa.argmax(1) == pb.argmax(1)).astype(float), "P_ref_top1": (ra.argmax(1) == rb.argmax(1)).astype(float)}
    x, an, c = xyz[m], anchor[m], cls[m]
    strata = {"C0_drive": np.zeros(n, np.int64), "C1_anchor": an,
              "C2_local": (an * 1000 + x[:, 2] // Z_BIN) * 1000000 + (x[:, 0] // XY_CELL) * 1000 + (x[:, 1] // XY_CELL),
              "C3_same_class_other_location": c}
    res["controls"] = {}; res["hist"] = {}
    for name, st in strata.items():
        perm, valid = shuffle_within(st, rng)
        if valid.sum() < 10:
            continue
        v = valid
        ctrl = {"F": cos_rows(fa[v], fb[perm][v]), "P_coarse": cos_rows(pa[v], pb[perm][v]), "P_ref": cos_rows(ra[v], rb[perm][v]),
                "P_coarse_top1": (pa[v].argmax(1) == pb[perm][v].argmax(1)).astype(float), "P_ref_top1": (ra[v].argmax(1) == rb[perm][v].argmax(1)).astype(float)}
        r = {"n": int(v.sum())}
        for k in ctrl:
            t_ = true[k][v]; c_ = ctrl[k]
            r[k] = {"control_mean": float(c_.mean()), "true_mean": float(t_.mean()), "margin": float(t_.mean() - c_.mean()),
                    "margin_over_control_std": float((t_.mean() - c_.mean()) / max(c_.std(), 1e-9)),
                    "auc_true_gt_control": auc(t_, c_) if not k.endswith("top1") else None}
        res["controls"][name] = r
        if name in ("C2_local", "C3_same_class_other_location"):
            bins = np.linspace(-1, 1, 41)
            res["hist"][name] = {"bins": bins.tolist(), "F_true": np.histogram(true["F"][v], bins)[0].tolist(), "F_control": np.histogram(ctrl["F"], bins)[0].tolist(),
                                 "P_coarse_true": np.histogram(true["P_coarse"][v], bins)[0].tolist(), "P_coarse_control": np.histogram(ctrl["P_coarse"], bins)[0].tolist()}
    return res


def fusion_metrics(D, a, b, c, text, cls):
    m = (D[a]["w"] > 0) & (D[b]["w"] > 0) & (D[c]["w"] > 0)
    if m.sum() < 10:
        return None
    wa, wb = D[a]["w"][m][:, None], D[b]["w"][m][:, None]
    p_ref_ab = (wa * D[a]["p_ref"][m] + wb * D[b]["p_ref"][m]) / (wa + wb)
    p_co_ab = (wa * D[a]["p_co"][m] + wb * D[b]["p_co"][m]) / (wa + wb)
    f_ab = l2n(wa * D[a]["f"][m] + wb * D[b]["f"][m])
    pf_ab = classify(f_ab, text)
    rc, pc, fc = D[c]["p_ref"][m], D[c]["p_co"][m], D[c]["f"][m]
    pf_c = classify(fc, text)                                          # C classified once from its fused feature
    single = {"A_only": (D[a]["p_co"][m], classify(D[a]["f"][m], text), l2n(D[a]["f"][m]))}
    def agree(p, q): return float((p.argmax(1) == q.argmax(1)).mean())
    out = {"n": int(m.sum()),
           "P_ref_fuse_vs_C_ref": {"top1": agree(p_ref_ab, rc), "cosine": float(cos_rows(p_ref_ab, rc).mean()), "js": float(js(p_ref_ab, rc).mean())},
           "P_coarse_fuse_vs_C_coarse": {"top1": agree(p_co_ab, pc), "cosine": float(cos_rows(p_co_ab, pc).mean()), "js": float(js(p_co_ab, pc).mean())},
           "F_fuse_classify_vs_C_coarse": {"top1": agree(pf_ab, pc), "cosine": float(cos_rows(pf_ab, pc).mean()), "js": float(js(pf_ab, pc).mean())},
           "F_fuse_classify_vs_C_feature_classified": {"top1": agree(pf_ab, pf_c), "cosine": float(cos_rows(pf_ab, pf_c).mean()), "js": float(js(pf_ab, pf_c).mean())},
           "P_coarse_fuse_vs_C_feature_classified": {"top1": agree(p_co_ab, pf_c)},
           "feature_cosine_fAB_fC": float(cos_rows(f_ab, fc).mean()),
           "single_view_A_vs_C": {"P_coarse_top1": agree(single["A_only"][0], pc), "F_classify_top1": agree(single["A_only"][1], pc),
                                  "feature_cosine_fA_fC": float(cos_rows(single["A_only"][2], fc).mean())},
           "per_class": {}}
    # pipelines agree with each other?
    out["P_coarse_vs_F_same_decision"] = agree(p_co_ab, pf_ab)
    c_ = cls[m]
    for k in np.unique(c_):
        s = c_ == k
        if s.sum() >= 100:
            out["per_class"][k] = {"n": int(s.sum()), "P_coarse_top1": agree(p_co_ab[s], pc[s]), "F_top1": agree(pf_ab[s], pc[s]),
                                   "P_ref_top1": agree(p_ref_ab[s], rc[s]), "feature_cosine": float(cos_rows(f_ab[s], fc[s]).mean())}
    # number of fused frames (A+B) -> held-out C
    nf = (D[a]["nf"][m] + D[b]["nf"][m]).astype(int)
    out["by_frames_fused"] = {}
    for lo, hi, name in ((1, 1, "1"), (2, 2, "2"), (3, 10 ** 6, "3+")):
        s = (nf >= lo) & (nf <= hi)
        if s.sum() >= 50:
            out["by_frames_fused"][name] = {"n": int(s.sum()), "P_coarse_top1": agree(p_co_ab[s], pc[s]), "F_top1": agree(pf_ab[s], pc[s]),
                                            "feature_cosine": float(cos_rows(f_ab[s], fc[s]).mean()), "P_coarse_cosine": float(cos_rows(p_co_ab[s], pc[s]).mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", default="eligible", choices=list(FF.EVIDENCE_DIRS))
    a = ap.parse_args()
    FF.require_data_root()
    v = vocab.load("kitti360"); names = list(v.phrases)
    into = V8.into_matrix("kitti360"); k360_of_union = into.argmax(1)      # channel c -> union index
    with np.load(FF.TEXT_PATH) as z:
        text = z["text"].astype(np.float32)
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    out = {"population": a.population, "grouping": GRP, "vocabulary": names, "drives": {}, "z_bin": Z_BIN, "xy_cell": XY_CELL}
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for d in drives:
            rng = np.random.default_rng(20260913)
            D = load_units(d, a.population, text, k360_of_union)
            cls = D["A"]["p_co"].argmax(1); cls = np.where(D["A"]["w"] > 0, cls, D["B"]["p_co"].argmax(1))
            xyz, anchor = D["meta"]["xyz"], D["meta"]["anchor"]
            r = {"support": D["support"], "pairs": {}, "fusion": {}, "classes_of_units": {names[k]: int((cls == k).sum()) for k in np.unique(cls)}}
            for g1, g2 in PAIRS:
                pm = pair_metrics(D, g1, g2, text, rng, xyz, anchor, cls)
                if pm:
                    r["pairs"][f"{g1}{g2}"] = pm
            for g1, g2, g3 in ROT:
                fm = fusion_metrics(D, g1, g2, g3, text, cls)
                if fm:
                    fm["per_class"] = {names[k]: v_ for k, v_ in fm["per_class"].items()}
                    r["fusion"][f"{g1}+{g2}->{g3}"] = fm
            # per-class same-surface vs C2/C3 margins on the AB pair
            m = (D["A"]["w"] > 0) & (D["B"]["w"] > 0)
            r["per_class_AB"] = {}
            for k in np.unique(cls[m]):
                s = m & (cls == k)
                if s.sum() >= 200:
                    Dk = {g: {kk: vv[s] for kk, vv in D[g].items()} for g in "ABC"}
                    pk = pair_metrics(Dk, "A", "B", text, rng, xyz[s], anchor[s], cls[s])
                    if pk:
                        r["per_class_AB"][names[k]] = {"n": pk["n"], "P_coarse_top1": pk["P_coarse"]["top1"], "P_ref_top1": pk["P_ref"]["top1"],
                                                       "F_cosine": pk["F"]["cosine"], "P_coarse_cosine": pk["P_coarse"]["cosine"],
                                                       "F_margin_C2": pk["controls"].get("C2_local", {}).get("F", {}).get("margin"),
                                                       "P_coarse_margin_C2": pk["controls"].get("C2_local", {}).get("P_coarse", {}).get("margin"),
                                                       "F_auc_C2": pk["controls"].get("C2_local", {}).get("F", {}).get("auc_true_gt_control"),
                                                       "P_coarse_auc_C2": pk["controls"].get("C2_local", {}).get("P_coarse", {}).get("auc_true_gt_control")}
            out["drives"][d] = r
            ab = r["pairs"].get("AB", {}); fu = r["fusion"].get("A+B->C", {})
            print(f"  {d[17:21]}: units {D['support']['n_units_ge2_groups']:,} AB {ab.get('n')} | top1 P_ref {ab.get('P_ref',{}).get('top1',0):.3f} P_co {ab.get('P_coarse',{}).get('top1',0):.3f} "
                  f"| cos P_co {ab.get('P_coarse',{}).get('cosine',0):.3f} F {ab.get('F',{}).get('cosine',0):.3f} | C2 margin P_co {ab.get('controls',{}).get('C2_local',{}).get('P_coarse',{}).get('margin',0):+.3f} "
                  f"F {ab.get('controls',{}).get('C2_local',{}).get('F',{}).get('margin',0):+.3f} AUC P_co {ab.get('controls',{}).get('C2_local',{}).get('P_coarse',{}).get('auc_true_gt_control',0):.3f} F {ab.get('controls',{}).get('C2_local',{}).get('F',{}).get('auc_true_gt_control',0):.3f} "
                  f"| fusion n {fu.get('n')} P_co->C {fu.get('P_coarse_fuse_vs_C_coarse',{}).get('top1',0):.3f} F->C {fu.get('F_fuse_classify_vs_C_coarse',{}).get('top1',0):.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    out["seconds"] = time.time() - t0; out["firewall"] = audit.summary(); out["env"] = FF.environment()
    os.makedirs(FF.ART, exist_ok=True)
    FF.write_json(os.path.join(FF.ART, f"analysis_{a.population}.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
