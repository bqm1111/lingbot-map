#!/usr/bin/env python
"""Open-vocabulary diagnostic on drive 0006: query the fused 3D feature with free-form text.

For every voxel with A&B evidence, ``f_AB = normalize(w_A f_A + w_B f_B)`` is compared by
cosine with 16 free-text queries encoded by Trident's own text tower; the best query's
mapped vocabulary class is compared with the fused feature's vocabulary decision and with
the independent view C. No training, no tuning: the queries were fixed before any result.

    python tools/feature_first_3d_semantics/openvocab.py --population eligible
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from tools.feature_first_3d_semantics.analyze import GRP, l2n, classify, cos_rows          # noqa: E402
from core.datasets import kitti360_splits as SRC, vocab                                   # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--population", default="eligible"); a = ap.parse_args()
    FF.require_data_root()
    v = vocab.load("kitti360"); names = list(v.phrases)
    with np.load(FF.TEXT_PATH) as z:
        text = z["text"].astype(np.float32)
    with np.load(os.path.join(FF.DATA_ROOT, "text_queries.npz")) as z:
        queries, target, qe = [str(x) for x in z["queries"]], [str(x) for x in z["target_class"]], z["embeddings"].astype(np.float32)
    tgt_idx = np.asarray([names.index(t) for t in target])
    fAB, fC, pcC = [], [], []
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in sorted(glob.glob(os.path.join(FF.LIFT_DIR, a.population, SRC.VAL_DRIVE, "*.npz"))):
            with np.load(f) as z:
                n = int(z["n_eligible"]); row = z["pair_row"].astype(np.int64); off = z["pair_frame"].astype(np.int64)
                w = z["pair_w"].astype(np.float64); feat = z["pair_feat"].astype(np.float32); pco = z["pair_pcoarse"].astype(np.float32)
            S = {}
            for g, offs in GRP.items():
                m = np.isin(off, offs); wg = np.zeros(n); fg = np.zeros((n, FF.FEAT_DIM), np.float32); pg = np.zeros((n, len(names)), np.float32)
                np.add.at(wg, row[m], w[m]); np.add.at(fg, row[m], feat[m]); np.add.at(pg, row[m], pco[m]); S[g] = (wg, fg, pg)
            keep = (S["A"][0] > 0) & (S["B"][0] > 0)
            if not keep.any():
                continue
            fAB.append(l2n(S["A"][1][keep] + S["B"][1][keep]))                       # sums are already weight-multiplied
            hasC = S["C"][0][keep] > 0
            fc = np.where(hasC[:, None], S["C"][1][keep] / np.maximum(S["C"][0][keep], 1e-9)[:, None], np.nan); fC.append(fc)
            pcC.append(np.where(hasC[:, None], S["C"][2][keep] / np.maximum(S["C"][0][keep], 1e-9)[:, None], np.nan))
    fAB, fC, pcC = np.concatenate(fAB), np.concatenate(fC), np.concatenate(pcC)
    vocab_dec = classify(fAB, text).argmax(1)
    qs = fAB @ qe.T; best = qs.argmax(1); mapped = tgt_idx[best]
    hasC = ~np.isnan(fC[:, 0])
    out = {"population": a.population, "n_units": int(len(fAB)), "n_with_C": int(hasC.sum()), "queries": queries, "target_class": target,
           "query_mapped_class_equals_vocab_decision": float((mapped == vocab_dec).mean()),
           "query_mapped_class_equals_C_view_decision": float((mapped[hasC] == pcC[hasC].argmax(1)).mean()),
           "vocab_decision_equals_C_view_decision": float((vocab_dec[hasC] == pcC[hasC].argmax(1)).mean()),
           "query_usage": {q: int((best == i).sum()) for i, q in enumerate(queries)},
           "mean_best_query_cosine": float(qs.max(1).mean()), "mean_best_vocab_cosine": float((fAB @ text.T).max(1).mean()),
           "per_vocab_class": {}}
    for k in np.unique(vocab_dec):
        s = vocab_dec == k
        if s.sum() >= 100:
            top_q = np.bincount(best[s], minlength=len(queries)); j = int(top_q.argmax())
            out["per_vocab_class"][names[k]] = {"n": int(s.sum()), "top_query": queries[j], "top_query_share": float(top_q[j] / s.sum()),
                                                "query_mapped_agree": float((mapped[s] == k).mean())}
    out["firewall"] = audit.summary()
    FF.write_json(os.path.join(FF.ART, f"openvocab_{a.population}.json"), out)
    print(json.dumps({k: out[k] for k in ("n_units", "query_mapped_class_equals_vocab_decision", "query_mapped_class_equals_C_view_decision", "vocab_decision_equals_C_view_decision")}, indent=1))
    for k, v_ in out["per_vocab_class"].items():
        print(f"  {k:16s} n {v_['n']:6d} top query '{v_['top_query']}' ({v_['top_query_share']:.2f}), mapped agree {v_['query_mapped_agree']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
