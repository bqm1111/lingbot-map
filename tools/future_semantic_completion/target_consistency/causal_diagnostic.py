#!/usr/bin/env python
"""Step 8, diagnostic only: is the future-consensus semantic identity already visible in the
causal map? For each retained voxel of a frozen filter, take the nearest causal-map voxel
carrying teacher evidence (Euclidean distance transform on the anchor's own 32-channel
input) and compare its class with the A+B target. Reported by distance and by that voxel's
evidence weight. Nothing here feeds back into retention.

    python tools/future_semantic_completion/target_consistency/causal_diagnostic.py
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion.target_consistency.analyze import (GROUPINGS, filters,   # noqa: E402
                                                                         group_sums, probs, retain_mask, cosine)
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from tools.future_semantic_completion import head as H                                    # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8                       # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.supervision import targets as TG                                                # noqa: E402

D_EDGES = np.asarray([0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 1e9]); D_NAMES = ["<=1", "1-2", "2-3", "3-5", "5-10", "10-15", ">15"]
W_EDGES = np.asarray([0, 1, 2, 4, 8, 16, 1e9]); W_NAMES = ["(0,1]", "(1,2]", "(2,4]", "(4,8]", "(8,16]", ">16"]
FILTERS = ("any_AB", "F1_agree", "F2_tau0.60")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grouping", default="interleaved")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    F = filters(); grouping = GROUPINGS[a.grouping]
    out = {"grouping": a.grouping, "drive": SRC.VAL_DRIVE, "stride": a.stride, "filters": {}}
    acc = {f: {"d": np.zeros((len(D_NAMES), 3)), "w": np.zeros((len(W_NAMES), 3))} for f in FILTERS}
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        files = sorted(glob.glob(os.path.join(TC.EVIDENCE_DIR, SRC.VAL_DRIVE, "*.npz")))[::a.stride]
        for f in files:
            with np.load(f) as z:
                z = {k: z[k] for k in z.files}
            idx = int(z["t"])
            with np.load(SRC.sample_path(SRC.VAL_DRIVE, idx)) as s:
                d = TG.unpack_sample(s, dev)
            x = d["input"]
            near, dist = H.b1_nearest_index(x[6].cpu().numpy(), 1e9)
            semw_all = (x[6].reshape(-1) * 20.0).cpu().numpy()
            probs_all = x[7:7 + V8.U].reshape(V8.U, -1).T.cpu().numpy()
            G = group_sums(z, grouping)
            (wA, sA, _), (wB, sB, _), _ = G["A"], G["B"], G["C"]
            pAB = probs(wA + wB, sA + sB); elig = z["elig"].astype(np.int64)
            for fname in FILTERS:
                fam, tau, gam = F[fname]
                keep = retain_mask(fam, tau, gam, wA, sA, wB, sB)
                if not keep.any():
                    continue
                rows = elig[keep]; src = near[rows]; ok_src = src >= 0
                if not ok_src.any():
                    continue
                rows, src = rows[ok_src], src[ok_src]
                tgt = pAB[keep][ok_src]; cur = probs_all[src]
                agree = (tgt.argmax(1) == cur.argmax(1)).astype(np.float64); cs = cosine(tgt, cur)
                dd = dist[rows]; ww = semw_all[src]
                bd = np.clip(np.searchsorted(D_EDGES, dd, side="right") - 1, 0, len(D_NAMES) - 1)
                bw = np.clip(np.searchsorted(W_EDGES, ww, side="left") - 1, 0, len(W_NAMES) - 1)
                np.add.at(acc[fname]["d"], bd, np.stack([np.ones(len(dd)), agree, cs], 1))
                np.add.at(acc[fname]["w"], bw, np.stack([np.ones(len(dd)), agree, cs], 1))
    for fname in FILTERS:
        A = acc[fname]
        tot = A["d"].sum(0)
        out["filters"][fname] = {
            "n": int(tot[0]), "top1_agreement": float(tot[1] / max(tot[0], 1)), "cosine": float(tot[2] / max(tot[0], 1)),
            "by_distance_vox": {D_NAMES[k]: {"n": int(A["d"][k, 0]), "fraction": float(A["d"][k, 0] / max(tot[0], 1)),
                                             "top1": float(A["d"][k, 1] / max(A["d"][k, 0], 1)),
                                             "cosine": float(A["d"][k, 2] / max(A["d"][k, 0], 1))} for k in range(len(D_NAMES))},
            "by_evidence_weight": {W_NAMES[k]: {"n": int(A["w"][k, 0]), "top1": float(A["w"][k, 1] / max(A["w"][k, 0], 1)),
                                                "cosine": float(A["w"][k, 2] / max(A["w"][k, 0], 1))} for k in range(len(W_NAMES)) if A["w"][k, 0] > 0}}
        r = out["filters"][fname]
        print(f"  {fname:12s} n {r['n']:7d} causal-vs-target top1 {r['top1_agreement']:.3f} cos {r['cosine']:.3f} | by dist "
              + " ".join(f"{k}:{v['top1']:.2f}({v['fraction']:.2f})" for k, v in r["by_distance_vox"].items()) + " | by w "
              + " ".join(f"{k}:{v['top1']:.2f}" for k, v in r["by_evidence_weight"].items()), flush=True)
    out["seconds"] = time.time() - t0; out["firewall"] = audit.summary()
    ST.write_json(os.path.join(TC.ART, "causal_diagnostic.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
