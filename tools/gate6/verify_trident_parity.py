#!/usr/bin/env python
"""Adapter parity: does the dense score tensor's argmax reproduce official Trident?

Two independent checks, on >= 100 deterministically chosen real frames:

1. **argmax parity** -- ``argmax(dense_probs) == official_hard`` pixel-for-pixel. The
   official label comes out of the official code path itself, so this tests exactly the
   reconstruction documented in ``gate6.trident_adapter``.
2. **recorder is a no-op** -- on a subset, the model is also run with the recording
   wrappers *removed*, and the two official labels must be bit-identical. Without this,
   check 1 could be satisfied by a recorder that quietly changed the prediction.

Frame selection is prediction-independent: evenly strided over each dataset's sorted
unique-frame list. No target is opened.

    python tools/gate6/verify_trident_parity.py --n 102
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                      # noqa: E402
from gates.gate6 import frames as F, vocab                                     # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, official_only, TRIDENT_ROOT  # noqa: E402

SAM_CKPT = os.environ.get(
    "GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def pick(dataset, n):
    u = F.unique_frames(dataset, REPO_ROOT)
    if n >= len(u):
        return u
    return [u[(i * len(u)) // n] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=102, help="total frames (split evenly)")
    ap.add_argument("--n-pristine", type=int, default=12,
                    help="frames also run with the recorder removed")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    per = a.n // len(vocab.DATASETS)
    rows, t0 = [], time.time()
    pristine_done = 0
    for ds in vocab.DATASETS:
        v = vocab.load(ds)
        T = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)
        sel = pick(ds, per)
        for i, (key, path) in enumerate(sel):
            out = T.predict_frame(path)
            n = int(out.hard.numel())
            row = {"dataset": ds, "frame": key, "pixels": n, "ties": int(out.ties),
                   "tie_fraction": out.ties / n, "lattice": list(out.lattice_hw),
                   "n_classes": int(out.probs.shape[0]),
                   "prob_sum_max_abs_err": float((out.probs.sum(0) - 1).abs().max())}
            if pristine_done < a.n_pristine and i % max(1, per // (a.n_pristine // 3 or 1)) == 0:
                ref = official_only(T, path)
                row["pristine_identical"] = bool(torch.equal(ref, out.hard))
                row["pristine_mismatch"] = int((ref != out.hard).sum())
                pristine_done += 1
            rows.append(row)
        del T
        torch.cuda.empty_cache()

    tied = [r for r in rows if r["ties"]]
    prist = [r for r in rows if "pristine_identical" in r]
    summary = {
        "n_frames": len(rows), "n_datasets": len(vocab.DATASETS),
        "frames_with_any_tie": len(tied),
        "total_tie_pixels": int(sum(r["ties"] for r in rows)),
        "total_pixels": int(sum(r["pixels"] for r in rows)),
        "max_tie_fraction": max((r["tie_fraction"] for r in rows), default=0.0),
        "argmax_parity_exact": len(tied) == 0,
        "n_pristine_checked": len(prist),
        "pristine_all_identical": all(r["pristine_identical"] for r in prist),
        "max_prob_sum_err": max(r["prob_sum_max_abs_err"] for r in rows),
        "trident_root": TRIDENT_ROOT,
        "seconds": time.time() - t0,
        "selection_rule": "evenly strided over each dataset's sorted unique-frame list",
        "targets_opened": "none",
    }
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6", "trident_parity.json"),
               {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))
    return 0 if summary["argmax_parity_exact"] and summary["pristine_all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
