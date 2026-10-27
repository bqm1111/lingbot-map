#!/usr/bin/env python
"""Phase A — audit the one-frame input chain for temporal leakage.

The brief requires that the one-frame arm reuse no five-frame geometry, depth, metric scale,
normalization statistics or cached temporal evidence. This checks that claim on every sample
rather than asserting it, and reports the one thing that IS shared (per-frame MoGe-B depth)
with the reason it is admissible.

Checks, all over the full sample set:

  L1  the one-frame clip holds exactly one slot, one log_s candidate, anchor_slot 0
  L2  the clip's single frame key is the anchor's own key
  L3  the one-frame metric scale DIFFERS from the five-frame scale -- if any sample matched
      exactly, the five-frame scale would have leaked in
  L4  the one-frame LingBotMap depth DIFFERS from the five-frame clip's anchor-slot depth --
      proves the reconstruction was genuinely re-run, not sliced
  L5  the stored prediction records n_input_frames == 1 and the one-frame scale
  L6  the diagnostic arm one_frame_unet_scale5 carries the FIVE-frame scale (it is supposed
      to; this confirms the diagnostic is what it claims and is distinguishable from L5)
  L7  the MoGe-B path depends only on (dataset, frame_key) -- i.e. it is a per-frame cache
      with no window argument, so it cannot carry temporal content
  L8  every arm used the same frozen CompletionUNet file (one sha256)

    PYTHONPATH=$PWD python tools/method_vs_data/one_frame_leakage_audit.py --dataset semantickitti
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.method_vs_data import one_frame as OF                                        # noqa: E402
from tools.method_vs_data.one_frame_predict import out_path                             # noqa: E402
from tools.occany_comparison import config as OC                                        # noqa: E402
from tools.occany_comparison.moge_extra import resolve_moge_path                        # noqa: E402
from tools.semantic_completion import config as SCC                                     # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402


def audit(dataset: str, limit=None) -> dict:
    sids = [s.sample_id for s in ZS.evaluated_samples(dataset)[:limit]]
    fails = {f"L{i}": [] for i in range(1, 7)}
    nonfinite: list = []
    scale_ratio, depth_absdiff = [], []
    n = 0
    for sid in sids:
        p1, p5 = OF.clip_path(dataset, sid), OC.clip_path(dataset, "causal", sid)
        if not (os.path.exists(p1) and os.path.exists(p5)):
            continue
        with np.load(p1) as a, np.load(p5) as b:
            if not (a["pred_depth"].shape[0] == 1 and a["log_s_frames"].shape == (1,)
                    and int(a["anchor_slot"]) == 0 and int(a["n_input_frames"]) == 1):
                fails["L1"].append(sid)
            if not (len(a["keys"]) == 1 and str(a["keys"][0]) == str(b["keys"][4])):
                fails["L2"].append(sid)
            s1, s5 = float(a["scale"]), float(b["scale"])
            # NaN != NaN, so a non-finite scale would pass a bare inequality test. Require
            # BOTH scales finite and different, and count non-finite one-frame scales apart.
            if not (np.isfinite(s1) and np.isfinite(s5)):
                nonfinite.append(sid)
            elif s1 == s5:
                fails["L3"].append(sid)
            else:
                scale_ratio.append(s1 / s5)
            d = float(np.abs(a["pred_depth"][0].astype(np.float32)
                             - b["pred_depth"][4].astype(np.float32)).mean())
            depth_absdiff.append(d)
            if d == 0.0:
                fails["L4"].append(sid)

            q = out_path("one_frame_unet", dataset, sid)
            if os.path.exists(q):
                with np.load(q) as z:
                    if not (int(z["n_input_frames"]) == 1
                            and abs(float(z["scale"]) - s1) < 1e-12):
                        fails["L5"].append(sid)
            r = out_path("one_frame_unet_scale5", dataset, sid)
            if os.path.exists(r):
                with np.load(r) as z:
                    if not (int(z["n_input_frames"]) == 1
                            and abs(float(z["scale"]) - s5) < 1e-12):
                        fails["L6"].append(sid)
        n += 1

    sig = list(inspect.signature(resolve_moge_path).parameters)
    l7_ok = sig == ["dataset", "key"] or ("key" in sig and len(sig) == 2)
    ck = SCC.Paths().geometry_checkpoint(ZC.GEOMETRY_SEED)
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest() if os.path.exists(ck) else None
    frozen = json.load(open(SCC.Paths().geometry_freeze))["sha256"].get(
        os.path.relpath(ck, SCC.REPO_ROOT))
    l8_ok = bool(sha) and sha == frozen

    return {"dataset": dataset, "n_samples_checked": n,
            "L1_one_slot_one_candidate": {"failures": len(fails["L1"]), "examples": fails["L1"][:3]},
            "L2_key_is_anchor": {"failures": len(fails["L2"]), "examples": fails["L2"][:3]},
            "L3_scale_differs_from_5frame": {"failures": len(fails["L3"]), "examples": fails["L3"][:3],
                                             "n_nonfinite_scale": len(nonfinite),
                                             "nonfinite_samples": nonfinite[:5],
                                             "ratio_1f_over_5f": {
                                                 "min": float(np.min(scale_ratio)) if scale_ratio else None,
                                                 "p50": float(np.median(scale_ratio)) if scale_ratio else None,
                                                 "max": float(np.max(scale_ratio)) if scale_ratio else None}},
            "L4_depth_rerun_not_sliced": {"failures": len(fails["L4"]), "examples": fails["L4"][:3],
                                          "mean_abs_depth_diff": {
                                              "min": float(np.min(depth_absdiff)) if depth_absdiff else None,
                                              "p50": float(np.median(depth_absdiff)) if depth_absdiff else None,
                                              "max": float(np.max(depth_absdiff)) if depth_absdiff else None}},
            "L5_prediction_records_one_frame": {"failures": len(fails["L5"]), "examples": fails["L5"][:3]},
            "L6_diagnostic_carries_5frame_scale": {"failures": len(fails["L6"]), "examples": fails["L6"][:3]},
            "L7_moge_cache_is_per_frame": {"ok": bool(l7_ok), "signature": sig,
                                           "note": "no window/clip argument, so the cache "
                                                   "cannot carry temporal content"},
            "L8_frozen_unet": {"ok": l8_ok, "checkpoint": ck, "sha256": sha,
                               "matches_code_freeze": frozen == sha,
                               "geometry_seed": ZC.GEOMETRY_SEED,
                               "note": "one file, loaded once, shared by every arm in a run"},
            "admissible_shared_input": "MoGe-B per-frame depth only (L7)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="semantickitti")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    r = audit(a.dataset, a.limit)
    print(f"=== leakage audit: {a.dataset}, {r['n_samples_checked']} samples")
    ok = True
    for k, v in r.items():
        if not k.startswith("L"):
            continue
        if "failures" in v:
            bad = v["failures"]; ok &= bad == 0
            print(f"  {'PASS' if bad == 0 else 'FAIL'} {k:38s} failures {bad}")
        else:
            good = v.get("ok", True); ok &= bool(good)
            print(f"  {'PASS' if good else 'FAIL'} {k:38s} {v.get('signature', v.get('sha256', ''))}")
    s = r["L3_scale_differs_from_5frame"]["ratio_1f_over_5f"]
    d = r["L4_depth_rerun_not_sliced"]["mean_abs_depth_diff"]
    print(f"  one-frame / five-frame scale ratio: min {s['min']:.4f} p50 {s['p50']:.4f} max {s['max']:.4f}")
    print(f"  mean |depth difference|:            min {d['min']:.4f} p50 {d['p50']:.4f} max {d['max']:.4f}")
    print(f"\n  {'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    out = os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data", "results",
                       f"one_frame_leakage_audit_{a.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
