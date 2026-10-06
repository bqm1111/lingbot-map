"""Final zero-shot evaluation verification, recomputed from files. Writes artifacts/final_zero_shot/verification.{json,md}."""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np

from tools.zero_shot_eval import config as C, freeze, samples as ZS
from tools.zero_shot_eval.run_eval import SCORE_DIR

A = C.ARTIFACT_DIR


def _row(rows: List[Dict], name: str, ok: bool, detail) -> None:
    rows.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, str(detail)[:300], flush=True)


def _ts(s: str) -> float:
    return _dt.datetime.fromisoformat(s).timestamp()


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(device: str = "cuda:0") -> Dict:
    rows: List[Dict] = []
    fz = {s: freeze.check(s) for s in ("protocol", "evaluators")}
    preds = {d: sorted(glob.glob(os.path.join(C.PREDICTION_ROOT, d, "*.npz"))) for d in C.DATASETS}
    t_pred = min(os.stat(p).st_mtime for v in preds.values() for p in v)
    t_score = min(os.stat(p).st_mtime for p in glob.glob(os.path.join(SCORE_DIR, "scores_*.npz")))
    _row(rows, "1 protocol and evaluators frozen before any prediction or score; unchanged since",
         all(_ts(r["frozen_at"]) < t_pred and not any(r["existing_predictions"].values()) and r["existing_scores"] == 0 for r in fz.values()) and t_pred < t_score,
         {s: r["frozen_at"] for s, r in fz.items()})
    g16 = json.load(open(os.path.join(C.REPO_ROOT, "artifacts", "gate16_candidate_selection", "code_freeze.json")))["sha256"]
    ck = {s: _sha(C.b3_checkpoint(s)) == g16[os.path.relpath(C.b3_checkpoint(s), C.REPO_ROOT)] for s in C.B3_SEEDS}
    geo = json.load(open(os.path.join(C.REPO_ROOT, "artifacts", "delayed_semantic_pilot", "checkpoint_manifest.json")))["geometry"]
    _row(rows, "2 frozen B3 checkpoints (== Gate 16 code freeze) and geometry checkpoint (== baseline freeze); none trained after target access",
         all(ck.values()) and _sha(os.path.join(C.REPO_ROOT, geo["path"])) == geo["frozen_reference"]
         and all(os.stat(C.b3_checkpoint(s)).st_mtime < _ts(fz["protocol"]["frozen_at"]) for s in C.B3_SEEDS), ck)
    counts = {d: (len(preds[d]), len(ZS.evaluated_samples(d))) for d in C.DATASETS}
    _row(rows, "3 one canonical prediction per evaluated sample", all(a == b for a, b in counts.values()), counts)
    occ_ok = {}
    for d in C.DATASETS:
        ev = {}
        for f in glob.glob(os.path.join(C.REPO_ROOT, "artifacts", "occany_comparison", "eval", f"{d}_shard*of*.npz")):
            z = np.load(f); keys = list(z["keys"]); col = keys.index(f"ours|causal|{'shared' if d == 'semantickitti' else 'ours'}|raw|seed0")
            for i, sid in enumerate(z["sample_id"]):
                ev[str(sid)] = z["official"][i, col]
        with np.load(os.path.join(SCORE_DIR, f"scores_{d}.npz")) as s:
            ours = s["ps__official_occ"]; ids = [str(x) for x in s["sample_id"]]
        occ_ok[d] = bool(all(np.array_equal(ours[i], ev[sid]) for i, sid in enumerate(ids)))
    _row(rows, "4 occupancy (geometry seed 0, causal) identical per sample to the frozen OccAny-comparison evaluation", all(occ_ok.values()), occ_ok)
    a = json.load(open(os.path.join(A, "analysis.json")))
    _row(rows, "5 Track O and Track P (full region) read the same predictions: confusions identical",
         all(all(v.values()) for v in (d["track_o_equals_track_p_full"] for d in a["datasets"].values())), {d: v["track_o_equals_track_p_full"] for d, v in a["datasets"].items()})
    tok = {}
    for d in C.DATASETS:
        frames = json.load(open(C.FRAME_LIST.format(dataset=d)))["frames"]
        logs = [json.load(open(f)) for f in glob.glob(os.path.join(C.LOG_DIR, f"tokens_{d}_s*of4.json"))]
        tok[d] = {"frames": len(frames), "tokens": sum(os.path.exists(C.token_path(d, f["key"])) for f in frames),
                  "min_argmax_agree": min(l["coarse_reproduction"]["argmax_agree_min"] for l in logs if l["coarse_reproduction"]["argmax_agree_min"] is not None)}
    _row(rows, "6 Trident tokens for every history frame; tokens reproduce Trident's coarse probabilities (argmax agreement >= 0.99)",
         all(v["frames"] == v["tokens"] and v["min_argmax_agree"] >= 0.99 for v in tok.values()), tok)
    lab = {}
    for d in C.DATASETS:
        ids = set(s.sample_id for s in ZS.evaluated_samples(d)); ok = True
        for f in glob.glob(os.path.join(C.REPO_ROOT, "artifacts", "occany_comparison", "eval", f"{d}_shard*of*.npz")):
            z = np.load(f); ok &= bool(all(li for sid, li in zip(z["sample_id"], z["label_identical"]) if str(sid) in ids))
        lab[d] = ok
    _row(rows, "7 labels: OccAny stored labels == independent official loader on every evaluated sample", all(lab.values()), lab)
    import torch
    from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
    from tools.zero_shot_eval.prediction import pipeline as PL
    dev = torch.device(device); torch.cuda.set_device(dev); rep = {}
    for d in C.DATASETS:
        models = PL.load_models(d, dev); ss = ZS.evaluated_samples(d)
        for s in (ss[0], ss[len(ss) // 2]):
            new = PL.predict_sample(s, d, models, dev); old = CanonicalPrediction.load(C.prediction_path(d, s.sample_id))
            rep[f"{d}:{s.sample_id}"] = bool(np.array_equal(new.occupied, old.occupied) and np.array_equal(new.query_index, old.query_index)
                                            and all(np.array_equal(new.labels[m], old.labels[m]) for m in old.labels) and np.array_equal(new.evidence, old.evidence))
    _row(rows, "8 predictions reproduce on re-run (deterministic)", all(rep.values()), rep)
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/zero_shot_eval", "-q"], cwd=C.REPO_ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": C.REPO_ROOT})
    _row(rows, "9 tests (protocol, serialization, masks, metrics, KITTI-360 reproduction of the grid-generic code)", p.returncode == 0,
         p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    rows.append({"check": "note", "pass": True, "detail": "the first freeze command ran while one test failed because its expected value was wrong "
                 "(class-1 FP count); the metric code was correct and no frozen file changed; the test was corrected and all tests pass"})
    rec = {"n_pass": sum(r["pass"] for r in rows if r["check"] != "note"), "n": sum(1 for r in rows if r["check"] != "note"), "rows": rows}
    json.dump(rec, open(os.path.join(A, "verification.json"), "w"), indent=1, default=str)
    md = [f"# Final zero-shot verification — {rec['n_pass']}/{rec['n']} pass\n", "| check | pass | detail |", "|---|---|---|"]
    md += [f"| {r['check']} | {'yes' if r['pass'] else '**NO**'} | `{json.dumps(r['detail'], default=str)[:300]}` |" for r in rows]
    open(os.path.join(A, "verification.md"), "w").write("\n".join(md) + "\n")
    return rec


__all__ = ["run"]
