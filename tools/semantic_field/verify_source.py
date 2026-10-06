"""Source-phase verification for the semantic-field experiment, recomputed from files. Writes artifacts/semantic_field/verification.{json,md}."""
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

from tools.delayed_semantic import analysis as PA, config as PILOT, rules as PILOT_RULES
from tools.delayed_semantic.gate16 import config as G16, rules as G16_RULES
from tools.semantic_field import freeze
from tools.semantic_field.config.schema import ARTIFACT_DIR, REPO_ROOT, SEEDS, Paths, TrainConfig
from tools.semantic_field.evaluation.analysis import PRIMARY, load

A = ARTIFACT_DIR


def _row(rows: List[Dict], name: str, ok: bool, detail) -> None:
    rows.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, str(detail)[:300], flush=True)


def _ts(s: str) -> float:
    return _dt.datetime.fromisoformat(s).timestamp()


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run() -> Dict:
    rows: List[Dict] = []; paths = Paths()
    fz = {s: freeze.check(s) for s in ("source_predeclaration", "centering", "source_code")}
    ckpts = sorted(glob.glob(os.path.join(paths.checkpoints, "B4_seed*_step*.pt")))
    t_ckpt = min(os.stat(p).st_mtime for p in ckpts); t_eval = min(os.stat(p).st_mtime for p in glob.glob(os.path.join(paths.eval, "source_eval_shard*.npz")))
    order = {"predeclaration < centering < first checkpoint": _ts(fz["source_predeclaration"]["frozen_at"]) < _ts(fz["centering"]["frozen_at"]) < t_ckpt,
             "source_code frozen before evaluation": _ts(fz["source_code"]["frozen_at"]) < t_eval,
             "no checkpoint at predeclaration / centering freeze": fz["source_predeclaration"]["existing"]["checkpoints"] == 0 and fz["centering"]["existing"]["checkpoints"] == 0}
    _row(rows, "1 frozen stages unchanged and in order", all(order.values()), order)
    pilot = [PILOT_RULES.check(s)["stage"] for s in ("predeclaration", "reliability", "addendum_A", "decision")]
    g16 = [G16_RULES.check(s)["stage"] for s in ("predeclaration", "ceiling", "code")]
    _row(rows, "2 delayed-target rules (pilot) and Gate 16 candidate rules unchanged", len(pilot) == 4 and len(g16) == 3, {"pilot": pilot, "gate16": g16})
    geo = json.load(open(os.path.join(PILOT.ARTIFACT_DIR, "checkpoint_manifest.json")))["geometry"]
    g16_freeze = json.load(open(os.path.join(G16.ARTIFACT_DIR, "code_freeze.json")))["sha256"]
    b3_ok = all(_sha(G16.head_path("B3", s)) == g16_freeze[os.path.relpath(G16.head_path("B3", s), REPO_ROOT)] for s in SEEDS)
    _row(rows, "3 geometry checkpoint == frozen baseline; B3 checkpoints == Gate 16 freeze",
         _sha(os.path.join(REPO_ROOT, geo["path"])) == geo["frozen_reference"] and b3_ok, {"geometry": geo["path"], "B3": b3_ok})
    mu_ok = _sha(paths.feature_mean) == fz["centering"]["payload"]["mu_sha256"]
    _row(rows, "4 centering mean unchanged since its freeze (training drives only)", mu_ok, {k: fz["centering"]["payload"][k] for k in ("n_voxels", "per_drive")})
    runs = {s: json.load(open(os.path.join(paths.runs, f"train_B4_seed{s}.json"))) for s in SEEDS}
    run_ok = {s: r["steps"] == TrainConfig().steps and r["checkpoints"][-1]["step"] == TrainConfig().steps and not r["firewall"]["violations"]
                 and _sha(os.path.join(REPO_ROOT, r["checkpoints"][-1]["path"])) == r["checkpoints"][-1]["sha256"] for s, r in runs.items()}
    geo_digests = {r["geometry_digest"] for r in runs.values()}
    _row(rows, "5 three B4 seeds, final-step checkpoints hashed, identical frozen geometry digest, no training firewall violation (labels, LiDAR, target datasets)",
         all(run_ok.values()) and len(geo_digests) == 1, {"seeds": run_ok, "params": runs[0]["params"]})
    evals = [json.load(open(f[:-4] + ".json")) for f in sorted(glob.glob(os.path.join(paths.eval, "source_eval_shard*.npz")))]
    targets_untouched = not glob.glob(os.path.join(paths.predictions, "semantickitti", "*")) and not glob.glob(os.path.join(paths.predictions, "occ3d", "*"))
    _row(rows, "6 no target dataset accessed before promotion (evaluation audits clean, no target predictions)",
         all(not e["firewall"]["violations"] for e in evals) and targets_untouched, {"shards": len(evals)})
    z = load(paths); ref = PA.gate15_reference(z["ids"])
    _row(rows, "7 occupancy TP/FP/FN identical to Gate 15 seed 0 on every anchor", bool(np.array_equal(z["occ"], ref["occ_seed0"])), {"anchors": len(z["ids"])})
    g16_parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(G16.EVAL, "eval_shard*of*.npz")))]
    gid = np.concatenate([p["ids"] for p in g16_parts]); order_ = np.argsort(gid); pos = np.searchsorted(gid[order_], z["ids"])
    b3_rep = {s: bool(np.array_equal(np.concatenate([p[f"conf__B3_s{s}__{PRIMARY}"] for p in g16_parts])[order_][pos], z[f"conf__B3_s{s}__{PRIMARY}"])) for s in SEEDS}
    _row(rows, "8 B0 reproduces Gate 15 and B3 reproduces Gate 16 per anchor (primary region)",
         bool(np.array_equal(z[f"conf__B0__{PRIMARY}"], ref["b0_primary"])) and all(b3_rep.values()), {"B3": b3_rep})
    _row(rows, "9 evidence path identical across methods on every anchor", bool(z["evidence_path_equal"].all()), {})
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/semantic_field", "tests/delayed_semantic", "-q"], cwd=REPO_ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": REPO_ROOT})
    _row(rows, "10 tests: B4 generative output, masking, centering, firewalls, frozen geometry; pilot/Gate 16 causality and prefix invariance",
         p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    rec = {"n_pass": sum(r["pass"] for r in rows), "n": len(rows), "rows": rows}
    json.dump(rec, open(os.path.join(A, "verification.json"), "w"), indent=1, default=str)
    md = [f"# Semantic-field source verification — {rec['n_pass']}/{rec['n']} pass\n", "| check | pass | detail |", "|---|---|---|"]
    md += [f"| {r['check']} | {'yes' if r['pass'] else '**NO**'} | `{json.dumps(r['detail'], default=str)[:300]}` |" for r in rows]
    open(os.path.join(A, "verification.md"), "w").write("\n".join(md) + "\n")
    return rec


__all__ = ["run"]
