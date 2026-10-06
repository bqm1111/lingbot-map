"""Gate 16 verification, recomputed from files. Writes artifacts/gate16_candidate_selection/verification.{json,md}."""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import inspect
import json
import os
import subprocess
import sys
from typing import Dict, List

from tools.delayed_semantic import config as PC
from tools.delayed_semantic.gate16 import config as G, rules as R

A = G.ARTIFACT_DIR


def _row(rows: List[Dict], name: str, ok: bool, detail) -> None:
    rows.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, str(detail)[:300], flush=True)


def _ts(s: str) -> float:
    return _dt.datetime.fromisoformat(s).timestamp()


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(device: str = "cuda:0") -> Dict:
    import torch
    rows: List[Dict] = []
    an = json.load(open(os.path.join(A, "analysis.json"))); rep = an["reproduction"]
    fz = {s: R.check(s) for s in ("predeclaration", "ceiling", "code")}
    P = PC.Paths(); drives = PC.TRAIN_DRIVES + (PC.DEV_DRIVE,)
    manifest = json.load(open(os.path.join(PC.ARTIFACT_DIR, "data_manifest.json")))["written_by_this_study"]
    count = lambda root, d: len(glob.glob(os.path.join(root, d, "*.npz")))
    caches = {"provenance": all(count(P.provenance_root, d) == manifest["provenance"][d] for d in drives),
              "delayed_targets": all(count(P.targets_root, d) == manifest["delayed_targets"][d] for d in drives),
              "train_cache": all(count(os.path.join(PC.DATA_ROOT, "train_cache", "student"), d) == manifest["train_cache_student"][d] for d in PC.TRAIN_DRIVES)}
    t_pre = _ts(fz["predeclaration"]["frozen_at"])
    newer = [f for d in drives for root in (P.provenance_root, P.targets_root) for f in glob.glob(os.path.join(root, d, "*.npz")) if os.stat(f).st_mtime > t_pre]
    _row(rows, "1 same geometry / evidence / target caches as the pilot (counts; none written after the Gate 16 predeclaration)", all(caches.values()) and not newer,
         {**caches, "files_newer_than_predeclaration": len(newer)})
    ck = json.load(open(os.path.join(PC.ARTIFACT_DIR, "checkpoint_manifest.json")))["geometry"]
    _row(rows, "2 same occupancy checkpoint (sha256 == frozen baseline)", _sha(os.path.join(G.REPO_ROOT, ck["path"])) == ck["frozen_reference"], ck["path"])
    _row(rows, "3 same Trident evidence: evidence path identical, B0 per-anchor confusions == pilot", rep["evidence_path_equal_all"] and all(v for k, v in rep.items() if k.startswith("B0_")),
         {k: v for k, v in rep.items() if k.startswith("B0_")})
    _row(rows, "4 same candidate sets: C0 and oracle == frozen ceiling; B1/B2 per-anchor confusions == pilot (same candidates, same heads)",
         rep["C0_equal_ceiling"] and rep["oracle_equal_ceiling"] and all(v for k, v in rep.items() if k.startswith(("B1_", "B2_"))),
         {k: rep[k] for k in ("C0_equal_ceiling", "oracle_equal_ceiling")})
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/delayed_semantic", "-q"], cwd=G.REPO_ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": G.REPO_ROOT})
    last = p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-200:]
    _row(rows, "5 same delayed targets (rebuild test) / 6 prefix-causal prediction (tests) / test suite", p.returncode == 0, last)
    runs = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(os.path.join(G.RUNS, "train_B3*_seed*.json")))}
    _row(rows, "7 no GT labels in training (TRAIN_FORBIDDEN audit, no violation; eval masks forbidden)",
         len(runs) == 6 and all(not r["firewall"]["violations"] for r in runs.values()) and "eval_masks" in PC.TRAIN_FORBIDDEN, {"runs": len(runs)})
    _row(rows, "8 B0 reproduces the pilot", all(v for k, v in rep.items() if k.startswith("B0_")), "per-anchor confusions, every region")
    evals = sorted(glob.glob(os.path.join(G.EVAL, "eval_shard*.npz"))); t_eval = min(os.stat(f).st_mtime for f in evals)
    ceil = json.load(open(os.path.join(A, "candidate_ceiling.json")))
    _row(rows, "9 candidate ceiling reconstructed and frozen before any B3 score", _ts(fz["ceiling"]["frozen_at"]) < t_eval and fz["ceiling"]["existing_results"]["eval"] == 0,
         {"ceiling_frozen": fz["ceiling"]["frozen_at"]})
    _row(rows, "10 selection threshold frozen before evaluation (payload == ceiling file == analysis)",
         fz["ceiling"]["payload"] == ceil["selection_threshold"] and abs(an["threshold"]["T"] - ceil["selection_threshold"]["T"]) < 1e-12 and _ts(fz["code"]["frozen_at"]) < t_eval,
         {"T": ceil["selection_threshold"]["T"], "code_frozen": fz["code"]["frozen_at"]})
    metas = [json.load(open(f[:-4] + ".json")) for f in evals]
    tokens = ("data/kitti/dataset", "occ3d_gt", "nuscenes", "lingbot_occ3d_zeroshot")
    _row(rows, "11 no target dataset accessed (eval and train audits forbid them, no violations)",
         all(not m["firewall"]["violations"] for m in metas) and all(t in PC.EVAL_FORBIDDEN and t in PC.TRAIN_FORBIDDEN for t in tokens), {"shards": len(metas)})
    _row(rows, "12 occupancy bit-identical to the pilot / Gate 15 seed 0 per anchor", rep["occupancy_equal_pilot"], "TP/FP/FN per anchor")
    heads = {m: torch.load(pth, map_location="cpu", weights_only=False) for m, pth in G.head_methods().items()}
    ks = {m: h["model_config"]["k_candidates"] for m, h in heads.items()}
    _row(rows, "13 candidate count fixed at 16", G.K == 16 and set(ks.values()) == {16}, ks)
    from tools.delayed_semantic.model import EvidenceAssociation
    params = {m: sum(v.numel() for v in h["state_dict"].values()) for m, h in heads.items()}
    _row(rows, "14 no architecture-size increase (B3 == pilot scorer; B3geom smaller)",
         all(v == params["B1"] for m, v in params.items() if m.startswith("B3_")) and all(v < params["B1"] for m, v in params.items() if m.startswith("B3geom")), params)
    from tools.delayed_semantic.gate16 import candidates as CS
    cfgs = {json.dumps(h["model_config"], sort_keys=True) for h in heads.values()}
    src = inspect.getsource(CS.build)
    _row(rows, "15 no candidate radius / search-rule tuning (one ModelConfig for all heads; candidates use model.nearest_candidates with K)",
         len(cfgs) == 1 and "nearest_candidates(voxel_centres(query_flat), evd[\"xyz\"][pool], G.K)" in src and rep["B1_completed_never_observed_equal_pilot"], json.loads(next(iter(cfgs))))
    order = {"predeclaration < first B3 checkpoint": t_pre < min(os.stat(f).st_mtime for f in glob.glob(os.path.join(G.CHECKPOINTS, "B3*.pt"))),
             "predeclaration < ceiling < code < evaluation": t_pre < _ts(fz["ceiling"]["frozen_at"]) < _ts(fz["code"]["frozen_at"]) < t_eval,
             "seed-0 B3 draws == pilot B2 draws": runs["train_B3_seed0.json"]["draw_digest"] == json.load(open(os.path.join(PC.ARTIFACT_DIR, "runs", "train_B2.json")))["draw_digest"]}
    _row(rows, "16 freeze order and matched draws", all(order.values()), order)
    rec = {"n_pass": sum(r["pass"] for r in rows), "n": len(rows), "rows": rows}
    json.dump(rec, open(os.path.join(A, "verification.json"), "w"), indent=1, default=str)
    md = [f"# Gate 16 verification — {rec['n_pass']}/{rec['n']} pass\n", "| check | pass | detail |", "|---|---|---|"]
    md += [f"| {r['check']} | {'yes' if r['pass'] else '**NO**'} | `{json.dumps(r['detail'], default=str)[:300]}` |" for r in rows]
    open(os.path.join(A, "verification.md"), "w").write("\n".join(md) + "\n")
    return rec


__all__ = ["run"]
