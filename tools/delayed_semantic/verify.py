"""Verification, recomputed from files (and a few live re-runs). Writes verification.json / verification.md.

 1 frozen stages unchanged              9 occupancy identical to Gate 15 seed 0 per anchor (K6)
 2 freeze order vs result timestamps   10 B0 reproduces Gate 15 B0 seed-0 confusions per anchor
 3 provenance == Gate 15 evidence      11 stored predictions-at-t reproduce; adding queries changes nothing
 4 no firewall violation in any log    12 test suite
 5 reliability gate (declared/strict)  13 live LingBot: future images processed first do not change the clip;
 6 B1/B2/B2s identical draws, final       the frozen mapper reproduces the cached hard map
   checkpoints hashed                  14 targets rebuild bit-identically
 7 evaluation covers the 590 anchors   15 the prediction path does not import target construction
 8 evidence path identical (K3)
"""
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
import torch

from tools.delayed_semantic import config as C, rules as R
from tools.delayed_semantic.analysis import gate15_reference, load_eval
from tools.delayed_semantic.regions import PRIMARY
from tools.semantic_completion.evidence import build as G15B

ART = C.ARTIFACT_DIR
LOGS = os.path.join(ART, "logs")


def _ts(s: str) -> float:
    return _dt.datetime.fromisoformat(s).timestamp()


def _row(rows: List[Dict], name: str, ok: bool, detail) -> None:
    rows.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, detail, flush=True)


def run(device: str = "cuda:0", paths: C.Paths = C.Paths()) -> Dict:
    rows: List[Dict] = []
    dev = torch.device(device); torch.cuda.set_device(dev)
    # 1
    frozen = {}
    try:
        for s in ("predeclaration", "reliability", "addendum_A", "decision"):
            frozen[s] = R.check(s)
        _row(rows, "1 frozen stages unchanged", True, {s: v["frozen_at"] for s, v in frozen.items()})
    except Exception as e:  # noqa: BLE001
        _row(rows, "1 frozen stages unchanged", False, str(e))
    # 2
    drives = C.TRAIN_DRIVES + (C.DEV_DRIVE,)
    t_targets = min(os.stat(f).st_mtime for d in drives for f in glob.glob(os.path.join(paths.targets_root, d, "*.npz")))
    t_ckpt = min(os.stat(f).st_mtime for f in glob.glob(os.path.join(paths.checkpoints, "*.pt")))
    t_eval = min(os.stat(f).st_mtime for f in glob.glob(os.path.join(paths.eval, "eval_shard*.npz")))
    order = {"predeclaration < targets": _ts(frozen["predeclaration"]["frozen_at"]) < t_targets,
             "reliability freeze < targets": _ts(frozen["reliability"]["frozen_at"]) < t_targets,
             "addendum A < first checkpoint": _ts(frozen["addendum_A"]["frozen_at"]) < t_ckpt,
             "decision freeze < first evaluation": _ts(frozen["decision"]["frozen_at"]) < t_eval,
             "no evaluation existed at decision freeze": frozen["decision"]["existing_results"]["eval"] == 0}
    _row(rows, "2 freeze order", all(order.values()), order)
    # 3
    prov = {}
    for d in drives:
        rec = json.load(open(os.path.join(LOGS, f"provenance_{d[17:21]}_s0.json")))
        fc = rec["feature_checks"]
        prov[d[17:21]] = {"files": len(glob.glob(os.path.join(paths.provenance_root, d, "*.npz"))), "anchors": len(G15B.anchor_ids(d)),
                          "feature_max_abs": max(c["max_abs"] for c in fc), "class_agree_min": min(c["class_agree"] for c in fc)}
    _row(rows, "3 provenance replay == Gate 15 evidence (voxels and counts asserted per anchor; features sampled)",
         all(v["files"] == v["anchors"] and v["feature_max_abs"] < 1e-3 and v["class_agree_min"] >= 0.999 for v in prov.values()), prov)
    # 4
    viol = {}
    for f in sorted(glob.glob(os.path.join(LOGS, "*.json")) + glob.glob(os.path.join(paths.runs, "train_*.json")) + glob.glob(os.path.join(paths.eval, "eval_shard*.json"))):
        rec = json.load(open(f))
        if "firewall" in rec:
            viol[os.path.basename(f)] = len(rec["firewall"]["violations"])
    _row(rows, "4 no firewall violation", len(viol) > 0 and not any(viol.values()), {"files": len(viol), "violations": sum(viol.values())})
    # 5
    rel = json.load(open(os.path.join(paths.reliability, "reliability.json")))["gate"]
    _row(rows, "5 reliability gate as declared (strict per-drive reading reported)", rel["pass"],
         {"declared": rel["pass"], "strict_per_drive_incl_dev": rel["G1_strict_pass"], "tiers": rel["delayed_loss_tiers"]})
    # 6
    runs = {m: json.load(open(os.path.join(paths.runs, f"train_{m}.json"))) for m in ("B1", "B2", "B2s")}
    ck_ok = all(hashlib.sha256(open(os.path.join(C.REPO_ROOT, r["checkpoints"][-1]["path"]), "rb").read()).hexdigest() == r["checkpoints"][-1]["sha256"] for r in runs.values())
    same = len({r["draw_digest"] for r in runs.values()}) == 1 and all(r["steps"] == C.TrainConfig().steps for r in runs.values())
    _row(rows, "6 B1/B2/B2s identical observed-term draws and steps; final checkpoints hashed", same and ck_ok,
         {m: {"digest": r["draw_digest"][:12], "steps": r["steps"], "tiers": r["delayed_tiers"]} for m, r in runs.items()})
    # 7-10
    z = load_eval(paths); ref = gate15_reference(z["ids"])
    _row(rows, "7 evaluation covers every 0006 anchor once", np.array_equal(z["ids"], np.asarray(G15B.anchor_ids(C.DEV_DRIVE))), {"anchors": len(z["ids"])})
    _row(rows, "8 evidence path identical across methods (K3)", bool(z["evidence_path_equal"].all()), {"anchors_equal": int(z["evidence_path_equal"].sum())})
    _row(rows, "9 occupancy TP/FP/FN == Gate 15 seed 0 per anchor (K6)", bool(np.array_equal(z["occ"], ref["occ_seed0"])), {"TP_FP_FN": z["occ"].sum(0).tolist()})
    _row(rows, "10 B0 == Gate 15 B0 seed 0 per anchor (primary, ssc_full)",
         bool(np.array_equal(z[f"conf__B0__{PRIMARY}"], ref["b0_primary"]) and np.array_equal(z["conf__B0__ssc_full"], ref["b0_ssc_full"])), {})
    # 11
    from tools.delayed_semantic import evaluate as EV
    from tools.delayed_semantic.stream import DriveStream
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    heads_meta = json.load(open(sorted(glob.glob(os.path.join(paths.eval, "eval_shard*.json")))[0]))["head_checkpoints"]
    heads = {m: EV.load_head(p, dev) for m, p in heads_meta.items()}
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev); stream = DriveStream(C.DEV_DRIVE)
    rep = {}
    for t in z["ids"][::250]:
        t = int(t)
        with np.load(paths.g15.eval_mask_path(t)) as zz:
            G = torch.from_numpy((zz["sem_gt"] >= 1) & (zz["sem_gt"] <= 18)).to(dev)
        a = EV.predict_anchor(stream, t, net, heads, text, G, dev); b = EV.predict_anchor(stream, t, net, heads, text, None, dev)
        with np.load(os.path.join(paths.predictions_root, C.DEV_DRIVE, f"{t:05d}.npz")) as st:
            idx = a["O"].nonzero(as_tuple=True)[0]
            stored = all(np.array_equal(st[f"label_{m}"], a["labels"][m][idx].to(torch.uint8).cpu().numpy()) for m in a["labels"])
        indep = all(torch.equal(a["labels"][m][idx], b["labels"][m][idx]) for m in a["labels"])
        rep[t] = {"stored_record_reproduced": stored, "query_independent": indep}
    _row(rows, "11 predictions-at-t reproduce from the stored record; adding GT-occupied queries changes no prediction on O",
         all(v["stored_record_reproduced"] and v["query_independent"] for v in rep.values()), rep)
    # 12
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/delayed_semantic", "-q"], cwd=C.REPO_ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": C.REPO_ROOT})
    _row(rows, "12 test suite", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    # 13
    rows_live = live_lingbot_check(stream, int(z["ids"][300]), dev, paths)
    _row(rows, "13 live LingBot: future images first leave the clip bit-identical; frozen mapper reproduces the cached hard map",
         rows_live["clip_equal_after_future"] and rows_live["clip_equal_cache"] and rows_live["hard_map_equal"], rows_live)
    # 14
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    from tools.delayed_semantic import targets as TG
    reb = {}
    for d, t in ((C.DEV_DRIVE, int(z["ids"][100])), (C.TRAIN_DRIVES[2], G15B.anchor_ids(C.TRAIN_DRIVES[2])[200])):
        s = DriveStream(d); geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        with np.load(paths.g15.evidence_path(d, t)) as zz:
            E = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); E[torch.from_numpy(zz["flat_index"].astype(np.int64)).to(dev)] = True
        out = TG.build_anchor(s, geo, t, TG.ViewCache(geo, text, C.TargetConfig(), dev), E, C.TargetConfig(), dev)["arrays"]
        with np.load(paths.target_path(d, t)) as st:
            # integer/bool/time arrays exact (NaN = "no second view" compares equal); the fp16 mean distribution is summed on
            # CUDA in a non-deterministic order, so it is compared to 1e-4 with an identical argmax
            exact = all(np.array_equal(st[k], v, equal_nan=True) for k, v in out.items() if k != "prob")
            p0, p1 = st["prob"].astype(np.float32), out["prob"].astype(np.float32)
            reb[f"{d[17:21]}:{t}"] = {"exact_except_prob": exact, "prob_max_abs": float(np.abs(p0 - p1).max()) if len(p0) else 0.0,
                                      "prob_argmax_equal": bool(np.array_equal(p0.argmax(1), p1.argmax(1)))}
    ok14 = all(v["exact_except_prob"] and v["prob_max_abs"] <= 1e-4 and v["prob_argmax_equal"] for v in reb.values())
    _row(rows, "14 delayed targets rebuild (exact; fp16 distribution within 1e-4 with identical argmax)", ok14, reb)
    # 15
    p = subprocess.run([sys.executable, "-c", "import sys, tools.delayed_semantic.evaluate as e; "
                        "print(any(m in sys.modules for m in ('tools.delayed_semantic.targets', 'core.supervision.rawtarget', 'tools.delayed_semantic.reliability')))"],
                       cwd=C.REPO_ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": C.REPO_ROOT})
    _row(rows, "15 prediction/evaluation path does not import target construction", p.stdout.strip() == "False", p.stdout.strip() or p.stderr[-300:])
    rec = {"n_pass": sum(r["pass"] for r in rows), "n": len(rows), "rows": rows}
    json.dump(rec, open(os.path.join(ART, "verification.json"), "w"), indent=1, default=str)
    md = [f"# Verification — {rec['n_pass']}/{rec['n']} pass\n", "| check | pass | detail |", "|---|---|---|"]
    md += [f"| {r['check']} | {'yes' if r['pass'] else '**NO**'} | `{json.dumps(r['detail'], default=str)[:400]}` |" for r in rows]
    open(os.path.join(ART, "verification.md"), "w").write("\n".join(md) + "\n")
    return rec


@torch.no_grad()
def live_lingbot_check(stream, t: int, dev, paths: C.Paths) -> Dict:
    """Run the frozen LingBot on H_t after it has processed FUTURE images (then clean_kv_cache), and fresh."""
    from core.datasets import kitti360 as K3
    from gates.gate7b import replay
    from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry
    from tools.semantic_completion.evidence.build import camera_to_grid
    from tools.semantic_completion.frozen_geometry import load_hard_map
    model, _ = replay.build_model(dev)
    hist = [f.image_path for f in stream.history(t).frames()]
    future = [f.image_path for f in stream.frames[t + 1: t + 6]]
    replay.replay_segment(model, replay.load_images(future), 1, dev)                    # contaminate the state with future images
    after = replay.replay_segment(model, replay.load_images(hist), 1, dev)             # replay_segment cleans the KV cache first
    model2, _ = replay.build_model(dev)
    fresh = replay.replay_segment(model2, replay.load_images(hist), 1, dev)
    keys = ("pred_depth", "pred_depth_conf", "pred_pose_c2w")
    with np.load(paths.g15.clip_path(stream.drive, t)) as z:
        cache_eq = all(np.array_equal(z[k], fresh[k]) for k in keys)
    clip = LocalClip.load(paths.g15.clip_path(stream.drive, t))
    S, F_, _ = build_local_geometry(clip, K3.SSCBENCH_KITTI360_GRID, camera_to_grid(), dev)
    S0, F0 = load_hard_map(stream.drive, t, dev, paths.g15)
    return {"t": t, "clip_equal_after_future": all(np.array_equal(after[k], fresh[k]) for k in keys), "clip_equal_cache": cache_eq,
            "hard_map_equal": bool(torch.equal(S.to(dev), S0) and torch.equal(F_.to(dev), F0)),
            "max_abs_depth_after_vs_fresh": float(np.abs(after["pred_depth"].astype(np.float32) - fresh["pred_depth"].astype(np.float32)).max())}


__all__ = ["run", "live_lingbot_check"]
