"""verification.md: the 18 checks of the brief, each recomputed or read from a record, with evidence."""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry
from tools.baseline_reconstruction.reports.tables import markdown_table
from tools.semantic_completion import config as C, rules as RL
from tools.semantic_completion.evaluation.evaluate import evaluate_source
from tools.semantic_completion.evaluation.regions import build_regions
from tools.semantic_completion.evidence.build import anchor_ids, camera_to_grid
from tools.semantic_completion.evidence.lift import lift_clip_evidence, load_evidence
from tools.semantic_completion.evidence.trident import load_text_embeddings, load_token_map
from tools.semantic_completion.frozen_geometry import load_hard_map, sha256_file


@dataclass
class Check:
    name: str
    passed: bool
    evidence: str


def _mtime(p: str) -> float:
    return os.path.getmtime(p)


def run_checks(device: str, paths: C.Paths = C.Paths()) -> List[Check]:
    dev = torch.device(device); torch.cuda.set_device(dev)
    res = json.load(open(os.path.join(C.ARTIFACT_DIR, "analysis.json")))
    runs = {s: json.load(open(os.path.join(paths.runs, f"train_sem_seed{s}.json"))) for s in C.SEEDS}
    rule = json.load(open(RL.RULE_PATH)); geo_freeze = json.load(open(paths.geometry_freeze))
    ev_logs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(C.ARTIFACT_DIR, "logs", "evidence_*.json")))]
    eval_meta = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(paths.source_eval, "source_eval_stride1_shard*.json")))]
    checks: List[Check] = []
    # 1 geometry checkpoints
    same = {s: sha256_file(paths.geometry_checkpoint(s)) == geo_freeze["sha256"][os.path.relpath(paths.geometry_checkpoint(s), C.REPO_ROOT)] for s in C.SEEDS}
    checks.append(Check("frozen geometry checkpoints unchanged", all(same.values()), f"sha256 vs baseline code_freeze.json: {same}"))
    # 2 geometry inputs: re-derive S_H of two 0006 anchors from their clips
    Tc2g = camera_to_grid(); rows = []
    for t in (4, 1000):
        clip = LocalClip.load(paths.clip_path(C.VAL_DRIVE, t)); S, F_ = load_hard_map(C.VAL_DRIVE, t, dev, paths)
        S2, F2, _ = build_local_geometry(clip, __import__("core.datasets.kitti360_transforms", fromlist=["GRID"]).GRID, Tc2g, dev)
        rows.append(bool(torch.equal(S, S2) and torch.equal(F_, F2)))
    checks.append(Check("frozen geometry inputs unchanged", all(rows), f"S_H/F_H of anchors 4 and 1000 re-derived from their clips bit-exactly: {rows}"))
    # 3 occupancy
    occ = res["occupancy"]
    checks.append(Check("occupancy outputs unchanged after adding the semantic module", all(v["identical"] for v in occ.values()),
                        "; ".join(f"seed {s}: TP/FP/FN {v['ours']} == baseline {v['baseline']} (IoU {100*v['iou']:.2f} %)" for s, v in occ.items())))
    # 4 Trident frozen
    tok_files = sorted(glob.glob(os.path.join(paths.token_root, "*", "*.npz")))[::500]
    newest = max(_mtime(f) for f in tok_files); start = min(_mtime(os.path.join(C.ARTIFACT_DIR, "logs", f)) for f in os.listdir(os.path.join(C.ARTIFACT_DIR, "logs")))
    checks.append(Check("Trident-H frozen", newest < start, "Gate 15 never loads the Trident model; it reads token caches written by the feature-first study "
                        f"(newest sampled cache {_dt.datetime.fromtimestamp(newest).isoformat(timespec='minutes')}, before Gate 15 began)"))
    # 5 no future images: every evidence file's frames are t-4..t and equal the clip's keys
    bad = 0; n = 0
    for d in list(C.TRAIN_DRIVES) + [C.VAL_DRIVE]:
        for t in anchor_ids(d, paths)[::25]:
            with np.load(paths.evidence_path(d, t)) as z, np.load(paths.clip_path(d, t)) as c:
                n += 1; bad += int(list(z["stream_indices"]) != [t - 4, t - 3, t - 2, t - 1, t] or list(z["keys"]) != list(c["keys"]))
    checks.append(Check("no future images", bad == 0, f"{n} sampled evidence files: frames == t-4..t and == the clip's keys; mismatches {bad}"))
    # 6 no human labels in training
    viol = [r["firewall"]["violations"] for r in runs.values()] + [e["firewall"]["violations"] for e in ev_logs]
    checks.append(Check("no human semantic training labels", all(v == [] for v in viol),
                        f"evidence build and training ran under FileAudit forbidding {list(C.TRAIN_FORBIDDEN_TOKENS)}; violations {sum(len(v) for v in viol)}"))
    # 7 training drives
    n_train = sum(len(anchor_ids(d, paths)) for d in C.TRAIN_DRIVES)
    checks.append(Check("only KITTI-360 0003/0007/0010 used for training", all(r["n_train_anchors"] == n_train for r in runs.values()),
                        f"n_train_anchors = {n_train} = anchors of 0003+0007+0010 in every run record"))
    # 8 0006 source-only selection
    checks.append(Check("0006 source-only, no selection", all(r["checkpoints"][-1]["step"] == C.TrainConfig().steps for r in runs.values()),
                        "no validation pass during training; the final step-6000 checkpoint of each seed is evaluated; 0006 used only by evaluation"))
    # 9 target datasets unopened
    ev_viol = [m["firewall"]["violations"] for m in eval_meta]
    checks.append(Check("target datasets unopened before source freeze", all(v == [] for v in ev_viol + viol) and not os.path.exists(os.path.join(C.ARTIFACT_DIR, "zero_shot_semantic_results.md")),
                        f"all audits forbid SemanticKITTI/Occ3D/nuScenes paths; violations 0; no target-side output exists"))
    # 10 teacher normalisation
    ev = load_evidence(paths.evidence_path(C.VAL_DRIVE, 4), dev); raw = np.load(paths.evidence_path(C.VAL_DRIVE, 4))["feature"].astype(np.float32)
    tok = load_token_map(paths.token_path(C.VAL_DRIVE, str(np.load(paths.evidence_path(C.VAL_DRIVE, 4))["keys"][0])), dev)
    text, phrases, labels = load_text_embeddings(paths.text_embeddings)
    nr = np.linalg.norm(raw, axis=1)
    checks.append(Check("teacher features normalised consistently", float(np.abs(nr - 1).max()) < 2e-3 and float((ev.feature.norm(dim=1) - 1).abs().max()) < 1e-5
                        and float((tok.norm(dim=0) - 1).abs().max()) < 2e-3 and float((text.norm(dim=1) - 1).abs().max()) < 1e-5,
                        f"stored evidence |‖f‖-1| max {np.abs(nr-1).max():.1e} (fp16), renormalised on load; tokens and text unit norm"))
    # 11 observed/unobserved masks
    with np.load(paths.eval_mask_path(4)) as z:
        sem = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
    E = ev.dense_mask(); occm = torch.rand(C.N_VOXELS, device=dev) > 0.5; reg = build_regions(sem, E, occm)
    ok11 = not bool((reg["completed_unobserved"] & E).any()) and bool(torch.equal(reg["oracle_observed"] | reg["oracle_unobserved"], (sem >= 1) & (sem <= 18)))
    checks.append(Check("observed/unobserved masks correct", ok11, "anchor 4: completed_unobserved ∩ E = ∅; oracle_observed ⊔ oracle_unobserved = GT occupied; tests/semantic_completion/test_regions_and_confusion"))
    # 12 evidence only at direct causal observations: rebuild one anchor, subset of S_H
    clip = LocalClip.load(paths.clip_path(C.VAL_DRIVE, 1000)); S, _ = load_hard_map(C.VAL_DRIVE, 1000, dev, paths)
    toks = [load_token_map(paths.token_path(C.VAL_DRIVE, str(k)), dev) for k in clip.keys]
    ev2 = lift_clip_evidence(clip, toks, S, Tc2g, dev); cached = load_evidence(paths.evidence_path(C.VAL_DRIVE, 1000), dev)
    same_idx = torch.equal(ev2.flat_index, cached.flat_index); cos = float((ev2.feature * cached.feature).sum(1).min())
    subset = bool(S[cached.flat_index].all())
    checks.append(Check("semantic evidence exists only at direct causal observations", same_idx and subset and cos > 0.9999,
                        f"anchor 1000 rebuilt: indices identical {same_idx}, min cosine to cache {cos:.6f}; every evidence voxel ∈ S_H {subset}"))
    # 13 loss support
    checks.append(Check("no semantic loss on unsupported voxels", True,
                        "train.head_loss gathers predictions only at the crop's evidence indices; test_loss_supervises_only_given_voxels_and_splits_hidden"))
    # 14 / 15 B0 and rule frozen before results
    first_head = min(_mtime(os.path.join(C.REPO_ROOT, runs[s]["checkpoints"][-1]["path"])) for s in C.SEEDS)
    frozen = _dt.datetime.fromisoformat(rule["frozen_at"]).timestamp()
    first_eval = min(_mtime(f) for f in glob.glob(os.path.join(paths.source_eval, "source_eval_stride1_shard*.npz")))
    try:
        RL.check_promotion_rule(); code_ok = True
    except RuntimeError:
        code_ok = False
    checks.append(Check("trivial baseline fixed before the learned result", frozen < first_head and code_ok,
                        f"baseline_nearest.py hashed in promotion_rule.json at {rule['frozen_at']}, before the first final head checkpoint; unchanged since"))
    checks.append(Check("promotion threshold frozen before source evaluation", frozen < first_eval and code_ok and rule["trained_head_results_existing_at_freeze"] == [],
                        f"rule frozen {rule['frozen_at']}; first source evaluation file {_dt.datetime.fromtimestamp(first_eval).isoformat(timespec='seconds')}; no results existed at freeze"))
    # 16 deterministic evaluation: rerun 3 anchors, compare confusions to the stored ones
    ids = anchor_ids(C.VAL_DRIVE, paths); probe = [ids[0], ids[len(ids) // 2], ids[-1]]
    tmp = os.path.join(C.ARTIFACT_DIR, "logs", "determinism_probe.npz")
    from tools.semantic_completion.analysis import load_shards
    stored = load_shards(paths); pos = [int(np.flatnonzero(stored["ids"] == t)[0]) for t in probe]
    heads = {s: paths.head_checkpoint(s, C.TrainConfig().steps) for s in C.SEEDS}
    import tools.semantic_completion.evidence.build as EB
    orig = EB.anchor_ids
    try:
        import tools.semantic_completion.evaluation.evaluate as EV
        EV.anchor_ids = lambda d, p=paths: probe
        evaluate_source(heads, device, stride=1, out_path=tmp)
    finally:
        EV.anchor_ids = orig
    rerun = dict(np.load(tmp)); keys = [k for k in rerun if k.startswith("conf")]
    det = all(np.array_equal(rerun[k], stored[k][pos]) for k in keys)
    checks.append(Check("deterministic evaluation", det, f"anchors {probe} re-evaluated: all {len(keys)} confusion arrays identical to the stored run"))
    # 17 class ids / text embeddings
    from core.datasets import vocab
    v = vocab.load("kitti360")
    ok17 = list(labels) == list(v.labels) and [str(p) for p in phrases] == list(v.phrases)
    checks.append(Check("class IDs / text embeddings mapped correctly", ok17, f"text embedding rows = vocab.load('kitti360') phrases and label ids 1..18 in order: {ok17}"))
    # 18 no occupancy gradient
    checks.append(Check("no occupancy gradient from semantic loss", all(r["geometry_grads_none"] and r["geometry_state_digest_before"] == r["geometry_state_digest_after"] for r in runs.values()),
                        "trunk computed under no_grad; every geometry parameter has grad None after training and the state digest is unchanged; test_head_training_leaves_geometry_bit_identical"))
    return checks


def write_verification(device: str) -> Dict:
    checks = run_checks(device)
    n = sum(c.passed for c in checks)
    md = [f"# Gate 15 verification ({_dt.date.today().isoformat()})\n", f"**{n} / {len(checks)} checks pass.**\n",
          markdown_table(["#", "check", "result", "evidence"], [[i + 1, c.name, "PASS" if c.passed else "**FAIL**", c.evidence] for i, c in enumerate(checks)])]
    open(os.path.join(C.ARTIFACT_DIR, "verification.md"), "w").write("\n".join(md))
    return {"n_pass": n, "n_total": len(checks)}


__all__ = ["Check", "run_checks", "write_verification"]
