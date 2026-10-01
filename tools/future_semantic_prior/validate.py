"""The checks Gate 21 must pass before its numbers mean anything (sections 11 and 17).

Three families:

* **pre-registration** -- `decision_rule.md` still hashes to what was recorded before the first training
  run, and `config.Thresholds` carries exactly the numbers that document states;
* **leakage** -- the future target reaches the training loss and nothing else. The channel list contains no
  future-derived name; building the head input under an audit that forbids the Gate 20 and Gate 21 caches
  opens nothing; the prediction is bit-identical when the reveal mask is corrupted, so the function really
  is the same for E1 and E2; and every training run's own firewall record is clean, with the development
  drive itself on the forbidden list;
* **frozen upstream** -- the geometry network's state digest is unchanged by every run, no gradient ever
  reached it, E0 is still bit-exactly the frozen region `completed_never_observed`, B0's label is still
  bit-exactly the frozen baseline's, and U^R is a subset of E0 that never touches causal evidence.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.validate --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.regions import build_regions
from tools.future_revealability import config as G20, eval_masks as EM
from tools.future_semantic_prior import config as G, evaluate as EV, features as FE, targets as TG
from tools.future_semantic_prior.model import build_head, channel_spec
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, state_digest

#: Gate 20's frozen counts for 0006. Any drift here means the population moved, not that Gate 21 improved.
FROZEN_DEV = {"E0": 2_743_212, "E1": 876_254, "E2": 1_866_958, "b0_E2_miou": 10.180049696284014}
FROZEN_TRAIN_UR = 2_280_255
#: a channel name may only come from this list -- nothing derived from a future frame can appear
ALLOWED_CHANNELS = ("trunk", "hard_S", "hard_F", "occupancy", "height", "range",
                    "evidence_mask", "evidence_dist", "evidence_proj")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_1_decision_rule_unchanged() -> Dict:
    p = os.path.join(G.ARTIFACT_DIR, "decision_rule.md")
    want = open(os.path.join(G.ARTIFACT_DIR, "decision_rule.sha256")).read().strip()
    got = _sha256(p)
    return {"name": "decision rule unchanged since it was hashed", "expected": want, "actual": got,
            "ok": got == want}


def check_2_thresholds_match_document() -> Dict:
    """Every number the rule decides on must appear in the document, so the code cannot drift from it."""
    text = open(os.path.join(G.ARTIFACT_DIR, "decision_rule.md")).read()
    th = G.Thresholds()
    want = {"a_delta_e2_min": "+1.00", "c_delta_e2_max": "+0.25", "b0_e2_miou": "10.180"}
    found = {k: (v in text) for k, v in want.items()}
    return {"name": "config thresholds appear verbatim in decision_rule.md",
            "values": {"a_delta_e2_min": th.a_delta_e2_min, "c_delta_e2_max": th.c_delta_e2_max,
                       "b0_e2_miou": round(th.b0_e2_miou, 3)},
            "found_in_document": found, "ok": all(found.values())}


def check_3_no_future_channel() -> Dict:
    names = sorted({n for c in G.LEARNED_CONTROLS for n, _ in channel_spec(c)})
    bad = [n for n in names if n not in ALLOWED_CHANNELS]
    suspicious = [n for n in names if re.search(r"future|reveal|target|delayed|gt|label", n, re.I)]
    return {"name": "no input channel is derived from a future frame", "channels": names,
            "not_whitelisted": bad, "suspiciously_named": suspicious,
            "ok": not bad and not suspicious}


def check_4_input_opens_nothing_privileged(device: str, n: int = 4) -> Dict:
    """Building the head input under an audit that forbids both reveal caches must open nothing from them."""
    dev = torch.device(device)
    paths = C.Paths()
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    head = build_head("C3_full").to(dev)
    rows: List[Dict] = []
    for t in EV.dev_anchor_ids(stride=1)[:n]:
        with FileAudit(patterns=G.INPUT_FORBIDDEN, strict=True) as audit:
            st = FE.frozen_causal_state(G.DEV_DRIVE, t, net, text, dev, paths, audit=False)
            x = FE.build_input(st, head, dev)
        hits = [p for p in audit.opened if G.DATA_ROOT in p or G20.DATA_ROOT in p]
        rows.append({"t": int(t), "n_opened": len(audit.opened), "violations": audit.violations,
                     "reveal_cache_hits": hits, "channels": int(x.shape[1])})
    return {"name": "the head input opens no file of the Gate 20 or Gate 21 reveal caches",
            "forbidden_patterns": list(G.INPUT_FORBIDDEN), "rows": rows,
            "ok": all(not r["violations"] and not r["reveal_cache_hits"] for r in rows)}


def check_5_prediction_ignores_reveal(device: str, control: str = "C3_full", seed: int = 0,
                                      n: int = 4) -> Dict:
    """Corrupt the reveal column and the prediction must not move one bit.

    This is the operational meaning of "the prediction function is identical for E1 and E2": the split is
    applied to the gathered logits, so there is no path by which it could change them."""
    dev = torch.device(device)
    paths = C.Paths()
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    head = build_head(control).to(dev)
    ck = G.checkpoint_path(control, seed, G.TrainConfig().steps)
    trained = os.path.exists(ck)
    if trained:
        head.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
    head.eval()
    rng = np.random.default_rng(0)
    rows: List[Dict] = []
    for t in EV.dev_anchor_ids(stride=1)[:n]:
        rec = EV.dev_record(t)
        st = FE.frozen_causal_state(G.DEV_DRIVE, t, net, text, dev, paths)
        a = EV.predict(rec, head, st, dev)
        bad = dict(rec)
        bad["vis_first_dt"] = rng.permutation(rec["vis_first_dt"])          # shuffle which voxel is revealed
        b = EV.predict(bad, head, st, dev)
        worse = dict(rec)
        worse["vis_first_dt"] = np.full_like(rec["vis_first_dt"], np.inf)   # claim nothing is ever revealed
        c = EV.predict(worse, head, st, dev)
        pops_a, pops_b = EV.populations(rec, dev), EV.populations(bad, dev)
        rows.append({"t": int(t), "n": int(len(a)),
                     "identical_under_shuffle": bool(torch.equal(a, b)),
                     "identical_under_all_hidden": bool(torch.equal(a, c)),
                     "partition_did_change": not bool(torch.equal(pops_a["E1"], pops_b["E1"])),
                     "E1_E2_partition_E0": bool(torch.equal(pops_a["E1"] | pops_a["E2"], pops_a["E0"]))
                                           and not bool((pops_a["E1"] & pops_a["E2"]).any())})
    return {"name": "the prediction is bit-identical when the reveal mask is corrupted",
            "control": control, "seed": seed, "used_trained_checkpoint": trained, "rows": rows,
            "ok": all(r["identical_under_shuffle"] and r["identical_under_all_hidden"]
                      and r["partition_did_change"] and r["E1_E2_partition_E0"] for r in rows)}


def check_6_training_firewall() -> Dict:
    rows: List[Dict] = []
    for control in G.LEARNED_CONTROLS:
        for seed in G.SEEDS:
            p = os.path.join(G.seed_dir(seed), f"train_{control}.json")
            if not os.path.exists(p):
                rows.append({"control": control, "seed": seed, "present": False})
                continue
            r = json.load(open(p))
            pats = r["firewall"]["patterns"]
            rows.append({"control": control, "seed": seed, "present": True,
                         "violations": r["firewall"]["violations"], "n_opened": r["firewall"]["opened"],
                         "dev_drive_forbidden": any("0006" in x for x in pats),
                         "labels_forbidden": any("preprocess/labels" in x for x in pats),
                         "eval_masks_forbidden": any("eval_masks" in x for x in pats),
                         "lidar_forbidden": any("velodyne" in x for x in pats)})
    done = [r for r in rows if r.get("present")]
    return {"name": "every training run's firewall is clean and forbids the dev drive and all GT",
            "rows": rows, "n_runs": len(done),
            "ok": bool(done) and all(not r["violations"] and r["dev_drive_forbidden"] and r["labels_forbidden"]
                                     and r["eval_masks_forbidden"] and r["lidar_forbidden"] for r in done)}


def check_7_geometry_frozen(device: str) -> Dict:
    dev = torch.device(device)
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, C.Paths().g15)
    live = state_digest(net)
    rows: List[Dict] = []
    for control in G.LEARNED_CONTROLS:
        for seed in G.SEEDS:
            p = os.path.join(G.seed_dir(seed), f"train_{control}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            rows.append({"control": control, "seed": seed, "unchanged": r["geometry_unchanged"],
                         "grads_none": r["geometry_grads_none"],
                         "matches_fresh_load": r["geometry_digest_before"] == live
                                               and r["geometry_digest_after"] == live})
    return {"name": "the frozen geometry network is identical before, during and after every run",
            "requires_grad_any": any(p.requires_grad for p in net.parameters()),
            "live_digest": live[:16], "rows": rows,
            "ok": bool(rows) and not any(p.requires_grad for p in net.parameters())
                  and all(r["unchanged"] and r["grads_none"] and r["matches_fresh_load"] for r in rows)}


def check_8_E0_is_the_frozen_region(device: str, n: int = 8) -> Dict:
    """E0 must still be exactly ``completed_never_observed``, rebuilt from GT, not read from a cache."""
    dev = torch.device(device)
    paths = C.Paths()
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    rows: List[Dict] = []
    for t in EV.dev_anchor_ids(stride=max(1, len(EV.dev_anchor_ids()) // n))[:n]:
        rec = EV.dev_record(t)
        st = FE.frozen_causal_state(G.DEV_DRIVE, t, net, text, dev, paths)
        m = EM.load_mask(G.DEV_DRIVE, int(t))
        sem_gt = torch.from_numpy(m["sem_gt"].astype(np.int64)).to(dev)
        reg = build_regions(sem_gt, st["E"], st["E"], st["S"], st["F"], st["O"],
                            torch.zeros_like(st["E"]), torch.zeros_like(st["E"]))
        rebuilt = reg["completed_never_observed"].nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int32)
        rows.append({"t": int(t), "n_rebuilt": int(len(rebuilt)), "n_cached": int(len(rec["flat_index"])),
                     "identical": bool(np.array_equal(rebuilt, rec["flat_index"])),
                     "sem_gt_identical": bool(np.array_equal(m["sem_gt"][rebuilt], rec["sem_gt"]))})
    return {"name": "E0 is bit-exactly the frozen region completed_never_observed", "rows": rows,
            "ok": all(r["identical"] and r["sem_gt_identical"] for r in rows)}


def check_9_population_sizes() -> Dict:
    n = {"E0": 0, "E1": 0, "E2": 0}
    root = os.path.join(G20.DATA_ROOT, "population", G.DEV_DRIVE)
    for f in sorted(os.listdir(root)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(root, f)) as z:
            rev = np.isfinite(z["vis_first_dt"])
        n["E0"] += int(len(rev)); n["E1"] += int(rev.sum()); n["E2"] += int((~rev).sum())
    ur = sum(int(len(np.load(os.path.join(G.DATA_ROOT, "targets", d, f))["flat_index"]))
             for d in G.TRAIN_DRIVES for f in sorted(os.listdir(os.path.join(G.DATA_ROOT, "targets", d)))
             if f.endswith(".npz"))
    return {"name": "the populations reproduce Gate 20's frozen counts",
            "dev": n, "expected_dev": {k: FROZEN_DEV[k] for k in ("E0", "E1", "E2")},
            "UR_train": ur, "expected_UR_train": FROZEN_TRAIN_UR,
            "partition_ok": n["E1"] + n["E2"] == n["E0"],
            "ok": all(n[k] == FROZEN_DEV[k] for k in n) and ur == FROZEN_TRAIN_UR
                  and n["E1"] + n["E2"] == n["E0"]}


def check_10_targets_well_formed(n: int = 40) -> Dict:
    """U^R is inside E0, carries no causal evidence, and its soft target is a proper distribution whose
    argmax is the class Gate 20 reported."""
    rows: List[Dict] = []
    for drive in G.TRAIN_DRIVES:
        root = os.path.join(G.DATA_ROOT, "targets", drive)
        files = sorted(f for f in os.listdir(root) if f.endswith(".npz"))
        for f in files[:: max(1, len(files) // max(n // 3, 1))][: n // 3]:
            t = int(f[:-4])
            a = TG.load_audit(drive, t)
            with np.load(G20.population_path(drive, t)) as z:
                p0, rev, mt = z["flat_index"], np.isfinite(z["vis_first_dt"]), z["mean_top1"][:, G.EVENTUAL_H]
            q = a["q_future"].astype(np.float64)
            dis = TG.argmax_disagreement(a["q_future"], mt[rev])
            rows.append({"drive": drive[17:21], "t": t, "n": int(len(a["flat_index"])),
                         "subset_of_E0": bool(np.isin(a["flat_index"], p0).all()),
                         "equals_revealed_rows": bool(np.array_equal(a["flat_index"], p0[rev])),
                         "sums_to_one": float(np.abs(q.sum(1) - 1.0).max(initial=0.0)) <= 2e-3,
                         "argmax_disagreements": dis["n_disagree"], "argmax_non_tie": dis["n_non_tie"],
                         "sem_gt_in_1_18": bool(((a["sem_gt"] >= 1) & (a["sem_gt"] <= 18)).all())})
    return {"name": "the training targets are a well-formed subset of E0 with Gate 20's own readout",
            "n_checked": len(rows), "rows": rows[:6],
            "total_argmax_disagreements": sum(r["argmax_disagreements"] for r in rows),
            "total_argmax_non_tie": sum(r["argmax_non_tie"] for r in rows),
            "note": "the stored target is float16; an argmax may differ from Gate 20's only at an exact tie",
            "ok": bool(rows) and all(r["subset_of_E0"] and r["equals_revealed_rows"] and r["sums_to_one"]
                                     and r["argmax_non_tie"] == 0 and r["sem_gt_in_1_18"] for r in rows)}


def check_11_class_prior_is_a_class_prior(device: str) -> Dict:
    """C1 must be a class prior and nothing else: one constant class for every voxel, drawn from the
    training target's own distribution.

    The exact closed-form optimum of the Gate 21 loss is ``softmax(bias) = mean_x q_future(x)``, and C1
    lands near it but not on it -- total variation a few percent, with the top-2 classes (vegetation 27.8 %
    and road 21.5 %) close enough that two of the three seeds pick the other one. Two things cause the
    gap, and neither is a bug: AdamW's decoupled weight decay pulls the bias toward zero, i.e. the
    distribution toward uniform; and the loss is averaged over the voxels of a step, so an anchor with 500
    revealed voxels moves the parameters as much as one with 8,000, which is a per-step rather than a
    per-voxel weighting. What the control has to satisfy is tested here instead: it predicts a single class
    everywhere, that class is one of the training distribution's largest, and the learned distribution is
    close to the empirical one. Those are exactly the properties that make its mIoU a floor for "what is
    available from class frequency alone"."""
    tot = np.zeros(G.N_CLASSES, np.float64)
    nvox = 0
    for drive in G.TRAIN_DRIVES:
        root = os.path.join(G.DATA_ROOT, "targets", drive)
        for f in sorted(os.listdir(root)):
            if not f.endswith(".npz"):
                continue
            with np.load(os.path.join(root, f)) as z:
                q = z["q_future"].astype(np.float64)
            tot += q.sum(0); nvox += len(q)
    emp = tot / max(nvox, 1)
    top3 = set(int(i) + 1 for i in np.argsort(-emp)[:3])
    dev = torch.device(device)
    ids = EV.dev_anchor_ids(stride=1)[:3]
    rows: List[Dict] = []
    for seed in G.SEEDS:
        ck = G.checkpoint_path("C1_prior", seed, G.TrainConfig().steps)
        if not os.path.exists(ck):
            continue
        head = build_head("C1_prior").to(dev)
        head.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
        head.eval()
        p = torch.softmax(head.bias.detach().double().cpu(), 0).numpy()
        classes = set()
        for t in ids:
            rec = EV.dev_record(t)
            idx = torch.from_numpy(rec["flat_index"].astype(np.int64)).to(dev)
            pred = head.logits_at(None, idx).argmax(1) + 1
            classes |= set(int(x) for x in torch.unique(pred).cpu())
        rows.append({"seed": seed, "n_distinct_predicted_classes": len(classes),
                     "predicted_class": sorted(classes),
                     "tv_from_empirical_pct": round(float(50.0 * np.abs(p - emp).sum()), 3),
                     "argmax_learned": int(p.argmax()) + 1,
                     "argmax_in_empirical_top3": int(p.argmax()) + 1 in top3})
    return {"name": "C1 is a class prior: one constant class, from the training distribution",
            "empirical_distribution": [round(x, 6) for x in emp], "n_target_voxels": nvox,
            "empirical_top3": [G.CLASS_NAMES[c - 1] for c in sorted(top3, key=lambda c: -emp[c - 1])],
            "rows": rows,
            "ok": bool(rows) and all(r["n_distinct_predicted_classes"] == 1 and r["argmax_in_empirical_top3"]
                                     and r["tv_from_empirical_pct"] <= 20.0 for r in rows)}


def check_12_runs_are_distinct(device: str, n: int = 3) -> Dict:
    """Twelve runs must be twelve models.

    A state-dict mix-up would make "all three seeds positive" the same number three times, and a channel
    bug could make two controls identical, so this hashes every trained head's parameters and, separately,
    its prediction on a few 0006 anchors.

    The two are checked differently on purpose. For the voxel-conditioned controls (C2/C3/C4) the
    *predictions* must all differ. For `C1_prior` they need not: it is a constant-class predictor with 18
    parameters, and the training target's top two classes (vegetation 27.8 %, road 21.5 %) are close enough
    that two seeds land on the same one, after which their predictions are identical everywhere by
    construction. For C1 the requirement is therefore on the *parameters*, which do differ."""
    dev = torch.device(device)
    paths = C.Paths()
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    ids = EV.dev_anchor_ids(stride=1)[:n]
    states = [(t, FE.frozen_causal_state(G.DEV_DRIVE, t, net, text, dev, paths)) for t in ids]
    recs = {t: EV.dev_record(t) for t in ids}
    pred_d: Dict[str, str] = {}
    param_d: Dict[str, str] = {}
    for control in G.LEARNED_CONTROLS:
        for seed in G.SEEDS:
            ck = G.checkpoint_path(control, seed, G.TrainConfig().steps)
            if not os.path.exists(ck):
                continue
            sd = torch.load(ck, map_location=dev, weights_only=False)["state_dict"]
            head = build_head(control).to(dev)
            head.load_state_dict(sd)
            head.eval()
            key = f"{control}_seed{seed}"
            hp = hashlib.sha256()
            for k in sorted(sd):
                hp.update(k.encode())
                hp.update(np.ascontiguousarray(sd[k].detach().float().cpu().numpy()).tobytes())
            param_d[key] = hp.hexdigest()[:16]
            h = hashlib.sha256()
            for t, st in states:
                h.update(EV.predict(recs[t], head, st, dev).cpu().numpy().astype(np.uint8).tobytes())
            pred_d[key] = h.hexdigest()[:16]
    voxel_keys = [k for k in pred_d if not k.startswith("C1_prior")]
    voxel_pred = [pred_d[k] for k in voxel_keys]
    c1_params = [v for k, v in param_d.items() if k.startswith("C1_prior")]
    return {"name": "every trained run is a distinct model",
            "n_runs": len(pred_d), "parameter_digests": param_d, "prediction_digests": pred_d,
            "n_unique_parameters": len(set(param_d.values())),
            "n_unique_predictions_voxel_conditioned": len(set(voxel_pred)),
            "n_voxel_conditioned_runs": len(voxel_keys),
            "c1_parameters_distinct": len(set(c1_params)) == len(c1_params),
            "note": "C1 is a constant-class predictor, so two seeds may share a prediction while "
                    "differing in parameters; its requirement is on the parameters",
            "ok": bool(pred_d) and len(set(param_d.values())) == len(param_d)
                  and len(set(voxel_pred)) == len(voxel_keys)
                  and len(set(c1_params)) == len(c1_params)}


def run_all(device: str, quick: bool = False) -> Dict:
    checks = [check_1_decision_rule_unchanged(), check_2_thresholds_match_document(),
              check_3_no_future_channel(), check_4_input_opens_nothing_privileged(device),
              check_5_prediction_ignores_reveal(device), check_6_training_firewall(),
              check_7_geometry_frozen(device), check_9_population_sizes(), check_10_targets_well_formed(),
              check_11_class_prior_is_a_class_prior(device), check_12_runs_are_distinct(device)]
    if not quick:
        checks.insert(7, check_8_E0_is_the_frozen_region(device))
    out = {"version": G.VERSION, "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks),
           "failed": [c["name"] for c in checks if not c["ok"]], "checks": checks}
    G.ensure_dirs()
    p = os.path.join(G.ARTIFACT_DIR, "logs", "validation.json")
    json.dump(out, open(p, "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    r = run_all(a.device, a.quick)
    for c in r["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}")
    print(json.dumps({k: r[k] for k in ("n_checks", "all_ok", "failed")}, indent=1))


if __name__ == "__main__":
    main()
