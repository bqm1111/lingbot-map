#!/usr/bin/env python
"""Gate 8B Stage 3 for one fold: pick {best,last} and one global threshold from the fold's
two **source-validation** domains, and freeze them before the target is opened.

Exactly Gate 8A's rule (``tools/gate8a/selection.py``) with the fold's own source pair:
macro-average AP over the two source-validation domains, AP > prevalence on each, then
one global final-logit threshold maximising the equally weighted macro full-grid binary
IoU; the same threshold search on the mapper's own log-odds. No semantic ground truth, no
target data of any kind.

    python tools/gate8b/select_fold.py --fold semantickitti
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402
from gates.gate8b.sources import FOLDS                                                 # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_block(source, tag, method, region):
    z = np.load(os.path.join(ART, f"scores_{source}_{tag}_{method}.npz"), allow_pickle=False)
    return {k[len(region) + 1:]: z[k] for k in z.files if k.startswith(region + "_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True, choices=["semantickitti", "occ3d"])
    a = ap.parse_args()
    fold = FOLDS[a.fold]; SOURCES = tuple(fold["val"]); target = fold["target"]
    assert target not in SOURCES
    tag = f"src_{a.fold}"
    ck = {f"fold_{a.fold}_{k}": f"artifacts/gate8b/checkpoints/fold_{a.fold}_{k}.pt"
          for k in ("best", "last")}
    t0 = time.time(); rows, blocks = [], {}
    for name in list(ck) + ["mapper"]:
        r = {"candidate": name, "checkpoint": ck.get(name), "sources": {}}
        blocks[name] = {}
        for src in SOURCES:
            for reg in ("full", "edit"):
                blocks[name][(src, reg)] = load_block(src, tag, name, reg)
            b = blocks[name][(src, "full")]
            sw = SC.sweep(b["pos"], b["neg"])
            prev = float(sw["n_pos"] / sw["n_tot"]); apv = SC.average_precision(sw)
            r["sources"][src] = {"average_precision": apv, "prevalence": prev,
                                 "ap_over_prevalence": apv / prev, "auroc": SC.auroc(sw),
                                 "n_anchors": len(b["pos"])}
        r["macro_ap"] = float(np.mean([r["sources"][s]["average_precision"] for s in SOURCES]))
        r["ap_exceeds_prevalence_on_both"] = all(
            r["sources"][s]["average_precision"] > r["sources"][s]["prevalence"] for s in SOURCES)
        rows.append(r)

    def tau_for(name):
        sws = [SC.sweep(blocks[name][(s, "full")]["pos"], blocks[name][(s, "full")]["neg"])
               for s in SOURCES]
        iou = np.mean([sw["iou"] for sw in sws], axis=0); j = int(np.argmax(iou))
        tau = float(sws[0]["tau"][j])
        per = {s: SC.at_threshold(sw, tau) for s, sw in zip(SOURCES, sws)}
        ed = {s: SC.at_threshold(SC.sweep(blocks[name][(s, "edit")]["pos"],
                                          blocks[name][(s, "edit")]["neg"]), tau) for s in SOURCES}
        return tau, float(iou[j]), per, ed

    for r in rows:
        t, mi, pf, pe = tau_for(r["candidate"])
        r.update({"own_macro_threshold": t, "own_macro_iou": mi, "full_at_own_threshold": pf,
                  "edit_at_own_threshold": pe,
                  "at_zero": {s: SC.at_threshold(SC.sweep(blocks[r["candidate"]][(s, "full")]["pos"],
                                                          blocks[r["candidate"]][(s, "full")]["neg"]),
                                                 0.0) for s in SOURCES}})
    learned = [r for r in rows if r["candidate"] != "mapper"]
    eligible = [r for r in learned if r["ap_exceeds_prevalence_on_both"]]
    assert eligible, "no candidate clears AP > prevalence on both source-validation domains"
    win = max(eligible, key=lambda r: r["macro_ap"])
    tau, macro_iou, per_full, per_edit = tau_for(win["candidate"])
    m_tau, m_macro, m_full, m_edit = tau_for("mapper")
    tr = json.load(open(os.path.join(ART, f"train_fold_{a.fold}.json")))
    sel = {"fold": a.fold, "target": target, "sources": list(SOURCES), "tag": tag,
           "candidates": rows,
           "rule": {"checkpoint": "max macro-average AP over the fold's two source-validation "
                                  "domains, subject to AP > prevalence on both",
                    "threshold": "single global final-logit threshold maximising the equally "
                                 "weighted mean of the two per-source pooled full-grid binary IoUs",
                    "forbidden": f"{target} data, labels, prevalence, metrics or predictions "
                                 f"play no part in any of the above"},
           "selected": {"candidate": win["candidate"], "checkpoint": ck[win["candidate"]],
                        "checkpoint_sha256": sha256(os.path.join(REPO_ROOT, ck[win["candidate"]])),
                        "macro_ap": win["macro_ap"], "threshold": tau,
                        "macro_full_iou_at_threshold": macro_iou,
                        "full_at_threshold": per_full, "edit_at_threshold": per_edit},
           "mapper_calibrated": {"threshold": m_tau, "macro_full_iou_at_threshold": m_macro,
                                 "full_at_threshold": m_full, "edit_at_threshold": m_edit},
           "seconds": time.time() - t0}
    write_json(os.path.join(ART, f"selection_{a.fold}.json"), sel)
    frozen = {"gate": "8B", "fold": a.fold, "target": target,
              "train_sources": fold["train"], "val_sources": list(SOURCES),
              "kitti360_train_drives": ["2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync",
                                        "2013_05_28_drive_0010_sync"],
              "kitti360_val_drive": "2013_05_28_drive_0006_sync",
              "selected_candidate": win["candidate"], "checkpoint": ck[win["candidate"]],
              "checkpoint_sha256": sel["selected"]["checkpoint_sha256"],
              "global_threshold_final_logodds": tau,
              "mapper_calibrated_threshold_logodds": m_tau,
              "selection_inputs": [f"artifacts/gate8b/scores_{s}_{tag}_*.npz" for s in SOURCES],
              "selection_criterion": sel["rule"], "hyperparameters": tr["config"],
              "config": tr["config_path"], "config_sha256": sha256(os.path.join(REPO_ROOT, tr["config_path"])),
              "target_used_in_selection": False}
    cfg_path = os.path.join(REPO_ROOT, "configs", "gate8b", f"frozen_fold_{a.fold}.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"# FROZEN before the target ({target}) was opened. Written by "
                "tools/gate8b/select_fold.py.\n# Nothing below may be re-tuned after the "
                "held-out evaluation.\n")
        yaml.safe_dump(frozen, f, sort_keys=False)
    write_json(os.path.join(ART, f"frozen_manifest_{a.fold}.json"),
               dict(frozen, frozen_sha256=sha256(cfg_path), frozen_at=time.strftime("%FT%T%z"),
                    all_candidate_checkpoints={k: sha256(os.path.join(REPO_ROOT, p))
                                               for k, p in ck.items()}))
    print(f"== Gate 8B fold {a.fold}: source-only selection on {', '.join(SOURCES)}")
    for r in rows:
        s = "  ".join(f"{k}: AP {r['sources'][k]['average_precision']:.4f} "
                      f"(prev {r['sources'][k]['prevalence']:.4f}, x{r['sources'][k]['ap_over_prevalence']:.2f})"
                      for k in SOURCES)
        print(f"  {r['candidate']:26s} macroAP {r['macro_ap']:.4f}  {s}  tau* {r['own_macro_threshold']:+.3f} "
              f"macroIoU {r['own_macro_iou']:.4f}" + ("" if r["ap_exceeds_prevalence_on_both"] else "  [REJECTED]"))
    print(f"  SELECTED {win['candidate']}  tau {tau:+.4f}  macro full IoU {macro_iou:.4f}")
    print(f"  mapper calibrated tau {m_tau:+.4f}  macro full IoU {m_macro:.4f}\n  frozen -> {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
