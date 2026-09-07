#!/usr/bin/env python
"""Gate 8A Stage 3: pick one cell, one checkpoint and one global threshold, from the two
**source** validation sets only, and freeze them before KITTI-360 is ever opened.

The rules, in the order the brief states them:

1. score every candidate on the full SemanticKITTI 08 and Occ3D validation grids -- not on
   the training crops, whose occupancy prior is a property of the sampler under test;
2. select by macro-average AP over the two sources (geometry only -- AP needs no semantic
   label and no threshold);
3. refuse any candidate whose AP does not exceed occupancy prevalence on **both** sources;
4. with the winner fixed, choose the single global final-logit threshold maximising the
   macro-average full-grid binary IoU, both sources weighted equally;
5. one threshold, not one per dataset;
6. no semantic ground truth anywhere in 1-5.

The same threshold search is run for the incremental-mapper log-odds so that the learned
completion is compared against a *calibrated* mapper rather than a straw one.

    python tools/gate8a/selection.py
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")
SOURCES = ("semantickitti", "occ3d")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_block(source, tag, method, region):
    z = np.load(os.path.join(ART, f"scores_{source}_{tag}_{method}.npz"), allow_pickle=False)
    return {k[len(region) + 1:]: z[k] for k in z.files if k.startswith(region + "_")}


def macro_threshold(sweeps):
    """The single tau maximising the equally weighted mean of the per-source pooled IoU."""
    iou = np.mean([s["iou"] for s in sweeps], axis=0)
    j = int(np.argmax(iou))
    return float(sweeps[0]["tau"][j]), float(iou[j]), j


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="source")
    ap.add_argument("--checkpoints", required=True, help="name=path[,name=path...]")
    ap.add_argument("--out", default="frozen")
    a = ap.parse_args()
    ck = dict(kv.split("=", 1) for kv in a.checkpoints.split(",") if kv)
    t0 = time.time()
    rows, blocks = [], {}
    for name in list(ck) + ["mapper"]:
        r = {"candidate": name, "checkpoint": ck.get(name), "sources": {}}
        blocks[name] = {}
        for src in SOURCES:
            for reg in ("full", "edit"):
                blocks[name][(src, reg)] = load_block(src, a.tag, name, reg)
            b = blocks[name][(src, "full")]
            sw = SC.sweep(b["pos"], b["neg"])
            prev = float(sw["n_pos"] / sw["n_tot"])
            apv = SC.average_precision(sw)
            r["sources"][src] = {"average_precision": apv, "prevalence": prev,
                                 "ap_over_prevalence": apv / prev,
                                 "auroc": SC.auroc(sw), "n_anchors": len(b["pos"])}
        r["macro_ap"] = float(np.mean([r["sources"][s]["average_precision"] for s in SOURCES]))
        r["ap_exceeds_prevalence_on_both"] = all(
            r["sources"][s]["average_precision"] > r["sources"][s]["prevalence"] for s in SOURCES)
        rows.append(r)

    learned = [r for r in rows if r["candidate"] != "mapper"]
    eligible = [r for r in learned if r["ap_exceeds_prevalence_on_both"]]
    assert eligible, "no candidate clears AP > prevalence on both sources"
    win = max(eligible, key=lambda r: r["macro_ap"])

    def tau_for(name):
        sws = [SC.sweep(blocks[name][(s, "full")]["pos"], blocks[name][(s, "full")]["neg"])
               for s in SOURCES]
        tau, macro_iou, j = macro_threshold(sws)
        per = {s: {"iou": float(sw["iou"][j]), "precision": float(sw["precision"][j]),
                   "recall": float(sw["recall"][j]), "density": float(sw["density"][j]),
                   "pred_over_gt": float((sw["tp"][j] + sw["fp"][j]) / sw["n_pos"])}
               for s, sw in zip(SOURCES, sws)}
        ed = {}
        for s in SOURCES:
            b = blocks[name][(s, "edit")]
            sw = SC.sweep(b["pos"], b["neg"])
            ed[s] = SC.at_threshold(sw, tau)
        return tau, macro_iou, per, ed

    tau, macro_iou, per_full, per_edit = tau_for(win["candidate"])
    m_tau, m_macro, m_full, m_edit = tau_for("mapper")
    for r in rows:
        t, mi, pf, pe = tau_for(r["candidate"])
        r["own_macro_threshold"] = t
        r["own_macro_iou"] = mi
        r["full_at_own_threshold"] = pf
        r["edit_at_own_threshold"] = pe
        r["at_zero"] = {s: SC.at_threshold(
            SC.sweep(blocks[r["candidate"]][(s, "full")]["pos"],
                     blocks[r["candidate"]][(s, "full")]["neg"]), 0.0) for s in SOURCES}

    sel = {"tag": a.tag, "sources": list(SOURCES), "candidates": rows,
           "rule": {"checkpoint": "max macro-average AP over the two source validation "
                                  "sets, subject to AP > prevalence on both",
                    "threshold": "single global final-logit threshold maximising the "
                                 "equally weighted mean of the two per-source pooled "
                                 "full-grid binary IoUs",
                    "forbidden": "KITTI-360 data, labels, prevalence, metrics or "
                                 "predictions play no part in any of the above"},
           "selected": {"candidate": win["candidate"], "checkpoint": ck[win["candidate"]],
                        "checkpoint_sha256": sha256(os.path.join(REPO_ROOT, ck[win["candidate"]])
                                                    if not os.path.isabs(ck[win["candidate"]])
                                                    else ck[win["candidate"]]),
                        "macro_ap": win["macro_ap"], "threshold": tau,
                        "macro_full_iou_at_threshold": macro_iou,
                        "full_at_threshold": per_full, "edit_at_threshold": per_edit},
           "mapper_calibrated": {"threshold": m_tau, "macro_full_iou_at_threshold": m_macro,
                                 "full_at_threshold": m_full, "edit_at_threshold": m_edit},
           "seconds": time.time() - t0}
    write_json(os.path.join(ART, "selection.json"), sel)

    frozen = {"gate": "8A", "selected_candidate": win["candidate"],
              "checkpoint": ck[win["candidate"]],
              "checkpoint_sha256": sel["selected"]["checkpoint_sha256"],
              "global_threshold_final_logodds": tau,
              "mapper_calibrated_threshold_logodds": m_tau,
              "selection_inputs": [f"artifacts/gate8a/scores_{s}_{a.tag}_*.npz"
                                   for s in SOURCES],
              "selection_criterion": sel["rule"],
              "kitti360_used_in_selection": False}
    # candidate names are "<cell>_<best|last>"; the training record is named after the
    # cell's full config tag ("cellB_uniform_focal"), so match on the cell prefix
    cell = win["candidate"].split("_")[0]
    tr = sorted(glob.glob(os.path.join(REPO_ROOT, "artifacts", "gate8a",
                                       f"train_{cell}_*.json")))
    if tr:
        j = json.load(open(tr[0]))
        frozen.update({"sampler": j["sampler"], "occ_loss": j["occ_loss"],
                       "config": j["config_path"]})
    else:                                     # cell A is the reused Gate 8 run
        frozen.update({"sampler": "occ_centred", "occ_loss": "focal_dice",
                       "config": "configs/gate8/completion.yaml",
                       "note": "cell A checkpoint reused from Gate 8, not retrained"})
    cfg_path = os.path.join(REPO_ROOT, "configs", "gate8a", f"{a.out}_selection.yaml")
    with open(cfg_path, "w") as f:
        f.write("# FROZEN before KITTI-360 was opened. Written by tools/gate8a/selection.py.\n"
                "# Nothing below may be re-tuned after the held-out evaluation.\n")
        yaml.safe_dump(frozen, f, sort_keys=False)
    write_json(os.path.join(ART, "frozen_manifest.json"),
               dict(frozen, config_sha256=sha256(cfg_path), frozen_at=time.strftime("%FT%T%z"),
                    all_candidate_checkpoints={k: sha256(os.path.join(REPO_ROOT, p)
                                                         if not os.path.isabs(p) else p)
                                               for k, p in ck.items()}))
    print(f"== Gate 8A source-only selection ({', '.join(SOURCES)})")
    for r in rows:
        s = "  ".join(f"{k}: AP {r['sources'][k]['average_precision']:.4f} "
                      f"(prev {r['sources'][k]['prevalence']:.4f}, "
                      f"x{r['sources'][k]['ap_over_prevalence']:.2f})" for k in SOURCES)
        print(f"  {r['candidate']:24s} macroAP {r['macro_ap']:.4f}  {s}  "
              f"tau* {r['own_macro_threshold']:+.3f} macroIoU {r['own_macro_iou']:.4f}"
              + ("" if r["ap_exceeds_prevalence_on_both"] else "   [REJECTED: AP <= prevalence]"))
    print(f"  SELECTED {win['candidate']}  tau {tau:+.4f}  macro full IoU {macro_iou:.4f}")
    print(f"  mapper calibrated tau {m_tau:+.4f}  macro full IoU {m_macro:.4f}")
    print(f"  frozen -> {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
