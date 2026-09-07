#!/usr/bin/env python
"""Gate 8D Phase 6: freeze the manifest. Only after this may the target firewall lift.

Records the selected checkpoint, its hash, the thresholds, the architecture, the
target-construction digest and the source results. ``tools/gate8d/eval_target.py`` refuses
to start without this file and accepts no checkpoint or threshold override.
"""
from __future__ import annotations
import datetime, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8d import protocol as P                                                 # noqa: E402

MANIFEST = os.path.join(ART, "frozen_manifest.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    sel = json.load(open(os.path.join(ART, "selection.json")))
    proto = json.load(open(os.path.join(ART, "protocol.json")))
    audit = json.load(open(os.path.join(ART, "target_audit.json")))
    win = sel["selected"]
    ck = os.path.join(REPO_ROOT, sel["candidates"][win]["checkpoint"])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
    except Exception:                                                   # noqa: BLE001
        commit = "unknown"
    seeds = {}
    for s in P.SEEDS:
        for which in ("best", "last"):
            k = f"seed{s}_{which}"
            if k in sel["candidates"]:
                p = os.path.join(REPO_ROOT, sel["candidates"][k]["checkpoint"])
                seeds[k] = {"checkpoint": sel["candidates"][k]["checkpoint"],
                            "sha256": sha256(p),
                            "selection_score": sel["candidates"][k]["selection_score"],
                            "occupancy_threshold": sel["candidates"][k]["identity_tau"]}
    man = {"gate": "8D",
           "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "code_commit": commit,
           "claim": proto["claim"],
           "protocol_sha256": P.source_digest(),
           "protocol_json": "artifacts/gate8d/protocol.json",
           "selected": win,
           "checkpoint": sel["candidates"][win]["checkpoint"],
           "checkpoint_sha256": sha256(ck),
           "occupancy_threshold": sel["occupancy_threshold"],
           "semantic_threshold": 0.0,
           "architecture": dict(P.ARCH),
           "target_construction": {"audit": "artifacts/gate8d/target_audit.json",
                                   "accepted": audit["accepted"],
                                   "pseudo_free_collision_rate":
                                       audit["pseudo_free_collision_rate"],
                                   "occupied_recall_vs_gate8c1":
                                       audit["occupied_recall_vs_gate8c1_plain_column"],
                                   "native_sweep_gain": audit["native_sweep_gain_mean"]},
           "source_selection": {"rule": sel["rule"], "weights": sel["weights"],
                                "n_val_samples": sel["n_val_samples"],
                                "per_candidate": {k: {kk: v[kk] for kk in
                                                      ("selection_score", "mean_stress_iou",
                                                       "worst_stress_iou", "identity_tau",
                                                       "pred_over_gt",
                                                       "surface_thickness_m",
                                                       "semantic_teacher_agreement",
                                                       "completed_semantic_accuracy")}
                                                  for k, v in sel["candidates"].items()}},
           "all_seeds": seeds,
           "eval_protocols": list(P.EVAL_PROTOCOLS),
           "success_criteria": dict(P.SUCCESS),
           "firewall": {"selection": sel.get("firewall"),
                        "target_audit": audit.get("firewall")}}
    write_json(MANIFEST, man)
    print(f"froze {MANIFEST}")
    print(f"  selected {win}  tau {man['occupancy_threshold']:+.4f}")
    print(f"  checkpoint {man['checkpoint']}  sha {man['checkpoint_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
