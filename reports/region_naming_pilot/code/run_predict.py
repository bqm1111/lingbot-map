"""Stage 4: write the five label sets of every anchor, inside a strict no-GT file audit."""
from __future__ import annotations

import json
import os
import sys
import time

import torch

import extension as EX
import legacy as L
import manifest as MF
import paths as P
import predict as PD


def main(device: str = "cuda:0") -> None:
    cal = os.path.join(P.ARTIFACTS, "calibration.json")
    hops = json.load(open(cal))["chosen_max_hops"] if os.path.exists(cal) else EX.SETTINGS["max_hops"]
    EX.SETTINGS["max_hops"] = int(hops)
    a = MF.select()
    ids = [a["golden"]] + a["calibration"] + a["evaluation"]
    ids = [t for t in ids if os.path.exists(os.path.join(P.QWEN_CACHE, f"{t:05d}.json"))]
    todo = [t for t in ids if not os.path.exists(os.path.join(P.PREDICTIONS, f"{t:05d}.npz"))]
    print(f"max_hops={hops}; {len(ids)} anchors ready, {len(todo)} to predict", flush=True)
    if not todo:
        return
    text, _, _ = L.TR.load_text_embeddings(); text = text.to(device)
    net = L.load_frozen_geometry(L.GEOMETRY_SEED, device)
    log = []
    with L.FileAudit(patterns=L.PREDICT_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(todo):
            t0 = time.time()
            out = PD.predict_anchor(t, net, text, device, max_hops=int(hops))
            PD.save(t, out)
            log.append({**out["stats"], "seconds": time.time() - t0})
            s = out["stats"]
            print(f"[{k+1}/{len(todo)}] t={t} O={s['n_O']} E={s['n_E']} "
                  f"seeds_P!=B0 {s['seeds_changed_P_vs_B0']} seeds_Q!=P {s['seeds_changed_Q_vs_P']} "
                  f"grown {s['extension']['n_grown']} contested {s['extension']['n_contested']} "
                  f"cover {s['extension']['occupied_coverage']:.3f} {time.time()-t0:.0f}s", flush=True)
        print("audit violations", audit.violations, flush=True)
        opened, violations = len(audit.opened), audit.violations
    with open(P.under_experiment(os.path.join(P.LOGS, "predict.json")), "w") as f:
        json.dump({"max_hops": int(hops), "log": log, "forbidden": list(L.PREDICT_FORBIDDEN),
                   "audit": {"opened": opened, "violations": violations}},
                  f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
