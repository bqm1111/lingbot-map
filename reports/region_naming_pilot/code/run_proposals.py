"""Stage 1: frozen SAM region proposals for every selected anchor. Resumable, idempotent."""
from __future__ import annotations

import json
import os
import sys
import time

import torch

import legacy as L
import manifest as MF
import paths as P
import proposals as PR


def main(device: str = "cuda:0") -> None:
    a = MF.select()
    ids = [a["golden"]] + a["calibration"] + a["evaluation"]
    todo = [t for t in ids if not os.path.exists(os.path.join(P.PROPOSALS, f"{t:05d}.npz"))]
    print(f"{len(ids)} anchors, {len(todo)} to build", flush=True)
    if not todo:
        return
    amg = PR.load_sam(device)
    T = L.camera_to_grid()
    log = []
    with L.FileAudit(patterns=L.PREDICT_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(todo):
            t0 = time.time()
            out = PR.build_anchor(t, amg, device, T)
            PR.save(t, out)
            s = out["stats"]
            log.append({"t": t, "seconds": time.time() - t0, **{k_: s[k_] for k_ in
                        ("n_frame_masks", "n_regions", "n_single_frame_regions",
                         "n_regions_merging_two_masks_of_one_frame", "n_regions_with_non_contiguous_frames",
                         "n_regions_with_rgb_support_no_evidence", "n_evidence_voxels",
                         "n_evidence_voxels_owned", "evidence_ownership_fraction")}})
            print(f"[{k+1}/{len(todo)}] t={t} regions={s['n_regions']} "
                  f"ownership={s['evidence_ownership_fraction']:.3f} {time.time()-t0:.0f}s", flush=True)
    with open(P.under_experiment(os.path.join(P.LOGS, "proposals.json")), "w") as f:
        json.dump({"log": log, "settings": PR.SETTINGS,
                   "audit": {"opened": len(audit.opened), "violations": audit.violations}}, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
