"""Stage 2: Qwen-VL names the queried regions. One cached call per region, reused by every variant.

The exact images shown are reconstructible from the proposal record alone (frame slot, owned-pixel bbox
and painted mask id), so they are not all written to disk; the views of the first
``SAVE_VIEWS_TOP_K`` regions of every anchor are saved as PNG for inspection, and all of them for the
golden and calibration anchors.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

import legacy as L
import manifest as MF
import paths as P
import proposals as PR
import recognize as RC

SAVE_VIEWS_TOP_K = 10


def run_anchor(t: int, model, proc, vocab: List[str], save_all_views: bool) -> Dict:
    prop = PR.load(t)
    regions = {r["region_id"]: r for r in prop["regions"]}
    qids = RC.queried_regions(prop["regions"])
    images = [PR.native_image(k) for k in prop["stats"]["keys"]]
    lab_native = prop["label_native"]
    order = sorted(qids, key=lambda r: (-regions[r]["n_evidence_voxels"], r))
    diag_ids = set(order[:RC.SETTINGS["qwen"]["diagnostic_single_view_top_k"]])
    out: Dict[str, Dict] = {}
    t0 = time.time()
    for n, r in enumerate(order):
        reg = regions[r]
        views = RC.region_views(reg, lab_native, images)
        tic = time.time()
        reply = RC.ask(model, proc, views, vocab)
        rec = {"n_views": len(views), "best_frame": reg["best_frame"], "best_key": reg["best_key"],
               "best_bbox": reg["best_bbox"], "best_mask_id": reg["best_mask_id"],
               "n_frames": len(reg["frames"]), "n_evidence_voxels": reg["n_evidence_voxels"],
               "reply": reply, "parsed": RC.parse(reply, vocab), "seconds": round(time.time() - tic, 2)}
        if r in diag_ids and len(reg["frames"]) > 1:
            sv = RC.ask(model, proc, RC.region_views(reg, lab_native, images, single_view=True), vocab)
            rec["single_view"] = {"reply": sv, "parsed": RC.parse(sv, vocab)}
        if save_all_views or n < SAVE_VIEWS_TOP_K:
            d = P.under_experiment(os.path.join(P.CROPS, f"{t:05d}", f"r{r:04d}_v0.png"))
            for i, v in enumerate(views):
                v.save(os.path.join(os.path.dirname(d), f"r{r:04d}_v{i}.png"))
        out[str(r)] = rec
    return {"t": t, "n_regions": len(prop["regions"]), "n_queried": len(qids),
            "seconds": round(time.time() - t0, 1), "settings": RC.SETTINGS["qwen"],
            "regions": out}


def main(device: str = "cuda:0", shard: int = 0, shards: int = 1) -> None:
    """``shard``/``shards`` split the remaining anchors across devices. Each anchor is one independent
    JSON file, so two workers never write the same output and the split changes no result."""
    a = MF.select()
    ids = [a["golden"]] + a["calibration"] + a["evaluation"]
    special = {a["golden"], *a["calibration"]}
    ids = [t for t in ids if os.path.exists(os.path.join(P.PROPOSALS, f"{t:05d}.npz"))]
    todo = [t for t in ids if not os.path.exists(os.path.join(P.QWEN_CACHE, f"{t:05d}.json"))]
    todo = todo[shard::shards]
    print(f"{len(ids)} anchors with proposals, {len(todo)} to query "
          f"(shard {shard}/{shards} on {device})", flush=True)
    if not todo:
        return
    _, phrases, _ = L.TR.load_text_embeddings()
    vocab = [str(p) for p in phrases]
    model, proc = RC.load_qwen(device)
    print(f"Qwen loaded: {P.QWEN_REPO}", flush=True)
    with L.FileAudit(patterns=L.PREDICT_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(todo):
            rec = run_anchor(t, model, proc, vocab, save_all_views=t in special)
            with open(P.under_experiment(os.path.join(P.QWEN_CACHE, f"{t:05d}.json")), "w") as f:
                json.dump(rec, f, indent=1)
            oc = {}
            for v in rec["regions"].values():
                oc[v["parsed"]["outcome"]] = oc.get(v["parsed"]["outcome"], 0) + 1
            print(f"[{k+1}/{len(todo)}] t={t} queried={rec['n_queried']} {oc} {rec['seconds']:.0f}s", flush=True)
        print("audit violations", audit.violations, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0",
         int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 1)
