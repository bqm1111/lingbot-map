"""Evaluate one Gate 19 checkpoint on the frozen 0006 population: geometry, then semantic SSC through B0.

Geometry reuses ``baseline_reconstruction.current.evaluate_source.evaluate_checkpoint`` unchanged, so the numbers are
directly comparable to the frozen baseline's (same regions, same range bins, same threshold).

Semantics reuse the frozen teacher and the frozen B0 rule. The ONLY thing that varies with training data is the
occupancy the semantics are read out on -- no propagation model is retrained, which is what makes this a clean
decomposition of "geometry improved" from "semantics improved".

    PYTHONPATH=$PWD python -m tools.data_scaling.evaluate --tag D_all_fixed_steps_seed0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from core.model.net import load_checkpoint
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction.current.evaluate_source import evaluate_checkpoint, predict
from tools.baseline_reconstruction.current.geometry import network_input
from tools.data_scaling import config as G19
from tools.delayed_semantic import config as DC
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.regions import SEMANTIC_REGIONS, SSC_REGIONS, build_regions
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR

SEM_REGIONS = ("never_observed", "completed_never_observed", "completed_geo_observed", "completed_geo_unobserved",
               "ssc_full", "ssc_never_observed")


@torch.no_grad()
def semantic_eval(checkpoint: str, device: str, paths: DC.Paths = DC.Paths()) -> Dict:
    """B0 semantics on the occupancy this checkpoint predicts. Teacher, evidence and B0 rule are all frozen."""
    dev = torch.device(device); torch.cuda.set_device(dev)
    net = load_checkpoint(checkpoint, dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    ids = G15B.anchor_ids(G19.VAL_DRIVE, paths.g15)
    conf = {r: [] for r in SEM_REGIONS}
    occ_counts = []
    t0 = time.time()
    with FileAudit(patterns=DC.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            with np.load(paths.g15.input_path(G19.VAL_DRIVE, t)) as z:
                S = torch.from_numpy(np.unpackbits(z["S_H"])[:DC.N_VOXELS].astype(bool)).to(dev)
                F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:DC.N_VOXELS].astype(bool)).to(dev)
            # the frozen decision rule, via the frozen evaluator's own helper
            O = predict(net, network_input(S, F_, DC.GRID_DIMS)).reshape(-1)
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            ev = load_observed(G19.VAL_DRIVE, t, dev, paths)
            E = ev.dense_mask(); Ec = torch.zeros_like(E); Ec[ev.flat_index[ev.current()]] = True
            ev_label = torch.zeros(DC.N_VOXELS, dtype=torch.long, device=dev)
            ev_label[ev.flat_index] = TR.classify(ev.feature, text)
            nearest, _ = B0.nearest_evidence(E)
            lab = torch.where(E, ev_label, B0.nearest_label(ev_label, nearest))
            R = torch.zeros(DC.N_VOXELS, dtype=torch.bool, device=dev)
            reg = build_regions(sem_gt, E, Ec, S, F_, O, R, R)
            ssc = torch.where(O, lab, torch.zeros_like(lab))
            for r in SEM_REGIONS:
                conf[r].append(SM.confusion(sem_gt, ssc if r in SSC_REGIONS else lab, reg[r]))
            occ_counts.append(int(O.sum()))
            if (k + 1) % 150 == 0:
                print(f"  [sem] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {}
    for r in SEM_REGIONS:
        c = np.stack(conf[r]).sum(0)
        s = SM.summarize(c.astype(np.int64), semantic_only=True)
        out[r] = {"miou": None if s["miou"] is None else 100 * s["miou"],
                  "accuracy": 100 * s["overall_accuracy"], "n_voxels": int(c[1:].sum()),
                  "per_class_iou": {c_: (None if s["per_class_iou"][c_] is None else 100 * s["per_class_iou"][c_])
                                    for c_ in range(1, 19)}}
    out["_meta"] = {"n_anchors": len(ids), "mean_predicted_occupied": float(np.mean(occ_counts)),
                    "seconds": time.time() - t0, "firewall": {"violations": audit.violations}}
    return out


def run(tag: str, device: str, stride: int = 1) -> Dict:
    rec = json.load(open(os.path.join(G19.ARTIFACT_DIR, "runs", f"train_{tag}.json")))
    ck = rec["final_checkpoint"]
    out_dir = os.path.join(G19.ARTIFACT_DIR, "eval"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(out_path):
        print(f"[{tag}] already evaluated", flush=True)
        return json.load(open(out_path))
    geo = evaluate_checkpoint(ck, device, stride=stride,
                              out_path=os.path.join(out_dir, f"{tag}_geometry.json"))
    sem = semantic_eval(ck, device)
    out = {"tag": tag, "scale": rec["scale"], "regime": rec["regime"], "seed": rec["seed"],
           "n_anchors": rec["n_train"], "steps": rec["steps"], "epochs": rec["epochs"],
           "gpu_hours": rec["gpu_hours"], "geometry": geo.summary, "map_only": geo.map_only, "semantic": sem}
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    g = geo.summary["full"]
    print(f"[{tag}] geometry full IoU {g['iou']:.4f} | unobs5_later {geo.summary['unobs5_later']['iou']:.4f} | "
          f"semantic completed-unobs mIoU {sem['completed_never_observed']['miou']:.2f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    r = run(a.tag, a.device, a.stride)
    print(json.dumps({k: v for k, v in r.items() if k not in ("geometry", "semantic", "map_only")}, indent=1))


if __name__ == "__main__":
    main()
