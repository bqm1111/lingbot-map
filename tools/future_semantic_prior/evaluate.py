"""Development evaluation on 0006: E0 / E1 / E2 and their P1 subsets, plus the required breakdowns.

The populations come from Gate 20's own per-anchor record, read not recomputed, so E0 is bit-exactly the
frozen region ``completed_never_observed`` and B0's label is bit-exactly the frozen baseline's::

    E0   every P0 voxel                               (correctly completed, GT occupied, never observed)
    E1   E0 and Gate 20 marks the voxel future-visible (where the training supervision would exist)
    E2   E0 and it never becomes future-visible        <-- THE PRIMARY GATE
    E1_P1 / E2_P1   the same two, restricted to voxels where B0's class is wrong

Future visibility is used ONLY to split the evaluation population. The model never sees it: one forward
runs over the whole grid and the split is applied to the gathered logits afterwards, so the prediction
function is provably identical for E1 and E2. ``validate.py`` asserts that by re-running with a corrupted
reveal mask and requiring bit-identical logits.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.evaluate --control C3_full --seed 0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.future_revealability import config as G20, eval_masks as EM
from tools.future_semantic_prior import config as G, features as FE
from tools.future_semantic_prior.model import build_head
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry

POPULATIONS: Sequence[str] = ("E0", "E1", "E2", "E1_P1", "E2_P1")
METHODS: Sequence[str] = ("B0", "model")
NB = len(G.DISTANCE_NAMES)


def dev_anchor_ids(stride: int = 1) -> List[int]:
    d = os.path.join(G20.DATA_ROOT, "population", G.DEV_DRIVE)
    return sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".npz"))[::stride]


def dev_record(t: int) -> Dict[str, np.ndarray]:
    """Gate 20's frozen record for one 0006 anchor: the population, its GT class and B0's answer."""
    with np.load(G20.population_path(G.DEV_DRIVE, t)) as z:
        r = {k: z[k] for k in ("flat_index", "sem_gt", "dist_bin", "b0_label", "b0_correct", "vis_first_dt")}
        r["stats"] = json.loads(str(z["stats_json"]))
    return r


def populations(rec: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    """The five evaluation masks over the record's rows."""
    revealed = torch.from_numpy(np.isfinite(rec["vis_first_dt"])).to(device)
    wrong = ~torch.from_numpy(rec["b0_correct"]).to(device)
    e0 = torch.ones_like(revealed)
    return {"E0": e0, "E1": revealed, "E2": ~revealed, "E1_P1": revealed & wrong, "E2_P1": (~revealed) & wrong}


@torch.no_grad()
def predict(rec: Dict[str, np.ndarray], head, st: Dict, device) -> torch.Tensor:
    """``[N]`` class id in 1..18 at the record's rows. One forward; the split happens afterwards."""
    rows = torch.from_numpy(rec["flat_index"].astype(np.int64)).to(device)
    x = FE.build_input(st, head, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = head.logits_at(x, rows)
    return logits.float().argmax(1) + 1          # channel c <-> semantic label c+1 (text index 0 == label 1)


@torch.no_grad()
def composite_ssc(drive: str, t: int, st: Dict, label: torch.Tensor, rows: torch.Tensor, text, device) -> np.ndarray:
    """19x19 SSC confusion of the deployable read over EVERY labelled voxel.

    The read is the frozen one: the teacher's own class at evidence voxels, the method's class elsewhere,
    and empty wherever the frozen occupancy says empty. It exists to show that training a semantic head on
    future-revealed voxels does not wreck the rest of the scene; it decides nothing."""
    m = EM.load_mask(drive, t)
    sem_gt = torch.from_numpy(m["sem_gt"].astype(np.int64)).to(device)
    dense = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    dense[st["ev"].flat_index] = TR.classify(st["ev"].feature, text)   # teacher where it measured, 1..18
    free = ~st["E"]
    dense[free] = st["b0"][free]                      # B0 everywhere else ...
    dense[rows] = label                               # ... except the rows the method answers for
    pred = torch.where(st["O"], dense, torch.zeros_like(dense))
    return SM.confusion(sem_gt, pred, sem_gt != 255)


@torch.no_grad()
def evaluate(control: str, seed: int, device: str, step: Optional[int] = None, stride: int = 1,
             limit: Optional[int] = None, with_ssc: bool = True, head=None, net=None, text=None,
             paths: C.Paths = C.Paths(), quiet: bool = False) -> Dict:
    dev = torch.device(device)
    step = step or G.TrainConfig().steps
    if head is None:
        ck = torch.load(G.checkpoint_path(control, seed, step), map_location=dev, weights_only=False)
        head = build_head(control).to(dev)
        head.load_state_dict(ck["state_dict"])
    head.eval()
    if net is None:
        net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    if text is None:
        text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
        text = text.to(dev)
    ids = dev_anchor_ids(stride)[:limit]
    thing = torch.tensor(G.THING_LABELS, device=dev)

    conf = {m: {p: [] for p in POPULATIONS} for m in METHODS}
    cdist = {m: [] for m in METHODS}
    cthing = {m: {"thing": [], "stuff": []} for m in METHODS}
    cssc = {m: [] for m in METHODS}
    acc = {"ids": [], "n": [], "geometry_ok": []}
    t0 = time.time()
    with FileAudit(patterns=G.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            rec = dev_record(t)
            st = FE.frozen_causal_state(G.DEV_DRIVE, t, net, text, dev, paths)
            rows = torch.from_numpy(rec["flat_index"].astype(np.int64)).to(dev)
            # the frozen stack must be bit-identical to the one Gate 20 recorded
            geo_ok = (int(st["O"].sum()) == rec["stats"]["n_pred_occ"]
                      and int(st["E"].sum()) == rec["stats"]["n_evidence"]
                      and bool((st["O"][rows] & ~st["E"][rows]).all())
                      and bool(torch.equal(st["b0"][rows].to(torch.uint8).cpu(),
                                           torch.from_numpy(rec["b0_label"]))))
            assert geo_ok, f"the frozen causal state at 0006:{t} differs from Gate 20's record"
            gt = torch.from_numpy(rec["sem_gt"].astype(np.int64)).to(dev)
            pops = populations(rec, dev)
            dbin = torch.from_numpy(rec["dist_bin"].astype(np.int64)).to(dev)
            is_thing = torch.isin(gt, thing)
            lab = {"B0": torch.from_numpy(rec["b0_label"].astype(np.int64)).to(dev),
                   "model": predict(rec, head, st, dev)}
            for m in METHODS:
                for p in POPULATIONS:
                    conf[m][p].append(SM.confusion(gt, lab[m], pops[p]))
                cdist[m].append(np.stack([SM.confusion(gt, lab[m], pops["E2"] & (dbin == b)) for b in range(NB)]))
                cthing[m]["thing"].append(SM.confusion(gt, lab[m], pops["E2"] & is_thing))
                cthing[m]["stuff"].append(SM.confusion(gt, lab[m], pops["E2"] & ~is_thing))
                if with_ssc:
                    cssc[m].append(composite_ssc(G.DEV_DRIVE, t, st, lab[m], rows, text, dev))
            acc["ids"].append(t); acc["n"].append(int(len(rows))); acc["geometry_ok"].append(bool(geo_ok))
            if not quiet and (k + 1) % 100 == 0:
                print(f"  [eval {control} s{seed}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)

    out = {"ids": np.asarray(acc["ids"]), "n": np.asarray(acc["n"], np.int64),
           "geometry_ok": np.asarray(acc["geometry_ok"])}
    for m in METHODS:
        for p in POPULATIONS:
            out[f"conf__{m}__{p}"] = np.stack(conf[m][p])
        out[f"confdist__{m}"] = np.stack(cdist[m])
        out[f"confthing__{m}"] = np.stack(cthing[m]["thing"])
        out[f"confstuff__{m}"] = np.stack(cthing[m]["stuff"])
        if with_ssc:
            out[f"confssc__{m}"] = np.stack(cssc[m])
    meta = {"control": control, "seed": seed, "step": step, "n_anchors": len(ids), "stride": stride,
            "all_geometry_identical": bool(all(acc["geometry_ok"])), "seconds": round(time.time() - t0, 1),
            "firewall": {"opened": len(audit.opened), "violations": audit.violations},
            "head": head.describe(), "version": G.VERSION}
    return {"arrays": out, "meta": meta}


def eval_path(control: str, seed: int, step: int, stride: int) -> str:
    return os.path.join(G.DATA_ROOT, "eval", f"{control}_seed{seed}_step{step:05d}_s{stride}.npz")


def run_and_save(control: str, seed: int, device: str, step: Optional[int] = None, stride: int = 1,
                 **kw) -> Dict:
    G.ensure_dirs()
    r = evaluate(control, seed, device, step, stride, **kw)
    p = eval_path(control, seed, r["meta"]["step"], stride)
    np.savez_compressed(p, **r["arrays"], meta_json=np.asarray(json.dumps(r["meta"])))
    json.dump(r["meta"], open(p[:-4] + ".json", "w"), indent=1)
    return r["meta"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--step", type=int)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(run_and_save(a.control, a.seed, a.device, a.step, a.stride, limit=a.limit), indent=1))


if __name__ == "__main__":
    main()
