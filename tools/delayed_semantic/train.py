"""Train the evidence association head. Geometry frozen (cached features).

Per step, ``anchors_per_step`` training anchors are drawn; for each:
  observed-only term (every method): Gate 15 hiding blocks (8, 8, 4), p = 0.5, remove evidence from the candidate
      set; up to 2,048 hidden evidence voxels are queries; candidates = 16 nearest VISIBLE evidence voxels;
      target = the hidden voxel's own teacher distribution softmax(40 cos(f, text)); loss = final-feature KL.
  delayed term (B2, B2s, Gate 16 B3): up to 2,048 delayed target voxels (not evidence, tiers allowed by the reliability
      gate) are queries; candidates = 16 nearest evidence voxels (all visible, exactly as at evaluation).
      objective "feature" (pilot B2): KL(delayed distribution || readout of the mixed feature);
      objective "select"  (Gate 16 B3): KL(candidate target || softmax(scorer logits)), see gate16/candidate_targets.py.
Loss = sum of the per-term means. AdamW + cosine, grad clip, final step kept (no selection).
Anchor picks, hiding and hidden-query subsamples come from one generator seeded ``seed``; the delayed subsample uses
its own generator seeded ``seed + 1``. Methods with the same seed therefore share every observed-term draw.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Callable, Dict, List

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C, rules
from tools.delayed_semantic import train_cache as TC
from tools.delayed_semantic.model import EvidenceAssociation, kl_to_target, nearest_candidates, teacher_distribution, voxel_centres
from tools.semantic_completion.dataset import hidden_block_mask
from tools.semantic_completion.evidence import build as G15B, trident as TR

METHODS = ("B1", "B2", "B2s")              # B2s: addendum A sensitivity (4 s targets), secondary
VARIANT = {"B1": None, "B2": "", "B2s": "4s"}


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """uint8 packbits (numpy big-endian bit order) -> [N_VOXELS] bool, on the packed tensor's device."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed[:, None] >> shifts) & 1).reshape(-1)[:C.N_VOXELS].bool()


def load_anchor(drive: str, t: int, device, delayed_tiers: List[int], variant: str = "", paths: C.Paths = C.Paths()) -> Dict:
    with np.load(paths.g15.evidence_path(drive, t)) as z:
        flat = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device); feat = torch.from_numpy(z["feature"]).to(device)
        count = torch.from_numpy(z["count"]).to(device)
    with np.load(paths.provenance_path(drive, t)) as z:
        assert np.array_equal(z["flat_index"], flat.cpu().numpy().astype(np.int32)); bits = torch.from_numpy(z["frame_bits"]).to(device).long()
    with np.load(TC.student_path(drive, t)) as z:
        masks = torch.stack([torch.from_numpy(z[k]) for k in ("F", "S", "O")]).to(device); trunk_E = torch.from_numpy(z["trunk_E"]).to(device)
    a = {"drive": drive, "t": t, "flat": flat, "feature": feat, "count": count, "current": (bits >> C.CURRENT_SLOT) & 1 == 1,
         "n_frames": sum((bits >> q) & 1 for q in range(5)), "trunk": trunk_E, "masks_packed": masks}
    if delayed_tiers:
        with np.load(TC.target_paths(variant).target_path(drive, t)) as z:
            keep = ~z["in_evidence"] & np.isin(z["tier"], delayed_tiers)
            a["D_flat"] = torch.from_numpy(z["flat_index"][keep].astype(np.int64)).to(device)
            a["D_prob"] = torch.from_numpy(z["prob"][keep]).to(device)
        with np.load(TC.query_path(drive, t, variant)) as z:
            a["D_trunk"] = torch.from_numpy(z["trunk_D"][keep]).to(device)
    return a


def load_training_anchors(device, tiers: List[int], variant: str = "", paths: C.Paths = C.Paths()) -> List[Dict]:
    return [load_anchor(d, t, device, tiers, variant, paths) for d in C.TRAIN_DRIVES for t in G15B.anchor_ids(d, paths.g15)]


def anchor_step(model, a: Dict, text, cfg: C.TrainConfig, gen: torch.Generator, gen_d: torch.Generator, delayed: bool, dig,
                delayed_objective: str = "feature") -> Dict:
    """Per term: (loss, n queries, readout agreement with the target argmax, mean weight on the nearest candidate[, selection stats])."""
    dev = a["flat"].device; K = model.cfg.k_candidates
    masks = torch.stack([unpack(m) for m in a["masks_packed"]]).reshape(3, *C.GRID_DIMS)
    xyz = voxel_centres(a["flat"])
    ev = {"feature": a["feature"].float(), "trunk": a["trunk"].float(), "xyz": xyz, "count": a["count"], "current": a["current"], "n_frames": a["n_frames"]}
    hid = hidden_block_mask(a["flat"], C.GRID_DIMS, cfg.hide_block_vox, cfg.hide_block_prob, gen)
    vis_rows = (~hid).nonzero(as_tuple=True)[0]; hid_rows = hid.nonzero(as_tuple=True)[0]
    hq = hid_rows[torch.randperm(len(hid_rows), generator=gen)[:cfg.hidden_queries].to(dev)]
    dig.update(f"{a['drive'][17:21]}:{a['t']}:{len(vis_rows)}:{len(hq)};".encode())
    out = {}
    if len(vis_rows) >= K and len(hq):
        idx, _ = nearest_candidates(xyz[hq], xyz[vis_rows], K)
        f, w = model(ev["trunk"][hq], xyz[hq], vis_rows[idx], ev, masks)
        tgt = teacher_distribution(ev["feature"][hq], text)
        out["hidden"] = (kl_to_target(tgt, f, text), len(hq), float((f @ text.T).argmax(1).eq(tgt.argmax(1)).float().mean()), float(w[:, 0].mean()))
    if delayed and len(a["D_flat"]) and len(xyz) >= K:
        dq = torch.randperm(len(a["D_flat"]), generator=gen_d)[:cfg.delayed_queries].to(dev)
        qxyz = voxel_centres(a["D_flat"][dq])
        idx, _ = nearest_candidates(qxyz, xyz, K)
        tgt = a["D_prob"][dq].float()
        if delayed_objective == "feature":
            f, w = model(a["D_trunk"][dq].float(), qxyz, idx, ev, masks)
            out["delayed"] = (kl_to_target(tgt, f, text), len(dq), float((f @ text.T).argmax(1).eq(tgt.argmax(1)).float().mean()), float(w[:, 0].mean()))
        elif delayed_objective == "select":
            from tools.delayed_semantic.gate16.candidate_targets import build_target
            from tools.delayed_semantic.gate16.selection_loss import selection_kl, selection_stats
            f, w, s = model(a["D_trunk"][dq].float(), qxyz, idx, ev, masks, return_logits=True)
            ct = build_target(teacher_distribution(ev["feature"][idx], text), tgt)
            out["delayed"] = (selection_kl(ct.distribution, s), len(dq), float((f @ text.T).argmax(1).eq(tgt.argmax(1)).float().mean()), float(w[:, 0].mean()),
                              selection_stats(ct.best_index, ct.distribution, w.detach()))
        else:
            raise ValueError(delayed_objective)
    return out


def fit(model, anchors: List[Dict], text, cfg: C.TrainConfig, steps: int, seed: int, delayed: bool, delayed_objective: str,
        ckpt_path: Callable[[int], str], tag: str, meta: Dict) -> Dict:
    """The optimisation loop shared by the pilot and Gate 16."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed); gen_d = torch.Generator().manual_seed(seed + 1)
    dig = hashlib.sha256(); log, ckpts = [], []; t0 = time.time()
    for it in range(1, steps + 1):
        picks = torch.randint(len(anchors), (cfg.anchors_per_step,), generator=gen).tolist()
        parts = [anchor_step(model, anchors[i], text, cfg, gen, gen_d, delayed, dig, delayed_objective) for i in picks]
        row = {"step": it}; loss = 0.0
        for term in ("hidden", "delayed"):
            ps = [p[term] for p in parts if term in p]
            if ps:
                n = sum(p[1] for p in ps)
                lt = sum(p[0] * p[1] for p in ps) / n; loss = loss + lt
                row[f"loss_{term}"] = float(lt); row[f"acc_{term}"] = sum(p[2] * p[1] for p in ps) / n; row[f"w0_{term}"] = sum(p[3] * p[1] for p in ps) / n
                if len(ps[0]) > 4:
                    for k in ps[0][4]:
                        row[f"sel_{k}"] = sum(p[4][k] * p[1] for p in ps) / n
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()
        log.append(row)
        if it % cfg.log_every == 0 or it == steps:
            print(f"  [{tag} {it}/{steps}] " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step") + f" {(time.time()-t0)/60:.1f} min", flush=True)
        if it % cfg.ckpt_every == 0 or it == steps:
            p = ckpt_path(it); os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model_config": model.config(), "step": it, "seed": seed,
                        "semantic_projection": model.semantic_projection, **meta}, p)
            ckpts.append({"step": it, "path": os.path.relpath(p, C.REPO_ROOT), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    return {"tag": tag, "seed": seed, "steps": steps, "train_config": asdict(cfg), "model_config": model.config(), "params": model.n_params(),
            "semantic_projection": model.semantic_projection, "delayed_objective": delayed_objective if delayed else None, "n_anchors": len(anchors),
            "draw_digest": dig.hexdigest(), "checkpoints": ckpts, "log": log, "seconds": time.time() - t0, **meta}


def train(method: str, device: str, steps: int | None = None, paths: C.Paths = C.Paths(), cfg: C.TrainConfig = C.TrainConfig()) -> Dict:
    """Pilot entry point (B1, B2, B2s; head seed 0)."""
    assert method in METHODS
    rules.check("reliability")
    rel = json.load(open(os.path.join(paths.reliability, "reliability.json")))["gate"]
    assert rel["pass"], "reliability gate failed: training is not allowed"
    tiers = rel["delayed_loss_tiers"] if VARIANT[method] is not None else []
    if method == "B2s":
        rules.check("addendum_A")
    dev = torch.device(device); torch.cuda.set_device(dev)
    steps = steps or cfg.steps; tag = method if steps == cfg.steps else f"smoke_{method}"
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    with FileAudit(patterns=C.TRAIN_FORBIDDEN, strict=True) as audit:
        anchors = load_training_anchors(dev, tiers, VARIANT[method] or "", paths)
        torch.manual_seed(C.HEAD_SEED); model = EvidenceAssociation().to(dev)
        rec = fit(model, anchors, text, cfg, steps, C.HEAD_SEED, VARIANT[method] is not None, "feature",
                  lambda it: paths.head_checkpoint(tag, it), tag, {"method": method, "tiers": tiers})
    rec.update({"delayed_tiers": tiers, "firewall": {"opened": len(audit.opened), "violations": audit.violations}})
    os.makedirs(paths.runs, exist_ok=True)
    json.dump(rec, open(os.path.join(paths.runs, f"train_{tag}.json"), "w"), indent=1)
    return rec


__all__ = ["METHODS", "unpack", "load_anchor", "load_training_anchors", "anchor_step", "fit", "train"]
