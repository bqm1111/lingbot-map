"""Target reliability before any training: label-free gate G1-G4 plus a labelled 0006 GT diagnostic.

G1 ``registration``   point-to-plane residual of a future sweep moved into the anchor grid by GT poses,
                      against planar neighbourhoods of the anchor's own sweep (plus a +0.2 m perturbed pose
                      as a sensitivity control: the metric must see a one-voxel misregistration).
G2 ``coverage``       |D| per anchor (D = target voxels that are not student evidence), class support.
G3 ``repeatability``  first vs second future view top-1 of the same voxel, against a within-anchor shuffle
                      of the second view (respects each anchor's class mix, breaks voxel identity).
G4 ``reobservation``  class of the H_t camera views of a voxel (LiDAR-projected) vs its future target class.
Also: time to reveal, tier and history-visibility split, and agreement between the student's own lifted
evidence class and the delayed target on re-observed evidence voxels.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C, rules as R
from tools.delayed_semantic.stream import DriveStream
from tools.delayed_semantic.targets import _voxel_of
from tools.semantic_completion.evidence import build as G15B, trident as TR

NONE = 255
DT_EDGES = (1.0, 2.0, 4.0)
DT_NAMES = ("0-1 s", "1-2 s", "2-4 s", "4-10 s")


# --------------------------------------------------------------------------- G1
@torch.no_grad()
def _plane_residuals(ref: torch.Tensor, qry: torch.Tensor, k: int = 10, radius: float = 0.5) -> torch.Tensor:
    out = []
    for s in range(0, len(qry), 2048):
        q = qry[s:s + 2048]
        d, i = torch.cdist(q.float(), ref.float()).topk(k, largest=False)
        ok = d[:, -1] <= radius
        nb = ref[i[ok]].double(); c = nb.mean(1)
        ev, evec = torch.linalg.eigh(((nb - c[:, None]) .transpose(1, 2) @ (nb - c[:, None])) / k)
        planar = ev[:, 0] < 0.1 * ev[:, 1]
        n = evec[:, :, 0]
        out.append(((q[ok].double() - c) * n).sum(1).abs()[planar])
    return torch.cat(out) if out else torch.zeros(0)


def _summary(xs) -> Dict:
    n = int(sum(len(x) for x in xs))
    return {"n": n, "median_m": float(torch.cat(xs).median()) if n else None, "p90_m": float(torch.cat(xs).quantile(0.9)) if n else None}


def registration(drive: str, device, anchor_stride: int = 20, offsets=(1, 5, 10, 20), n_query: int = 20000) -> Dict:
    dev = torch.device(device)
    stream = DriveStream(drive); geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    # sensitivity controls: shifts across the dominant planes (ground: z, facades: y); an x shift slides along both
    shifts = {"z+0.2m": [0.0, 0.0, 0.2], "y+0.2m": [0.0, 0.2, 0.0]}
    res = {k: [] for k in offsets}; pert = {s: {k: [] for k in offsets} for s in shifts}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for t in G15B.anchor_ids(drive)[::anchor_stride]:
        a = stream.frames[t]; fut = {f.index - t: f for f in stream.future(t)}
        ref = torch.as_tensor(geo.read_velodyne(a.native), device=dev); ref = ref[_voxel_of(ref) >= 0]
        for k in offsets:
            if k not in fut or fut[k].native not in geo.cam0_to_world:
                continue
            T = TF.inv(geo.velo_to_world(a.native)) @ geo.velo_to_world(fut[k].native)
            p = torch.as_tensor(TF.apply(T, geo.read_velodyne(fut[k].native)), device=dev); p = p[_voxel_of(p) >= 0]
            p = p[torch.randperm(len(p), generator=gen)[:n_query].to(dev)]
            res[k].append(_plane_residuals(ref, p).cpu())
            for s, v in shifts.items():
                pert[s][k].append(_plane_residuals(ref, p + torch.tensor(v, dtype=p.dtype, device=dev)).cpu())
    return {"drive": drive, "anchor_stride": anchor_stride, "offsets": {str(k): _summary(res[k]) for k in offsets},
            "perturbed": {s: {str(k): _summary(v[k]) for k in offsets} for s, v in pert.items()},
            "_raw": {str(k): torch.cat(res[k]) if res[k] else torch.zeros(0) for k in offsets}}


# --------------------------------------------------------------------------- G2-G4
def target_statistics(drive: str, evidence_stride: int = 10, paths: C.Paths = C.Paths(), device="cuda:0") -> Dict:
    ids = G15B.anchor_ids(drive, paths.g15)
    rng = np.random.default_rng(0)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(device)
    per_anchor, cls = [], {t: np.zeros(C.N_CLASSES, np.int64) for t in ("A", "B")}
    rep = {t: np.zeros(3, np.int64) for t in ("A", "B")}            # [pairs, agree, shuffled agree]
    rep_dt = {t: np.zeros((len(DT_NAMES), 2), np.int64) for t in ("A", "B")}
    reob = {"all": np.zeros(2, np.int64), "D": np.zeros(2, np.int64)}
    reveal = {t: np.zeros(len(DT_NAMES), np.int64) for t in ("A", "B")}
    ev_agree = np.zeros(2, np.int64)
    for k, t in enumerate(ids):
        with np.load(paths.target_path(drive, t)) as z:
            a = {n: z[n] for n in z.files if n != "stats_json"}
        D = ~a["in_evidence"]; top = a["prob"].astype(np.float32).argmax(1).astype(np.uint8)
        row = {"t": t, "n_target": int(len(D)), "n_D": int(D.sum())}
        for name, tier in (("A", 1), ("B", 2)):
            m = D & (a["tier"] == tier)
            row[f"n_D_{name}"] = int(m.sum()); row[f"n_D_{name}_hist_seen"] = int((m & a["hist_seen"]).sum())
            cls[name] += np.bincount(top[m], minlength=C.N_CLASSES)
            two = m & (a["n_views"] >= 2)
            v0, v1 = a["view0_top1"][two], a["view1_top1"][two]
            rep[name] += [len(v0), int((v0 == v1).sum()), int((v0 == rng.permutation(v1)).sum())]
            b = np.digitize((a["second_dt_s"][two] - a["first_dt_s"][two]).astype(np.float32), DT_EDGES)
            np.add.at(rep_dt[name], (b, 0), 1); np.add.at(rep_dt[name], (b, 1), (v0 == v1).astype(np.int64))
            reveal[name] += np.bincount(np.digitize(a["first_dt_s"][m].astype(np.float32), DT_EDGES), minlength=len(DT_NAMES))
        h = a["hist_top1"] != NONE
        reob["all"] += [int(h.sum()), int((a["hist_top1"][h] == top[h]).sum())]
        reob["D"] += [int((h & D).sum()), int((a["hist_top1"][h & D] == top[h & D]).sum())]
        if k % evidence_stride == 0 and (~D).any():
            with np.load(paths.g15.evidence_path(drive, t)) as z:
                flat, feat = z["flat_index"].astype(np.int64), z["feature"]
            pos = np.searchsorted(flat, a["flat_index"][~D].astype(np.int64))
            ecls = (TR.classify(torch.from_numpy(feat[pos].astype(np.float32)).to(device), text) - 1).cpu().numpy()
            ev_agree += [len(pos), int((ecls == top[~D]).sum())]
        per_anchor.append(row)
    frac = lambda x: float(x[1] / max(x[0], 1))
    return {"drive": drive, "n_anchors": len(ids), "per_anchor": per_anchor,
            "mean_D": float(np.mean([r["n_D"] for r in per_anchor])),
            "mean_D_A": float(np.mean([r["n_D_A"] for r in per_anchor])), "mean_D_B": float(np.mean([r["n_D_B"] for r in per_anchor])),
            "D_hist_seen_fraction": {n: float(sum(r[f"n_D_{n}_hist_seen"] for r in per_anchor) / max(sum(r[f"n_D_{n}"] for r in per_anchor), 1)) for n in ("A", "B")},
            "class_counts": {n: v.tolist() for n, v in cls.items()},
            "repeatability": {n: {"pairs": int(v[0]), "agreement": float(v[1] / max(v[0], 1)), "shuffled": float(v[2] / max(v[0], 1))} for n, v in rep.items()},
            "repeatability_by_view_gap": {n: {DT_NAMES[i]: {"pairs": int(v[i, 0]), "agreement": float(v[i, 1] / max(v[i, 0], 1))} for i in range(len(DT_NAMES))} for n, v in rep_dt.items()},
            "reobservation": {n: {"voxels": int(v[0]), "agreement": frac(v)} for n, v in reob.items()},
            "time_to_reveal": {n: dict(zip(DT_NAMES, v.tolist())) for n, v in reveal.items()},
            "student_evidence_vs_target": {"voxels": int(ev_agree[0]), "agreement": frac(ev_agree), "anchor_stride": evidence_stride}}


def pooled(stats: List[Dict]) -> Dict:
    out = {"mean_D": float(np.mean([r["n_D"] for s in stats for r in s["per_anchor"]])), "class_counts": {}, "repeatability": {}, "reobservation": {}}
    for n in ("A", "B"):
        out["class_counts"][n] = np.sum([s["class_counts"][n] for s in stats], 0).tolist()
        p = sum(s["repeatability"][n]["pairs"] for s in stats)
        out["repeatability"][n] = {"pairs": p, "agreement": sum(s["repeatability"][n]["agreement"] * s["repeatability"][n]["pairs"] for s in stats) / max(p, 1),
                                   "shuffled": sum(s["repeatability"][n]["shuffled"] * s["repeatability"][n]["pairs"] for s in stats) / max(p, 1)}
    for n in ("all", "D"):
        v = sum(s["reobservation"][n]["voxels"] for s in stats)
        out["reobservation"][n] = {"voxels": v, "agreement": sum(s["reobservation"][n]["agreement"] * s["reobservation"][n]["voxels"] for s in stats) / max(v, 1)}
    return out


def gate(reg: Dict[str, Dict], train: Dict, dev: Dict) -> Dict:
    g = R.RELIABILITY_GATE
    ok = lambda v: v["median_m"] <= g["G1_registration"]["median_max_m"] and v["p90_m"] <= g["G1_registration"]["p90_max_m"]
    # as predeclared: training drives pooled, every offset that has data (offset 20 = 10.4 s lies outside the 10 s window)
    pooled_rows = {k: _summary([reg[d]["_raw"][k] for d in C.TRAIN_DRIVES]) for k in reg[C.TRAIN_DRIVES[0]]["_raw"]}
    g1 = all(ok(v) for v in pooled_rows.values() if v["n"])
    # stricter reading of the first implementation (each drive separately, development drive included) -- reported, disclosed
    strict = {f"{d[17:21]}@{k}": ok(v) for d, r in reg.items() for k, v in r["offsets"].items() if v["n"]}

    def g3(tier):
        tr, dv = train["repeatability"][tier], dev["repeatability"][tier]
        c = g["G3_repeatability"]
        return (tr["agreement"] >= c["train_agreement_min"] and tr["agreement"] - tr["shuffled"] >= c["train_margin_over_shuffle_min"]
                and dv["agreement"] >= c["dev_agreement_min"] and dv["agreement"] - dv["shuffled"] >= c["dev_margin_over_shuffle_min"])
    counts = np.asarray(train["class_counts"]["A"]) + np.asarray(train["class_counts"]["B"])
    c2 = g["G2_coverage"]
    g2 = train["mean_D"] >= c2["mean_D_per_train_anchor_min"] and int((counts >= c2["voxels_per_class_min"]).sum()) >= c2["classes_min"]
    g4 = train["reobservation"]["all"]["agreement"] >= g["G4_reobservation"]["agreement_min"]
    passed = g1 and g2 and g3("A") and g4
    return {"G1": g1, "G1_train_pooled": pooled_rows, "G1_strict_per_drive_incl_dev": strict, "G1_strict_pass": all(strict.values()), "G2": g2, "G2_classes_with_support": int((counts >= c2["voxels_per_class_min"]).sum()), "G3_tierA": g3("A"), "G3_tierB": g3("B"),
            "G4": g4, "pass": passed, "delayed_loss_tiers": ([1, 2] if g3("B") else [1]) if passed else [],
            "outcome": "targets usable" if passed else "Stop: delayed targets are not reliable enough"}


# --------------------------------------------------------------------------- GT diagnostic (0006, evaluation only)
def gt_diagnostic(paths: C.Paths = C.Paths()) -> Dict:
    """LABELLED DIAGNOSTIC -- never feeds a rule. Target class (and view classes) vs SSCBench semantic GT."""
    conf = {n: np.zeros((C.N_CLASSES + 1, C.N_CLASSES + 1), np.int64) for n in ("D_A", "D_B", "D_view0", "evidence_reobserved")}
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True):
        for t in G15B.anchor_ids(C.DEV_DRIVE, paths.g15):
            with np.load(paths.g15.eval_mask_path(t)) as z:
                gt = z["sem_gt"].astype(np.int64)
            with np.load(paths.target_path(C.DEV_DRIVE, t)) as z:
                flat, top, tier, inE, v0 = z["flat_index"].astype(np.int64), z["prob"].astype(np.float32).argmax(1) + 1, z["tier"], z["in_evidence"], z["view0_top1"].astype(np.int64) + 1
            g = gt[flat]; lab = (g >= 1) & (g <= C.N_CLASSES)
            for name, m, p in (("D_A", ~inE & (tier == 1), top), ("D_B", ~inE & (tier == 2), top), ("D_view0", ~inE, v0), ("evidence_reobserved", inE, top)):
                sel = m & lab
                np.add.at(conf[name], (g[sel], p[sel]), 1)
    out = {}
    for n, cm in conf.items():
        tp = np.diag(cm)[1:]; gt_n = cm[1:].sum(1); pr = cm[:, 1:].sum(0)
        sup = gt_n > 0
        out[n] = {"voxels": int(gt_n.sum()), "accuracy": float(tp.sum() / max(gt_n.sum(), 1)),
                  "miou": float(np.mean(tp[sup] / (gt_n[sup] + pr[sup] - tp[sup]))), "per_class_accuracy": (tp / np.maximum(gt_n, 1)).tolist(),
                  "gt_counts": gt_n.tolist()}
    return out


def run_all(device: str) -> Dict:
    R.check("reliability")
    with FileAudit(patterns=C.TARGET_FORBIDDEN, strict=True) as audit:
        reg = {d: registration(d, device) for d in C.TRAIN_DRIVES + (C.DEV_DRIVE,)}
        stats = {d: target_statistics(d, device=device) for d in C.TRAIN_DRIVES + (C.DEV_DRIVE,)}
    train = pooled([stats[d] for d in C.TRAIN_DRIVES]); dev = pooled([stats[C.DEV_DRIVE]])
    g_ = gate(reg, train, dev)
    for r in reg.values():
        r.pop("_raw")
    rec = {"registration": reg, "per_drive": {d: {k: v for k, v in s.items() if k != "per_anchor"} for d, s in stats.items()},
           "train_pooled": train, "dev_pooled": dev, "gate": g_,
           "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    rec["gt_diagnostic_0006"] = gt_diagnostic()
    os.makedirs(C.Paths().reliability, exist_ok=True)
    json.dump(rec, open(os.path.join(C.Paths().reliability, "reliability.json"), "w"), indent=1)
    json.dump({d: s["per_anchor"] for d, s in stats.items()}, open(os.path.join(C.Paths().reliability, "per_anchor.json"), "w"))
    return rec


__all__ = ["registration", "target_statistics", "pooled", "gate", "gt_diagnostic", "run_all"]
