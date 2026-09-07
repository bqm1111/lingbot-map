#!/usr/bin/env python
"""Gate 8C-1: evaluate the frozen KITTI-360-trained model on an untouched target dataset.

Refuses to run unless `artifacts/gate8c1/frozen_manifest.json` exists: the firewall lifts
only after every checkpoint and threshold is frozen. Takes no checkpoint and no threshold
argument -- both come from the manifest.

Three protocols, all on identical anchors and identical official masks:

* ``stream``      -- Protocol A, all-past causal streaming: one persistent map per segment,
                     scale fixed from the first five stream frames, every frame integrated
                     exactly once, queried at each official anchor.
* ``past5``       -- Protocol A primary: a fresh map over the five stream frames **ending**
                     at the target, same frozen stream scale, no future observation.
* ``occany_fwd``  -- Protocol B, comparison only and explicitly **non-causal**: OccAny's
                     published target-first sampling, stream offsets 0, +2, +4, +6, +8.

    python tools/gate8c1/eval_target.py --dataset semantickitti --mode past5 --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from scale_gate.config import write_json                                         # noqa: E402
from gate6 import grids as G6G, metrics as G6M, targets as G6T, vocab            # noqa: E402
from gate8 import sources as S, vocab as V8                                      # noqa: E402
from gate8.feed import CachedFeed                                                # noqa: E402
from gate8.mapper import FrameInput, IncrementalMapper                           # noqa: E402
from gate8.net import load_checkpoint, LOCK_LOGODDS                              # noqa: E402
from gate8a import baselines as BL, scores as SC                                 # noqa: E402
from gate8a.regions import to_eval_grid, occ_to_eval_grid                        # noqa: E402
from gate8b import pooling as PL                                                 # noqa: E402
from tools.gate8a.evaluate import dilate_with_semantics, _save_counts, RANDOM_SEEDS  # noqa: E402

#: the frozen record. ``G8C1_MANIFEST`` lets a variant (e.g. the padding-fix checkpoints)
#: be evaluated through the identical code path without touching the gate's own manifest.
MANIFEST = os.environ.get("G8C1_MANIFEST",
                          os.path.join(ART, "frozen_manifest.json"))
OCCANY_FWD_OFFSETS = (0, 2, 4, 6, 8)          # OccAny's published target-first sampling
PAST5_OFFSETS = (-4, -3, -2, -1, 0)


def _t(x, dev):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def window_map(feed, seg, i, offsets, scale, into, C, dev):
    """A fresh map over ``i + offsets`` with the frozen stream scale. Each frame once."""
    m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
    m.scale_state.force(scale)
    used = []
    for o in offsets:
        j = i + o
        if 0 <= j < len(seg):
            m.step(feed.frame(j)); used.append(j)
    return m, used


def run(dataset, mode, seed, device, limit=None, art=None):
    art = art or ART
    man = json.load(open(MANIFEST))
    fz = man["seeds"][str(seed)]
    ckpt, tau, sem_tau = fz["checkpoint"], fz["occupancy_threshold"], fz["semantic_threshold"]
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.init()
    ds = dataset
    v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims)); edims = tuple(int(d) for d in EVAL.dims)
    dist, height = G6M.band_masks(EVAL)
    labels = np.asarray(v.labels, np.int32)
    into = V8.into_matrix(ds); out_ch = torch.as_tensor(V8.out_channel(ds), device=dev)
    net = load_checkpoint(os.path.join(REPO_ROOT, ckpt), dev)
    tag = f"{mode}_seed{seed}"
    acc = {m_: {r: SC.ScoreAccumulator() for r in ("full", "edit")}
           for m_ in ("mapper", "completion")}
    fixed = ["mapper_native", "mapper_dilate", "all_valid_occupied", "editable_fill",
             "completion_occany_pool", "completion_occany_vote"] + \
            [f"random_editable_s{s}" for s in RANDOM_SEEDS]
    bins = {k: [] for k in fixed}
    meta = {"clip_id": [], "group": [], "n_valid": [], "n_gt": [], "n_editable": [],
            "n_gt_editable": [], "random_q": []}
    sem_counts, newly, lat = {}, [], {"map": [], "net": []}
    n_done, n_skip, t0 = 0, 0, time.time()
    for seg in S.segments(ds, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(ds, seg.name)) as z:
            scale = float(np.exp(np.median(z["log_s"][:5])))     # frozen five-frame anchor
        anchors = sorted(seg.anchors)
        persistent = None
        if mode == "stream":
            persistent = IncrementalMapper(dev, sem_into=into, n_teacher=C)
        for i in range(len(seg)):
            if mode == "stream":
                ts = time.time(); persistent.step(feed.frame(i))
                torch.cuda.synchronize(); lat["map"].append(time.time() - ts)
            if i not in anchors:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
            if target is None:
                n_skip += 1; continue
            if mode == "stream":
                if not persistent.scale_state.frozen:
                    n_skip += 1; continue
                m = persistent; used = list(range(i + 1)); sc = m.scale_state.scale
            else:
                offs = PAST5_OFFSETS if mode == "past5" else OCCANY_FWD_OFFSETS
                if mode == "occany_fwd" and i + max(offs) >= len(seg):
                    n_skip += 1; continue
                if mode == "past5" and i + min(offs) < 0:
                    n_skip += 1; continue
                ts = time.time()
                m, used = window_map(feed, seg, i, offs, scale, into, C, dev)
                torch.cuda.synchronize(); lat["map"].append((time.time() - ts) / max(len(used), 1))
                sc = scale
            P = feed.pose[i].copy(); P[:3, 3] *= sc
            Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
            q = m.query(MAP, Tgw)
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            base = q["logodds"]
            ts = time.time(); fin, pr_net = net.raw(q, MAP)
            torch.cuda.synchronize(); lat["net"].append(time.time() - ts)
            valid = _t(keep.reshape(-1), dev)
            gt = _t(((target != EVAL.empty_class) & keep).reshape(-1), dev)
            s_map, editable, forced = to_eval_grid(base, base, ds)
            s_fin = to_eval_grid(fin, base, ds)[0]
            cid, grp = str(f.gt_ref["clip_id"]), str(seg.name)
            regions = {"full": valid, "edit": valid & editable}
            for r, mk in regions.items():
                acc["mapper"][r].add(s_map, gt, mk, cid, grp)
                acc["completion"][r].add(s_fin, gt, mk, cid, grp)
            occ_native = q["occupied"] & (q["sem_w"] > 0)
            probs_map = q["sem"] / q["sem_w"].clamp_min(1e-9).unsqueeze(1)
            mp = occ_to_eval_grid(occ_native, ds)
            flat_n = occ_native.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
            dil_flat, dil_pr = dilate_with_semantics(flat_n, probs_map[occ_native],
                                                     tuple(MAP.dims), dev)
            dil_native = torch.zeros_like(occ_native)
            if len(dil_flat):
                dil_native[_t(dil_flat, dev)] = True
            mdil = occ_to_eval_grid(dil_native, ds)
            comp = (s_fin >= tau) & valid
            pred = {"mapper_native": mp, "mapper_dilate": mdil,
                    "all_valid_occupied": BL.all_valid_occupied(valid),
                    "editable_fill": BL.editable_fill(mp, editable),
                    "completion_occany_pool": PL.geometry_dilation(comp.reshape(edims)).reshape(-1),
                    "completion_occany_vote": PL.geometry_majority(comp.reshape(edims)).reshape(-1)}
            qq = BL.matched_density_q(int(comp.sum()), mp, editable, forced, valid)
            meta["random_q"].append(qq)
            for sd in RANDOM_SEEDS:
                pred[f"random_editable_s{sd}"] = BL.random_editable(mp, editable, forced,
                                                                    valid, qq, sd, n_done)
            for k, p_ in pred.items():
                bins[k].append(BL.binary_counts(p_, gt, valid))
            meta["clip_id"].append(cid); meta["group"].append(grp)
            meta["n_valid"].append(int(valid.sum())); meta["n_gt"].append(int((gt & valid).sum()))
            meta["n_editable"].append(int(regions["edit"].sum()))
            meta["n_gt_editable"].append(int((gt & regions["edit"]).sum()))
            # ---- semantics at the frozen threshold --------------------------------------
            sem_lab = {}
            for name, (occ_sel, prv) in (("completion", (fin >= tau, pr_net)),
                                         ("mapper_native", (occ_native, probs_map))):
                fl = occ_sel.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
                pp = prv[occ_sel]
                if G6G.NEEDS_REDUCTION[ds] and len(fl):
                    fl, pp = G6G.reduce_occ3d_probs(fl, pp)
                ch = out_ch[pp.argmax(1)] if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
                pl = np.full(n_eval, v.empty_label, np.int32)
                if len(fl):
                    pl[fl] = labels[ch.cpu().numpy()]
                sem_lab[name] = pl
                sem_counts.setdefault(name, []).append(G6M.clip_counts(
                    pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label,
                    dist, height, cid, grp, name))
            fl, pp = dil_flat, dil_pr
            if G6G.NEEDS_REDUCTION[ds] and len(fl):
                fl, pp = G6G.reduce_occ3d_probs(fl, pp)
            ch = out_ch[pp.argmax(1)] if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
            pl = np.full(n_eval, v.empty_label, np.int32)
            if len(fl):
                pl[fl] = labels[ch.cpu().numpy()]
            sem_counts.setdefault("mapper_dilate", []).append(G6M.clip_counts(
                pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label,
                dist, height, cid, grp, "mapper_dilate"))
            kf = keep.reshape(-1); tf = target.reshape(-1)
            gt_np = (tf != v.empty_label) & kf
            pc_, pm_ = sem_lab["completion"], sem_lab["mapper_native"]
            new_tp = (pc_ != v.empty_label) & (pm_ == v.empty_label) & gt_np
            obs_tp = (pc_ != v.empty_label) & (pm_ != v.empty_label) & gt_np
            newly.append((int(new_tp.sum()), int((new_tp & (pc_ == tf)).sum()),
                          int(obs_tp.sum()), int((obs_tp & (pc_ == tf)).sum())))
            n_done += 1
            if limit and n_done >= limit:
                break
            if n_done % 200 == 0:
                print(f"  [{ds} {tag}] {n_done} {(time.time()-t0)/n_done:.2f}s each", flush=True)
        if limit and n_done >= limit:
            break
    os.makedirs(art, exist_ok=True)
    out = {"dataset": ds, "mode": mode, "seed": seed, "tag": tag, "checkpoint": ckpt,
           "threshold": tau, "semantic_threshold": sem_tau, "lock_logodds": LOCK_LOGODDS,
           "causal": mode in ("stream", "past5"), "n_clips": n_done, "n_skipped": n_skip,
           "seconds": time.time() - t0, "random_seeds": list(RANDOM_SEEDS),
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "latency_ms": {k: {"median": 1e3 * float(np.median(x)),
                              "p95": 1e3 * float(np.percentile(x, 95))}
                          for k, x in lat.items() if x},
           "methods": {}, "baselines": {}, "semantic": {}}
    for name, regs in acc.items():
        blk = {}
        for r, a_ in regs.items():
            for k, val in a_.block().items():
                blk[f"{r}_{k}"] = val
        np.savez_compressed(os.path.join(art, f"scores_{ds}_{tag}_{name}.npz"), **blk)
        out["methods"][name] = {r: SC.summarize({k[len(r) + 1:]: vv for k, vv in blk.items()
                                                 if k.startswith(r + "_")}) for r in ("full", "edit")}
    np.savez_compressed(os.path.join(art, f"bincounts_{ds}_{tag}.npz"),
                        **{f"pred_{k}": np.asarray(val, np.int64) for k, val in bins.items()},
                        **{k: np.asarray(val) for k, val in meta.items() if val})
    nv, ng = np.asarray(meta["n_valid"]).sum(), np.asarray(meta["n_gt"]).sum()
    for k, val in bins.items():
        b = np.asarray(val, np.int64).sum(0)
        out["baselines"][k] = {"binary_iou": float(b[0] / max(b.sum(), 1)),
                               "precision": float(b[0] / max(b[0] + b[1], 1)),
                               "recall": float(b[0] / max(b[0] + b[2], 1)),
                               "density": float((b[0] + b[1]) / max(nv, 1)),
                               "pred_over_gt": float((b[0] + b[1]) / max(ng, 1)),
                               "tp": int(b[0]), "fp": int(b[1]), "fn": int(b[2])}
    n_ = np.asarray(newly, np.int64)
    np.savez_compressed(os.path.join(art, f"newly_{ds}_{tag}.npz"), counts=n_,
                        clip_id=np.asarray(meta["clip_id"]))
    out["newly_completed"] = {
        "n_new_true_positives": int(n_[:, 0].sum()), "n_named_correctly": int(n_[:, 1].sum()),
        "semantic_accuracy": float(n_[:, 1].sum() / max(n_[:, 0].sum(), 1)),
        "n_observed_true_positives": int(n_[:, 2].sum()),
        "observed_semantic_accuracy": float(n_[:, 3].sum() / max(n_[:, 2].sum(), 1))}
    for name, cc in sem_counts.items():
        _save_counts(ds, tag, name, cc, v, labels, art)
        s_ = G6M.summarize(G6M.aggregate(cc), v.names)
        out["semantic"][name] = {
            "binary_iou": s_["binary_iou"], "binary_precision": s_["binary_precision"],
            "binary_recall": s_["binary_recall"], "ssc_miou": s_["ssc_miou"],
            "tp_accuracy": s_["tp_conditioned"]["top1_accuracy"],
            "tp_balanced_recall": s_["tp_conditioned"]["balanced_recall"],
            "coverage_miss": s_["decomposition"]["coverage_miss_fraction"],
            "naming_error": s_["decomposition"]["naming_error_fraction"],
            "per_class_iou": {k: x["iou"] for k, x in s_["per_class"].items()}}
    write_json(os.path.join(art, f"eval_{ds}_{tag}.json"), out)
    fu = out["methods"]["completion"]["full"]
    cb = out["semantic"]["completion"]
    print(f"[{ds} {tag}] AP {fu['average_precision']:.4f} prev {fu['prevalence']:.4f} "
          f"x{fu['ap_over_prevalence']:.2f} AUROC {fu['auroc']:.4f} | completion@tau "
          f"IoU {cb['binary_iou']:.4f} P {cb['binary_precision']:.4f} R {cb['binary_recall']:.4f} "
          f"SSC {cb['ssc_miou']:.4f} | mapper {out['baselines']['mapper_native']['binary_iou']:.4f} "
          f"dil {out['baselines']['mapper_dilate']['binary_iou']:.4f} "
          f"allocc {out['baselines']['all_valid_occupied']['binary_iou']:.4f} "
          f"({n_done} clips)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--mode", required=True, choices=["stream", "past5", "occany_fwd"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"refusing to open {a.dataset}: {MANIFEST} does not exist yet")
    run(a.dataset, a.mode, a.seed, a.device, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
