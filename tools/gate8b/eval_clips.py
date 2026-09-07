#!/usr/bin/env python
"""Gate 8B matched five-frame evaluation: fresh map per official clip, completion applied
after frame five, scored threshold-free in both regions with every declared baseline and
with OccAny's exact post-processing as separate columns.

Per clip: ``gate8b.clips.build_map`` (five real frames of one camera, scale fixed from those
five, each integrated once, no memory across clips) -> query -> completion. The score
histograms, fixed baselines and semantic counts follow ``tools/gate8a/evaluate.py`` so the
two settings are scored by identical code; the only additions are the OccAny-pooled
columns (``gate8b.pooling``), applied on the evaluation grid, never tuned.

    python tools/gate8b/eval_clips.py --dataset semantickitti --checkpoint <ckpt> \
        --threshold <tau> --mapper-threshold <tau_m> --tag clips_semantickitti
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G, metrics as G6M, targets as G6T, vocab            # noqa: E402
from gates.gate8 import vocab as V8                                                    # noqa: E402
from gates.gate8.net import load_checkpoint, LOCK_LOGODDS                              # noqa: E402
from gates.gate8a import baselines as BL, scores as SC                                 # noqa: E402
from gates.gate8a.regions import to_eval_grid, occ_to_eval_grid                        # noqa: E402
from gates.gate8b import pooling as PL                                                 # noqa: E402
from gates.gate8b.clips import ClipFeed, build_map, grid_to_world                      # noqa: E402
from tools.gate8a.evaluate import dilate_with_semantics, _save_counts, RANDOM_SEEDS  # noqa: E402


def _t(x, dev):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def _labels_of(occ_native, probs, ds, out_ch, labels, v, n_eval, dev):
    fl = occ_native.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
    pp = probs[occ_native]
    if G6G.NEEDS_REDUCTION[ds] and len(fl):
        fl, pp = G6G.reduce_occ3d_probs(fl, pp)
    ch = pp.argmax(1) if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
    if len(fl):
        ch = out_ch[ch]
    pl = np.full(n_eval, v.empty_label, np.int32)
    if len(fl):
        pl[fl] = labels[ch.cpu().numpy()]
    return pl


def run(dataset, device, checkpoint, threshold, mapper_threshold, tag, limit=None, art=None):
    art = art or ART
    dev = torch.device(device); torch.cuda.set_device(dev)
    ds = dataset; v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims)); edims = tuple(int(d) for d in EVAL.dims)
    dist, height = G6M.band_masks(EVAL)
    labels = np.asarray(v.labels, np.int32)
    into = V8.into_matrix(ds); out_ch = torch.as_tensor(V8.out_channel(ds), device=dev)
    net = load_checkpoint(checkpoint, dev)
    feed = ClipFeed(ds, REPO_ROOT, dev)
    acc = {m: {r: SC.ScoreAccumulator() for r in ("full", "edit")} for m in ("mapper", "selected")}
    fixed = ["mapper_native", "mapper_dilate", "all_valid_occupied", "editable_fill",
             "completion_occany_dilation", "completion_occany_majority",
             "mapper_native_occany_dilation", "mapper_native_occany_majority",
             "mapper_dilate_occany_dilation"] + [f"random_editable_s{s}" for s in RANDOM_SEEDS]
    bins = {m: [] for m in fixed}
    meta = {"clip_id": [], "group": [], "n_valid": [], "n_gt": [], "n_editable": [],
            "n_gt_editable": [], "random_q": []}
    sem_counts, newly, n_done, n_skip, t0 = {}, [], 0, 0, time.time()
    lat = []
    for i in range(len(feed)):
        clip = feed.clip(i)
        if clip is None:
            n_skip += 1; continue
        target, keep = G6T.semantic_target(ds, clip.gt_ref, REPO_ROOT)
        if target is None:
            n_skip += 1; continue
        ts = time.time()
        m = build_map(clip, dev, into, C)
        q = m.query(MAP, grid_to_world(clip, m.scale_state.scale))
        q["age"] = torch.where(q["last_time"] >= 0, (len(clip.frames) - 1 - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        base = q["logodds"]
        fin, probs_net = net.raw(q, MAP)
        torch.cuda.synchronize(); lat.append(time.time() - ts)
        valid = _t(keep.reshape(-1), dev)
        gt = _t(((target != EVAL.empty_class) & keep).reshape(-1), dev)
        s_map, editable, forced = to_eval_grid(base, base, ds)
        s_fin = to_eval_grid(fin, base, ds)[0]
        cid, grp = clip.clip_id, clip.group
        regions = {"full": valid, "edit": valid & editable}
        for r, mask in regions.items():
            acc["mapper"][r].add(s_map, gt, mask, cid, grp)
            acc["selected"][r].add(s_fin, gt, mask, cid, grp)
        occ_native = q["occupied"] & (q["sem_w"] > 0)
        probs_map = q["sem"] / q["sem_w"].clamp_min(1e-9).unsqueeze(1)
        mp = occ_to_eval_grid(occ_native, ds)
        flat_n = occ_native.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
        dil_flat, dil_pr = dilate_with_semantics(flat_n, probs_map[occ_native], tuple(MAP.dims), dev)
        dil_native = torch.zeros_like(occ_native)
        if len(dil_flat):
            dil_native[_t(dil_flat, dev)] = True
        mdil = occ_to_eval_grid(dil_native, ds)
        comp = (s_fin >= threshold) & valid
        pred = {"mapper_native": mp, "mapper_dilate": mdil,
                "all_valid_occupied": BL.all_valid_occupied(valid),
                "editable_fill": BL.editable_fill(mp, editable),
                "completion_occany_dilation": PL.geometry_dilation(comp.reshape(edims)).reshape(-1),
                "completion_occany_majority": PL.geometry_majority(comp.reshape(edims)).reshape(-1),
                "mapper_native_occany_dilation": PL.geometry_dilation((mp & valid).reshape(edims)).reshape(-1),
                "mapper_native_occany_majority": PL.geometry_majority((mp & valid).reshape(edims)).reshape(-1),
                "mapper_dilate_occany_dilation": PL.geometry_dilation((mdil & valid).reshape(edims)).reshape(-1)}
        n_target = int(comp.sum())
        qq = BL.matched_density_q(n_target, mp, editable, forced, valid)
        meta["random_q"].append(qq)
        for s in RANDOM_SEEDS:
            pred[f"random_editable_s{s}"] = BL.random_editable(mp, editable, forced, valid, qq, s, n_done)
        for k, p in pred.items():
            bins[k].append(BL.binary_counts(p, gt, valid))
        meta["clip_id"].append(cid); meta["group"].append(grp)
        meta["n_valid"].append(int(valid.sum())); meta["n_gt"].append(int((gt & valid).sum()))
        meta["n_editable"].append(int(regions["edit"].sum()))
        meta["n_gt_editable"].append(int((gt & regions["edit"]).sum()))
        # ---- semantics at the locked threshold (">=" matches the histogram bin rule) --
        sem_variants = {"completion": (fin >= threshold, probs_net),
                        "mapper_native": (occ_native, probs_map),
                        "mapper_calibrated": (base >= mapper_threshold, probs_map)}
        sem_labels = {}
        for name, (occ_sel, pr) in sem_variants.items():
            pl = _labels_of(occ_sel, pr, ds, out_ch, labels, v, n_eval, dev)
            sem_labels[name] = pl
            sem_counts.setdefault(name, []).append(G6M.clip_counts(
                pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label,
                dist, height, cid, grp, name))
        # the 0.4 m dilation carries Gate 6's nearest-source semantics
        fl, pp = dil_flat, dil_pr
        if G6G.NEEDS_REDUCTION[ds] and len(fl):
            fl, pp = G6G.reduce_occ3d_probs(fl, pp)
        ch = out_ch[pp.argmax(1)] if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
        pl = np.full(n_eval, v.empty_label, np.int32)
        if len(fl):
            pl[fl] = labels[ch.cpu().numpy()]
        sem_counts.setdefault("mapper_dilate", []).append(G6M.clip_counts(
            pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label, dist, height,
            cid, grp, "mapper_dilate"))
        # OccAny semantic-mode separate pooling on the completion's labels (occupancy unchanged)
        lab_c = torch.from_numpy(sem_labels["completion"]).to(dev)
        chan = torch.full((int(labels.max()) + 2,), -1, dtype=torch.long, device=dev)
        chan[torch.as_tensor(labels, device=dev)] = torch.arange(len(labels), device=dev)
        occ_c = lab_c != v.empty_label
        cl = torch.where(occ_c, chan[lab_c.clamp(min=0)], torch.full_like(lab_c, len(labels)))
        pooled = PL.semantic_separate(cl.reshape(edims), len(labels) + 1, len(labels),
                                      len(labels)).reshape(-1)
        pl2 = np.full(n_eval, v.empty_label, np.int32)
        occ_np = occ_c.cpu().numpy(); pooled_np = pooled.cpu().numpy()
        pl2[occ_np] = labels[np.clip(pooled_np[occ_np], 0, len(labels) - 1)]
        sem_counts.setdefault("completion_occany_semantic", []).append(G6M.clip_counts(
            pl2.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label, dist, height,
            cid, grp, "completion_occany_semantic"))
        kf = keep.reshape(-1); tf = target.reshape(-1)
        gt_np = (tf != v.empty_label) & kf
        pc_, pm_ = sem_labels["completion"], sem_labels["mapper_native"]
        new_tp = (pc_ != v.empty_label) & (pm_ == v.empty_label) & gt_np
        obs_tp = (pc_ != v.empty_label) & (pm_ != v.empty_label) & gt_np
        newly.append((int(new_tp.sum()), int((new_tp & (pc_ == tf)).sum()),
                      int(obs_tp.sum()), int((obs_tp & (pc_ == tf)).sum())))
        n_done += 1
        if limit and n_done >= limit:
            break
        if n_done % 200 == 0:
            print(f"  [{ds} clips] {n_done} done {(time.time()-t0)/n_done:.2f}s each", flush=True)
    os.makedirs(art, exist_ok=True)
    out = {"dataset": ds, "setting": "matched_five_frame", "tag": tag, "n_clips": n_done,
           "n_skipped": n_skip, "checkpoint": checkpoint, "threshold": threshold,
           "mapper_threshold": mapper_threshold, "lock_logodds": LOCK_LOGODDS,
           "seconds": time.time() - t0, "random_seeds": list(RANDOM_SEEDS),
           "latency_ms_map_plus_net": {"median": 1e3 * float(np.median(lat)),
                                       "p95": 1e3 * float(np.percentile(lat, 95))},
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "methods": {}, "baselines": {}, "semantic": {}}
    for name, regs in acc.items():
        blk = {}
        for r, a in regs.items():
            for k, val in a.block().items():
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
    n = np.asarray(newly, np.int64)
    np.savez_compressed(os.path.join(art, f"newly_completed_{ds}_{tag}.npz"), counts=n,
                        clip_id=np.asarray(meta["clip_id"]))
    out["newly_completed"] = {"n_new_true_positives": int(n[:, 0].sum()),
                              "n_named_correctly": int(n[:, 1].sum()),
                              "semantic_accuracy": float(n[:, 1].sum() / max(n[:, 0].sum(), 1)),
                              "n_observed_true_positives": int(n[:, 2].sum()),
                              "observed_semantic_accuracy": float(n[:, 3].sum() / max(n[:, 2].sum(), 1))}
    for name, cc in sem_counts.items():
        _save_counts(ds, tag, name, cc, v, labels, art)
        s = G6M.summarize(G6M.aggregate(cc), v.names)
        out["semantic"][name] = {
            "binary_iou": s["binary_iou"], "binary_precision": s["binary_precision"],
            "binary_recall": s["binary_recall"], "ssc_miou": s["ssc_miou"],
            "tp_accuracy": s["tp_conditioned"]["top1_accuracy"],
            "tp_balanced_recall": s["tp_conditioned"]["balanced_recall"],
            "coverage_miss": s["decomposition"]["coverage_miss_fraction"],
            "naming_error": s["decomposition"]["naming_error_fraction"],
            "per_class_iou": {k: x["iou"] for k, x in s["per_class"].items()}}
    write_json(os.path.join(art, f"eval_{ds}_{tag}.json"), out)
    for name, r in out["methods"].items():
        fu = r["full"]
        print(f"[{ds} clips {name}] AP {fu['average_precision']:.4f} prev {fu['prevalence']:.4f} "
              f"AP/prev {fu['ap_over_prevalence']:.2f} AUROC {fu['auroc']:.4f} "
              f"IoU@0 {fu['at_zero']['iou']:.4f}", flush=True)
    for k in ("mapper_native", "mapper_dilate", "completion_occany_dilation"):
        b = out["baselines"][k]
        print(f"   {k:30s} IoU {b['binary_iou']:.4f} P {b['precision']:.4f} R {b['recall']:.4f}")
    sc = out["semantic"].get("completion", {})
    print(f"   completion@tau IoU {sc.get('binary_iou', float('nan')):.4f} SSC mIoU "
          f"{sc.get('ssc_miou', float('nan')):.4f}  ({n_done} clips, {n_skip} skipped)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d", "kitti360"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--mapper-threshold", type=float, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.dataset, a.device, a.checkpoint, a.threshold, a.mapper_threshold,
        a.tag or f"clips_{a.dataset}", a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
