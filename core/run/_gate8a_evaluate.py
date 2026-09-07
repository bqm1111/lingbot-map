# extracted from tools/gate8a/evaluate.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Gate 8A evaluation: continuous occupancy scores, two regions, every baseline, one pass.

The incremental map is built once per segment exactly as in Gate 8 (one frame at a time,
queried at each official anchor, never rebuilt). What changes is what comes out: instead
of a single binarized prediction this dumps the **final occupancy log-odds** for the
mapper and for every checkpoint supplied, histogrammed per anchor over

* region ``full`` -- the complete valid evaluation grid, and
* region ``edit`` -- the editable completion region, ``abs(base_logodds) < 2.0`` (exactly
  the gate of ``gate8.net.apply_residual``, via ``gate8a.regions``).

Every threshold-dependent number in the report is then derived from those histograms, so
no threshold is ever baked into a stored artifact. Fixed baselines that carry no score
(mapper's own rule, its 0.4 m dilation, all-valid-occupied, editable-fill) are counted in
the same pass so that all methods see byte-identical maps and masks.

    # Stage 1/3: source validation, all four ablation checkpoints in one map build
    python tools/gate8a/evaluate.py --source semantickitti \
        --checkpoints cellA=artifacts/gate8/checkpoints/completion_best.pt,... --device cuda:1

    # Stage 4: the single locked KITTI-360 run (adds semantics + random baselines)
    python tools/gate8a/evaluate.py --source kitti360 --checkpoints sel=<ckpt> \
        --locked sel --threshold <tau> --mapper-threshold <tau_map> --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.config import REPO_ROOT, write_json                              # noqa: E402
from core.datasets import grids as G6G, vocab            # noqa: E402
from core.evaluation import metrics as G6M, official_targets as G6T            # noqa: E402
from core.supervision import lifting as G6L                                                 # noqa: E402
from core.datasets import sources as S, union_vocab as V8                                      # noqa: E402
from core.mapping.feed import CachedFeed                                                # noqa: E402
from core.mapping.mapper import IncrementalMapper                                       # noqa: E402
from core.model.net import load_checkpoint, LOCK_LOGODDS                              # noqa: E402
from core.evaluation import baselines as BL, scores as SC                                 # noqa: E402
from core.evaluation.regions import to_eval_grid, occ_to_eval_grid                        # noqa: E402
from core.run._gate8_evaluate import dilate_with_semantics, _grid_to_world, _default_device  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")
RANDOM_SEEDS = (0, 1, 2, 3, 4)


def _t(x, dev):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def run(source, device, checkpoints, tag="scores", limit_anchors=None, limit_segments=None,
        locked=None, threshold=None, mapper_threshold=None, art=None):
    art = art or ART
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.init()
    ds = S.DATASET_OF[source]
    v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims))
    dist, height = G6M.band_masks(EVAL)
    labels = np.asarray(v.labels, np.int32)
    into = V8.into_matrix(ds); out_ch = torch.as_tensor(V8.out_channel(ds), device=dev)
    nets = {k: load_checkpoint(p, dev) for k, p in checkpoints.items()}
    acc = {m: {r: SC.ScoreAccumulator() for r in ("full", "edit")}
           for m in ["mapper"] + list(nets)}
    fixed = ["mapper_native", "mapper_dilate", "all_valid_occupied", "editable_fill"]
    if locked:
        fixed += [f"random_editable_s{s}" for s in RANDOM_SEEDS]
    bins = {m: [] for m in fixed}
    meta = {"clip_id": [], "group": [], "n_valid": [], "n_gt": [], "n_editable": [],
            "n_gt_editable": [], "random_q": []}
    sem_counts, newly = {}, []
    segs = S.segments(source, REPO_ROOT)
    if limit_segments:
        segs = segs[:limit_segments]
    n_anchor, t0 = 0, time.time()
    for seg in segs:
        feed = CachedFeed(seg, dev)
        m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
        anchors = set(seg.anchors)
        for i in range(len(seg)):
            m.step(feed.frame(i))
            if i not in anchors or not m.scale_state.frozen:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
            if target is None:
                continue
            q = m.query(MAP, _grid_to_world(m, f, feed))
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            base = q["logodds"]
            valid = _t(keep.reshape(-1), dev)
            gt = _t(((target != EVAL.empty_class) & keep).reshape(-1), dev)
            s_map, editable, forced = to_eval_grid(base, base, ds)
            cid, grp = str(f.gt_ref["clip_id"]), str(seg.name)
            regions = {"full": valid, "edit": valid & editable}
            for r, mask in regions.items():
                acc["mapper"][r].add(s_map, gt, mask, cid, grp)
            finals, probs_of = {}, {}
            for name, net in nets.items():
                fin, pr = net.raw(q, MAP)
                finals[name] = fin; probs_of[name] = pr
                s_e, _, _ = to_eval_grid(fin, base, ds)
                for r, mask in regions.items():
                    acc[name][r].add(s_e, gt, mask, cid, grp)
            # ---- fixed, score-free baselines, same map and same masks -------------
            occ_native = q["occupied"] & (q["sem_w"] > 0)
            probs_map = q["sem"] / q["sem_w"].clamp_min(1e-9).unsqueeze(1)
            mp = occ_to_eval_grid(occ_native, ds)
            flat_n = occ_native.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
            dil_flat, dil_pr = dilate_with_semantics(flat_n, probs_map[occ_native],
                                                     tuple(MAP.dims), dev)
            dil_native = torch.zeros_like(occ_native)
            if len(dil_flat):
                dil_native[_t(dil_flat, dev)] = True
            pred = {"mapper_native": mp,
                    "mapper_dilate": occ_to_eval_grid(dil_native, ds),
                    "all_valid_occupied": BL.all_valid_occupied(valid),
                    "editable_fill": BL.editable_fill(mp, editable)}
            if locked:
                n_target = int(((to_eval_grid(finals[locked], base, ds)[0] >= threshold)
                                & valid).sum())
                qq = BL.matched_density_q(n_target, mp, editable, forced, valid)
                meta["random_q"].append(qq)
                for s in RANDOM_SEEDS:
                    pred[f"random_editable_s{s}"] = BL.random_editable(
                        mp, editable, forced, valid, qq, s, n_anchor)
            for k, p in pred.items():
                assert BL.check_protected(p, mp, editable, valid) or k in (
                    "all_valid_occupied", "mapper_dilate"), k
                bins[k].append(BL.binary_counts(p, gt, valid))
            meta["clip_id"].append(cid); meta["group"].append(grp)
            meta["n_valid"].append(int(valid.sum())); meta["n_gt"].append(int((gt & valid).sum()))
            meta["n_editable"].append(int(regions["edit"].sum()))
            meta["n_gt_editable"].append(int((gt & regions["edit"]).sum()))
            # ---- locked mode: full Gate-6 semantic counts at the frozen threshold --
            if locked:
                # thresholding on the prediction grid and then applying the frozen
                # any-sub-voxel reduction is identical to max-reducing the score first.
                # ">=" matches the score histogram's bin rule exactly, so the semantic
                # counts and the histogram-derived binary counts cannot disagree.
                sem_variants = {
                    "completion": (finals[locked] >= threshold, probs_of[locked], False),
                    "mapper_native": (occ_native, probs_map, False),
                    "mapper_dilate": (dil_native, None, True),
                }
                if mapper_threshold is not None:
                    sem_variants["mapper_calibrated"] = (base >= mapper_threshold, probs_map, False)
                sem_labels = {}
                for name, (occ_sel, pr, is_dil) in sem_variants.items():
                    if is_dil:
                        fl, pp = dil_flat, dil_pr
                    else:
                        fl = occ_sel.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
                        pp = pr[occ_sel]
                    if G6G.NEEDS_REDUCTION[ds] and len(fl):
                        fl, pp = G6G.reduce_occ3d_probs(fl, pp)
                    ch = pp.argmax(1) if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
                    if len(fl):
                        ch = out_ch[ch]
                    pl = np.full(n_eval, v.empty_label, np.int32)
                    if len(fl):
                        pl[fl] = labels[ch.cpu().numpy()]
                    sem_counts.setdefault(name, []).append(G6M.clip_counts(
                        pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label,
                        dist, height, cid, grp, name))
                    sem_labels[name] = pl
                # semantic accuracy on the voxels the completion *added*: correct
                # occupancy the mapper did not have, and whether it is also named right
                kf = keep.reshape(-1); tf = target.reshape(-1)
                gt_np = (tf != v.empty_label) & kf
                pc_, pm_ = sem_labels["completion"], sem_labels["mapper_native"]
                new_tp = (pc_ != v.empty_label) & (pm_ == v.empty_label) & gt_np
                newly.append((int(new_tp.sum()), int((new_tp & (pc_ == tf)).sum())))
            n_anchor += 1
            if limit_anchors and n_anchor >= limit_anchors:
                break
        if limit_anchors and n_anchor >= limit_anchors:
            break
    # ------------------------------------------------------------------ write out
    os.makedirs(art, exist_ok=True)
    out = {"source": source, "dataset": ds, "tag": tag, "n_anchors": n_anchor,
           "checkpoints": checkpoints, "locked": locked, "threshold": threshold,
           "mapper_threshold": mapper_threshold, "lock_logodds": LOCK_LOGODDS,
           "seconds": time.time() - t0, "random_seeds": list(RANDOM_SEEDS),
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "score_bins": {"lo": SC.LOGIT_LO, "hi": SC.LOGIT_HI, "n": SC.N_BINS},
           "methods": {}, "baselines": {}}
    for name, regs in acc.items():
        blk = {}
        for r, a in regs.items():
            for k, val in a.block().items():
                blk[f"{r}_{k}"] = val
        np.savez_compressed(os.path.join(art, f"scores_{source}_{tag}_{name}.npz"), **blk)
        out["methods"][name] = {r: SC.summarize({k[len(r) + 1:]: v for k, v in blk.items()
                                                 if k.startswith(r + "_")})
                                for r in ("full", "edit")}
    bmeta = {k: np.asarray(val) for k, val in meta.items() if val}
    np.savez_compressed(os.path.join(art, f"bincounts_{source}_{tag}.npz"),
                        **{f"pred_{k}": np.asarray(val, np.int64) for k, val in bins.items()},
                        **bmeta)
    nv, ng = np.asarray(meta["n_valid"]).sum(), np.asarray(meta["n_gt"]).sum()
    for k, val in bins.items():
        b = np.asarray(val, np.int64).sum(0)
        out["baselines"][k] = {"binary_iou": float(b[0] / max(b.sum(), 1)),
                               "precision": float(b[0] / max(b[0] + b[1], 1)),
                               "recall": float(b[0] / max(b[0] + b[2], 1)),
                               "density": float((b[0] + b[1]) / max(nv, 1)),
                               "pred_over_gt": float((b[0] + b[1]) / max(ng, 1)),
                               "tp": int(b[0]), "fp": int(b[1]), "fn": int(b[2])}
    if newly:
        n = np.asarray(newly, np.int64)
        np.savez_compressed(os.path.join(art, f"newly_completed_{source}_{tag}.npz"),
                            counts=n, clip_id=np.asarray(meta["clip_id"]))
        out["newly_completed"] = {
            "n_new_true_positives": int(n[:, 0].sum()),
            "n_named_correctly": int(n[:, 1].sum()),
            "semantic_accuracy": float(n[:, 1].sum() / max(n[:, 0].sum(), 1))}
    for name, cc in sem_counts.items():
        _save_counts(source, tag, name, cc, v, labels, art)
        s = G6M.summarize(G6M.aggregate(cc), v.names)
        out.setdefault("semantic", {})[name] = {
            "binary_iou": s["binary_iou"], "binary_precision": s["binary_precision"],
            "binary_recall": s["binary_recall"], "ssc_miou": s["ssc_miou"],
            "tp_accuracy": s["tp_conditioned"]["top1_accuracy"],
            "tp_balanced_recall": s["tp_conditioned"]["balanced_recall"],
            "coverage_miss": s["decomposition"]["coverage_miss_fraction"],
            "naming_error": s["decomposition"]["naming_error_fraction"],
            "per_class_iou": {k: x["iou"] for k, x in s["per_class"].items()}}
    write_json(os.path.join(art, f"eval_{source}_{tag}.json"), out)
    for name, r in out["methods"].items():
        fu, ed = r["full"], r["edit"]
        print(f"[{source} {name}] full AP {fu['average_precision']:.4f} "
              f"prev {fu['prevalence']:.4f} AP/prev {fu['ap_over_prevalence']:.2f} "
              f"IoU@0 {fu['at_zero']['iou']:.4f} bestIoU {fu['best_iou']['iou']:.4f}"
              f"@{fu['best_iou']['threshold']:+.2f} | edit AP {ed['average_precision']:.4f} "
              f"prev {ed['prevalence']:.4f}", flush=True)
    for k, b in out["baselines"].items():
        print(f"   baseline {k:24s} IoU {b['binary_iou']:.4f} P {b['precision']:.4f} "
              f"R {b['recall']:.4f} pred/gt {b['pred_over_gt']:.2f}")
    return out


def _save_counts(source, tag, name, cc, v, labels, art=None):
    np.savez_compressed(os.path.join(art or ART, f"counts_{source}_{tag}_{name}.npz"),
                        clip_id=np.asarray([c.clip_id for c in cc]),
                        group=np.asarray([c.group for c in cc]),
                        tp=np.stack([c.tp for c in cc]).astype(np.int32),
                        fp=np.stack([c.fp for c in cc]).astype(np.int32),
                        fn=np.stack([c.fn for c in cc]).astype(np.int32),
                        conf=np.stack([c.conf for c in cc]).astype(np.int32),
                        fp_empty=np.stack([c.fp_empty for c in cc]).astype(np.int32),
                        decomp=np.stack([c.decomp for c in cc]).astype(np.int32),
                        binary=np.asarray([[c.btp, c.bfp, c.bfn, c.n_valid, c.n_gt_occupied,
                                            c.n_pred_occupied] for c in cc], np.int64),
                        labels=labels, names=np.asarray(v.names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--checkpoints", default="", help="name=path[,name=path...]")
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--tag", default="scores")
    ap.add_argument("--limit-anchors", type=int, default=None)
    ap.add_argument("--limit-segments", type=int, default=None)
    ap.add_argument("--locked", default=None, help="checkpoint name to lock (Stage 4)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mapper-threshold", type=float, default=None)
    a = ap.parse_args()
    ck = dict(kv.split("=", 1) for kv in a.checkpoints.split(",") if kv)
    if a.locked:
        assert a.locked in ck and a.threshold is not None, "--locked needs its checkpoint + tau"
    run(a.source, a.device, ck, a.tag, a.limit_anchors, a.limit_segments, a.locked,
        a.threshold, a.mapper_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
