# extracted from tools/gate8/evaluate.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Gate 8 evaluation: run the incremental mapper over a benchmark and score every official
timestamp with Gate 6's own metric code, with or without the completion module.

The map is built once per segment, one frame at a time, and *queried* at each anchor --
never rebuilt. Raw and fixed-dilation results are reported separately and never mixed.

    python tools/gate8/evaluate.py --source kitti360 --device cuda:1            # mapper only
    python tools/gate8/evaluate.py --source kitti360 --checkpoint ckpt.pt ...   # with completion
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.config import REPO_ROOT, write_json                              # noqa: E402
from core.datasets import grids as G6G, vocab            # noqa: E402
from core.evaluation import metrics as G6M, official_targets as G6T            # noqa: E402
from core.supervision import lifting as G6L                                                 # noqa: E402
from core.mapping import voxmap as VM                                                  # noqa: E402
from core.datasets import sources as S, union_vocab as V8                                      # noqa: E402
from core.mapping.feed import CachedFeed                                                # noqa: E402
from core.mapping.mapper import IncrementalMapper                                       # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8")


def dilate_with_semantics(occ_flat: np.ndarray, probs: torch.Tensor, dims, device):
    """The frozen ``dilate_r2`` plus Gate 6's nearest-source semantic propagation."""
    from core.evaluation.voxels import dilate
    n = int(np.prod(dims))
    occ = torch.zeros(n, dtype=torch.bool, device=device)
    if len(occ_flat):
        occ[torch.from_numpy(occ_flat).to(device)] = True
    dil = dilate(occ.view(dims), 2).reshape(-1)
    only = (dil & ~occ).nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
    if len(only) == 0 or len(occ_flat) == 0:
        return occ_flat, probs
    op, _ = G6L.propagate_dilation(tuple(dims), occ_flat, probs, only, 2, device)
    return np.concatenate([occ_flat, only]), torch.cat([probs, op], 0)


def run(source, device, checkpoint=None, limit_anchors=None, tag=None, dilate=True,
        vocab_mode="native", limit_segments=None):
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.init()
    ds = S.DATASET_OF[source]
    v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims))
    dist, height = G6M.band_masks(EVAL)
    labels = np.asarray(v.labels, np.int32)
    into = V8.into_matrix(ds) if vocab_mode == "union" else None
    out_ch = torch.as_tensor(V8.out_channel(ds), device=dev) if vocab_mode == "union" else None
    net = None
    if checkpoint:
        from core.model.net import load_checkpoint
        net = load_checkpoint(checkpoint, dev)
    segs = S.segments(source, REPO_ROOT)
    if limit_segments:
        segs = segs[:limit_segments]
    tag = tag or (f"{'complete' if net else 'mapper'}_{vocab_mode}")
    store = {"raw": [], "dil": []}
    lat = {"step": [], "query": [], "net": [], "dilate": []}
    mem, n_anchor, t0 = [], 0, time.time()
    for seg in segs:
        feed = CachedFeed(seg, dev)
        m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
        anchors = set(seg.anchors)
        for i in range(len(seg)):
            fr = feed.frame(i)
            ts = time.time()
            m.step(fr)                      # buffers until the scale is fixed, then integrates
            torch.cuda.synchronize(); lat["step"].append(time.time() - ts)
            if i not in anchors or not m.scale_state.frozen:
                continue
            f = seg.frames[i]
            ts = time.time()
            q = m.query(MAP, _grid_to_world(m, f, feed))
            torch.cuda.synchronize(); lat["query"].append(time.time() - ts)
            occ = q["occupied"] & (q["sem_w"] > 0)
            probs = q["sem"] / q["sem_w"].clamp_min(1e-9).unsqueeze(1)
            if net is not None:
                q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                       torch.full_like(q["last_time"], -1).float())
                ts = time.time()
                occ, probs = net.complete(q, MAP)
                torch.cuda.synchronize(); lat["net"].append(time.time() - ts)
            flat = occ.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
            pr = probs[occ]
            variants = {"raw": (flat, pr)}
            if dilate:
                ts = time.time()
                variants["dil"] = dilate_with_semantics(flat, pr, tuple(MAP.dims), dev)
                torch.cuda.synchronize(); lat["dilate"].append(time.time() - ts)
            target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
            if target is None:
                continue
            for name, (fl, pp) in variants.items():
                if G6G.NEEDS_REDUCTION[ds] and len(fl):
                    fl, pp = G6G.reduce_occ3d_probs(fl, pp)
                ch = pp.argmax(1) if len(fl) else torch.zeros(0, dtype=torch.long, device=dev)
                if out_ch is not None and len(fl):
                    ch = out_ch[ch]
                pl = np.full(n_eval, v.empty_label, np.int32)
                if len(fl):
                    pl[fl] = labels[ch.cpu().numpy()]
                store[name].append(G6M.clip_counts(
                    pl.reshape(EVAL.dims), None, target, keep, v.labels, v.empty_label,
                    dist, height, f.gt_ref["clip_id"], seg.name, name))
            mem.append(m.table.memory_bytes())
            n_anchor += 1
            if limit_anchors and n_anchor >= limit_anchors:
                break
        if limit_anchors and n_anchor >= limit_anchors:
            break
    res = {"source": source, "dataset": ds, "tag": tag, "checkpoint": checkpoint,
           "n_anchors": n_anchor, "vocab_mode": vocab_mode, "seconds": time.time() - t0,
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "map_bytes_median": float(np.median(mem)) if mem else 0.0,
           "map_bytes_max": float(max(mem)) if mem else 0.0,
           "latency_ms": {k: {"median": 1e3 * float(np.median(x)),
                              "p95": 1e3 * float(np.percentile(x, 95))}
                          for k, x in lat.items() if x},
           "variants": {}}
    for name, cc in store.items():
        if not cc:
            continue
        agg = G6M.aggregate(cc)
        s = G6M.summarize(agg, v.names)
        pc = np.array([[c.btp, c.bfp, c.bfn] for c in cc], np.float64)
        den = pc.sum(axis=1)
        s["binary_iou_mean_per_clip"] = float(np.mean(np.where(den > 0, pc[:, 0] / np.maximum(den, 1), 0)))
        res["variants"][name] = s
        np.savez_compressed(os.path.join(ART, f"counts_{source}_{tag}_{name}.npz"),
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
    write_json(os.path.join(ART, f"eval_{source}_{tag}.json"), res)
    for name, s in res["variants"].items():
        print(f"[{source} {tag} {name}] {n_anchor} anchors binIoU {s['binary_iou']:.4f} "
              f"P {s['binary_precision']:.4f} R {s['binary_recall']:.4f} mIoU {s['ssc_miou']:.4f} "
              f"TPacc {s['tp_conditioned']['top1_accuracy']:.4f}")
    return res


def _grid_to_world(m, f, feed):
    """grid -> anchor camera (calibrated) -> scaled world (frozen LingBot pose at anchor)."""
    T_cam_to_grid = np.asarray(f.T_cam_to_grid, np.float64)
    P = feed.pose[f.index].copy()
    P[:3, 3] *= float(m.scale_state.scale)
    return P @ np.linalg.inv(T_cam_to_grid)


def _default_device() -> str:
    """First GPU of the Gate-8 pool. ``GATE8_GPUS`` (default "1 2 3") reserves
    GPU 0 for other users; ``GATE8_DEVICE`` overrides outright."""
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit-anchors", type=int, default=None)
    ap.add_argument("--limit-segments", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--vocab", default="native", choices=["native", "union"])
    ap.add_argument("--no-dilate", action="store_true")
    a = ap.parse_args()
    run(a.source, a.device, a.checkpoint, a.limit_anchors, a.tag, not a.no_dilate,
        a.vocab, a.limit_segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
