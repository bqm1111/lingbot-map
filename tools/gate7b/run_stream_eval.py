#!/usr/bin/env python
"""Gate 7B — evaluate one (variant, scale policy, horizon) over a whole benchmark.

At every official Gate-6 clip anchor the persistent map is materialised from the causal
prefix of the stream that ends at that frame, transformed into the benchmark's own
evaluation grid, and scored with **Gate 6's own metric code**, so every number here is
directly comparable with Gates 6 and 7A.

    python tools/gate7b/run_stream_eval.py --dataset kitti360 --variant S1 \
        --scale G-A --horizon all --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G, metrics as G6M, pipelines as G6P, \
    targets as G6T, vocab                                                        # noqa: E402
from gates.gate7b import config as C, depth as D7, evidence as EV, fuse as FZ, \
    scale as SC, streams as ST                                                   # noqa: E402
from gates.gate7b.voxmap import EvidenceVolume                                         # noqa: E402
from tools.gate7b.stream_lingbot import seg_path                                 # noqa: E402
from tools.gate7b.scale_candidates import OUT_ROOT as SCALE_ROOT                 # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")


class LRU(OrderedDict):
    """Frame-keyed cache; consecutive anchors share most of their causal window."""

    def __init__(self, cap):
        super().__init__()
        self.cap = int(cap)

    def fetch(self, key, make):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = make(key)
        self[key] = v
        while len(self) > self.cap:
            self.popitem(last=False)
        return v


def anchor_transform(dataset, rec, repo_root, cache):
    """Anchor camera -> the benchmark's grid frame. Exactly Gate 6's convention."""
    if dataset == "semantickitti":
        seq = rec["sequence"]
        if seq not in cache:
            from prompted_lingbot.occ_datasets import SemanticKittiOccSpec
            root = os.path.join(repo_root, G6F.SEMANTICKITTI_ROOT)
            cache[seq] = SemanticKittiOccSpec.build(root, seq).cam_to_velo
        return cache[seq]
    cid = rec["clip_id"]
    p = G6F.lingbot_cache_path(dataset, cid, repo_root)
    with np.load(p) as z:
        if dataset == "kitti360":
            return z["rect_cam_to_velo"].astype(np.float64)
        return z["T_camera_to_ego"][-1].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--variant", default="S1", choices=["S1", "S2", "S3", "S4"])
    ap.add_argument("--scale", default="G-A", choices=list(C.SCALE_POLICIES))
    ap.add_argument("--horizon", default="all")
    ap.add_argument("--conf", type=float, default=EV.S2_PRIMARY_CONF)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-anchors", type=int, default=None)
    ap.add_argument("--skip-anchors", type=int, default=0)
    ap.add_argument("--limit-segments", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    ds, variant, pol = a.dataset, a.variant, a.scale
    horizon = "all" if a.horizon == "all" else int(a.horizon)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(dev)

    v = vocab.load(ds)
    Cn = len(v)
    MAP_GRID = G6G.PREDICTION_GRID[ds]          # canonical map lattice (Gate-6 identical)
    EVAL = G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims))
    dist_bands, height_bands = G6M.band_masks(EVAL)
    recs = {r.clip_id: r.raw for r in G6F.read_manifest(ds, REPO_ROOT)}
    groups = G6P.load_groups(ds, REPO_ROOT)
    segs = ST.build(ds, REPO_ROOT)
    if a.limit_segments:
        segs = segs[:a.limit_segments]
    moge_index = D7.moge_frame_index(ds, REPO_ROOT)
    calib_cache = {}
    labels = np.asarray(v.labels, np.int32)

    tag = a.tag or (f"{variant}_{pol}_h{a.horizon}"
                    + (f"_c{a.conf:g}" if variant == "S2" else ""))
    store, per_clip, fuse_stats = [], [], []
    scale_series = {}
    t0, n_anchor = time.time(), 0
    vol_reuse = None

    for seg in segs:
        sp = seg_path(ds, seg.name)
        scp = os.path.join(SCALE_ROOT, ds, f"{seg.name}.npz")
        if not (os.path.exists(sp) and os.path.exists(scp)):
            raise SystemExit(f"missing stream/scale cache for {ds}/{seg.name}")
        with np.load(sp) as z:
            stream = {k: z[k] for k in ("pred_depth", "pred_depth_conf", "pred_pose_c2w",
                                        "pred_K")}
            keys = [str(k) for k in z["keys"]]
        cand = np.load(scp)["log_s"]

        # ---- causal gauge: one pass over the segment, strictly forward in time
        st = SC.ScaleState(policy=pol)
        s_of = np.zeros(len(keys))
        for i in range(len(keys)):
            s_of[i] = st.observe(i, {"log_s": float(cand[i])})
        scale_series[seg.name] = st.diagnostics()
        scale_series[seg.name]["series"] = st.series.tolist()[:2000]

        win = len(keys) if horizon == "all" else int(horizon)
        sem_cache = LRU(min(win + 8, 160))
        moge_cache = LRU(min(win + 8, 160))

        def sem_loader(f, _keys=keys):
            def make(k):
                p = G6F.semantic_cache_path(ds, _keys[k])
                if not os.path.exists(p):
                    return None
                with np.load(p) as zz:
                    return torch.from_numpy(
                        zz["probs"].astype(np.float32)).permute(1, 2, 0).to(dev)
            return sem_cache.fetch(f, make)

        def moge_loader(f, _keys=keys):
            def make(k):
                return D7.load_moge_frame(ds, _keys[k], moge_index)
            return moge_cache.fetch(f, make)

        anchors = sorted(seg.anchors.items(), key=lambda kv: kv[1])
        if a.skip_anchors:
            anchors = anchors[a.skip_anchors:]
        for cid, t in anchors:
            rec = recs.get(cid)
            if rec is None:
                continue
            T_ag = anchor_transform(ds, rec, REPO_ROOT, calib_cache)
            lo = FZ.horizon_start(t, horizon)
            reach = FZ.reach_mask(stream["pred_pose_c2w"].astype(np.float64), t,
                                  float(s_of[t]), T_ag, MAP_GRID, D7.MAX_DEPTH_M)
            frames = [f for f in range(lo, t + 1) if reach[f]]
            assert not frames or max(frames) <= t, "a future frame entered the window"

            if vol_reuse is None:
                vol_reuse = EvidenceVolume(tuple(MAP_GRID.dims), dev, Cn)
            else:
                vol_reuse.reset()
            vol = vol_reuse
            ts = time.time()
            fs = FZ.fuse_window(vol, stream, frames, t, lambda _f: s_of[t], T_ag,
                                MAP_GRID, variant, a.conf, dev,
                                moge_loader=moge_loader, sem_loader=sem_loader,
                                veto_free=(variant == "S4"))
            fs["seconds"] = time.time() - ts
            fs["clip_id"] = cid
            fs["n_window"] = len(frames)
            fuse_stats.append(fs)

            occ = vol.occupied()
            ch = vol.semantic_channel()
            n_lb = vol.n_occ_lingbot
            first = vol.first_time
            vols = vol.volumes()

            # A voxel with no semantic evidence at all is not emitted: labelling it
            # channel 0 would invent a class. With band-wide semantic fusion this set is
            # empty in practice, and the count is recorded so the claim can be checked.
            n_unlabelled = int((occ & (vol.sem_w <= 0)).sum())
            occ = occ & (vol.sem_w > 0)
            flat = occ.nonzero(as_tuple=True)[0]
            chan = ch[flat].clamp(min=0)
            sup = (n_lb[flat] > 0).to(torch.uint8)
            newer = (first[flat] < max(t - 4, 0)).to(torch.uint8)   # outside the 5-frame set
            if G6G.NEEDS_REDUCTION[ds]:
                fc = flat.cpu().numpy().astype(np.int64)
                probs = vol.sem[flat]
                if len(fc):
                    fn, pn = G6G.reduce_occ3d_probs(fc, probs)
                    idx = G6G.unflat(fc, MAP_GRID) // G6G.RATIO
                    nat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]),
                                               EVAL.dims)
                    uq, inv = np.unique(nat, return_inverse=True)
                    supn = np.zeros(len(uq), np.uint8)
                    np.maximum.at(supn, inv, sup.cpu().numpy())
                    newn = np.zeros(len(uq), np.uint8)
                    np.maximum.at(newn, inv, newer.cpu().numpy())
                    flat_e = fn
                    chan_e = pn.argmax(1).cpu().numpy().astype(np.int64)
                    sup_e, new_e = supn, newn
                else:
                    flat_e = np.zeros(0, np.int64)
                    chan_e = np.zeros(0, np.int64)
                    sup_e = np.zeros(0, np.uint8)
                    new_e = np.zeros(0, np.uint8)
            else:
                flat_e = flat.cpu().numpy().astype(np.int64)
                chan_e = chan.cpu().numpy().astype(np.int64)
                sup_e = sup.cpu().numpy()
                new_e = newer.cpu().numpy()

            target, keep = G6T.semantic_target(ds, rec, REPO_ROOT)
            if target is None:
                continue
            pl = np.full(n_eval, v.empty_label, np.int32)
            ps = np.zeros(n_eval, np.uint8)
            if len(flat_e):
                pl[flat_e] = labels[chan_e]
                ps[flat_e] = sup_e
            cc = G6M.clip_counts(pl.reshape(EVAL.dims), ps.reshape(EVAL.dims), target,
                                 keep, v.labels, v.empty_label, dist_bands, height_bands,
                                 cid, groups.get(cid, "?"), tag)
            store.append(cc)

            # provenance of the occupied set, for the newly-recovered semantic tables
            per_clip.append({
                "clip_id": cid, "t": int(t), "n_window": len(frames),
                "scale": float(s_of[t]),
                "n_occ_eval": int(len(flat_e)),
                "n_occ_lingbot": int(sup_e.sum()) if len(flat_e) else 0,
                "n_occ_moge_only": int((sup_e == 0).sum()) if len(flat_e) else 0,
                "n_occ_beyond_five_frames": int(new_e.sum()) if len(flat_e) else 0,
                "map_occupied": vols["occupied"], "map_free": vols["free"],
                "map_unknown": vols["unknown"],
                "map_provisional": vols["provisional"],
                "map_bytes": vol.memory_bytes(),
                "n_occupied_without_semantics": n_unlabelled,
                "fuse_seconds": fs["seconds"]})
            n_anchor += 1
            if a.limit_anchors and n_anchor >= a.limit_anchors:
                break
            if n_anchor % 100 == 0:
                print(f"  [{ds} {tag}] {n_anchor} anchors "
                      f"{(time.time()-t0)/n_anchor:.2f}s/anchor", flush=True)
        if a.limit_anchors and n_anchor >= a.limit_anchors:
            break

    agg = G6M.aggregate(store)
    summary = G6M.summarize(agg, v.names) if agg else {}
    pc = np.array([[c.btp, c.bfp, c.bfn] for c in store], np.float64)
    if len(pc):
        den = pc.sum(axis=1)
        summary["binary_iou_mean_per_clip"] = float(
            np.mean(np.where(den > 0, pc[:, 0] / np.maximum(den, 1), 0.0)))
        summary["binary_iou_pooled"] = summary["binary_iou"]
    out = {"dataset": ds, "variant": variant, "scale_policy": pol,
           "horizon": a.horizon, "s2_conf": a.conf if variant == "S2" else None,
           "tag": tag, "n_anchors": len(store), "grid": EVAL.name,
           "map_grid": MAP_GRID.name,
           "seconds": time.time() - t0,
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "summary": summary,
           "scale_diagnostics": scale_series,
           "per_clip": per_clip,
           "fuse": {"median_seconds_per_anchor":
                    float(np.median([f["seconds"] for f in fuse_stats]))
                    if fuse_stats else 0.0,
                    "median_window": float(np.median([f["n_window"]
                                                      for f in fuse_stats]))
                    if fuse_stats else 0.0,
                    "total_rays": int(sum(f["n_rays"] for f in fuse_stats)),
                    "total_moge_rays": int(sum(f["n_moge_rays"] for f in fuse_stats)),
                    "total_occ_updates": int(sum(f["n_occ_updates"]
                                                 for f in fuse_stats)),
                    "total_free_updates": int(sum(f["n_free_updates"]
                                                  for f in fuse_stats))}}
    write_json(os.path.join(ART, f"eval_{ds}_{tag}.json"), out)
    np.savez_compressed(
        os.path.join(ART, f"counts_{ds}_{tag}.npz"),
        clip_id=np.asarray([c.clip_id for c in store]),
        group=np.asarray([c.group for c in store]),
        tp=np.stack([c.tp for c in store]).astype(np.int32) if store else np.zeros((0, Cn), np.int32),
        fp=np.stack([c.fp for c in store]).astype(np.int32) if store else np.zeros((0, Cn), np.int32),
        fn=np.stack([c.fn for c in store]).astype(np.int32) if store else np.zeros((0, Cn), np.int32),
        conf=np.stack([c.conf for c in store]).astype(np.int32) if store else np.zeros((0, Cn, Cn), np.int32),
        conf_support=np.stack([c.conf_support for c in store]).astype(np.int32) if store else np.zeros((0, Cn, Cn), np.int32),
        conf_dilonly=np.stack([c.conf_dilonly for c in store]).astype(np.int32) if store else np.zeros((0, Cn, Cn), np.int32),
        decomp=np.stack([c.decomp for c in store]).astype(np.int32) if store else np.zeros((0, Cn, 3), np.int32),
        decomp_dist=np.stack([c.decomp_dist for c in store]).astype(np.int32) if store else np.zeros((0, len(dist_bands), 3), np.int32),
        fp_empty=np.stack([c.fp_empty for c in store]).astype(np.int32) if store else np.zeros((0, Cn), np.int32),
        binary=np.asarray([[c.btp, c.bfp, c.bfn, c.n_valid, c.n_gt_occupied,
                            c.n_pred_occupied] for c in store], np.int64),
        labels=np.asarray(v.labels, np.int32), names=np.asarray(v.names))
    s = summary
    print(f"[{ds} {tag}] {len(store)} anchors  binIoU {s.get('binary_iou', 0):.4f} "
          f"mIoU {s.get('ssc_miou', 0):.4f}  P {s.get('binary_precision', 0):.4f} "
          f"R {s.get('binary_recall', 0):.4f}  "
          f"miss {s.get('decomposition', {}).get('coverage_miss_fraction', 0):.4f}  "
          f"{out['seconds']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
