#!/usr/bin/env python
"""Dump per-anchor SemanticKITTI predictions from a frozen Gate 8C-1 checkpoint.

The completion path is the one ``core/run/eval_target.py`` scores, reproduced here so the
saved volumes are exactly what produced the reported IoU -- the script recomputes TP/FP/FN
as it writes and prints the SC IoU, which must match the eval run for the same manifest,
mode and seed.

    G8C1_MANIFEST=artifacts/gate8c1/frozen_manifest_sky.json \
      python tools/skyfix/predict_semantickitti.py --seed 2 --out output/skyfix/pred

One ``.npz`` per anchor, all on the official 256x256x32 @ 0.2 m grid, XYZ order:

    pred_occ    packed bits, the thresholded completion (what SC IoU scores)
    pred_label  uint8, benchmark class id, 0 where the prediction is empty
    logodds     float16, final log-odds before the threshold -- re-threshold from this
    gt_label    uint8, the official target (255 = ignored, 0 = free)
    keep        packed bits, the official evaluation mask
    map_occ     packed bits, the raw incremental map before completion, for reference
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.run._common import REPO_ROOT, default_device                                # noqa: E402
from core.run.eval_target import MANIFEST, PAST5_OFFSETS, OCCANY_FWD_OFFSETS, window_map, _t  # noqa: E402
from core.datasets import grids as G6G, vocab, sources as S, union_vocab as V8        # noqa: E402
from core.evaluation import official_targets as G6T                                   # noqa: E402
from core.mapping.feed import CachedFeed                                              # noqa: E402
from core.mapping.mapper import IncrementalMapper                                     # noqa: E402
from core.model.net import load_checkpoint                                            # noqa: E402
from core.evaluation.regions import to_eval_grid                                      # noqa: E402

DS = "semantickitti"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", default="past5", choices=["stream", "past5", "occany_fwd"])
    ap.add_argument("--out", required=True, help="directory for the per-anchor npz files")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    ckpt, tau = fz["checkpoint"], fz["occupancy_threshold"]
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    v = vocab.load(DS); C = len(v); labels = np.asarray(v.labels, np.int32)
    MAP, EVAL = G6G.PREDICTION_GRID[DS], G6G.EVAL_GRID[DS]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(DS); out_ch = torch.as_tensor(V8.out_channel(DS), device=dev)
    net = load_checkpoint(os.path.join(REPO_ROOT, ckpt), dev)
    os.makedirs(a.out, exist_ok=True)
    print(f"manifest {MANIFEST}\ncheckpoint {ckpt}  tau {tau}  mode {a.mode}", flush=True)

    tp = fp = fn = 0
    n_done, n_skip, t0 = 0, 0, time.time()
    index = []
    for seg in S.segments(DS, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(DS, seg.name)) as z:
            scale = float(np.exp(np.median(z["log_s"][:5])))
        anchors = sorted(seg.anchors)
        persistent = IncrementalMapper(dev, sem_into=into, n_teacher=C) \
            if a.mode == "stream" else None
        for i in range(len(seg)):
            if a.mode == "stream":
                persistent.step(feed.frame(i))
            if i not in anchors:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(DS, f.gt_ref, REPO_ROOT)
            if target is None:
                n_skip += 1; continue
            if a.mode == "stream":
                if not persistent.scale_state.frozen:
                    n_skip += 1; continue
                m, sc = persistent, persistent.scale_state.scale
            else:
                offs = PAST5_OFFSETS if a.mode == "past5" else OCCANY_FWD_OFFSETS
                if a.mode == "occany_fwd" and i + max(offs) >= len(seg):
                    n_skip += 1; continue
                if a.mode == "past5" and i + min(offs) < 0:
                    n_skip += 1; continue
                m, _u = window_map(feed, seg, i, offs, scale, into, C, dev)
                sc = scale
            P = feed.pose[i].copy(); P[:3, 3] *= sc
            q = m.query(MAP, P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64)))
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            base = q["logodds"]
            fin, pr_net = net.raw(q, MAP)

            valid = _t(keep.reshape(-1), dev)
            s_fin = to_eval_grid(fin, base, DS)[0]
            comp = (s_fin >= tau) & valid                       # exactly what SC IoU scores

            # semantics on the completed voxels, benchmark label ids, empty elsewhere
            fl = comp.nonzero(as_tuple=True)[0]
            pl = np.full(int(np.prod(edims)), v.empty_label, np.int32)
            if len(fl):
                ch = out_ch[pr_net[fl].argmax(1)]
                pl[fl.cpu().numpy()] = labels[ch.cpu().numpy()]

            gt = _t(((target != EVAL.empty_class) & keep).reshape(-1), dev)
            c = comp.cpu().numpy(); g = gt.cpu().numpy()
            tp += int((c & g).sum()); fp += int((c & ~g).sum()); fn += int((~c & g).sum())

            cid = str(f.gt_ref["clip_id"])
            np.savez_compressed(
                os.path.join(a.out, f"{cid}.npz"),
                pred_occ=np.packbits(c), pred_label=pl.reshape(edims).astype(np.uint8),
                logodds=s_fin.cpu().numpy().astype(np.float16).reshape(edims),
                gt_label=target.astype(np.uint8), keep=np.packbits(keep.reshape(-1)),
                map_occ=np.packbits(to_eval_grid(base, base, DS)[0].cpu().numpy() >= 0.0),
                dims=np.asarray(edims, np.int32), origin=np.asarray(EVAL.origin, np.float64),
                voxel_size=np.float64(EVAL.voxel_size), threshold=np.float64(tau),
                sequence=str(f.gt_ref["sequence"]), frame=np.int32(f.gt_ref["frame_ids"][-1]))
            index.append({"clip_id": cid, "sequence": str(f.gt_ref["sequence"]),
                          "frame": int(f.gt_ref["frame_ids"][-1]),
                          "n_pred": int(c.sum()), "n_gt": int(g.sum())})
            n_done += 1
            if a.limit and n_done >= a.limit:
                break
            if n_done % 40 == 0:
                print(f"  {n_done} anchors, {(time.time()-t0)/n_done:.2f}s each", flush=True)
        if a.limit and n_done >= a.limit:
            break

    iou = tp / max(tp + fp + fn, 1)
    meta = {"dataset": DS, "mode": a.mode, "seed": a.seed, "manifest": MANIFEST,
            "checkpoint": ckpt, "threshold": tau, "n_anchors": n_done, "n_skipped": n_skip,
            "sc_iou": iou, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "seconds": time.time() - t0, "anchors": index}
    with open(os.path.join(a.out, "index.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n{n_done} anchors -> {a.out}\nSC IoU {iou*100:.2f}  "
          f"P {meta['precision']*100:.2f}  R {meta['recall']*100:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
