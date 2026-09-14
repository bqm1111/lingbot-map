#!/usr/bin/env python
"""Frozen evaluator: one checkpoint on the development drive 0006.

    python tools/kitti360_data_scaling_7drives/evaluate.py --checkpoint <path> --device cuda:0 [--stride 5]

Per anchor the 32-channel causal input of the sample is run through the checkpoint under
the adopted bf16 inference convention. Occupancy: final log-odds = apply_residual(base,
residual) > 0 (one fixed threshold for every run). Semantics: the adopted readout
(``Completer.raw``) -- the map's own teacher distribution where the causal map has
evidence, the network's softmax elsewhere -- argmax over the 25-way union, mapped to the
18 KITTI-360 classes by the frozen ``union_vocab.out_channel``. Predicted label = empty (0)
where the occupancy decision is empty.

Regions (frozen masks, ``build_eval_masks.py``): ``unobserved_later`` (PRIMARY), ``observed``,
``full`` (= target valid). Semantic scoring uses voxels with a valid semantic ground truth
(free, or LiDAR-occupied with a human class label); a GT-occupied voxel predicted empty is
an error of its class. mIoU averages the 18 classes with GT support in the region; the
empty class is scored but excluded from the mean. Occupancy IoU uses the LiDAR target over
all valid voxels of the region. Teacher agreement: argmax agreement with the GT-registered
Trident target on target voxels. Per-anchor confusion matrices are saved for resampling.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import union_vocab as V8, vocab                                        # noqa: E402
from core.model.net import load_checkpoint, apply_residual                                # noqa: E402
from core.supervision import targets as TG                                                # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

REGIONS = ("unobserved_later", "observed", "full")
NC = 19                                                                                   # 0 = empty, 1..18 classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    net = load_checkpoint(a.checkpoint, dev).net.eval()
    v = vocab.load("kitti360"); labels = torch.as_tensor(np.asarray(v.labels), device=dev)
    out_ch = torch.as_tensor(V8.out_channel("kitti360"), device=dev)                       # union -> channel
    lab_of_union = labels[out_ch]                                                          # union -> label id 1..18
    d = drives7.VAL_DRIVE
    mask_files = sorted(glob.glob(os.path.join(drives7.DATA_ROOT, "eval_masks", d, "*.npz")))[::a.stride]
    conf = {r: np.zeros((len(mask_files), NC, NC), np.int64) for r in REGIONS}             # rows GT, cols pred
    occ = {r: np.zeros((len(mask_files), 3), np.int64) for r in REGIONS}                   # tp fp fn
    teach = {r: np.zeros((len(mask_files), 2), np.int64) for r in REGIONS}                 # n, agree
    ids = []; t0 = time.time()
    for k, mf in enumerate(mask_files):
        t = int(os.path.basename(mf)[:-4]); ids.append(t)
        with np.load(os.path.join(drives7.DATA_ROOT, "samples_gtsem", d, f"{t:05d}.npz")) as z:
            s = TG.unpack_sample(z, dev)
        with np.load(mf) as z:
            n = int(np.prod(s["gt_occ"].shape))
            M = {r: torch.from_numpy(np.unpackbits(z[r if r != "full" else "valid"])[:n].astype(bool)).to(dev) for r in REGIONS}
            occ_t = torch.from_numpy(np.unpackbits(z["occupied"])[:n].astype(bool)).to(dev)
            sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
        x = s["input"][None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            occ_res, sem_log = net(x)
        final = apply_residual(s["base_logodds"], occ_res[0, 0].float()).reshape(-1)
        semw = x[0, 6].reshape(-1); mp = x[0, 7:7 + V8.U].reshape(V8.U, -1).T
        pn = F.softmax(sem_log[0].float().reshape(V8.U, -1).T, 1)
        probs = torch.where((semw > 0).unsqueeze(1), mp, pn)
        pocc = final > 0
        plab = torch.where(pocc, lab_of_union[probs.argmax(1)], torch.zeros_like(sem_gt))
        tvalid = s["fut_valid"].reshape(-1); tcls = s["fut_p"].reshape(-1, V8.U).argmax(1)
        for r in REGIONS:
            m = M[r]
            sm = m & (sem_gt != 255)
            idx = sem_gt[sm] * NC + plab[sm]
            conf[r][k] = torch.bincount(idx, minlength=NC * NC).reshape(NC, NC).cpu().numpy()
            gt = occ_t & m; pr = pocc & m
            occ[r][k] = [int((gt & pr).sum()), int((~gt & pr & m).sum()), int((gt & ~pr).sum())]
            tm = m & tvalid
            teach[r][k] = [int(tm.sum()), int((probs.argmax(1)[tm] == tcls[tm]).sum())]
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(mask_files)} anchors, {(time.time()-t0)/(k+1):.2f}s each", flush=True)

    def summary(r):
        cm = conf[r].sum(0); tp = np.diag(cm).astype(float); gt_n = cm.sum(1); pr_n = cm.sum(0)
        iou = np.where(gt_n + pr_n - tp > 0, tp / np.maximum(gt_n + pr_n - tp, 1), np.nan)
        present = gt_n[1:] > 0; o = occ[r].sum(0); te = teach[r].sum(0)
        return {"semantic_miou": float(np.nanmean(iou[1:][present])), "empty_iou": float(iou[0]),
                "per_class": {v.names[c]: {"label": int(v.labels[c]), "iou": (None if np.isnan(iou[c + 1]) else float(iou[c + 1])), "gt_support": int(gt_n[c + 1]),
                                           "predicted": int(pr_n[c + 1])} for c in range(18)},
                "n_classes_with_support": int(present.sum()), "occupancy_iou": float(o[0] / max(o.sum(), 1)),
                "occupancy_precision": float(o[0] / max(o[0] + o[1], 1)), "occupancy_recall": float(o[0] / max(o[0] + o[2], 1)),
                "semantic_voxels": int(cm.sum()), "gt_occupied_labelled_voxels": int(gt_n[1:].sum()),
                "teacher_agreement": float(te[1] / max(te[0], 1)), "teacher_voxels": int(te[0])}
    res = {"checkpoint": os.path.relpath(a.checkpoint, drives7.REPO_ROOT), "checkpoint_sha256": C0.sha256_file(a.checkpoint), "stride": a.stride,
           "n_anchors": len(mask_files), "anchor_ids": ids, "threshold": 0.0, "regions": {r: summary(r) for r in REGIONS}, "seconds": time.time() - t0}
    out = a.out or os.path.join(drives7.ART, "eval", os.path.basename(a.checkpoint)[:-3] + (f"_stride{a.stride}" if a.stride > 1 else "") + ".json")
    C0.write_json(out, res)
    np.savez_compressed(out[:-5] + "_per_anchor.npz", ids=np.asarray(ids), **{f"conf_{r}": conf[r] for r in REGIONS}, **{f"occ_{r}": occ[r] for r in REGIONS},
                        **{f"teach_{r}": teach[r] for r in REGIONS})
    for r in REGIONS:
        s_ = res["regions"][r]
        print(f"  [{r}] mIoU {s_['semantic_miou']:.4f} ({s_['n_classes_with_support']} classes) occ IoU {s_['occupancy_iou']:.4f} teacher {s_['teacher_agreement']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
