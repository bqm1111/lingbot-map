#!/usr/bin/env python
"""Gate 8C-1: end-to-end check of the convolution-padding fix.

For each seed: re-select the occupancy threshold with the gate's own rule (final-logit
threshold maximising full-grid IoU on KITTI-360 drive 0006 -- source domain only, so the
dataset firewall holds), then evaluate the target benchmark and report the occupancy
profile by height, which is where the bug showed itself.

    python tools/gate8c1/padfix_evaluate.py --dataset semantickitti --device cuda:2
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, targets as TG, vocab as V8                       # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import apply_residual, load_checkpoint                            # noqa: E402
from gates.gate8a.regions import to_eval_grid                                          # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map        # noqa: E402
from tools.gate8c1.figure_3d_gallery import anchors                              # noqa: E402

VAL_DIR = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/2013_05_28_drive_0006_sync"
CKPT = "artifacts/gate8c1/checkpoints/{tag}_{which}.pt"


def select_tau(comp, files, dev):
    """The gate's rule, unchanged: maximise full-grid IoU on KITTI-360 drive 0006."""
    taus = np.round(np.arange(-2.0, 2.0001, 0.0625), 4)
    acc = np.zeros((len(taus), 3), dtype=np.int64)
    for p_ in files:
        with np.load(p_) as z:
            d = TG.unpack_sample(z, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ_res, _ = comp.net(d["input"][None])
        final = apply_residual(d["base_logodds"], occ_res[0, 0].float()).reshape(-1)
        valid = d["gt_valid"].reshape(-1).bool()
        f, g = final[valid], (d["gt_occ"] > 0).reshape(-1)[valid]
        for j, t in enumerate(taus):
            pr = f >= float(t)
            acc[j, 0] += int((pr & g).sum()); acc[j, 1] += int((pr & ~g).sum())
            acc[j, 2] += int((~pr & g).sum())
    iou = acc[:, 0] / np.maximum(acc.sum(1), 1)
    j = int(np.argmax(iou))
    return float(taus[j]), float(iou[j])


def evaluate(ds, comp, tau, dev, n):
    C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
    tp = fp = fn = 0
    prof = np.zeros(edims[2]); gtp = np.zeros(edims[2]); nz = np.zeros(edims[2])
    for seg, i in anchors(ds, n):
        f = seg.frames[i]
        target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
        if target is None:
            continue
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(ds, seg.name)) as z:
            sc = float(np.exp(np.median(z["log_s"][:5])))
        m, _ = window_map(feed, seg, i, PAST5_OFFSETS, sc, into, C, dev)
        P = feed.pose[i].copy(); P[:3, 3] *= sc
        Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
        q = m.query(MAP, Tgw)
        q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        fin, _ = comp.raw(q, MAP)
        pr = (to_eval_grid(fin, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
        gt = ((target != EVAL.empty_class) & keep).reshape(edims)
        val = keep.reshape(edims)
        p, g = pr & val, gt & val
        tp += int((p & g).sum()); fp += int((p & ~g).sum()); fn += int((~p & g).sum())
        prof += p.sum(axis=(0, 1)); gtp += g.sum(axis=(0, 1)); nz += val.sum(axis=(0, 1))
        del m; torch.cuda.empty_cache()
    return ({"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
             "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn},
            100 * prof / np.maximum(nz, 1), 100 * gtp / np.maximum(nz, 1), edims, EVAL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="semantickitti",
                    choices=["semantickitti", "occ3d"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.npz")))[:a.n_val]
    frozen = json.load(open(MANIFEST))["seeds"]
    out = {"dataset": a.dataset, "n_anchors": a.n, "n_source_val": len(files), "seeds": {}}
    for s in (0, 1, 2):
        for name, tag, ftau in (("zeros (frozen)", f"seed{s}", frozen[str(s)]["occupancy_threshold"]),
                                ("replicate (fixed)", f"padfix_seed{s}", None)):
            ck = os.path.join(REPO_ROOT, CKPT.format(tag=tag, which=a.which))
            if not os.path.exists(ck):
                continue
            comp = load_checkpoint(ck, dev)
            tau, src = select_tau(comp, files, dev)
            m, prof, gtp, edims, EVAL = evaluate(a.dataset, comp, tau, dev, a.n)
            key = f"seed{s}/{name}"
            out["seeds"][key] = {"checkpoint": tag, "selected_tau": tau,
                                 "source_iou": src, "frozen_tau": ftau, "target": m,
                                 "pred_pct_by_z": prof.tolist(),
                                 "gt_pct_by_z": gtp.tolist()}
            print(f"{key:26s} tau {tau:+.4f}  source IoU {100*src:5.2f}  "
                  f"target IoU {100*m['iou']:5.2f}  P {100*m['precision']:5.2f}  "
                  f"R {100*m['recall']:5.2f}", flush=True)
            del comp; torch.cuda.empty_cache()
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    zc = (np.arange(edims[2]) + .5) * vs + z0
    out["z_centres_m"] = zc.tolist()
    ks = [k for k in out["seeds"] if k.startswith("seed0")]
    print(f"\nseed 0 occupancy by height (%)\n{'z (m)':>7} {'GT':>7}" +
          "".join(f"{k.split('/')[1][:9]:>12}" for k in ks))
    for j in range(edims[2]):
        print(f"{zc[j]:+7.1f} {out['seeds'][ks[0]]['gt_pct_by_z'][j]:7.2f}" +
              "".join(f"{out['seeds'][k]['pred_pct_by_z'][j]:12.2f}" for k in ks))
    with open(os.path.join(ART, f"padfix_evaluate_{a.dataset}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote artifacts/gate8c1/padfix_evaluate_{a.dataset}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
