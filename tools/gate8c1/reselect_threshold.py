#!/usr/bin/env python
"""Gate 8C-1: re-select the occupancy threshold after the boundary-padding fix.

The frozen tau was chosen by the gate's own rule -- *the final-logit threshold maximising
full-grid IoU on KITTI-360 drive 0006* -- while the zero-padding boundary bug was present,
so it is calibrated against a model whose ceiling and floor were systematically inflated.
Fixing the padding changes the score distribution, which makes the old tau wrong.

This re-runs **exactly that rule**, unchanged, on **exactly that data** (the source-domain
validation drive). No target dataset is opened, so the re-selection is legitimate under the
gate's dataset firewall.

    python tools/gate8c1/reselect_threshold.py --pad-z 8 --device cuda:2
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate8 import targets as TG                                                 # noqa: E402
from gates.gate8.net import apply_residual, load_checkpoint                            # noqa: E402
from tools.gate8c1.eval_target import MANIFEST                                   # noqa: E402

VAL_DIR = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/2013_05_28_drive_0006_sync"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pad-z", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    comp = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.npz")))[:a.n]
    print(f"KITTI-360 drive 0006: {len(files)} validation samples", flush=True)

    taus = np.round(np.arange(-2.0, 2.0001, 0.0625), 4)
    acc = {p: np.zeros((len(taus), 3), dtype=np.int64) for p in a.pad_z}
    for k, p_ in enumerate(files):
        with np.load(p_) as z:
            d = TG.unpack_sample(z, dev)
        inp = d["input"][None]
        gt = (d["gt_occ"] > 0).reshape(-1)
        valid = d["gt_valid"].reshape(-1).bool()
        for pad in a.pad_z:
            x = comp._pad_z(inp, pad) if pad else inp
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ_res, _ = comp.net(x)
            if pad:
                occ_res = occ_res[..., pad:pad + inp.shape[-1]]
            final = apply_residual(d["base_logodds"], occ_res[0, 0].float()).reshape(-1)
            f, g = final[valid], gt[valid]
            for j, t in enumerate(taus):
                pr = f >= float(t)
                acc[pad][j, 0] += int((pr & g).sum())
                acc[pad][j, 1] += int((pr & ~g).sum())
                acc[pad][j, 2] += int((~pr & g).sum())
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    out = {"rule": ("final-logit threshold maximising full-grid IoU on KITTI-360 drive "
                    "0006 -- the gate's own rule, unchanged"),
           "n_val_samples": len(files), "frozen_tau": float(fz["occupancy_threshold"]),
           "by_pad": {}}
    print(f"\n{'pad_z':>6} {'best tau':>10} {'source IoU':>12} "
          f"{'IoU at frozen tau':>19}")
    for pad in a.pad_z:
        tp, fp, fn = acc[pad][:, 0], acc[pad][:, 1], acc[pad][:, 2]
        iou = tp / np.maximum(tp + fp + fn, 1)
        j = int(np.argmax(iou))
        jf = int(np.argmin(np.abs(taus - float(fz["occupancy_threshold"]))))
        out["by_pad"][str(pad)] = {
            "best_tau": float(taus[j]), "best_source_iou": float(iou[j]),
            "iou_at_frozen_tau": float(iou[jf]),
            "precision_at_best": float(tp[j] / max(tp[j] + fp[j], 1)),
            "recall_at_best": float(tp[j] / max(tp[j] + fn[j], 1)),
            "curve": {"tau": taus.tolist(), "iou": iou.tolist()}}
        print(f"{pad:6d} {taus[j]:+10.4f} {100*iou[j]:12.2f} {100*iou[jf]:19.2f}")
    with open(os.path.join(ART, "reselect_threshold.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote artifacts/gate8c1/reselect_threshold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
