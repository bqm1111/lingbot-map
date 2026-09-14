#!/usr/bin/env python
"""Fixed qualitative comparisons on development drive 0006: human-label ground truth vs the
small and full seed-0 final checkpoints, bird's-eye (lowest non-empty label per column)
restricted to the valid mask, with the unobserved->later-measured region outlined.

    python tools/kitti360_data_scaling_7drives/qualitative.py --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import union_vocab as V8, vocab                                        # noqa: E402
from core.model.net import load_checkpoint, apply_residual                                # noqa: E402
from core.supervision import targets as TG                                                # noqa: E402

FIXED = (4, 604, 1204)
DIMS = (256, 256, 32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = drives7.VAL_DRIVE
    steps = json.load(open(os.path.join(drives7.ART, "budget.json")))["steps"]
    nets = {r: load_checkpoint(os.path.join(drives7.ART, "checkpoints", f"kds7_{r}_seed0_step{steps:05d}.pt"), dev).net.eval() for r in ("small", "full")}
    v = vocab.load("kitti360"); labels = torch.as_tensor(np.asarray(v.labels), device=dev); lab_of_union = labels[torch.as_tensor(V8.out_channel("kitti360"), device=dev)]
    cmap = plt.get_cmap("tab20", 19); od = os.path.join(drives7.ART, "figures", "qualitative"); os.makedirs(od, exist_ok=True)
    def bev(vol):
        vol = vol.reshape(DIMS); out = np.zeros(DIMS[:2], np.int32)
        for z in range(DIMS[2] - 1, -1, -1):
            s = vol[:, :, z]; out = np.where((s > 0) & (s < 255), s, out)
        return out
    for t in FIXED:
        with np.load(os.path.join(drives7.DATA_ROOT, "samples_gtsem", d, f"{t:05d}.npz")) as z:
            s = TG.unpack_sample(z, dev)
        with np.load(os.path.join(drives7.DATA_ROOT, "eval_masks", d, f"{t:05d}.npz")) as z:
            n = int(np.prod(DIMS)); gt = z["sem_gt"].astype(np.int64); valid = np.unpackbits(z["valid"])[:n].astype(bool)
            later = np.unpackbits(z["unobserved_later"])[:n].astype(bool)
        preds = {}
        for r, net in nets.items():
            x = s["input"][None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                occ_res, sem_log = net(x)
            fin = apply_residual(s["base_logodds"], occ_res[0, 0].float()).reshape(-1)
            semw = x[0, 6].reshape(-1); mp = x[0, 7:7 + V8.U].reshape(V8.U, -1).T; pn = F.softmax(sem_log[0].float().reshape(V8.U, -1).T, 1)
            probs = torch.where((semw > 0).unsqueeze(1), mp, pn)
            pl = torch.where(fin > 0, lab_of_union[probs.argmax(1)], torch.zeros_like(fin, dtype=torch.long)).cpu().numpy()
            preds[r] = np.where(valid, pl, 0)
        gtv = np.where(valid & (gt < 255), gt, 0)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        for k, (name, vol) in enumerate((("human labels (LiDAR-occupied, valid)", gtv), ("small seed 0", preds["small"]), ("full seed 0", preds["full"]))):
            img = bev(vol); rgb = np.ones(img.shape + (3,)); nz = img > 0; rgb[nz] = np.asarray([cmap(int(c))[:3] for c in img[nz]])
            ax[k].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower"); ax[k].contour(later.reshape(DIMS).any(2).T, levels=[0.5], colors="k", linewidths=0.4)
            ax[k].set_title(name, fontsize=10); ax[k].set_xticks([]); ax[k].set_yticks([])
        present = sorted(set(np.unique(bev(gtv))) | set(np.unique(bev(preds["full"]))) - {0})
        fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=cmap(int(c)), label=v.names[list(v.labels).index(c)]) for c in present if c in v.labels], loc="lower center", ncol=9, fontsize=7)
        fig.suptitle(f"0006 anchor {t}: bird's-eye semantic completion (black outline = unobserved at t, later measured)"); fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(os.path.join(od, f"t{t:05d}.png"), dpi=90); plt.close(fig)
    print(f"qualitative -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
