#!/usr/bin/env python
"""Gate 13 figures 6-7: representative M and O inputs and completion outputs on 0006.

    python tools/gate13_oracle/figures13.py --device cuda:0

Deterministic selection: per-anchor full-region completion IoU of M seed 0 at the 90th / 50th / 10th
percentile (good / median / bad). Seed-0 final checkpoints of both arms. Bird's-eye max over height.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
from g13 import g12                                                                          # noqa: E402
sys.path.insert(0, g13.REPO_ROOT)
from core.model.net import load_checkpoint, apply_residual                                   # noqa: E402

DIMS = (256, 256, 32); N = int(np.prod(DIMS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(g13.ART, "eval", "g13_M_seed0_step06000_counts.npz")); ids = z["ids"]; per = z["counts"][:, 0, 5]
    iou = per[:, 0] / np.maximum(per.sum(1), 1); order = np.argsort(iou, kind="stable")
    reps = {lab: int(ids[order[int(q * (len(order) - 1))]]) for lab, q in (("good", 0.9), ("median", 0.5), ("bad", 0.1))}
    nets = {c: load_checkpoint(os.path.join(g13.ART, "checkpoints", f"g13_{c}_seed0_step06000.pt"), dev).net.eval() for c in g13.CONDITIONS}
    od = os.path.join(g13.ART, "figures", "maps"); os.makedirs(od, exist_ok=True)
    u = lambda zz, k: np.unpackbits(zz[k])[:N].astype(bool)
    fig6, ax6 = plt.subplots(3, 2, figsize=(9, 13))
    for i, (lab, t) in enumerate(reps.items()):
        zm = np.load(g13.m_path(g13.VAL_DRIVE, t)); zo = np.load(g13.o_path(g13.VAL_DRIVE, t)); zr = np.load(os.path.join(g12.DATA_ROOT, "regions", g13.VAL_DRIVE, f"{t:05d}.npz"))
        inp = {"M": (u(zm, "S_H"), u(zm, "F_H")), "O": (u(zo, "S_O"), u(zo, "F_O"))}
        valid, occ = u(zr, "valid"), u(zr, "occupied"); gt = (valid & occ).reshape(DIMS).any(2)
        preds, ious = {}, {}
        for c, (S, F) in inp.items():
            x = torch.from_numpy(np.stack([S, np.zeros(N, bool), F]).astype(np.float32).reshape((1, 3) + DIMS)).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res, _ = nets[c](x)
            p = (apply_residual(torch.zeros(DIMS, device=dev), res[0, 0].float()).reshape(-1) > 0).cpu().numpy()
            tp = (p & occ & valid).sum(); fp = (p & ~occ & valid).sum(); fn = (~p & occ & valid).sum(); ious[c] = tp / max(tp + fp + fn, 1); preds[c] = p
        for j, c in enumerate(g13.CONDITIONS):
            S, F = inp[c]; img = np.ones(DIMS[:2] + (3,)); img[F.reshape(DIMS).any(2)] = (0.78, 0.88, 0.97); img[S.reshape(DIMS).any(2)] = (0.80, 0.15, 0.15)
            ax6[i, j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax6[i, j].set_xticks([]); ax6[i, j].set_yticks([])
            ax6[i, j].set_title(f"{lab} t={t}: input {c} (S {S.sum():,}, F {F.sum():,})", fontsize=9)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
        for j, c in enumerate(g13.CONDITIONS):
            p2 = (preds[c] & valid).reshape(DIMS).any(2); img = np.ones(DIMS[:2] + (3,))
            img[gt & ~p2] = (0.75, 0.75, 0.75); img[p2 & ~gt] = (0.85, 0.25, 0.25); img[p2 & gt] = (0.10, 0.55, 0.20)
            ax[j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[j].set_xticks([]); ax[j].set_yticks([]); ax[j].set_title(f"completion {c} seed 0: anchor IoU {ious[c]:.3f}", fontsize=10)
        img = np.ones(DIMS[:2] + (3,)); img[gt] = (0.2, 0.2, 0.2)
        ax[2].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].set_title("target occupied (valid)", fontsize=10)
        fig.suptitle(f"7. {lab} case (M seed 0 anchor IoU percentile rule), 0006 t={t}; bird's-eye over height: green hit, red false occupied, grey missed", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(od, f"7_completion_{lab}_t{t:05d}.png"), dpi=85); plt.close(fig)
    fig6.suptitle("6. inputs for the same anchors: red S (surface), pale blue F (free)", fontsize=10); fig6.tight_layout()
    fig6.savefig(os.path.join(od, "6_inputs_M_vs_O.png"), dpi=85); plt.close(fig6)
    print("representative", reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
