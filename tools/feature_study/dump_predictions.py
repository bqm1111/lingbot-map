#!/usr/bin/env python
"""Predictions of all three conditions on the frozen comparison set, side by side.

The eight drive-0006 anchors come from ``prediction_set.json``, written before any
checkpoint existed by a rule that consults no model output.

Each anchor produces one NPZ (the raw masks, for anyone who wants to recount) and one PNG:
a bird's-eye view of the map-unobserved/later-measured population only -- the population the
study is about -- with the causal map's own occupancy beside it for reference.

For C it also renders the **predicted feature field** projected to three channels. That is a
feature visualisation: the colours are the first three principal directions of a SAM
embedding, they are not class labels, and nothing in this study licenses reading them as
object names.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import apply_residual, load_checkpoint                        # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from tools.feature_study import feature_targets as FT                             # noqa: E402
from tools.feature_study.losses import features_at                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device(a.device); torch.cuda.set_device(dev)
    sel = json.load(open(os.path.join(ST.ART, "prediction_set.json")))
    out_dir = os.path.join(ST.ART, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    nets, taus = {}, {}
    for c in ST.CONDITIONS:
        nets[c] = load_checkpoint(ST.checkpoint_path(c, a.seed), dev).net
        taus[c] = json.load(open(os.path.join(
            ST.ART, "source", f"source_{ST.run_tag(c, a.seed)}.json")))["occupancy_threshold"]

    for idx in sel["stream_indices"]:
        with np.load(SRC.sample_path(SRC.VAL_DRIVE, idx)) as z:
            d = TG.unpack_sample(z, dev)
        n = d["gt_occ"].numel()
        with np.load(ST.eval_mask_path(SRC.VAL_DRIVE, idx)) as z:
            um = lambda k: torch.from_numpy(np.unpackbits(z[k])[:n].astype(bool)).to(dev)
            later = um("later_measured"); obs = um("map_observed")
        X, Y, Z = d["gt_occ"].shape
        gt = (d["gt_occ"] > 0).reshape(-1) & d["gt_valid"].reshape(-1)
        store = {"later_measured": np.packbits(later.cpu().numpy()),
                 "map_observed": np.packbits(obs.cpu().numpy()),
                 "gt_occupied": np.packbits(gt.cpu().numpy()),
                 "map_occupied": np.packbits((d["base_logodds"].reshape(-1) > 0).cpu().numpy()),
                 "dims": np.asarray([X, Y, Z], np.int32), "seed": np.int32(a.seed)}
        feat_rgb = None
        for c in ST.CONDITIONS:
            with torch.no_grad():
                occ_res, _, trunk = nets[c](d["input"][None], return_trunk=True)
                fin = apply_residual(d["base_logodds"], occ_res[0, 0].float()).reshape(-1)
                pred = (fin >= taus[c])
                if c == "C":
                    ix = torch.arange(n, device=dev)[later]
                    pf = features_at(nets[c], trunk, ix)
                    pf = pf / pf.norm(dim=1, keepdim=True).clamp_min(1e-9)
                    full = torch.zeros(n, 3, device=dev)
                    full[later] = pf[:, :3]
                    feat_rgb = full.reshape(X, Y, Z, 3).cpu().numpy()
            store[f"pred_{c}"] = np.packbits(pred.cpu().numpy())
            store[f"logodds_{c}"] = fin.cpu().numpy().astype(np.float16)
            store[f"tau_{c}"] = np.float32(taus[c])
        np.savez_compressed(os.path.join(out_dir, f"anchor_{idx:05d}_seed{a.seed}.npz"), **store)

        # ---- figure: bird's eye of the later-measured population -------------------
        L = later.reshape(X, Y, Z).cpu().numpy()
        G = (gt.reshape(X, Y, Z).cpu().numpy() & L)
        M = (d["base_logodds"] > 0).cpu().numpy()
        panels = [("causal map (any z)", M.max(2)),
                  ("later-measured GT occupied", G.max(2))]
        for c in ST.CONDITIONS:
            p = np.unpackbits(store[f"pred_{c}"])[:n].astype(bool).reshape(X, Y, Z) & L
            panels.append((f"{c}: predicted occupied", p.max(2)))
        ncol = len(panels) + (1 if feat_rgb is not None else 0)
        fig, ax = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.4), constrained_layout=True)
        for k, (t, img) in enumerate(panels):
            ax[k].imshow(img.T, origin="lower", cmap="magma", interpolation="nearest")
            ax[k].set_title(t, fontsize=8); ax[k].set_xticks([]); ax[k].set_yticks([])
        if feat_rgb is not None:
            v = feat_rgb.max(2)
            v = (v - v.min()) / max(float(v.max() - v.min()), 1e-9)
            ax[-1].imshow(np.transpose(v, (1, 0, 2)), origin="lower", interpolation="nearest")
            ax[-1].set_title("C: predicted FEATURE field\n(PCA 1-3, not class labels)",
                             fontsize=8)
            ax[-1].set_xticks([]); ax[-1].set_yticks([])
        fig.suptitle(f"drive 0006 anchor {idx}  -  seed {a.seed}  -  "
                     f"map-unobserved / later-measured voxels only", fontsize=9)
        fig.savefig(os.path.join(out_dir, f"anchor_{idx:05d}_seed{a.seed}.png"), dpi=130)
        plt.close(fig)
        print(f"  anchor {idx}: written", flush=True)
    ST.write_json(os.path.join(ST.ART, "predictions", "index.json"),
                  {"seed": a.seed, "anchors": sel["stream_indices"],
                   "thresholds": taus, "selection": sel["rule"],
                   "note": "PCA colours in the C panel are a feature visualisation, not "
                           "semantic labels", "env": ST.environment()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
