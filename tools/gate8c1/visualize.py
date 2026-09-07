#!/usr/bin/env python
"""Gate 8C-1: qualitative panels of what the completion actually predicts.

Runs the *same* frozen inference as ``tools/gate8c1/eval_target.py`` -- the checkpoint and
occupancy threshold come from ``frozen_manifest.json``, the map is the causal five-past-frame
window, nothing is re-tuned -- and renders, per anchor:

    camera at t | causal map (input) | our completion | ground truth | error

The error panel is an RGB composite over the bird's-eye plane: **green = correct occupied
(TP)**, **red = predicted but empty (FP)**, **blue = real but missed (FN)**. Intensity is
the count of voxels in that column, so a bright red patch is a solidly hallucinated wall and
a faint one is a single stray voxel.

    python tools/gate8c1/visualize.py --dataset semantickitti --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import occ_to_eval_grid, to_eval_grid                        # noqa: E402
from tools.gate8a.evaluate import dilate_with_semantics                          # noqa: E402
from tools.gate8c1.eval_target import PAST5_OFFSETS, window_map, MANIFEST        # noqa: E402

NICE = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}
FWD = {"semantickitti": "x — forward (m)", "occ3d": "x — forward (m)"}


def bev_count(vol3d: np.ndarray) -> np.ndarray:
    """Occupied voxels per bird's-eye column."""
    return vol3d.sum(axis=2).astype(np.float32)


def bev_height(vol3d: np.ndarray, z0: float, vs: float) -> np.ndarray:
    """Height of the topmost occupied voxel per column; NaN where the column is empty."""
    any_ = vol3d.any(axis=2)
    top = vol3d.shape[2] - 1 - np.argmax(vol3d[:, :, ::-1], axis=2)
    h = np.where(any_, z0 + (top + 0.5) * vs, np.nan)
    return h


def error_rgb(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, axis: int = 2) -> np.ndarray:
    """TP green / FP red / FN blue, intensity = voxels collapsed along ``axis``."""
    p, g, v = pred & valid, gt & valid, valid
    tp = (p & g).sum(axis=axis).astype(np.float32)
    fp = (p & ~g).sum(axis=axis).astype(np.float32)
    fn = (~p & g).sum(axis=axis).astype(np.float32)
    m = max(np.percentile(np.concatenate([tp.ravel(), fp.ravel(), fn.ravel()]), 99.0), 1.0)
    img = np.stack([np.clip(fp / m, 0, 1), np.clip(tp / m, 0, 1), np.clip(fn / m, 0, 1)], -1)
    img[~v.any(axis=axis)] = 0.12                    # grey where nothing is evaluated
    return img


def panel(ax, img, title, cmap=None, vmin=None, vmax=None, grid=None, axes_pair=(0, 1)):
    """One bird's-eye (x,y) or side (x,z) panel, forward to the right."""
    show = np.transpose(img, (1, 0) if img.ndim == 2 else (1, 0, 2))
    ext = None
    if grid is not None:
        o, vs, d = np.asarray(grid.origin, float), float(grid.voxel_size), grid.dims
        a0, a1 = axes_pair
        ext = [o[a0], o[a0] + d[a0] * vs, o[a1], o[a1] + d[a1] * vs]
    im = ax.imshow(show, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=ext, interpolation="nearest",
                   aspect="equal" if axes_pair == (0, 1) else "auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x — forward (m)", fontsize=7)
    ax.set_ylabel("y — left (m)" if axes_pair == (0, 1) else "z — up (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.plot([0], [0], marker="^", ms=7, color="#ffcc00", mec="k", mew=.6, zorder=5)
    return im


def render(ds, seed, device, n_anchors, out_png, art=None):
    art = art or ART
    man = json.load(open(MANIFEST))
    fz = man["seeds"][str(seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(device); torch.cuda.set_device(dev)
    v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    segs = S.segments(ds, REPO_ROOT)
    # deterministic, spread over the benchmark
    cand = []
    for seg in segs:
        for i in sorted(seg.anchors):
            if i + min(PAST5_OFFSETS) >= 0:
                cand.append((seg, i))
    picks = [cand[int(q * (len(cand) - 1))] for q in np.linspace(0.15, 0.85, n_anchors)]
    rows = []
    for seg, i in picks:
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(ds, seg.name)) as z:
            scale = float(np.exp(np.median(z["log_s"][:5])))
        f = seg.frames[i]
        target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
        if target is None:
            continue
        m, used = window_map(feed, seg, i, PAST5_OFFSETS, scale, into, C, dev)
        P = feed.pose[i].copy(); P[:3, 3] *= scale
        Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
        q = m.query(MAP, Tgw)
        q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                               torch.full_like(q["last_time"], -1).float())
        base = q["logodds"]
        fin, _ = net.raw(q, MAP)
        s_fin = to_eval_grid(fin, base, ds)[0]
        occ_native = q["occupied"] & (q["sem_w"] > 0)
        mp = occ_to_eval_grid(occ_native, ds).cpu().numpy().reshape(edims)
        comp = (s_fin >= tau).cpu().numpy().reshape(edims)
        gt = ((target != EVAL.empty_class) & keep).reshape(edims)
        val = keep.reshape(edims)
        rows.append({"seg": seg.name, "i": i, "clip": str(f.gt_ref["clip_id"]),
                     "img": f.path, "mapper": mp & val, "completion": comp & val,
                     "gt": gt, "valid": val, "used": used})
        del m
        torch.cuda.empty_cache()
    if not rows:
        return None
    z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
    n = len(rows)
    NC = 6
    fig, axes = plt.subplots(n, NC, figsize=(25, 4.2 * n), squeeze=False)
    # one shared colour scale for the three occupancy panels, so they are comparable
    vmax = max(int(np.percentile(bev_count(r[k]), 99.5))
               for r in rows for k in ("mapper", "completion", "gt"))
    vmax = max(vmax, 2)
    for r, rec in enumerate(rows):
        ax = axes[r]
        try:
            from PIL import Image
            with Image.open(rec["img"]) as im:
                ax[0].imshow(np.asarray(im))
        except Exception:
            ax[0].text(.5, .5, "image unavailable", ha="center")
        ax[0].set_axis_off()
        ax[0].set_title(f"camera at t  ·  {rec['clip'][:34]}", fontsize=9)
        for c, (key, ttl) in enumerate((("mapper", "1. causal map — what the module is given"),
                                        ("completion", "2. our completion — what it predicts"),
                                        ("gt", "3. ground truth"))):
            im = panel(ax[c + 1], bev_count(rec[key]), ttl, cmap="magma", vmin=0, vmax=vmax,
                       grid=EVAL)
            if r == 0 and c == 2:
                cb = fig.colorbar(im, ax=ax[c + 1], fraction=.046, pad=.02)
                cb.set_label("occupied voxels in the column", fontsize=7)
                cb.ax.tick_params(labelsize=6)
        panel(ax[4], error_rgb(rec["completion"], rec["gt"], rec["valid"], axis=2),
              "4. error, seen from above", grid=EVAL)
        panel(ax[5], error_rgb(rec["completion"], rec["gt"], rec["valid"], axis=1),
              "5. error, seen from the side", grid=EVAL, axes_pair=(0, 2))
        tp = int((rec["completion"] & rec["gt"] & rec["valid"]).sum())
        fp = int((rec["completion"] & ~rec["gt"] & rec["valid"]).sum())
        fn = int((~rec["completion"] & rec["gt"] & rec["valid"]).sum())
        lab = (f"IoU {tp / max(tp + fp + fn, 1):.3f}   P {tp / max(tp + fp, 1):.3f}   "
               f"R {tp / max(tp + fn, 1):.3f}   predicted/true volume "
               f"{(tp + fp) / max(tp + fn, 1):.2f}×")
        ax[4].set_xlabel(lab, fontsize=8)
        ax[5].set_xlabel("x — forward (m)", fontsize=7)
    handles = [Patch(color="#00ff00", label="correct occupied (TP)"),
               Patch(color="#ff0000", label="predicted, actually empty (FP)"),
               Patch(color="#0000ff", label="real, missed (FN)"),
               Patch(color="#1f1f1f", label="not evaluated (outside the official mask)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.002))
    cut = ("   ·   the rear half is outside the official single-camera mask"
           if ds == "occ3d" else "")
    fig.suptitle(
        f"Gate 8C-1 — {NICE[ds]}   ·   seed {seed}   ·   causal 5-past-frame protocol   ·   "
        f"occupancy threshold {tau:+.4f}\n"
        f"the 0.99 M completion module was trained on KITTI-360 only — this benchmark was "
        f"never seen in training, validation or threshold selection{cut}", fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["semantickitti", "occ3d", "summary"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    t0 = time.time()
    if a.dataset == "summary":
        out = summary(os.path.join(ART, "fig_summary.png"))
        print(f"wrote {out}  ({time.time()-t0:.0f}s)")
        return 0
    p = os.path.join(ART, f"fig_qualitative_{a.dataset}.png")
    out = render(a.dataset, a.seed, a.device, a.n, p)
    print(f"wrote {out}  ({time.time()-t0:.0f}s)")
    return 0




def summary(out_png, art=None):
    """Three plain-language panels: how good, over- or under-confident, and is it ranking."""
    art = art or ART
    res = json.load(open(os.path.join(art, "gate8c1_results.json")))
    pub = res["published_occany"]
    ds_list = ["semantickitti", "occ3d"]
    keys = [("completion", "ours (completion)"),
            ("mapper_dilate", "map + 0.4 m dilation"),
            ("frozen_5frame_dil", "frozen 5-frame + dilation"),
            ("all_valid_occupied", "fill everything"),
            ("mapper_native", "map only, no completion")]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    w = 0.36
    for j, ds in enumerate(ds_list):
        pt = res["per_target"][ds]["past5"]
        vals, labs = [], []
        for k, lab in keys:
            v = (pt["completion"]["binary_iou"]["median"] if k == "completion"
                 else pt["baselines"][k]["mean"])
            vals.append(100 * v); labs.append(lab)
        vals.append(100 * pub[ds]["sc_iou"]); labs.append("OccAny (published)")
        x = np.arange(len(vals))
        cols = ["#2e7d32" if i == 0 else ("#8e24aa" if i == len(vals) - 1 else "#90a4ae")
                for i in range(len(vals))]
        ax[0].bar(x + (j - 0.5) * w, vals, w, color=cols,
                  edgecolor="k", linewidth=.4, label=NICE[ds] if j == 0 else NICE[ds])
        if j == 1:
            ax[0].set_xticks(x); ax[0].set_xticklabels(labs, rotation=22, ha="right", fontsize=8)
    ax[0].set_ylabel("SC IoU %  (higher is better)")
    ax[0].set_title("How much of the scene do we get right?\n"
                    "left bar = SemanticKITTI, right bar = Occ3D", fontsize=10)
    ax[0].grid(alpha=.3, axis="y")
    for j, ds in enumerate(ds_list):
        pt = res["per_target"][ds]["past5"]["completion"]
        ax[1].bar(j, pt["pred_over_gt_volume"]["median"], .5,
                  color="#c62828" if pt["pred_over_gt_volume"]["median"] > 1 else "#1565c0",
                  edgecolor="k", linewidth=.4)
        ax[1].text(j, pt["pred_over_gt_volume"]["median"] + .06,
                   f"{pt['pred_over_gt_volume']['median']:.2f}×", ha="center", fontsize=10)
        ax[2].bar(j - .18, pt["ap_over_prevalence"]["median"], .34, color="#00695c",
                  edgecolor="k", linewidth=.4, label="AP / prevalence" if j == 0 else None)
        ax[2].bar(j + .18, pt["auroc"]["median"], .34, color="#ef6c00",
                  edgecolor="k", linewidth=.4, label="AUROC" if j == 0 else None)
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(-0.42, 1.04, "exactly the right amount", fontsize=8, ha="left")
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels([NICE[d] for d in ds_list], fontsize=9)
    ax[1].set_ylabel("predicted occupied volume ÷ true occupied volume")
    ax[1].set_title("Do we predict too much or too little?\n"
                    "above 1 = hallucinating, below 1 = missing", fontsize=10)
    ax[1].grid(alpha=.3, axis="y")
    ax[2].axhline(1.0, color="#00695c", ls=":", lw=1.2)
    ax[2].axhline(0.5, color="#ef6c00", ls=":", lw=1.2)
    ax[2].text(-0.45, 1.04, "chance for AP/prevalence", fontsize=8, ha="left", color="#00695c")
    ax[2].text(-0.45, 0.54, "chance for AUROC", fontsize=8, ha="left", color="#ef6c00")
    ax[2].set_xticks(range(2)); ax[2].set_xticklabels([NICE[d] for d in ds_list], fontsize=9)
    ax[2].set_title("Is the model actually ranking voxels,\nor guessing?", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
    fig.suptitle("Gate 8C-1 — completion trained on KITTI-360 only, evaluated on two "
                 "benchmarks it never saw   ·   causal 5-past-frame protocol   ·   "
                 "median of 3 seeds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


if __name__ == "__main__":
    raise SystemExit(main())
