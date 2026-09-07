#!/usr/bin/env python
"""Gate 8C-1: head-to-head comparison against the released OccAny checkpoint.

OccAny's own released model was run on this machine over the Occ3D-nuScenes validation
split (`extract_output_occany.py`, EXP_ID 2 settings: `occany_must3r`, 5 frames, CAM_FRONT,
`--gen`, conf 1.1). This tool reads its saved voxel predictions, runs **our** frozen module
on the same target samples, scores both with **OccAny's own** `SSCMetrics`, and draws the
2D and 3D comparisons.

Two of our protocols are shown because they are not the same input:

* ``past5`` -- our deployed causal setting: five frames *ending* at the target, no future.
* ``occany_fwd`` -- OccAny's own sampling: the target first, four frames *after* it. This is
  non-causal and exists only so the input budgets match.

OccAny's stored ``voxel_label`` is used as the ground truth for both, after verifying it is
byte-identical to the target our own evaluator builds.

    python tools/gate8c1/compare_occany.py --n 4 --device cuda:1
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G, targets as G6T, vocab             # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8a.regions import to_eval_grid                                          # noqa: E402
from gates.gate8b import pooling as PL                                                  # noqa: E402
from gates.gate8c1 import occany_eval as OE, render3d as R3                            # noqa: E402
from tools.gate8c1.eval_target import (MANIFEST, OCCANY_FWD_OFFSETS,             # noqa: E402
                                       PAST5_OFFSETS, window_map)

OCCANY_PRED = ("/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred/"
               "OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2")
PRED_KEY = "render_recon_gen_recon1.1_gen1.1"
FREE, IGNORE = 17, 255
DIMS = (200, 200, 16)
VOXEL, ORIGIN = 0.4, (-40.0, -40.0, -1.0)
#: colours shared by the 2D and 3D panels
C_TP, C_FP, C_FN = (0.13, 0.72, 0.33), (0.87, 0.16, 0.24), (0.16, 0.42, 0.85)


def occany_samples():
    out = {}
    for f in sorted(glob.glob(os.path.join(OCCANY_PRED, "*", "voxel_predictions.pkl"))):
        name = os.path.basename(os.path.dirname(f))
        scene, tok = name.split("_", 1)
        out[tok] = f
    return out


def load_occany(path):
    d = pickle.load(open(path, "rb"))
    return {"pred": np.asarray(d[PRED_KEY]).reshape(DIMS) != FREE,
            "label": np.asarray(d["voxel_label"]).reshape(DIMS),
            "images": np.asarray(d.get("estimated_input_images"))}


def our_prediction(ds, seed, dev, net, tau, seg, i, offsets, into, C, MAP):
    with np.load(S.scale_path(ds, seg.name)) as z:
        scale = float(np.exp(np.median(z["log_s"][:5])))
    feed = CachedFeed(seg, dev)
    m, used = window_map(feed, seg, i, offsets, scale, into, C, dev)
    f = seg.frames[i]
    P = feed.pose[i].copy(); P[:3, 3] *= scale
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    fin, _ = net.raw(q, MAP)
    occ = (to_eval_grid(fin, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(DIMS)
    del m
    return occ, used


def counts(pred, gt, valid):
    p, g = pred & valid, gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "pred_over_gt": (tp + fp) / max(tp + fn, 1)}


def official(pred, label):
    """Score through OccAny's own SSCMetrics: occupied -> class 1, free -> 17."""
    lab = np.where(pred, 1, FREE).astype(np.uint8)
    return OE.official_sc_counts(lab, label.astype(np.uint8), n_classes=18, empty_class=FREE)


def bev(vol):
    return vol.sum(axis=2).astype(np.float32)


def err_rgb(pred, gt, valid, axis=2):
    p, g = pred & valid, gt & valid
    tp = (p & g).sum(axis=axis).astype(np.float32)
    fp = (p & ~g).sum(axis=axis).astype(np.float32)
    fn = (~p & g).sum(axis=axis).astype(np.float32)
    m = max(np.percentile(np.concatenate([tp.ravel(), fp.ravel(), fn.ravel()]), 99.0), 1.0)
    img = np.stack([np.clip(fp / m, 0, 1), np.clip(tp / m, 0, 1), np.clip(fn / m, 0, 1)], -1)
    img[~valid.any(axis=axis)] = 0.12
    return img


def panel(ax, img, title, cmap=None, vmin=None, vmax=None, pair=(0, 1)):
    show = np.transpose(img, (1, 0) if img.ndim == 2 else (1, 0, 2))
    a0, a1 = pair
    ext = [ORIGIN[a0], ORIGIN[a0] + DIMS[a0] * VOXEL,
           ORIGIN[a1], ORIGIN[a1] + DIMS[a1] * VOXEL]
    ax.imshow(show, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=ext,
              interpolation="nearest", aspect="equal" if pair == (0, 1) else "auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x — forward (m)", fontsize=7)
    ax.set_ylabel("y — left (m)" if pair == (0, 1) else "z — up (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.plot([0], [0], marker="^", ms=8, color="#ffcc00", mec="k", mew=.6, zorder=6, ls="none")


def err_mesh(pred, gt, valid):
    """Three coloured cube meshes: TP, FP, FN."""
    out = []
    for m, col in ((pred & gt & valid, C_TP), (pred & ~gt & valid, C_FP),
                   (~pred & gt & valid, C_FN)):
        idx = np.argwhere(m)
        out.append(R3.cube_mesh(idx, np.tile(col, (len(idx), 1)), VOXEL, ORIGIN)
                   if len(idx) else None)
    return out


def height_mesh(vol):
    idx = np.argwhere(vol)
    if not len(idx):
        return None
    return R3.cube_mesh(idx, R3.height_colors(idx, VOXEL, ORIGIN[2], DIMS[2]), VOXEL, ORIGIN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--max-metric", type=int, default=100000)
    a = ap.parse_args()
    ds = "occ3d"
    man = json.load(open(MANIFEST)); fz = man["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    v = vocab.load(ds); C = len(v)
    MAP = G6G.PREDICTION_GRID[ds]
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    segs = {s.name: s for s in S.segments(ds, REPO_ROOT)}
    recs = {r.raw["anchor_token"]: r.raw for r in G6F.read_manifest(ds, REPO_ROOT)}
    idx_of = {}
    for s in segs.values():
        for i in sorted(s.anchors):
            idx_of[(s.name, s.frames[i].gt_ref["anchor_token"])] = i
    oc = occany_samples()
    shared = [(tok, p) for tok, p in oc.items() if tok in recs]
    print(f"OccAny samples on disk: {len(oc)}   shared with our manifest: {len(shared)}",
          flush=True)
    if not shared:
        print("no overlap yet -- is the OccAny run still early?"); return 1

    # their published number applies apply_majority_pooling(geometry_only) whose default
    # is a 3x3x3 max-pool, so every method is reported raw AND pooled
    VARIANTS = ("occany", "occany_pooled", "ours_past5", "ours_past5_pooled",
                "ours_fwd", "ours_fwd_pooled")
    agg = {k: [0, 0, 0] for k in VARIANTS}
    label_identical = 0
    rows, t0 = [], time.time()
    shared.sort()
    use = shared[:a.max_metric]
    for k, (tok, path) in enumerate(use):
        rec = recs[tok]; scene = rec["scene"]
        seg = segs.get(scene)
        key = (scene, tok)
        if seg is None or key not in idx_of:
            continue
        i = idx_of[key]
        d = load_occany(path)
        label = d["label"]
        gt = (label != FREE) & (label != IGNORE)
        valid = label != IGNORE
        ours_t, keep_t = G6T.semantic_target(ds, rec, REPO_ROOT)
        if ours_t is not None:
            mine = np.where(keep_t, ours_t, IGNORE).astype(np.uint8).reshape(DIMS)
            label_identical += int(bool((mine == label.astype(np.uint8)).all()))
        preds = {"occany": d["pred"]}
        for nm, offs in (("ours_past5", PAST5_OFFSETS), ("ours_fwd", OCCANY_FWD_OFFSETS)):
            if i + min(offs) < 0 or i + max(offs) >= len(seg):
                continue
            preds[nm], _ = our_prediction(ds, a.seed, dev, net, tau, seg, i, offs, into, C, MAP)
        if len(preds) < 3:
            continue
        for nm in list(preds):
            preds[nm + "_pooled"] = PL.geometry_dilation(
                torch.from_numpy(preds[nm])).numpy()
        row = {"token": tok, "scene": scene, "stream_index": i}
        for nm, p in preds.items():
            tp, fp, fn = official(p, label)
            for j, x in enumerate((tp, fp, fn)):
                agg[nm][j] += x
            row[nm] = counts(p, gt, valid)
            row[nm]["official"] = {"tp": tp, "fp": fp, "fn": fn,
                                   "iou": tp / max(tp + fp + fn, 1)}
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} scored  ({(time.time()-t0)/len(rows):.2f}s each)", flush=True)
        if a.metrics_only:
            continue
    out = {"n_scored": len(rows), "label_identical_count": label_identical,
           "occany_pred_key": PRED_KEY, "occany_pred_dir": OCCANY_PRED,
           "threshold": tau, "seed": a.seed,
           "scored_with": "OccAny SSCMetrics.get_score_completion",
           "pooled": {}, "per_sample": rows[:200], "seconds": time.time() - t0}
    for nm, (tp, fp, fn) in agg.items():
        out["pooled"][nm] = {"tp": tp, "fp": fp, "fn": fn,
                             "iou": tp / max(tp + fp + fn, 1),
                             "precision": tp / max(tp + fp, 1),
                             "recall": tp / max(tp + fn, 1),
                             "pred_over_gt": (tp + fp) / max(tp + fn, 1)}
    write_json(os.path.join(ART, "occany_headtohead.json"), out)
    print(f"\npooled over {len(rows)} shared samples (OccAny's own evaluator):")
    for nm, m in out["pooled"].items():
        print(f"   {nm:12s} IoU {100*m['iou']:6.2f}  P {100*m['precision']:6.2f}  "
              f"R {100*m['recall']:6.2f}  pred/GT {m['pred_over_gt']:.2f}")
    print(f"   GT label identical to ours on {label_identical}/{len(rows)} samples")
    if a.metrics_only:
        return 0

    # ------------------------------------------------------------------ figures
    pick = [rows[int(q * (len(rows) - 1))] for q in np.linspace(0.1, 0.9, a.n)]
    published = OE.PUBLISHED_5FRAME["occ3d"]["sc_iou"]
    fig, axes = plt.subplots(len(pick), 6, figsize=(25, 4.1 * len(pick)), squeeze=False)
    for r, row in enumerate(pick):
        tok = row["token"]; rec = recs[tok]; seg = segs[rec["scene"]]
        i = row["stream_index"]
        d = load_occany(oc[tok]); label = d["label"]
        gt = (label != FREE) & (label != IGNORE); valid = label != IGNORE
        p_oc = d["pred"]
        p_p5, _ = our_prediction(ds, a.seed, dev, net, tau, seg, i, PAST5_OFFSETS, into, C, MAP)
        ax = axes[r]
        try:
            from PIL import Image
            with Image.open(os.path.join(G6F.image_root(ds, REPO_ROOT),
                                         rec["image_paths"][-1])) as im:
                ax[0].imshow(np.asarray(im))
        except Exception:
            if d["images"].size:
                ax[0].imshow(d["images"][0])
        ax[0].set_axis_off(); ax[0].set_title(f"camera at target t\n{rec['scene']}", fontsize=9)
        # panels show only what is scored; the metrics keep OccAny's pool-then-mask order
        show_oc, show_p5 = p_oc & valid, p_p5 & valid
        vmax = max(2, int(np.percentile(np.concatenate(
            [bev(show_oc).ravel(), bev(show_p5).ravel(), bev(gt).ravel()]), 99.5)))
        for c, (vol, ttl) in enumerate(((show_oc, "OccAny (released ckpt)"),
                                        (show_p5, "ours — causal 5 past frames"),
                                        (gt, "ground truth"))):
            panel(ax[c + 1], bev(vol), ttl, cmap="magma", vmin=0, vmax=vmax)
        panel(ax[4], err_rgb(p_oc, gt, valid), "OccAny error (above)")
        panel(ax[5], err_rgb(p_p5, gt, valid), "our error (above)")
        for cc, p_, nm in ((4, p_oc, "occany"), (5, p_p5, "ours_past5")):
            m = counts(p_, gt, valid)
            ax[cc].set_xlabel(f"IoU {m['iou']:.3f}  P {m['precision']:.3f}  "
                              f"R {m['recall']:.3f}  pred/GT {m['pred_over_gt']:.2f}", fontsize=8)
    handles = [Patch(color=C_TP, label="correct occupied (TP)"),
               Patch(color=C_FP, label="predicted, actually empty (FP)"),
               Patch(color=C_FN, label="real, missed (FN)"),
               Patch(color="#1f1f1f", label="not evaluated (outside the official mask)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.002))
    fig.suptitle("Gate 8C-1 — ours vs the released OccAny checkpoint on Occ3D-nuScenes val "
                 "(bird's-eye)\nsame target frame, same official mask, both scored by "
                 "OccAny's own evaluator   ·   OccAny sees 4 s of FUTURE, we see 2 s of PAST",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    fig.subplots_adjust(hspace=0.28)
    p2 = os.path.join(ART, "fig_occany_2d.png")
    fig.savefig(p2, dpi=110); plt.close(fig); print("wrote", p2)

    # ------------------------------------------------------------------ 3D
    n3 = min(3, len(pick))
    fig, axes = plt.subplots(n3, 3, figsize=(19, 5.4 * n3), squeeze=False)
    for r, row in enumerate(pick[:n3]):
        tok = row["token"]; rec = recs[tok]; seg = segs[rec["scene"]]
        i = row["stream_index"]
        d = load_occany(oc[tok]); label = d["label"]
        gt = (label != FREE) & (label != IGNORE); valid = label != IGNORE
        p_oc = d["pred"]
        p_p5, _ = our_prediction(ds, a.seed, dev, net, tau, seg, i, PAST5_OFFSETS, into, C, MAP)
        for c, (vol, ttl) in enumerate(((p_oc, "OccAny (released ckpt)"),
                                        (p_p5, "ours — causal 5 past frames"),
                                        (gt, "ground truth"))):
            img = R3.render(err_mesh(vol, gt, valid) if c < 2 else [height_mesh(vol & valid)],
                            size=(940, 660))
            axes[r][c].imshow(img); axes[r][c].set_axis_off()
            m = counts(vol, gt, valid)
            axes[r][c].set_title(
                (f"{ttl}\nIoU {m['iou']:.3f}   P {m['precision']:.3f}   R {m['recall']:.3f}"
                 if c < 2 else f"{ttl}\n{int((vol & valid).sum()):,} occupied voxels"),
                fontsize=10)
        axes[r][0].text(0.01, 0.98, rec["scene"], transform=axes[r][0].transAxes,
                        fontsize=9, va="top", bbox=dict(fc="w", ec="none", alpha=.75))
    fig.legend(handles=handles[:3], loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Gate 8C-1 — the same scenes in 3D.  Left two panels are coloured by "
                 "error (green correct, red hallucinated, blue missed);\nthe right panel is "
                 "the ground truth coloured by height.  Viewpoint is identical in every panel.",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.subplots_adjust(hspace=0.16)
    p3 = os.path.join(ART, "fig_occany_3d.png")
    fig.savefig(p3, dpi=110); plt.close(fig); print("wrote", p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
