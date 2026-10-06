"""Tables and figures from analysis.json / reliability.json / eval shards (no model is re-run except for the
deterministic visual examples, which read the stored predictions-at-t and the caches).

    tables/results_table.md, tables/per_class.csv, figures/1_reliability.png ... 6_target_construction.png
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.delayed_semantic import config as C  # noqa: E402
from tools.delayed_semantic.analysis import fast_miou, load_eval  # noqa: E402
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS  # noqa: E402

ART = C.ARTIFACT_DIR
FIG = os.path.join(ART, "figures"); TAB = os.path.join(ART, "tables")
NAMES = ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking", "sidewalk", "other ground", "building",
         "fence", "vegetation", "terrain", "pole", "traffic sign", "other structure", "other object"]
PALETTE = np.array([[255, 255, 255], [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180], [0, 0, 255], [255, 30, 30], [255, 0, 255],
                    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0], [255, 120, 50], [0, 175, 0], [150, 240, 80], [255, 240, 150],
                    [255, 0, 0], [50, 255, 255], [160, 160, 160]], np.uint8)
COLORS = {"B0": "#7f7f7f", "B0g": "#bcbd22", "B1": "#1f77b4", "B2": "#d62728", "B2s": "#ff9896"}
REGION_LABEL = {"observed_current": "observed in frame t", "observed_past_only": "observed only in t-4..t-1", "never_observed": "never observed (GT occupancy)",
                "completed_never_observed": "**never observed ∩ predicted occupied (primary)**", "completed_geo_observed": "… geometry observed (S or F)",
                "completed_geo_unobserved": "… geometry unobserved", "completed_later_revealed": "… later revealed (0006 delayed target)",
                "completed_revealed_unseen": "… later revealed, no H_t camera return", "completed_not_revealed": "… not revealed within 10 s",
                "ssc_full": "SSC mIoU, all labelled voxels", "ssc_never_observed": "SSC mIoU, voxels without evidence"}
pct = lambda x: "—" if x is None else f"{100 * x:.2f}"


def _table(head: List[str], rows: List[List]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def results_table(a: Dict, rel: Dict) -> str:
    ms = a["methods"]; out = [f"# Delayed-supervision pilot — 0006 development evaluation ({a['n_anchors']} anchors, geometry seed 0, head seed 0)\n",
                               "mIoU in %. Semantic regions score GT-occupied voxels; SSC regions score all labelled voxels under the frozen occupancy.\n"]
    rows = [[REGION_LABEL[r]] + [pct(a["pooled"][m][r]["miou"]) for m in ms] + [f"{a['pooled']['B0'][r]['n_voxels']:,}"] for r in SEMANTIC_REGIONS + SSC_REGIONS]
    out += ["## mIoU by region\n", _table(["region"] + ms + ["voxels"], rows)]
    rows = [[REGION_LABEL[r]] + [pct(a["pooled"][m][r]["overall_accuracy"]) for m in ms] for r in SEMANTIC_REGIONS]
    out += ["\n## Overall accuracy by region\n", _table(["region"] + ms, rows)]
    keys = [PRIMARY, "completed_geo_observed", "completed_geo_unobserved", "completed_later_revealed", "completed_revealed_unseen", "completed_not_revealed", "ssc_full"]
    rows = []
    for k, per in a["deltas"].items():
        for r in keys:
            d = per[r]; rows.append([k.replace("-", " − "), REGION_LABEL[r], f"{100*d['delta']:+.2f}", f"[{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]",
                                     f"{d['block_wins'][0]}/{d['block_wins'][1]}" if d.get("block_wins") else ""])
    out += ["\n## Paired differences (mIoU points, 95 % block bootstrap, 20-anchor blocks)\n", _table(["pair", "region", "Δ", "CI", "blocks won"], rows)]
    if "criteria" in a:
        for name, K in (("B2 (decision)", a["criteria"]), ("B2s (addendum A sensitivity, not a criterion)", a.get("criteria_B2s_sensitivity"))):
            if K is None:
                continue
            rows = [["K1 Δ vs B1 > 0, CI lo > 0", f"{100*K['K1']['delta']:+.2f} [{100*K['K1']['ci95'][0]:+.2f}, {100*K['K1']['ci95'][1]:+.2f}]", K["K1"]["pass"]],
                    ["K2 Δ vs B0 ≥ +2.00, CI lo > 0", f"{100*K['K2']['delta']:+.2f} [{100*K['K2']['ci95'][0]:+.2f}, {100*K['K2']['ci95'][1]:+.2f}]", K["K2"]["pass"]],
                    ["K3 evidence path identical", f"{K['K3']['anchors_equal']}/{K['K3']['anchors']}", K["K3"]["pass"]],
                    ["K4 ≥ 5 classes gain; gain without top class > 0", f"{K['K4']['classes_positive']} classes; without {NAMES[K['K4']['largest_gain_class']-1]}: {100*K['K4']['gain_without_it']:+.2f}", K["K4"]["pass"]],
                    ["K5 SSC mIoU ≥ B0", f"{pct(K['K5'][name.split()[0]])} vs {pct(K['K5']['B0'])}", K["K5"]["pass"]],
                    ["K6 occupancy identical", "", K["K6"]["pass"]]]
            out += [f"\n## Decision criteria — {name}\n", _table(["criterion", "value", "pass"], rows)]
    out += [f"\n**Decision: {a['decision']}.**\n"]
    c = [(i + 1, NAMES[i]) for i in range(18)]
    pr = a["pooled"]
    rows = [[n] + [pct(pr[m][PRIMARY]["per_class_iou"][str(k)] if isinstance(next(iter(pr[m][PRIMARY]["per_class_iou"])), str) else pr[m][PRIMARY]["per_class_iou"][k]) for m in ms]
            + [f"{100*pr['B0'][PRIMARY]['gt_frequency'][k-1]:.2f}"] + [f"{100*pr[m][PRIMARY]['pred_frequency'][k-1]:.2f}" for m in ms] for k, n in c]
    out += ["\n## Per-class IoU on the primary region (%), GT share and prediction share (%)\n", _table(["class"] + ms + ["GT share"] + [f"{m} pred share" for m in ms], rows)]
    rows = [[C.DISTANCE_NAMES[b], f"{a['distance']['B0'][b]['n_voxels']:,}"] + [f"{pct(a['distance'][m][b]['miou'])} / {pct(a['distance'][m][b]['overall_accuracy'])}" for m in ms] for b in range(len(C.DISTANCE_NAMES))]
    out += ["\n## Primary region by distance to the nearest evidence (mIoU / accuracy %)\n", _table(["distance", "voxels"] + ms, rows)]
    rows = [[C.REVEAL_NAMES[b], f"{a['reveal']['B0'][b]['n_voxels']:,}"] + [f"{pct(a['reveal'][m][b]['miou'])} / {pct(a['reveal'][m][b]['overall_accuracy'])}" for m in ms] for b in range(len(C.REVEAL_NAMES))]
    out += ["\n## Primary region by time until a later camera view (mIoU / accuracy %)\n", _table(["time to reveal", "voxels"] + ms, rows)]
    rows = [[m, f"{v['voxels']:,}", pct(v["agreement"]), pct(v["miou"])] for m, v in a["delayed_target_agreement"].items()]
    out += ["\n## Label-free: agreement with the 0006 delayed target class (all non-evidence target voxels)\n", _table(["method", "voxels", "agreement %", "mIoU vs target %"], rows)]
    o = a["occupancy"]
    out += [f"\n## Occupancy\n\nTP/FP/FN {o['TP_FP_FN']}, IoU {100*o['iou']:.2f} %, identical to Gate 15 seed 0 per anchor: {o['equal_to_gate15_seed0']}. "
            f"B0 reproduces Gate 15 per anchor: {a['b0_reproduces_gate15']}. Geodesic spread reaches {100*a['geodesic_reached_fraction']:.1f} % of completed non-evidence voxels.\n"]
    return "\n".join(out)


def reliability_table(rel: Dict) -> str:
    g = rel["gate"]; out = ["# Target reliability (label-free gate, then a labelled 0006 diagnostic)\n"]
    rows = [[k, v["n"], f"{v['median_m']:.3f}" if v["median_m"] is not None else "—", f"{v['p90_m']:.3f}" if v["p90_m"] is not None else "—"] for k, v in g["G1_train_pooled"].items()]
    out += ["## G1 registration, training drives pooled (point-to-plane, m)\n", _table(["offset (frames)", "points", "median", "p90"], rows)]
    rows = []
    for d, r in rel["registration"].items():
        for k, v in r["offsets"].items():
            if v["n"]:
                z = r["perturbed"]["z+0.2m"][k]; y = r["perturbed"]["y+0.2m"][k]
                rows.append([d[17:21], k, v["n"], f"{v['median_m']:.3f}", f"{v['p90_m']:.3f}", f"{z['median_m']:.3f}", f"{y['median_m']:.3f}", g["G1_strict_per_drive_incl_dev"].get(f"{d[17:21]}@{k}")])
    out += ["\n## G1 per drive (strict reading) with perturbation controls (median after +0.2 m shift)\n",
            _table(["drive", "offset", "points", "median", "p90", "z+0.2 median", "y+0.2 median", "per-drive pass"], rows)]
    tp, dp = rel["train_pooled"], rel["dev_pooled"]
    rows = [[f"tier {t}", f"{tp['repeatability'][t]['pairs']:,}", pct(tp["repeatability"][t]["agreement"]), pct(tp["repeatability"][t]["shuffled"]),
             f"{dp['repeatability'][t]['pairs']:,}", pct(dp["repeatability"][t]["agreement"]), pct(dp["repeatability"][t]["shuffled"])] for t in ("A", "B")]
    out += ["\n## G3 cross-view repeatability (first vs second later view, top-1 %)\n", _table(["tier", "train pairs", "train agree", "train shuffled", "0006 pairs", "0006 agree", "0006 shuffled"], rows)]
    rows = [[d[17:21], f"{s['mean_D']:.0f}", f"{s['mean_D_A']:.0f}", f"{s['mean_D_B']:.0f}", pct(s["D_hist_seen_fraction"]["A"]), pct(s["reobservation"]["all"]["agreement"]),
             pct(s["student_evidence_vs_target"]["agreement"]), ", ".join(f"{k} {v:,}" for k, v in {**{f'A {k}': v for k, v in s['time_to_reveal']['A'].items()}}.items())] for d, s in rel["per_drive"].items()]
    out += ["\n## Coverage (G2), reobservation (G4), student evidence vs target\n",
            _table(["drive", "mean |D|/anchor", "tier A", "tier B", "tier A seen by an H_t camera %", "H_t view vs target %", "student evidence vs target %", "tier A time to first view"], rows)]
    cc = np.asarray(tp["class_counts"]["A"]) + np.asarray(tp["class_counts"]["B"])
    out += ["\nTraining target class counts (argmax): " + ", ".join(f"{NAMES[i]} {cc[i]:,}" for i in np.argsort(-cc)) + "\n"]
    gt = rel["gt_diagnostic_0006"]
    rows = [[k, f"{v['voxels']:,}", pct(v["accuracy"]), pct(v["miou"])] for k, v in gt.items()]
    out += ["\n## LABELLED DIAGNOSTIC (0006 semantic GT, never used by a rule): target class vs GT\n", _table(["set", "voxels", "accuracy %", "mIoU %"], rows)]
    out += [f"\nGate: G1 {g['G1']} (strict per drive {g['G1_strict_pass']}), G2 {g['G2']} ({g['G2_classes_with_support']} classes), G3 A {g['G3_tierA']} / B {g['G3_tierB']}, "
            f"G4 {g['G4']} → **{g['outcome']}**, delayed loss tiers {g['delayed_loss_tiers']}.\n"]
    return "\n".join(out)


def figures(a: Dict, rel: Dict) -> List[str]:
    os.makedirs(FIG, exist_ok=True); out = []
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for d, r in rel["registration"].items():
        ks = [k for k, v in r["offsets"].items() if v["n"]]
        ax[0].plot([int(k) for k in ks], [r["offsets"][k]["p90_m"] for k in ks], "o-", label=f"{d[17:21]} p90")
        ax[0].plot([int(k) for k in ks], [r["perturbed"]["z+0.2m"][k]["median_m"] for k in ks], "x:", color=ax[0].lines[-1].get_color(), alpha=0.6)
    ax[0].axhline(0.20, color="k", lw=0.8, ls="--"); ax[0].set_xlabel("future offset (stream frames, ~0.5 s each)"); ax[0].set_ylabel("m")
    ax[0].set_title("G1: point-to-plane p90 (o) vs +0.2 m z-shift median (x)"); ax[0].legend(fontsize=7)
    for t, col in (("A", "#1f77b4"), ("B", "#ff7f0e")):
        s = rel["per_drive"]
        gaps = list(next(iter(s.values()))["repeatability_by_view_gap"][t])
        pairs = np.array([[sum(v["repeatability_by_view_gap"][t][g]["pairs"] for d, v in s.items() if d != C.DEV_DRIVE),
                           sum(v["repeatability_by_view_gap"][t][g]["pairs"] * v["repeatability_by_view_gap"][t][g]["agreement"] for d, v in s.items() if d != C.DEV_DRIVE)] for g in gaps])
        ax[1].plot(gaps, pairs[:, 1] / np.maximum(pairs[:, 0], 1), "o-", color=col, label=f"tier {t}")
        ax[1].axhline(rel["train_pooled"]["repeatability"][t]["shuffled"], color=col, ls=":", lw=1)
    ax[1].axhline(0.65, color="k", lw=0.8, ls="--"); ax[1].set_ylim(0, 1); ax[1].set_title("G3: view-to-view agreement by gap (train; dotted = shuffle)"); ax[1].legend()
    for i, t in enumerate(("A", "B")):
        v = rel["per_drive"][C.TRAIN_DRIVES[1]]["time_to_reveal"][t]
        tot = {k: sum(rel["per_drive"][d]["time_to_reveal"][t][k] for d in C.TRAIN_DRIVES) for k in v}
        ax[2].bar(np.arange(len(tot)) + (i - 0.5) * 0.4, list(tot.values()), 0.4, label=f"tier {t}")
        ax[2].set_xticks(range(len(tot))); ax[2].set_xticklabels(list(tot))
    ax[2].set_title("time from t to the first later view (train)"); ax[2].legend()
    fig.tight_layout(); p = os.path.join(FIG, "1_reliability.png"); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)

    keys = [PRIMARY, "completed_geo_observed", "completed_geo_unobserved", "completed_later_revealed", "completed_revealed_unseen", "completed_not_revealed", "ssc_full"]
    pairs = [k for k in a["deltas"]]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(keys) * len(pairs) / 2 + 2))
    y = 0; ticks, labels = [], []
    for r in keys:
        for k in pairs:
            d = a["deltas"][k][r]; col = COLORS.get(k.split("-")[0], "k")
            ax.errorbar(100 * d["delta"], y, xerr=[[100 * (d["delta"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["delta"])]], fmt="o", color=col, ms=4, capsize=2)
            ticks.append(y); labels.append(f"{r.replace('completed_', 'c_')}: {k}"); y -= 1
        y -= 0.6
    ax.axvline(0, color="k", lw=0.8); ax.axvline(2, color="k", lw=0.6, ls="--"); ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ mIoU (points), 95 % block bootstrap"); ax.set_title("Paired differences on 0006 (dashed: +2 point threshold vs B0)")
    fig.tight_layout(); p = os.path.join(FIG, "2_deltas.png"); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for m in a["methods"]:
        ax[0].plot(C.DISTANCE_NAMES, [100 * (s["overall_accuracy"] or 0) for s in a["distance"][m]], "o-", color=COLORS[m], label=m)
        ax[1].plot(C.REVEAL_NAMES, [100 * (s["overall_accuracy"] or 0) for s in a["reveal"][m]], "o-", color=COLORS[m], label=m)
    ax[0].set_title("primary region: accuracy vs distance to nearest evidence"); ax[1].set_title("primary region: accuracy vs time until a later view")
    for x in ax:
        x.set_ylabel("overall accuracy %"); x.legend()
    fig.tight_layout(); p = os.path.join(FIG, "3_distance_reveal.png"); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(figsize=(15, 4.5)); w = 0.8 / len(a["methods"])
    for i, m in enumerate(a["methods"]):
        pc = a["pooled"][m][PRIMARY]["per_class_iou"]
        vals = [100 * (pc[str(k)] if str(k) in pc else pc.get(k)) if (pc.get(str(k), pc.get(k)) is not None) else 0 for k in range(1, 19)]
        ax.bar(np.arange(18) + (i - len(a["methods"]) / 2 + 0.5) * w, vals, w, color=COLORS[m], label=m)
    ax.set_xticks(range(18)); ax.set_xticklabels(NAMES, rotation=45, ha="right"); ax.set_ylabel("IoU %"); ax.legend(); ax.set_title("per-class IoU, primary region")
    fig.tight_layout(); p = os.path.join(FIG, "4_per_class.png"); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    return out


def _bev(vol_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Top-most masked voxel's label per (x, y) column -> RGB, rotated so forward is up."""
    X, Y, Z = C.GRID_DIMS
    lab = np.where(mask, vol_labels, 0).reshape(X, Y, Z)
    zi = np.where(lab > 0, np.arange(Z)[None, None], -1).argmax(2)
    top = np.take_along_axis(lab, zi[..., None], 2)[..., 0]
    return np.rot90(PALETTE[top])


def examples(device: str = "cuda:0", paths: C.Paths = C.Paths()) -> List[str]:
    """Deterministic: 0006 anchors with >= 2,000 primary voxels at the 10/50/90th percentile of B2 - B0 accuracy on the primary region."""
    from tools.delayed_semantic.memory import load_observed
    from tools.semantic_completion.evidence import trident as TR
    z = load_eval(paths)
    acc = lambda cm: np.trace(cm[:, 1:, 1:], axis1=1, axis2=2) / np.maximum(cm[:, 1:].sum((1, 2)), 1)
    b0, b2 = z[f"conf__B0__{PRIMARY}"], z[f"conf__B2__{PRIMARY}"]
    ok = b0[:, 1:].sum((1, 2)) >= 2000
    gain = acc(b2) - acc(b0); ids = z["ids"][ok]; order = np.argsort(gain[ok], kind="stable")
    picks = {q: int(ids[order[int(round(q / 100 * (len(order) - 1)))]]) for q in (10, 50, 90)}
    dev = torch.device(device); text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    out = []
    for q, t in picks.items():
        with np.load(paths.g15.eval_mask_path(t)) as e:
            gt = e["sem_gt"].astype(np.int64)
        with np.load(os.path.join(paths.predictions_root, C.DEV_DRIVE, f"{t:05d}.npz")) as pr:
            O = np.unpackbits(pr["O"])[:C.N_VOXELS].astype(bool); labs = {m: np.zeros(C.N_VOXELS, np.int64) for m in ("B0", "B1", "B2")}
            for m in labs:
                labs[m][O] = pr[f"label_{m}"]
        ev = load_observed(C.DEV_DRIVE, t, dev, paths)
        E = np.zeros(C.N_VOXELS, bool); E[ev.flat_index.cpu().numpy()] = True
        evl = np.zeros(C.N_VOXELS, np.int64); evl[ev.flat_index.cpu().numpy()] = TR.classify(ev.feature, text).cpu().numpy()
        with np.load(paths.target_path(C.DEV_DRIVE, t)) as tg:
            keep = ~tg["in_evidence"]; tl = np.zeros(C.N_VOXELS, np.int64); tl[tg["flat_index"][keep]] = tg["prob"][keep].astype(np.float32).argmax(1) + 1
        G = (gt >= 1) & (gt <= 18); prim = G & O & ~E
        from tools.delayed_semantic.stream import DriveStream
        img = plt.imread(DriveStream(C.DEV_DRIVE).frames[t].image_path)
        pan = [("evidence in H_t (teacher class)", _bev(evl, E)), ("delayed target (later views, not evidence)", _bev(tl, tl > 0)),
               ("B0 on primary voxels", _bev(labs["B0"], prim)), ("B1 on primary voxels", _bev(labs["B1"], prim)),
               ("B2 on primary voxels", _bev(labs["B2"], prim)), ("GT on primary voxels (evaluation only)", _bev(gt, prim))]
        fig = plt.figure(figsize=(18, 7.5)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 2.6])
        axi = fig.add_subplot(gs[0, 1:5]); axi.imshow(img, cmap="gray"); axi.set_title(f"0006 stream frame t = {t} (RGB)"); axi.axis("off")
        for i, (title, im) in enumerate(pan):
            ax = fig.add_subplot(gs[1, i]); ax.imshow(im, interpolation="nearest"); ax.set_title(title, fontsize=8); ax.axis("off")
        a0, a2 = acc(b0[z["ids"] == t])[0], acc(b2[z["ids"] == t])[0]
        fig.suptitle(f"{q}th percentile of B2 − B0 accuracy on the primary region: B0 {100*a0:.1f} %, B2 {100*a2:.1f} % (bird's-eye, forward = up)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[k] / 255) for k in range(1, 19)]
        fig.legend(handles, NAMES, loc="lower center", ncol=9, fontsize=7); fig.tight_layout(rect=(0, 0.06, 1, 1))
        p = os.path.join(FIG, f"5_example_p{q}_t{t:05d}.png"); fig.savefig(p, dpi=110); plt.close(fig); out.append(p)
    return out


def target_construction_figure(device: str = "cuda:0", paths: C.Paths = C.Paths()) -> str:
    """One future view of 0006 (deterministic: anchor index 300, offset +4): visible returns by teacher class, z-buffer rejected in grey."""
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    from tools.delayed_semantic import targets as TG
    from tools.delayed_semantic.stream import DriveStream
    from tools.semantic_completion.evidence import build as G15B, trident as TR
    dev = torch.device(device); text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    s = DriveStream(C.DEV_DRIVE); t = G15B.anchor_ids(C.DEV_DRIVE)[300]; f = s.future(t)[3]
    geo = TF.DriveGeometry(C.DEV_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    cv = TG.camera_view(geo, f, text, C.TargetConfig(), dev)
    T = torch.as_tensor(np.asarray(geo.calib.velo_to_rect_cam, np.float64), device=dev); cam = cv.pts_velo @ T[:3, :3].T + T[:3, 3]
    K = geo.calib.K; H, W = geo.calib.native_hw
    u = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).cpu().numpy(); v = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).cpu().numpy(); zc = cam[:, 2].cpu().numpy()
    inimg = (zc >= 1) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    vis = np.zeros(len(u), bool); vis[cv.vis_index.cpu().numpy()] = True
    cls = np.zeros(len(u), np.int64); cls[cv.vis_index.cpu().numpy()] = cv.cos.argmax(1).cpu().numpy() + 1
    fig, ax = plt.subplots(2, 1, figsize=(16, 8.5))
    for a in ax:
        a.imshow(plt.imread(f.image_path), cmap="gray"); a.axis("off")
    rej = inimg & ~vis
    ax[0].scatter(u[vis], v[vis], s=1, c=PALETTE[cls[vis]] / 255); ax[0].set_title(f"future frame j = t+4 ({f.time_s - s.frames[t].time_s:.2f} s after t={t}): camera-visible LiDAR returns coloured by the Trident-H class at their pixel")
    ax[1].scatter(u[rej], v[rej], s=1, c="k"); ax[1].scatter(u[vis], v[vis], s=0.3, c="#2ca02c", alpha=0.3)
    ax[1].set_title(f"z-buffer: {int(vis.sum()):,} visible (green), {int(rej.sum()):,} rejected as occluded from the camera (black) of {int(inimg.sum()):,} in the image")
    fig.tight_layout(); p = os.path.join(FIG, "6_target_construction.png"); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def write_all(device: str = "cuda:0") -> Dict:
    os.makedirs(TAB, exist_ok=True)
    a = json.load(open(os.path.join(ART, "analysis.json"))); rel = json.load(open(os.path.join(C.Paths().reliability, "reliability.json")))
    open(os.path.join(ART, "results_table.md"), "w").write(results_table(a, rel))
    open(os.path.join(ART, "target_reliability.md"), "w").write(reliability_table(rel))
    with open(os.path.join(TAB, "per_class.csv"), "w") as fh:
        fh.write("region,class_id,class," + ",".join(f"{m}_iou" for m in a["methods"]) + ",gt_share\n")
        for r in SEMANTIC_REGIONS:
            for k in range(1, 19):
                vals = [a["pooled"][m][r]["per_class_iou"].get(str(k)) for m in a["methods"]]
                fh.write(f"{r},{k},{NAMES[k-1]}," + ",".join("" if x is None else f"{x:.6f}" for x in vals) + f",{a['pooled']['B0'][r]['gt_frequency'][k-1]:.6f}\n")
    figs = figures(a, rel) + examples(device) + [target_construction_figure(device)]
    return {"figures": figs}


__all__ = ["results_table", "reliability_table", "figures", "examples", "target_construction_figure", "write_all"]
