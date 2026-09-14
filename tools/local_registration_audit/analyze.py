#!/usr/bin/env python
"""Aggregate pose, reprojection, depth, geometry and semantic results; figures; tables.

    python tools/local_registration_audit/analyze.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

DRIVES = (LR.PRIMARY,) + LR.ROBUST
SHORT = {d: d[17:21] for d in DRIVES}
POSITION = {"early (0-299)": (0, 300), "middle (300-899)": (300, 900), "late (>=900)": (900, 10 ** 6)}


def q(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None, "mean": float(x.mean()) if len(x) else None,
            "p90": float(np.percentile(x, 90)) if len(x) else None, "p95": float(np.percentile(x, 95)) if len(x) else None}


def pose_block(z, clip_mask, pt_mask):
    out = {}
    for c in ("S", "L"):
        out[c] = {"trans_err_m": q(z[f"clip_{c}_trans_err_m"][clip_mask]), "trans_rel": q(z[f"clip_{c}_trans_rel"][clip_mask]),
                  "trans_dir_deg": q(z[f"clip_{c}_trans_dir_deg"][clip_mask]), "rot_err_deg": q(z[f"clip_{c}_rot_err_deg"][clip_mask]),
                  "trans_scale_ratio": q(z[f"clip_{c}_trans_scale_ratio"][clip_mask])}
    for c in ("S", "L", "G"):
        out.setdefault(c, {})["reproj_px"] = q(z[f"pt_{c}_reproj_px"][pt_mask]); out[c]["endpoint_pose_m"] = q(z[f"pt_{c}_endpoint_pose_m"][pt_mask])
    for c in ("S", "L", "Lp", "G", "GLd"):
        out.setdefault(c, {})["endpoint_depthpose_m"] = q(z[f"pt_{c}_endpoint_depthpose_m"][pt_mask])
    return out


def main() -> int:
    res = {"pose": {}, "depth": {}, "geometry": {}, "semantic": None}
    for d in DRIVES:
        p = os.path.join(LR.DATA_ROOT, "pose_audit", f"{d}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p); off = z["clip_offset"]; tt = z["clip_t"]; row = z["pt_row"]
        r = {"all": pose_block(z, np.ones(len(off), bool), np.ones(len(row), bool)), "by_offset": {}, "by_position": {},
             "gt_motion": {"trans_m": q(z["clip_gt_trans_m"]), "rot_deg": q(z["clip_gt_rot_deg"])}}
        for o in (-1, -2, -3, -4):
            cm = off == o; r["by_offset"][str(o)] = pose_block(z, cm, cm[row])
        for name, (lo, hi) in POSITION.items():
            cm = (tt >= lo) & (tt < hi)
            if cm.sum():
                r["by_position"][name] = pose_block(z, cm, cm[row])
        res["pose"][d] = r
        D = z["depth"]; F = [str(x) for x in z["depth_fields"]]
        res["depth"][d] = {k: q(D[:, F.index(k)]) for k in F if k != "t"}
        res["depth"][d]["by_position"] = {name: {k: q(D[(D[:, 0] >= lo) & (D[:, 0] < hi), F.index(k)]) for k in ("absrel_S", "absrel_L", "absrel_S_vs_L")}
                                          for name, (lo, hi) in POSITION.items() if ((D[:, 0] >= lo) & (D[:, 0] < hi)).any()}
    conds = LR.CONDITIONS
    for d in DRIVES:
        files = sorted(glob.glob(os.path.join(LR.DATA_ROOT, "geometry", d, "*.npz")))
        if not files:
            continue
        acc = {c: {k: np.zeros(3, np.int64) for k in ("past5_raw", "past5_dil", "target_raw", "target_dil", "head_target")} for c in conds}
        extra = {c: {"on_free": 0, "n_occ": 0, "conflict": 0, "mv": []} for c in conds}; ts = []; per_clip = {c: [] for c in conds}; jac = []; ratios = []
        for f in files:
            g = np.load(f); ts.append(int(g["t"])); jac.append(float(g["S_vs_streamframe_jaccard"])); ratios.append(float(g["oracle_ratio"]))
            for c in conds:
                for k in acc[c]:
                    acc[c][k] += g[f"{c}_{k}"]
                extra[c]["on_free"] += int(g[f"{c}_occ_on_lidar_free"]); extra[c]["n_occ"] += int(g[f"{c}_n_occ"]); extra[c]["conflict"] += int(g[f"{c}_conflict_voxels"])
                extra[c]["mv"].append(g[f"{c}_mv_nn_median_m"]); a_ = g[f"{c}_past5_raw"]; per_clip[c].append(a_[0] / max(a_.sum(), 1))
        ts = np.asarray(ts)
        met = lambda a_: {"iou": float(a_[0] / max(a_.sum(), 1)), "precision": float(a_[0] / max(a_[0] + a_[1], 1)), "recall": float(a_[0] / max(a_[0] + a_[2], 1))}
        gr = {"n_clips": len(files), "S_vs_streamframe_jaccard": q(jac), "oracle_depth_ratio_lidar_over_S": q(ratios), "conditions": {}}
        for c in conds:
            mv = np.stack([np.pad(np.asarray(x, np.float64), (0, 4 - len(x)), constant_values=np.nan) for x in extra[c]["mv"]])
            gr["conditions"][c] = {k: met(v) for k, v in acc[c].items()} | {
                "occupied_on_lidar_free_fraction": extra[c]["on_free"] / max(extra[c]["n_occ"], 1), "conflict_voxels_per_clip": extra[c]["conflict"] / len(files),
                "multiview_nn_median_m_by_offset": {str(o): float(np.nanmedian(mv[:, k])) for k, o in enumerate((-4, -3, -2, -1))},
                "past5_iou_per_clip": q(per_clip[c]),
                "by_position": {name: q(np.asarray(per_clip[c])[(ts >= lo) & (ts < hi)]) for name, (lo, hi) in POSITION.items() if ((ts >= lo) & (ts < hi)).any()}}
        res["geometry"][d] = gr
    sp = os.path.join(LR.ART, "semantic_check.json")
    if os.path.exists(sp):
        res["semantic"] = json.load(open(sp))["results"]
    C0.write_json(os.path.join(LR.ART, "analysis.json"), res)
    figures(res)
    tables(res)
    return 0


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(LR.ART, "figures"); os.makedirs(fd, exist_ok=True)
    col = {"S": "#c0392b", "L": "#2c6fbb", "Lp": "#5dade2", "G": "#3a9d5d", "GLd": "#82c09a", "Gs": "#7a5ea8"}
    P = res["pose"]; d0 = LR.PRIMARY
    if d0 in P:
        # 1 reprojection error S/L/G per drive
        fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(P)); w = 0.25
        for j, c in enumerate(("S", "L", "G")):
            med = [P[d]["all"][c]["reproj_px"]["median"] for d in P]; p90 = [P[d]["all"][c]["reproj_px"]["p90"] for d in P]
            ax.bar(x + (j - 1) * w, med, w, color=col[c], label=f"{c} median"); ax.scatter(x + (j - 1) * w, p90, marker="_", s=200, color="k")
        ax.scatter([], [], marker="_", s=200, color="k", label="p90"); ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in P]); ax.set_ylabel("reprojection error of verified LiDAR points (px)")
        ax.set_yscale("symlog", linthresh=1); ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_title("pose-only reprojection into the anchor camera, five-frame clips")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "1_reprojection.png"), dpi=120); plt.close(fig)
        # 2, 3 translation / rotation vs offset
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for d in P:
            ls = "-" if d == d0 else ":"
            for c in ("S", "L"):
                offs = ["-1", "-2", "-3", "-4"]
                ax[0].plot(offs, [P[d]["by_offset"][o][c]["trans_err_m"]["median"] for o in offs], ls, marker="o", color=col[c], label=f"{c} {SHORT[d]}", alpha=1 if d == d0 else 0.5)
                ax[1].plot(offs, [P[d]["by_offset"][o][c]["rot_err_deg"]["median"] for o in offs], ls, marker="o", color=col[c], label=f"{c} {SHORT[d]}", alpha=1 if d == d0 else 0.5)
                ax[2].plot(offs, [P[d]["by_offset"][o][c]["trans_rel"]["median"] for o in offs], ls, marker="o", color=col[c], label=f"{c} {SHORT[d]}", alpha=1 if d == d0 else 0.5)
        for a_, lab in zip(ax, ("translation error (m, median)", "rotation error (deg, median)", "relative translation error (median)")):
            a_.set_xlabel("frame offset from anchor"); a_.set_ylabel(lab); a_.grid(alpha=0.3); a_.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(fd, "2_3_pose_vs_offset.png"), dpi=120); plt.close(fig)
        # 4 error vs stream position (0006)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4)); names = list(P[d0]["by_position"])
        for c in ("S", "L", "G"):
            ax[0].plot(names, [P[d0]["by_position"][n][c]["reproj_px"]["median"] for n in names], "-o", color=col[c], label=c)
        for c in ("S", "L"):
            ax[1].plot(names, [P[d0]["by_position"][n][c]["trans_rel"]["median"] for n in names], "-o", color=col[c], label=c)
        ax[0].set_ylabel("reprojection px (median)"); ax[1].set_ylabel("relative translation error (median)")
        G = res["geometry"].get(d0)
        if G:
            a2 = ax[1].twinx()
            for c in ("S", "L", "G"):
                a2.plot(names, [G["conditions"][c]["by_position"].get(n, {}).get("median") for n in names], "--s", color=col[c], alpha=0.6, label=f"{c} raw IoU")
            a2.set_ylabel("raw five-frame map IoU vs LiDAR (median, dashed)"); a2.legend(fontsize=7, loc="lower right")
        for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=8); a_.set_xlabel("anchor stream position (0006)")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "4_stream_position.png"), dpi=120); plt.close(fig)
        # 7 depth S vs L
        fig, ax = plt.subplots(figsize=(9, 4)); D = res["depth"]; x = np.arange(len(D))
        for j, (k, lab, cc) in enumerate((("absrel_S", "AbsRel S vs LiDAR", col["S"]), ("absrel_L", "AbsRel L vs LiDAR", col["L"]), ("absrel_S_vs_L", "AbsRel S vs L (canonical)", "#999999"))):
            ax.bar(x + (j - 1) * 0.25, [D[d][k]["median"] for d in D], 0.25, color=cc, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in D]); ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_title("anchor-frame depth, median per clip")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "7_depth_S_vs_L.png"), dpi=120); plt.close(fig)
    if res["geometry"]:
        Gm = res["geometry"]; conds = LR.CONDITIONS
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.2)); x = np.arange(len(Gm)); w = 0.8 / len(conds)
        for j, c in enumerate(conds):
            ax[0].bar(x + (j - 2.5) * w, [Gm[d]["conditions"][c]["past5_raw"]["iou"] for d in Gm], w, color=col[c], label=c)
            ax[1].bar(x + (j - 2.5) * w, [Gm[d]["conditions"][c]["target_dil"]["iou"] for d in Gm], w, color=col[c], label=c)
            ax[2].bar(x + (j - 2.5) * w, [np.nanmean(list(Gm[d]["conditions"][c]["multiview_nn_median_m_by_offset"].values())) for d in Gm], w, color=col[c], label=c)
        for a_, lab in zip(ax, ("raw map IoU vs the same five LiDAR sweeps", "dilated map IoU vs future raw-LiDAR target", "multi-view NN distance to anchor-frame surface (m)")):
            a_.set_xticks(x); a_.set_xticklabels([SHORT[d] for d in Gm]); a_.set_title(lab, fontsize=9); a_.grid(alpha=0.3, axis="y"); a_.legend(fontsize=7, ncol=3)
        fig.suptitle("Gs = GT pose + LiDAR-oracle depth scale (diagnostic only)"); fig.tight_layout(); fig.savefig(os.path.join(fd, "5_6_geometry.png"), dpi=120); plt.close(fig)
    if res["semantic"]:
        Sm = res["semantic"]; cs = list(Sm); fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].bar(cs, [Sm[c]["support_fraction"] for c in cs], color=[col[c] for c in cs]); ax[0].set_title("same-voxel two-frame support")
        ax[1].bar(cs, [Sm[c]["class_agreement"] for c in cs], color=[col[c] for c in cs]); ax[1].plot(cs, [Sm[c]["local_control_agreement"] for c in cs], "k_", ms=25, mew=2, label="local control"); ax[1].set_title("class agreement"); ax[1].legend()
        ax[2].bar(cs, [Sm[c]["cosine_margin_over_local"] for c in cs], color=[col[c] for c in cs]); ax[2].set_title("feature cosine margin over local control")
        for a_ in ax: a_.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(fd, "8_semantic_check.png"), dpi=120); plt.close(fig)


def tables(res):
    L = ["# Local registration audit -- result tables\n"]
    for d, r in res["pose"].items():
        L.append(f"## {SHORT[d]}: relative pose and reprojection (five-frame clips)\n")
        L.append(f"GT motion anchor<-frame: translation median {r['gt_motion']['trans_m']['median']:.2f} m, rotation median {r['gt_motion']['rot_deg']['median']:.2f} deg\n")
        L.append("| offset | S trans err m | L trans err m | S rel | L rel | S dir deg | L dir deg | S rot deg | L rot deg | S reproj px med/p90 | L reproj px med/p90 | G reproj px |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for o, b in list(r["by_offset"].items()) + [("all", r["all"])]:
            L.append(f"| {o} | {b['S']['trans_err_m']['median']:.2f} | {b['L']['trans_err_m']['median']:.2f} | {b['S']['trans_rel']['median']:.3f} | {b['L']['trans_rel']['median']:.3f} | "
                     f"{b['S']['trans_dir_deg']['median']:.1f} | {b['L']['trans_dir_deg']['median']:.1f} | {b['S']['rot_err_deg']['median']:.2f} | {b['L']['rot_err_deg']['median']:.2f} | "
                     f"{b['S']['reproj_px']['median']:.1f} / {b['S']['reproj_px']['p90']:.0f} | {b['L']['reproj_px']['median']:.1f} / {b['L']['reproj_px']['p90']:.0f} | {b['G']['reproj_px']['median']:.2f} |")
        L.append("\nEndpoint error after mapping into the anchor camera (median m): pose only S / L / G = "
                 f"{r['all']['S']['endpoint_pose_m']['median']:.2f} / {r['all']['L']['endpoint_pose_m']['median']:.2f} / {r['all']['G']['endpoint_pose_m']['median']:.2f}; "
                 "LingBot depth + pose: " + ", ".join(f"{c} {r['all'][c]['endpoint_depthpose_m']['median']:.2f}" for c in ("S", "L", "Lp", "G", "GLd")) + "\n")
        L.append("| stream position | n pairs | S reproj px | L reproj px | S rel trans | L rel trans | S rot deg | L rot deg |"); L.append("|---|---|---|---|---|---|---|---|")
        for n_, b in r["by_position"].items():
            L.append(f"| {n_} | {b['S']['trans_err_m']['n']} | {b['S']['reproj_px']['median']:.1f} | {b['L']['reproj_px']['median']:.1f} | {b['S']['trans_rel']['median']:.3f} | {b['L']['trans_rel']['median']:.3f} | {b['S']['rot_err_deg']['median']:.2f} | {b['L']['rot_err_deg']['median']:.2f} |")
        L.append("")
    L.append("## Depth (anchor frame, median over clips)\n"); L.append("| drive | AbsRel S vs LiDAR | AbsRel L vs LiDAR | AbsRel S vs L | corr S,L | L/S canonical ratio | L/S scale | S metric/LiDAR | L metric/LiDAR |"); L.append("|---|---|---|---|---|---|---|---|---|")
    for d, r in res["depth"].items():
        L.append(f"| {SHORT[d]} | {r['absrel_S']['median']:.3f} | {r['absrel_L']['median']:.3f} | {r['absrel_S_vs_L']['median']:.3f} | {r['corr_S_L_canonical']['median']:.3f} | {r['median_ratio_L_over_S_canonical']['median']:.3f} | {r['scale_L_over_S']['median']:.3f} | {r['median_ratio_S_metric_over_lidar']['median']:.3f} | {r['median_ratio_L_metric_over_lidar']['median']:.3f} |")
    for d, g in res["geometry"].items():
        L.append(f"\n## {SHORT[d]}: five-frame maps ({g['n_clips']} clips; anchor-relative S vs stream-frame S Jaccard median {g['S_vs_streamframe_jaccard']['median']:.3f}; LiDAR/S depth ratio median {g['oracle_depth_ratio_lidar_over_S']['median']:.3f})\n")
        L.append("| condition | raw vs past-5 LiDAR IoU / P / R | dilated vs past-5 IoU | raw vs future target IoU | dilated vs target IoU / P / R | occupied on LiDAR-free | conflicts / clip | multi-view NN m (-4,-3,-2,-1) | frozen head vs target IoU / P / R |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c, v in g["conditions"].items():
            mv = v["multiview_nn_median_m_by_offset"]; tag = " (oracle)" if c in LR.DIAGNOSTIC_ORACLE else ""
            L.append(f"| {c}{tag} | {v['past5_raw']['iou']:.3f} / {v['past5_raw']['precision']:.3f} / {v['past5_raw']['recall']:.3f} | {v['past5_dil']['iou']:.3f} | {v['target_raw']['iou']:.3f} | "
                     f"{v['target_dil']['iou']:.3f} / {v['target_dil']['precision']:.3f} / {v['target_dil']['recall']:.3f} | {v['occupied_on_lidar_free_fraction']:.3f} | {v['conflict_voxels_per_clip']:.0f} | "
                     f"{mv['-4']:.2f} / {mv['-3']:.2f} / {mv['-2']:.2f} / {mv['-1']:.2f} | {v['head_target']['iou']:.3f} / {v['head_target']['precision']:.3f} / {v['head_target']['recall']:.3f} |")
        L.append("\n| condition | " + " | ".join(g["conditions"]["S"]["by_position"]) + " |"); L.append("|---|" + "---|" * len(g["conditions"]["S"]["by_position"]))
        for c, v in g["conditions"].items():
            L.append(f"| {c} raw past-5 IoU (median per clip) | " + " | ".join(f"{v['by_position'][n]['median']:.3f}" for n in g["conditions"]["S"]["by_position"]) + " |")
    if res["semantic"]:
        L.append("\n## Semantic sanity check (0006, verified same-point pairs inside clips)\n"); L.append("| condition | landmarks | two-frame same-voxel support | class agreement | local-control agreement | feature cosine | local-control cosine | cosine margin |"); L.append("|---|---|---|---|---|---|---|---|")
        for c, v in res["semantic"].items():
            L.append(f"| {c} | {v['landmarks']:,} | {v['support_fraction']:.3f} | {v['class_agreement']:.3f} | {v['local_control_agreement']:.3f} | {v['feature_cosine']:.3f} | {v['local_control_cosine']:.3f} | {v['cosine_margin_over_local']:+.3f} |")
    open(os.path.join(LR.ART, "results_table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
