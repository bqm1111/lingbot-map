#!/usr/bin/env python
"""Collect every number the report cites into ``results.json`` and ``results_table.md``.

    python tools/future_semantic_completion/summarize.py --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402

CONDS = ("B0", "B1", "B2", "B2ref_existing_head")
LABEL = {"B0": "B0 global mean", "B1": "B1 nearest map evidence", "B2": "B2 learned head",
         "B2ref_existing_head": "ref: existing joint head"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = ST.run_tag(a.seed)
    sup = json.load(open(os.path.join(ST.ART, "supervision.json")))
    src = json.load(open(os.path.join(ST.ART, "source", f"source_{tag}.json")))
    diag = json.load(open(os.path.join(ST.ART, "target_diagnosis.json")))
    tr = json.load(open(os.path.join(ST.ART, f"train_{tag}.json")))
    ver = json.load(open(os.path.join(ST.ART, "verification.json")))
    transfer = {}
    for ds in ("semantickitti", "occ3d"):
        p = os.path.join(ST.ART, "transfer", f"eval_{ds}_{tag}.json")
        if os.path.exists(p):
            transfer[ds] = json.load(open(p))
    L = []
    L.append(f"# Future semantic completion -- seed {a.seed}\n")
    L.append("## Source: KITTI-360 drive 0006, oracle-occupied voxels (590 anchors)\n")
    for pn, title in (("key", "unobserved at t -> later LiDAR-occupied (PRIMARY)"), ("observed", "map-observed, LiDAR-occupied (contrast)")):
        P = src["populations"][pn]
        L.append(f"**{title}** -- n = {P['B0']['n_voxels']:,}\n")
        L.append("| condition | cosine to future teacher | KL | top-1 agreement | mIoU (present classes) | cosine to B0 | classes predicted |")
        L.append("|---|---|---|---|---|---|---|")
        for c in CONDS:
            s = P[c]; npred = sum(1 for v in s["per_class"].values() if v["predicted"] > 0)
            L.append(f"| {LABEL[c]} | {s['cosine_mean']:.4f} | {s['kl_mean']:.3f} | {s['top1_agreement']:.4f} | {s['miou_present_classes']:.4f} | {s['cosine_to_b0_pred']:.3f} | {npred} |")
        L.append("")
    L.append("Paired differences on the primary set (block bootstrap, 20-anchor blocks, 2 000 resamples):\n")
    L.append("| pair | Δ cosine [95 % CI] | Δ top-1 [95 % CI] | Δ KL [95 % CI] |")
    L.append("|---|---|---|---|")
    for pr in ("B2-B0", "B2-B1", "B2-B2ref_existing_head", "B1-B0"):
        def f(k):
            v = src["paired"][f"key:{pr}:{k}"]; return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
        L.append(f"| {pr} | {f('sum_cos')} | {f('n_top1')} | {f('sum_kl')} |")
    d = src["distance_to_evidence"]; h = np.asarray(d["key_hist"]) / sum(d["key_hist"])
    L.append("\nTop-1 agreement by distance to the nearest map voxel with teacher evidence (primary set):\n")
    L.append("| distance (voxels) | " + " | ".join(d["names"]) + " |")
    L.append("|---|" + "---|" * len(d["names"]))
    L.append("| share of voxels | " + " | ".join(f"{x:.3f}" for x in h) + " |")
    for c in CONDS:
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{x:.3f}" for x in d["by_condition"][c]["top1"]) + " |")
    o = diag["overall"]; sh = diag["shuffled_pairing"]
    L.append(f"\n## Target repeatability (drive 0006, {diag['n_anchors']} anchors)\n")
    L.append(f"Causal-map teacher vector vs future teacher vector at the same LiDAR-occupied voxel: top-1 agreement "
             f"**{o['top1_agreement']:.3f}**, cosine {o['cosine']:.3f} (shuffled pairing {sh['top1_agreement']:.3f} / {sh['cosine']:.3f}), n = {o['n']:,}.\n")
    L.append("| evidence weight (rays) on the map side | " + " | ".join(diag["by_current_map_weight"]) + " |")
    L.append("|---|" + "---|" * len(diag["by_current_map_weight"]))
    L.append("| top-1 agreement | " + " | ".join(f"{v['top1_agreement']:.3f}" for v in diag["by_current_map_weight"].values()) + " |")
    L.append("\n| future class | " + " | ".join(k for k, _ in sorted(diag["by_future_class"].items(), key=lambda x: -x[1]["n"])[:8]) + " |")
    L.append("|---|" + "---|" * 8)
    L.append("| repeatability | " + " | ".join(f"{v['top1_agreement']:.2f}" for _, v in sorted(diag["by_future_class"].items(), key=lambda x: -x[1]["n"])[:8]) + " |")
    if transfer:
        L.append("\n## Zero-shot transfer, past5, benchmark GT labels on GT-occupied voxels (seed 0, measurement only)\n")
        L.append("| dataset / population | n | " + " | ".join(f"{LABEL[c]} top-1 / mIoU" for c in CONDS) + " |")
        L.append("|---|---|" + "---|" * len(CONDS))
        for ds, t in transfer.items():
            for pn in ("unobserved", "observed"):
                P = t["populations"][pn]
                L.append(f"| {ds} {pn} | {P['B0']['n_voxels']:,} | " + " | ".join(
                    f"{P[c]['top1_accuracy']:.3f} / {P[c]['miou_present_classes']:.3f}" for c in CONDS) + " |")
        L.append("\nPaired Δ top-1 on unobserved voxels (scene bootstrap):\n")
        for ds, t in transfer.items():
            for pr in ("B2-B0", "B2-B1", "B1-B0"):
                v = t["paired"][f"unobserved:{pr}:n_top1"]
                L.append(f"- {ds} {pr}: {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    L.append(f"\n## Supervision\n")
    tot_lost = sum(v["key_no_target"] for k, v in sup["drives"].items() if k != "2013_05_28_drive_0006_sync")
    L.append(f"Training key set {sup['train_total_key']:,} voxels over {sup['train_total_anchors']} anchors; "
             f"{tot_lost:,} further unobserved->future-observed->occupied voxels carry no teacher vector "
             f"({sup['train_total_key']/(sup['train_total_key']+tot_lost):.1%} coverage). ")
    for k, v in sup["drives"].items():
        L.append(f"- {k[17:21]}: {v['key']:,} key voxels ({v['key_per_anchor_mean']:.0f}/anchor), top-1 class share {v['argmax_top1_share']:.3f}, "
                 f"top-2 {v['argmax_top2_share']:.3f}, entropy {v['argmax_entropy_nats']:.2f} nats, {v['n_classes_present']} classes")
    L.append(f"\nTraining: {tr['steps']} steps, {tr['head_params']:,} head parameters, {tr['seconds']/60:.1f} min; "
             f"final val KL {[m['val']['loss'] for m in tr['log'] if 'val' in m][-1]:.3f} vs B0 "
             f"{[m['val']['b0_loss'] for m in tr['log'] if 'val' in m][-1]:.3f}; mean train KL over the last 500 steps "
             f"{np.mean([m['loss'] for m in tr['log'] if 'val' not in m][-500:]):.3f}.")
    L.append(f"\nVerification: {ver['n_pass']}/{ver['n_total']} checks pass; failing: "
             + ", ".join(c["name"] for c in ver["checks"] if not c["pass"]) + ".")
    open(os.path.join(ST.ART, "results_table.md"), "w").write("\n".join(L) + "\n")
    ST.write_json(os.path.join(ST.ART, "results.json"),
                  {"source": src["populations"], "paired_source": src["paired"], "distance": d,
                   "target_repeatability": {k: diag[k] for k in ("overall", "shuffled_pairing", "by_current_map_weight", "by_future_weight", "by_future_class")},
                   "transfer": {ds: {"populations": t["populations"], "paired": t["paired"], "n_clips": t["n_clips"]} for ds, t in transfer.items()},
                   "supervision": {k: v for k, v in sup.items() if k != "b0"}, "verification": ver})
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
