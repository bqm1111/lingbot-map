#!/usr/bin/env python
"""The three figures of the streaming phase, each built from the recorded measurements only.

``accuracy_vs_latency``   every point's accuracy and latency come from the SAME operating
                          mode, read from one record per configuration. A point is drawn only
                          if both halves exist for that configuration; nothing is paired
                          across modes and nothing is interpolated.
``performance_over_time`` per-anchor SC IoU against position in the physical stream, clip-reset
                          against persistent-state, with a rolling median. This is the figure
                          that would show drift if the startup-only gauge decayed.
``regional``              the five history-aware visibility regions, both modes, with the
                          region's share of GT occupancy underneath.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/figures.py --which all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import matplotlib                                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                          # noqa: E402
from tools.method_vs_data.streaming import config as SV, streams as ST                   # noqa: E402

C_A = "#4C78A8"
C_B = "#E45756"
C_O = "#54A24B"
C_M = "#9C755F"


def _rec(path: str) -> Optional[dict]:
    return json.load(open(path)) if os.path.exists(path) else None


def accuracy_vs_latency(out_dir: str) -> Optional[dict]:
    """One point per configuration; accuracy and latency from the same configuration."""
    sk = _rec(os.path.join(SV.SCORE_DIR, "streaming_semantickitti.json"))
    if sk is None:
        return None
    pts: List[dict] = []
    # ours, clip-reset: accuracy from mode A, latency from the clip-reset runtime record
    clip = _rec(os.path.join(SV.RUNTIME_DIR, "clipreset_semantickitti_e2e.json"))
    strm = _rec(os.path.join(SV.RUNTIME_DIR, "stream_semantickitti_e2e.json"))
    if clip:
        pts.append({"name": "ours · clip-reset", "colour": C_A, "marker": "o",
                    "latency_ms": clip["latency_per_completed_occupancy_grid"]["mean_ms"],
                    "iou_raw": sk["iou"]["raw"]["all"]["A:k360_s0"]["iou"],
                    "iou_pooled": sk["iou"]["pooled"]["all"]["A:k360_s0"]["iou"],
                    "latency_source": "clipreset_semantickitti_e2e.json",
                    "latency_definition": "per completed occupancy grid, state reset per anchor"})
    if strm:
        pts.append({"name": "ours · streaming (per prediction)", "colour": C_B, "marker": "s",
                    "latency_ms": strm["per_prediction_s"]["mean_ms"],
                    "iou_raw": sk["iou"]["raw"]["all"]["B:k360_s0"]["iou"],
                    "iou_pooled": sk["iou"]["pooled"]["all"]["B:k360_s0"]["iou"],
                    "latency_source": "stream_semantickitti_e2e.json",
                    "latency_definition": "prediction step only, on top of the per-frame update"})
        pts.append({"name": "ours · streaming (sequence average)", "colour": C_B, "marker": "D",
                    "latency_ms": strm["sequence_average_per_prediction_s"]["mean_ms"],
                    "iou_raw": sk["iou"]["raw"]["all"]["B:k360_s0"]["iou"],
                    "iou_pooled": sk["iou"]["pooled"]["all"]["B:k360_s0"]["iou"],
                    "latency_source": "stream_semantickitti_e2e.json",
                    "latency_definition": ("total sequence computation, startup and every "
                                           "intervening update included, per emitted grid")})
    # OccAny: prefer the reruns taken in THIS measurement session, on the same idle GPU as ours;
    # fall back to the previous phase's files, which used the same GPU model but a different run.
    def _occany(preset: str):
        rerun = os.path.join(SV.RUNTIME_DIR, "occany_%s_e2e_rerun.json" % preset)
        r = _rec(rerun)
        if r is not None:
            return r, "streaming_scale_validation/runtime/occany_%s_e2e_rerun.json" % preset, \
                "per completed occupancy grid, rerun on the same idle GPU as our measurement"
        prior = os.path.join(SV.REPO_ROOT, "artifacts", "method_vs_data", "runtime",
                             "occany_%s_e2e.json" % preset)
        return _rec(prior), "method_vs_data/runtime/occany_%s_e2e.json" % preset, \
            "per completed occupancy grid, previous phase, same GPU model"

    occ, occ_src, occ_def = _occany("K_1FRAME_GEOM")
    occ5, occ5_src, occ5_def = _occany("K_5FRAMES_GEOM")
    for rec, src, defn, key, nm, mk in (
            (occ, occ_src, occ_def, "occany_1f", "OccAny · 1 causal frame", "^"),
            (occ5, occ5_src, occ5_def, "occany_5f_fwd",
             "OccAny · 5 forward frames [+4 s future]", "v")):
        if rec and key in sk["iou"]["raw"]["all"]:
            pts.append({"name": nm, "colour": C_O, "marker": mk,
                        "latency_ms": rec["latency_per_completed_occupancy_grid"]["mean_ms"],
                        "iou_raw": sk["iou"]["raw"]["all"][key]["iou"],
                        "iou_pooled": sk["iou"]["pooled"]["all"][key]["iou"],
                        "latency_source": src,
                        "latency_definition": defn})
    if not pts:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, post in zip(axes, ("raw", "pooled")):
        for p in pts:
            ax.scatter(p["latency_ms"], p[f"iou_{post}"], s=150, c=p["colour"],
                       marker=p["marker"], edgecolor="black", linewidth=0.7,
                       label=p["name"], zorder=3)
            ax.annotate(f"{p[f'iou_{post}']:.1f} @ {p['latency_ms']:.0f} ms",
                        (p["latency_ms"], p[f"iou_{post}"]), textcoords="offset points",
                        xytext=(8, -12), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("latency per completed occupancy grid (ms, log scale)")
        ax.set_ylabel(f"SC IoU, {post} (%)")
        ax.set_title(f"SemanticKITTI 08 · {post}")
        ax.grid(alpha=0.25, zorder=0)
    axes[0].legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.suptitle("Accuracy against measured latency — each point's accuracy and latency come "
                 "from the same operating mode", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "accuracy_vs_latency.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"figure": p, "points": pts,
            "warning": ("the streaming and clip-reset points are different operating modes, "
                        "not a speed-up of one configuration; OccAny's forward point consumes "
                        "4 s of future frames and is not temporally matched to either of ours")}


def performance_over_time(out_dir: str, dataset: str = "semantickitti") -> Optional[dict]:
    npz = os.path.join(SV.SCORE_DIR, f"streaming_{dataset}_per_anchor.npz")
    if not os.path.exists(npz):
        return None
    z = np.load(npz, allow_pickle=True)
    anchors = [str(x) for x in z["anchors"]]
    segs = {s.name: s for s in ST.build(dataset)}
    pos, tim = [], []
    by = {}
    for s in segs.values():
        for sid, p in s.anchors:
            by[sid] = (s.name, p, float(s.times_s[p]))
    for a in anchors:
        _n, p, t = by[a]
        pos.append(p); tim.append(t)
    pos, tim = np.asarray(pos), np.asarray(tim)
    out = {}
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    for ax, post in zip(axes, ("raw", "pooled")):
        for key, c, nm in (("A:k360_s0", C_A, "clip-reset"), ("B:k360_s0", C_B, "streaming"),
                           ("B:map_only", C_M, "map-only (streaming)")):
            k = f"{key}|{post}|all"
            if k not in z:
                continue
            cnt = z[k].astype(np.float64)
            iou = 100 * cnt[:, 0] / np.maximum(cnt.sum(1), 1)
            o = np.argsort(tim)
            w = 25
            med = np.convolve(iou[o], np.ones(w) / w, mode="valid")
            ax.plot(tim[o], iou[o], lw=0.5, alpha=0.25, color=c)
            ax.plot(tim[o][w - 1:], med, lw=2.0, color=c, label=f"{nm} (rolling mean, {w})")
            out[f"{key}|{post}"] = {"mean": float(iou.mean()),
                                    "first_quarter_mean": float(iou[o][:len(o)//4].mean()),
                                    "last_quarter_mean": float(iou[o][-len(o)//4:].mean())}
        ax.set_ylabel(f"per-anchor SC IoU, {post} (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel(f"time since the stream reset (s) — one unbroken "
                        f"{len(segs['08']) if '08' in segs else 0}-frame stream")
    fig.suptitle("Per-anchor accuracy along the physical stream: the startup-only gauge is "
                 "fixed at t = 2.1 s and never refreshed", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "performance_over_time.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"figure": p, "summary": out,
            "note": ("a declining streaming curve against a flat clip-reset curve would be "
                     "gauge or pose drift; the quarter means above are the numbers behind "
                     "that reading")}


PRETTY = {"current_frame": "observed in the\ncurrent frame",
          "recent_other_four": "observed in the other\nfour recent frames",
          "earlier_in_stream": "observed earlier in\nthe stream only",
          "never_in_frustum": "never observed,\nin anchor frustum",
          "never_out_frustum": "never observed,\noutside that frustum"}


def regional(out_dir: str, dataset: str = "semantickitti") -> Optional[dict]:
    rec = _rec(os.path.join(SV.VISIBILITY_DIR, f"regional_{dataset}.json"))
    if rec is None:
        return None
    from tools.method_vs_data.streaming.visibility import HISTORY
    arms = [a for a in ("A:k360_s0", "B:k360_s0", "B:map_only", "occany_1f", "occany_5f_fwd")
            if a in rec["arms"]]
    labels = {"A:k360_s0": "ours · clip-reset", "B:k360_s0": "ours · streaming",
              "B:map_only": "map-only · streaming", "occany_1f": "OccAny · 1 causal frame",
              "occany_5f_fwd": "OccAny · 5 forward [+4 s]"}
    cols = {"A:k360_s0": C_A, "B:k360_s0": C_B, "B:map_only": C_M,
            "occany_1f": C_O, "occany_5f_fwd": "#B279A2"}
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 8.0))
    x = np.arange(len(HISTORY))
    w = 0.8 / max(len(arms), 1)
    for ax, post in zip(axes, ("raw", "pooled")):
        for i, a in enumerate(arms):
            v = [rec["iou"][post][k][a]["iou"] for k in HISTORY]
            ax.bar(x + i * w - 0.4 + w / 2, v, w, color=cols[a], edgecolor="black",
                   linewidth=0.5, label=labels[a])
        ax.set_xticks(x)
        # the share goes INSIDE the tick label: a separate annotation at a fixed offset
        # collides with the wrapped region names, which are up to three lines tall
        ax.set_xticklabels([f"{PRETTY.get(k, k.replace('_', ' '))}\n"
                            f"{100*rec['composition'][k]['share_of_gt_occupied']:.0f}% of GT occ."
                            for k in HISTORY], fontsize=9)
        for t in ax.get_xticklabels():
            t.set_linespacing(1.6)
        ax.set_ylabel(f"SC IoU, {post} (%)")
        # explicit headroom so the legend never sits on top of the tallest bar
        top = max(rec["iou"][post][k][a]["iou"] for k in HISTORY for a in arms)
        ax.set_ylim(0, top * 1.24)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, ncol=len(arms), loc="upper center", framealpha=0.95)
    fig.suptitle(f"{dataset}: accuracy by what the PROCESSED STREAM had seen "
                 f"({rec['n_anchors']} anchors)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, f"regional_{dataset}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return {"figure": p, "arms": arms,
            "iou": {post: {k: {a: rec["iou"][post][k][a]["iou"] for a in arms}
                           for k in HISTORY} for post in ("raw", "pooled")},
            "composition": {k: rec["composition"][k] for k in HISTORY}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    choices=["all", "latency", "time", "regional"])
    a = ap.parse_args()
    SV.ensure_dirs()
    out = SV.FIGURE_DIR
    rec_path = os.path.join(out, "figure_data.json")
    data = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    if a.which in ("all", "latency"):
        r = accuracy_vs_latency(out)
        if r:
            data["accuracy_vs_latency"] = r
            print("->", r["figure"])
        else:
            print("accuracy_vs_latency: skipped (missing runtime or score record)")
    if a.which in ("all", "time"):
        r = performance_over_time(out)
        if r:
            data["performance_over_time"] = r
            print("->", r["figure"])
        else:
            print("performance_over_time: skipped")
    if a.which in ("all", "regional"):
        for d in SV.DATASETS:
            r = regional(out, d)
            if r:
                data[f"regional_{d}"] = r
                print("->", r["figure"])
            else:
                print(f"regional_{d}: skipped")
    json.dump(data, open(rec_path, "w"), indent=1, default=float)
    print("->", rec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
