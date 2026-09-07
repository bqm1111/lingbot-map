#!/usr/bin/env python
"""Generate every number in the Gate-7B report from the artifacts."""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                          # noqa: E402
from gates.gate7b import config as C, evidence as EV, rays as RY, replay as RP, \
    scale as SC, streams as ST, voxmap as VM                                     # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")
TEMPLATE = os.path.join(REPO_ROOT, "reports", "gate7b",
                        "_streaming_metric_semantic_replay.template.md")
OUT = os.path.join(REPO_ROOT, "reports", "gate7b",
                   "streaming_metric_semantic_replay.md")
ORDER = ["semantickitti", "occ3d", "kitti360"]
LABEL = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes",
         "kitti360": "SSCBench-KITTI-360"}
_S = {}


def S():
    if "s" not in _S:
        p = os.path.join(ART, "summary.json")
        _S["s"] = json.load(open(p)) if os.path.exists(p) else {"datasets": {}}
    return _S["s"]


def have():
    s = S()
    return [(ds, s["datasets"][ds]) for ds in ORDER
            if ds in s.get("datasets", {}) and s["datasets"][ds]["configs"]]


def cfg(d, tag):
    return d["configs"].get(tag)


def j(name):
    p = os.path.join(ART, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def andlist(xs):
    xs = list(xs)
    if not xs:
        return "none"
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# --------------------------------------------------------------------------- #
def t_headline():
    r = ["| benchmark | five-frame B-R | five-frame B-D (S0) | S1 streaming, all past | S2 relaxed | S3 MoGe rescue | S4 gated fusion |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        row = [LABEL[ds]]
        for src in ("B-R", "B-D"):
            v = d["reference"][src]
            row.append(f"{v['binary_iou']:.4f} / {v['ssc_miou']:.4f}")
        for tag in ("S1_G-A_hall", "S2_G-A_hall_c0.5", "S3_G-A_hall", "S4_G-A_hall"):
            c = cfg(d, tag)
            row.append(f"**{c['binary_iou']:.4f}** / **{c['ssc_miou']:.4f}**"
                       if c else "—")
        r.append("| " + " | ".join(row) + " |")
    r.append("")
    r.append("_Each cell is pooled **binary IoU / SSC mIoU**. S0–S4 use the identical "
             "Gate-6 metric code, so these are directly comparable with Gates 6 and 7A._")
    return "\n".join(r)


def t_interface():
    r = ["| component | what it is | horizon |", "|---|---|---|"]
    r.append(f"| input at *t* | one RGB image, {RP.INFERENCE_RESOLUTION} px wide, "
             f"patch {RP.PATCH_SIZE} | — |")
    r.append(f"| anchor context | the first {RP.NUM_SCALE_FRAMES} frames of the segment, "
             "pinned in the KV cache | whole segment |")
    r.append(f"| pose-reference window | KV sliding window | "
             f"{RP.KV_CACHE_SLIDING_WINDOW} blocks |")
    r.append("| trajectory / camera tokens | cross-frame special tokens | sliding |")
    r.append("| persistent map | log-odds occupancy + separate semantic accumulator "
             "| unbounded |")
    r.append(f"| entry point | `GCTStream.inference_streaming` | — |")
    r.append(f"| reset | `clean_kv_cache()` once per segment | genuine boundary only |")
    return "\n".join(r)


def t_stream():
    r = ["| benchmark | segments | stream frames | official anchors | keyframe interval | RoPE slots used | streaming FPS | peak VRAM |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ORDER:
        m = j(f"stream_{ds}.json")
        if m is None:
            continue
        segs = [s for s in m["segments"] if not s.get("cached")]
        fps = [s["fps"] for s in segs if "fps" in s]
        k = sorted({s.get("keyframe_interval", 1) for s in segs})
        slots = max((s.get("rope_slots", 0) for s in segs), default=0)
        st = ST.summarize(ds, REPO_ROOT)
        r.append(f"| {LABEL[ds]} | {m['n_segments']} | {m['n_frames_streamed']:,} "
                 f"| {st['n_anchor_clips']:,} | {'/'.join(str(x) for x in k)} "
                 f"| {slots:,} | {np.median(fps) if fps else 0:.1f} "
                 f"| {m['peak_gpu_gib']:.2f} GiB |")
    return "\n".join(r)


def t_scale():
    r = ["| benchmark | policy | median scale | log MAD | adjacent-frame log MAD | first→final drift | min | max |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds in ORDER:
        first = True
        for pol in C.SCALE_POLICIES:
            p = os.path.join(ART, f"eval_{ds}_S1_{pol}_hall.json")
            if not os.path.exists(p):
                continue
            e = json.load(open(p))
            segs = list(e["scale_diagnostics"].values())
            def med(k):
                v = [s[k] for s in segs if k in s]
                return float(np.median(v)) if v else float("nan")
            r.append(f"| {LABEL[ds] if first else ''} | {pol} "
                     f"| {med('median_scale'):.3f} | {med('log_mad'):.4f} "
                     f"| {med('adjacent_log_mad'):.4f} "
                     f"| {med('first_to_final_log_drift'):+.4f} "
                     f"| {med('min_scale'):.2f} | {med('max_scale'):.2f} |")
            first = False
    return "\n".join(r)


def t_thickness():
    r = ["| benchmark | policy | map thickness (voxels/column) | duplicate-surface rate | occupied voxels | binary IoU | SSC mIoU |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        t = j(f"thickness_{ds}.json")
        first = True
        for pol in C.SCALE_POLICIES:
            c = cfg(d, f"S1_{pol}_hall")
            tp = (t or {}).get("policies", {}).get(pol)
            if c is None and tp is None:
                continue
            cells = [LABEL[ds] if first else "", pol,
                     f"{tp['thickness_mean']:.3f}" if tp else "—",
                     f"{tp['duplicate_rate_mean']:.4f}" if tp else "—",
                     f"{tp['occupied_mean']:,.0f}" if tp else "—",
                     f"{c['binary_iou']:.4f}" if c else "—",
                     f"{c['ssc_miou']:.4f}" if c else "—"]
            r.append("| " + " | ".join(cells) + " |")
            first = False
    return "\n".join(r)


def t_horizon():
    hs = [1, 5, 20, 50, "all"]
    r = ["| benchmark | history | window (median) | precision | recall | binary IoU | SSC mIoU | coverage miss |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        first = True
        for h in hs:
            c = cfg(d, f"S1_G-A_h{h}")
            if not c:
                continue
            b = "**" if h == "all" else ""
            r.append(f"| {LABEL[ds] if first else ''} | {b}{h}{b} "
                     f"| {c['median_window']:.0f} | {c['binary_precision']:.4f} "
                     f"| {c['binary_recall']:.4f} | {b}{c['binary_iou']:.4f}{b} "
                     f"| {b}{c['ssc_miou']:.4f}{b} | {c['coverage_miss']:.4f} |")
            first = False
        for ref in ("B-R", "B-D"):
            v = d["reference"][ref]
            r.append(f"|  | _five-frame {ref}_ | 5 | {v['binary_precision']:.4f} "
                     f"| {v['binary_recall']:.4f} | {v['binary_iou']:.4f} "
                     f"| {v['ssc_miou']:.4f} | {v['coverage_miss']:.4f} |")
    return "\n".join(r)


def t_s2():
    r = ["| benchmark | confidence threshold | precision | recall | binary IoU | SSC mIoU | occupied voxels |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        first = True
        base = cfg(d, "S1_G-A_hall")
        if base:
            r.append(f"| {LABEL[ds]} | 1.5 (S1, frozen) | {base['binary_precision']:.4f} "
                     f"| {base['binary_recall']:.4f} | {base['binary_iou']:.4f} "
                     f"| {base['ssc_miou']:.4f} "
                     f"| {base['occ_source']['lingbot']:,} |")
            first = False
        for c_ in (1.0, 0.5, 0.0):
            c = cfg(d, f"S2_G-A_hall_c{c_:g}")
            if not c:
                continue
            r.append(f"| {LABEL[ds] if first else ''} | {c_:g} "
                     f"| {c['binary_precision']:.4f} | {c['binary_recall']:.4f} "
                     f"| {c['binary_iou']:.4f} | {c['ssc_miou']:.4f} "
                     f"| {c['occ_source']['lingbot']:,} |")
            first = False
    return "\n".join(r)


def t_s34():
    r = ["| benchmark | variant | precision | recall | binary IoU | SSC mIoU | MoGe-only voxels | provisional | TP-acc (LingBot) | TP-acc (MoGe-only) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        first = True
        for var, tag in (("S1", "S1_G-A_hall"), ("S3", "S3_G-A_hall"),
                         ("S4", "S4_G-A_hall")):
            c = cfg(d, tag)
            if not c:
                continue
            r.append(f"| {LABEL[ds] if first else ''} | {var} "
                     f"| {c['binary_precision']:.4f} | {c['binary_recall']:.4f} "
                     f"| {c['binary_iou']:.4f} | {c['ssc_miou']:.4f} "
                     f"| {c['occ_source']['moge_only']:,} "
                     f"| {c['volumes']['provisional']:,} "
                     f"| {c['tp_reconstruction_support']:.4f} "
                     f"| {f(c['tp_moge_only']) if c['n_moge_only_voxels'] else '—'} |")
            first = False
    return "\n".join(r)


def t_recovery():
    r = ["| benchmark | B-D coverage misses | already in the causal past | recovered from a later viewpoint | in-frustum, never reconstructed | outside all frusta |",
         "|---|---:|---:|---:|---:|---:|"]
    for ds in ORDER:
        d = j(f"recoverability_{ds}.json")
        if d is None:
            continue
        fr = d["fractions"]
        r.append(f"| {LABEL[ds]} | {d['n_b_d_coverage_misses']:,} "
                 f"| {fr['recovered_from_causal_past']:.4f} "
                 f"| {fr['recovered_from_a_later_viewpoint']:.4f} "
                 f"| {fr['in_frustum_but_never_reconstructed']:.4f} "
                 f"| {fr['outside_all_frusta']:.4f} |")
    r.append("")
    r.append("| benchmark | +1 frame | +5 | +10 | +20 | +50 | any later frame |")
    r.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in ORDER:
        d = j(f"recoverability_{ds}.json")
        if d is None:
            continue
        L = d["recovery_latency_fraction"]
        r.append(f"| {LABEL[ds]} | " + " | ".join(
            f"{L[k]:.4f}" for k in ("within_1_frames", "within_5_frames",
                                    "within_10_frames", "within_20_frames",
                                    "within_50_frames", "any_later_frame")) + " |")
    return "\n".join(r)


def t_semantic():
    r = ["| benchmark | variant | TP-conditioned accuracy | balanced recall | on LingBot voxels | on MoGe-only voxels | naming error | correct |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        first = True
        v = d["reference"]["B-D"]
        r.append(f"| {LABEL[ds]} | _five-frame B-D_ | {v['tp_accuracy']:.4f} "
                 f"| {v['tp_balanced_recall']:.4f} | — | — "
                 f"| {v['naming_error']:.4f} | — |")
        for var, tag in (("S1", "S1_G-A_hall"), ("S2", "S2_G-A_hall_c0.5"),
                         ("S3", "S3_G-A_hall"), ("S4", "S4_G-A_hall")):
            c = cfg(d, tag)
            if not c:
                continue
            r.append(f"|  | {var} | {c['tp_accuracy']:.4f} "
                     f"| {c['tp_balanced_recall']:.4f} "
                     f"| {c['tp_reconstruction_support']:.4f} "
                     f"| {f(c['tp_moge_only']) if c['n_moge_only_voxels'] else '—'} "
                     f"| {c['naming_error']:.4f} | {c['correct']:.4f} |")
        first = False
    return "\n".join(r)


def t_precision():
    r = ["| benchmark | configuration | occupied | free | unknown | precision | false positives per anchor | added over S1 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds, d in have():
        s1 = cfg(d, "S1_G-A_hall")
        first = True
        for var, tag in (("S1", "S1_G-A_hall"), ("S2 (c=0.5)", "S2_G-A_hall_c0.5"),
                         ("S3", "S3_G-A_hall"), ("S4", "S4_G-A_hall")):
            c = cfg(d, tag)
            if not c:
                continue
            v = c["volumes"]
            fpn = None
            cmp = d["comparisons"].get(f"{tag.split('_')[0]}|G-A|all vs S1|G-A|all")
            r.append(f"| {LABEL[ds] if first else ''} | {var} | {v['occupied']:,} "
                     f"| {v['free']:,} | {v['unknown']:,} "
                     f"| {c['binary_precision']:.4f} | "
                     + (f"{(1 - c['binary_precision']) * v['occupied']:,.0f} | ")
                     + (f"{c['binary_precision'] - s1['binary_precision']:+.4f} |"
                        if s1 else "— |"))
            first = False
    return "\n".join(r)


def t_bootstrap():
    r = ["| benchmark | comparison | unit (n) | Δ binary IoU | 95% CI | excludes 0 | Δ SSC mIoU | 95% CI | excludes 0 |",
         "|---|---|---|---:|---|---|---:|---|---|"]
    for ds, d in have():
        first = True
        for key, c in d["comparisons"].items():
            if not c or "binary_iou" not in c:
                continue
            bi, mi = c["binary_iou"], c["ssc_miou"]
            r.append(f"| {LABEL[ds] if first else ''} | {key} "
                     f"| {c['unit']} ({c['n_units']}) | {bi['difference']:+.4f} "
                     f"| [{bi['ci'][0]:+.4f}, {bi['ci'][1]:+.4f}] "
                     f"| {'yes' if bi['excludes_zero'] else 'no'} "
                     f"| {mi['difference']:+.4f} "
                     f"| [{mi['ci'][0]:+.4f}, {mi['ci'][1]:+.4f}] "
                     f"| {'yes' if mi['excludes_zero'] else 'no'} |")
            first = False
    return "\n".join(r)


def t_runtime():
    r = ["| stage | SemanticKITTI | Occ3D-nuScenes | SSCBench-KITTI-360 | total |",
         "|---|---:|---:|---:|---:|"]
    def row(name, get, fmt="{:.1f} min"):
        vals, tot = [], 0.0
        for ds in ORDER:
            v = get(ds)
            vals.append(fmt.format(v) if v is not None else "—")
            tot += v or 0.0
        return f"| {name} | " + " | ".join(vals) + f" | {fmt.format(tot)} |"
    r.append(row("LingBot native streaming",
                 lambda ds: (j(f"stream_{ds}.json") or {}).get("seconds", 0) / 60))
    r.append(row("MoGe G51-B cache",
                 lambda ds: (j(f"moge_b_cache_{ds}.json") or {}).get("seconds", 0) / 60))
    r.append(row("metric-gauge candidates",
                 lambda ds: (j(f"scale_candidates_{ds}.json") or {}).get("seconds", 0)
                 / 60))
    def matrix(ds):
        return sum(json.load(open(p))["seconds"]
                   for p in glob.glob(os.path.join(ART, f"eval_{ds}_*.json"))) / 60
    r.append(row("evaluation matrix (12 configs)", matrix))
    r.append(row("map thickness",
                 lambda ds: (j(f"thickness_{ds}.json") or {}).get("seconds", 0) / 60))
    r.append(row("temporal recoverability",
                 lambda ds: (j(f"recoverability_{ds}.json") or {}).get("seconds", 0)
                 / 60))
    r.append("")
    r.append("| quantity | value |")
    r.append("|---|---|")
    fps = []
    for ds in ORDER:
        m = j(f"stream_{ds}.json")
        if m:
            fps += [s["fps"] for s in m["segments"] if "fps" in s]
    r.append(f"| LingBot geometry throughput | {np.median(fps) if fps else 0:.1f} FPS "
             f"(median over segments) |")
    lat, mem, peak = [], [], []
    for _ds, d in have():
        for c in d["configs"].values():
            lat.append(c["median_seconds_per_anchor"] * 1000)
            mem.append(c["map_bytes_mean"] / 2 ** 20)
            peak.append(c["peak_gpu_gib"])
    if lat:
        r.append(f"| map-update latency per timestamp | median "
                 f"{np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms |")
        r.append(f"| evidence volume in memory | {np.median(mem):.0f} MiB "
                 f"(dense over the evaluation grid) |")
        r.append(f"| peak VRAM, map stage | {max(peak):.2f} GiB |")
    r.append("| semantic update | Trident is **not** run online here; its per-frame cache "
             "is read. Fusing a cached vector costs microseconds, running the teacher does "
             "not. |")
    sz = 0
    for root in ("/media/SSD1/MINH_DATASETS/lingbot_gate7b",):
        for dp, _dn, fn in os.walk(root):
            sz += sum(os.path.getsize(os.path.join(dp, x)) for x in fn)
    art = sum(os.path.getsize(os.path.join(ART, x)) for x in os.listdir(ART)
              if os.path.isfile(os.path.join(ART, x)))
    r.append(f"| new bulk storage | {sz / 2**30:.1f} GB on `/media/SSD1` |")
    r.append(f"| in-repo artifacts | {art / 2**20:.1f} MB |")
    return "\n".join(r)


def t_cross():
    return t_bootstrap()


def t_files():
    rows = ["| path | role |", "|---|---|"]
    for p, role in [
        ("gates/gate7b/config.py", "every predeclared choice; the precommit is generated from it"),
        ("gates/gate7b/streams.py", "chronological, deduplicated streams and genuine boundaries"),
        ("gates/gate7b/replay.py", "native direct-mode replay and the RoPE capacity rule"),
        ("gates/gate7b/depth.py", "depth conventions, the frozen gate, the per-frame MoGe index"),
        ("gates/gate7b/scale.py", "`ScaleState`: G-A, G-B and G-C"),
        ("gates/gate7b/voxmap.py", "occupied / free / unknown / provisional evidence volume"),
        ("gates/gate7b/rays.py", "ray casting; free before the surface, nothing behind it"),
        ("gates/gate7b/evidence.py", "the S1-S4 acceptance rules and the reliability weight"),
        ("gates/gate7b/fuse.py", "causal window selection and rematerialisation"),
        ("tools/gate7b/stage0.py", "hashes everything Gate 7B must not disturb"),
        ("tools/gate7b/precommit.py", "writes and pins the configuration"),
        ("tools/gate7b/stream_lingbot.py", "the native streaming pass"),
        ("tools/gate7b/cache_moge_b.py", "the calibrated-FOV MoGe cache"),
        ("tools/gate7b/scale_candidates.py", "per-frame gauge candidates"),
        ("tools/gate7b/run_stream_eval.py", "one (variant, gauge, horizon) evaluation"),
        ("tools/gate7b/thickness.py", "map thickness and duplicate surfaces"),
        ("tools/gate7b/recoverability.py", "the forward-looking diagnostic (isolated)"),
        ("tools/gate7b/aggregate.py", "pooled results, paired bootstrap, decision rules"),
        ("tools/gate7b/figures.py", "the seven figures"),
        ("tools/gate7b/report_tables.py", "this report's tables"),
        ("configs/gate7b/streaming_replay_precommit.yaml", "the pinned configuration"),
        ("tests/gate7b/test_gate7b.py", "the Gate-7B test suite"),
        ("artifacts/gate7b/", "count blocks, summaries, figures, logs"),
        ("reports/gate7b/streaming_metric_semantic_replay.md", "this report"),
    ]:
        rows.append(f"| `{p}` | {role} |")
    return "\n".join(rows)


TABLES = {"TABLE_HEADLINE": t_headline, "TABLE_INTERFACE": t_interface,
          "TABLE_STREAM": t_stream, "TABLE_SCALE": t_scale,
          "TABLE_THICKNESS": t_thickness, "TABLE_HORIZON": t_horizon,
          "TABLE_S2": t_s2, "TABLE_S34": t_s34, "TABLE_RECOVERY": t_recovery,
          "TABLE_SEMANTIC": t_semantic, "TABLE_PRECISION": t_precision,
          "TABLE_CROSS": t_cross, "TABLE_RUNTIME": t_runtime, "TABLE_FILES": t_files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    text = open(TEMPLATE).read()
    import tools.gate7b.report_prose as P
    allt = dict(TABLES)
    allt.update(P.PROSE)
    for k, fn in allt.items():
        tok = "{{" + k + "}}"
        if tok in text:
            try:
                text = text.replace(tok, fn())
            except Exception as exc:                        # noqa: BLE001
                text = text.replace(tok, f"_generator `{k}` failed: {exc}_")
    if a.write:
        open(OUT, "w").write(text)
        print(f"wrote {OUT} ({len(text)} chars)")
    else:
        print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
