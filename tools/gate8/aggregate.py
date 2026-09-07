#!/usr/bin/env python
"""Gate 8 results: the three-way comparison per benchmark with paired, scene-aware
bootstrap intervals (Gate 6 units, seed 0), raw and dilated reported separately.

    python tools/gate8/aggregate.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate7a import stats as S7                                                   # noqa: E402
from tools.gate6.analyze import unit_ids                                         # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8")
G6 = os.path.join(REPO_ROOT, "artifacts", "gate6")
DS = ("semantickitti", "occ3d", "kitti360")
N_BOOT, SEED = 10000, 0


def load(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}


_UNITS = {}


def gate6_units(ds):
    """clip_id -> bootstrap unit, taken from Gate 6's own count block.

    The Gate-8 evaluator records the *segment* name as the group (one drive on the KITTI
    family), which is not Gate 6's resampling unit. Reading the units from Gate 6
    guarantees Gate 8 resamples over exactly the same blocks/scenes as every prior gate.
    """
    if ds not in _UNITS:
        z = np.load(os.path.join(G6, f"counts_{ds}_B-D.npz"), allow_pickle=False)
        cid = [str(c) for c in z["clip_id"]]
        u, name = unit_ids(ds, cid, [str(g) for g in z["group"]])
        _UNITS[ds] = (dict(zip(cid, u)), name)
    return _UNITS[ds]


def paired(ds, A, B):
    ka = {str(c): i for i, c in enumerate(A["clip_id"])}; kb = {str(c): i for i, c in enumerate(B["clip_id"])}
    umap, uname = gate6_units(ds)
    common = sorted(set(ka) & set(kb) & set(umap))
    if not common:
        return None
    ia = np.array([ka[c] for c in common]); ib = np.array([kb[c] for c in common])
    units = [umap[c] for c in common]
    uids = sorted(set(units)); ui = {u: i for i, u in enumerate(uids)}
    def per_unit(blk, idx):
        out = {k: np.zeros((len(uids),) + blk[k].shape[1:], np.float64) for k in ("binary", "tp", "fp", "fn")}
        for j, i in enumerate(idx):
            for k in out:
                out[k][ui[units[j]]] += blk[k][i]
        return out
    ua, ub = per_unit(A, ia), per_unit(B, ib)
    mult = S7.multiplicities(len(uids), N_BOOT, SEED)
    def m(u):
        b = S7.resample(u["binary"][:, :3], mult)
        pc = np.stack([S7.resample(u[k], mult) for k in ("tp", "fp", "fn")], -1)
        pt = np.stack([u[k].sum(0) for k in ("tp", "fp", "fn")], -1)
        return S7.binary_iou(b), S7.miou(pc), float(S7.binary_iou(u["binary"][:, :3].sum(0))), float(S7.miou(pt))
    ba, ma, pa, pma = m(ua); bb, mb, pb, pmb = m(ub)
    return {"n_clips": len(common), "unit": uname, "n_units": len(uids),
            "binary_iou": S7.paired(pa, pb, ba, bb), "ssc_miou": S7.paired(pma, pmb, ma, mb)}


def summary_of(ds, tag, variant):
    p = os.path.join(ART, f"eval_{ds}_{tag}.json")
    if not os.path.exists(p):
        return None
    e = json.load(open(p)); s = e["variants"].get(variant)
    if not s:
        return None
    d = s["decomposition"]; t = s["tp_conditioned"]
    return {"n_anchors": e["n_anchors"], "binary_iou": s["binary_iou"],
            "binary_precision": s["binary_precision"], "binary_recall": s["binary_recall"],
            "tp": s["per_class"] and int(sum(v["tp"] for v in s["per_class"].values())),
            "btp": None, "ssc_miou": s["ssc_miou"], "tp_accuracy": t["top1_accuracy"],
            "tp_balanced_recall": t["balanced_recall"],
            "coverage_miss": d["coverage_miss_fraction"], "naming_error": d["naming_error_fraction"],
            "per_class_iou": {k: v["iou"] for k, v in s["per_class"].items()},
            "latency_ms": e.get("latency_ms"), "peak_gpu_gib": e.get("peak_gpu_gib"),
            "map_bytes_max": e.get("map_bytes_max")}


def main() -> int:
    res = {"bootstrap": {"n_boot": N_BOOT, "seed": SEED}, "datasets": {}}
    for ds in DS:
        d = {"methods": {}, "comparisons": {}}
        g6 = json.load(open(os.path.join(G6, f"summary_{ds}.json")))
        for cond, name in (("B-R", "frozen_5frame_raw"), ("B-D", "frozen_5frame_dil")):
            c = g6["conditions"][cond]
            d["methods"][name] = {"binary_iou": c["binary_iou_pooled"], "binary_precision": c["binary_precision"],
                                  "binary_recall": c["binary_recall"], "ssc_miou": c["ssc_miou"],
                                  "tp_accuracy": c["tp_conditioned"]["top1_accuracy"],
                                  "tp_balanced_recall": c["tp_conditioned"]["balanced_recall"],
                                  "coverage_miss": c["decomposition"]["coverage_miss_fraction"],
                                  "naming_error": c["decomposition"]["naming_error_fraction"],
                                  "per_class_iou": {k: v["iou"] for k, v in c["per_class"].items()},
                                  "n_anchors": g6["n_clips"]}
        for tag, name in (("mapper_native", "mapper"), ("mapper_union", "mapper_union"),
                          ("complete_union", "mapper_plus_completion")):
            for var in ("raw", "dil"):
                s = summary_of(ds, tag, var)
                if s:
                    d["methods"][f"{name}_{var}"] = s
        # binary counts for the report: TP/FP/FN
        for name, tag, var in (("mapper_raw", "mapper_native", "raw"), ("mapper_dil", "mapper_native", "dil"),
                               ("mapper_plus_completion_raw", "complete_union", "raw"),
                               ("mapper_plus_completion_dil", "complete_union", "dil")):
            p = os.path.join(ART, f"counts_{ds}_{tag}_{var}.npz")
            if os.path.exists(p) and name in d["methods"]:
                b = load(p)["binary"].sum(0)
                d["methods"][name].update({"btp": int(b[0]), "bfp": int(b[1]), "bfn": int(b[2])})
        # Trivial sanity baseline: mark EVERY valid voxel occupied. A completion model that
        # inflates its predictions can raise binary IoU without learning anything; if it
        # does not clear this line, the gain is over-prediction, not structure.
        bb = load(os.path.join(G6, f"counts_{ds}_B-D.npz"))["binary"].sum(0)
        nval, ngt = int(bb[3]), int(bb[4])
        d["methods"]["trivial_all_occupied"] = {
            "binary_iou": ngt / nval, "binary_precision": ngt / nval, "binary_recall": 1.0,
            "ssc_miou": None, "tp_accuracy": None, "tp_balanced_recall": None,
            "coverage_miss": 0.0, "naming_error": None, "per_class_iou": {},
            "btp": ngt, "bfp": nval - ngt, "bfn": 0, "n_anchors": None,
            "note": "no model: every valid voxel declared occupied"}
        for cond in ("B-R", "B-D"):
            b = load(os.path.join(G6, f"counts_{ds}_{cond}.npz"))["binary"].sum(0)
            d["methods"]["frozen_5frame_raw" if cond == "B-R" else "frozen_5frame_dil"].update(
                {"btp": int(b[0]), "bfp": int(b[1]), "bfn": int(b[2])})
        pairs = [("mapper_raw vs frozen_5frame_raw", os.path.join(G6, f"counts_{ds}_B-R.npz"),
                  os.path.join(ART, f"counts_{ds}_mapper_native_raw.npz")),
                 ("mapper_dil vs frozen_5frame_dil", os.path.join(G6, f"counts_{ds}_B-D.npz"),
                  os.path.join(ART, f"counts_{ds}_mapper_native_dil.npz")),
                 ("completion_raw vs mapper_raw", os.path.join(ART, f"counts_{ds}_mapper_union_raw.npz"),
                  os.path.join(ART, f"counts_{ds}_complete_union_raw.npz")),
                 ("completion_dil vs mapper_dil", os.path.join(ART, f"counts_{ds}_mapper_union_dil.npz"),
                  os.path.join(ART, f"counts_{ds}_complete_union_dil.npz")),
                 ("completion_raw vs frozen_5frame_raw", os.path.join(G6, f"counts_{ds}_B-R.npz"),
                  os.path.join(ART, f"counts_{ds}_complete_union_raw.npz")),
                 ("completion_dil vs frozen_5frame_dil", os.path.join(G6, f"counts_{ds}_B-D.npz"),
                  os.path.join(ART, f"counts_{ds}_complete_union_dil.npz"))]
        for key, pa, pb in pairs:
            if os.path.exists(pa) and os.path.exists(pb):
                d["comparisons"][key] = paired(ds, load(pa), load(pb))
        res["datasets"][ds] = d
    for name in ("runtime_kitti360", "train_completion"):
        p = os.path.join(ART, f"{name}.json")
        if os.path.exists(p):
            res[name] = json.load(open(p))
    write_json(os.path.join(ART, "gate8_results.json"), res)
    for ds, d in res["datasets"].items():
        print(f"== {ds}")
        for k, m in d["methods"].items():
            mi = "  n/a " if m["ssc_miou"] is None else f"{m['ssc_miou']:.4f}"
            ta = "  n/a " if m["tp_accuracy"] is None else f"{m['tp_accuracy']:.4f}"
            flag = ""
            triv = d["methods"].get("trivial_all_occupied", {}).get("binary_iou")
            if triv and "completion" in k and m["binary_iou"] < triv:
                flag = "   <-- BELOW the trivial all-occupied baseline"
            print(f"   {k:30s} binIoU {m['binary_iou']:.4f} P {m['binary_precision']:.4f} "
                  f"R {m['binary_recall']:.4f} mIoU {mi} TPacc {ta}{flag}")
        for k, c in d["comparisons"].items():
            if c: print(f"   Δ {k}: IoU {c['binary_iou']['difference']:+.4f} {c['binary_iou']['ci']} mIoU {c['ssc_miou']['difference']:+.4f} {c['ssc_miou']['ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
