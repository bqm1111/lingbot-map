#!/usr/bin/env python
"""Gate 7B — pooled results, paired bootstrap and the predeclared decision rules.

Every count block is the Gate-6 one, so a Gate-7B number and a Gate-6 number mean exactly
the same thing. The bootstrap reuses Gate 6's resampling units and its seed, and every
comparison in ``gate7b.config.PAIRED_COMPARISONS`` was declared before a result existed.

    python tools/gate7b/aggregate.py
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate7a import stats as S7                                                   # noqa: E402
from gates.gate7b import config as C                                                   # noqa: E402
from tools.gate6.analyze import unit_ids                                         # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")
G6ART = os.path.join(REPO_ROOT, "artifacts", "gate6")


def load_counts(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def align(a, b):
    """Restrict two count blocks to their common clips, in one order."""
    ka = {str(c): i for i, c in enumerate(a["clip_id"])}
    kb = {str(c): i for i, c in enumerate(b["clip_id"])}
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka[c] for c in common])
    ib = np.array([kb[c] for c in common])
    return common, ia, ib


def per_unit(block, idx, units, keys):
    uids = sorted(set(units))
    ui = {u: i for i, u in enumerate(uids)}
    out = {k: np.zeros((len(uids),) + block[k].shape[1:], np.float64) for k in keys}
    for j, i in enumerate(idx):
        u = ui[units[j]]
        for k in keys:
            out[k][u] += block[k][i]
    return out, uids


def paired(ds, block_a, block_b, n_boot, seed):
    """Paired bootstrap of (b - a) for pooled binary IoU and SSC mIoU."""
    common, ia, ib = align(block_a, block_b)
    if not common:
        return None
    groups = [str(g) for g in np.asarray(block_a["group"])[ia]]
    units, unit_name = unit_ids(ds, common, groups)
    ua, uids = per_unit(block_a, ia, units, ("binary", "tp", "fp", "fn"))
    ub, _ = per_unit(block_b, ib, units, ("binary", "tp", "fp", "fn"))
    mult = S7.multiplicities(len(uids), n_boot, seed)

    def metrics(u):
        b = S7.resample(u["binary"][:, :3], mult)
        pc = np.stack([S7.resample(u["tp"], mult), S7.resample(u["fp"], mult),
                       S7.resample(u["fn"], mult)], axis=-1)
        pt = np.stack([u["tp"].sum(0), u["fp"].sum(0), u["fn"].sum(0)], axis=-1)
        return (S7.binary_iou(b), S7.miou(pc),
                float(S7.binary_iou(u["binary"][:, :3].sum(0))),
                float(S7.miou(pt)))

    ba, ma, pia, pma = metrics(ua)
    bb, mb, pib, pmb = metrics(ub)
    return {"n_clips": len(common), "unit": unit_name, "n_units": len(uids),
            "binary_iou": S7.paired(pia, pib, ba, bb),
            "ssc_miou": S7.paired(pma, pmb, ma, mb)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP["n_boot"])
    a = ap.parse_args()
    res = {"bootstrap": {"n_boot": a.n_boot, "seed": C.BOOTSTRAP["seed"],
                         "units": C.BOOTSTRAP["units"],
                         "independence_caveat": C.BOOTSTRAP["independence_caveat"]},
           "datasets": {}}
    for ds in C.DATASETS:
        d = {"configs": {}, "reference": {}, "comparisons": {}}
        g6 = json.load(open(os.path.join(G6ART, f"summary_{ds}.json")))
        for cond in ("B-R", "B-D"):
            c = g6["conditions"][cond]
            d["reference"][cond] = {
                "binary_iou": c["binary_iou_pooled"], "ssc_miou": c["ssc_miou"],
                "binary_precision": c["binary_precision"],
                "binary_recall": c["binary_recall"],
                "coverage_miss": c["decomposition"]["coverage_miss_fraction"],
                "naming_error": c["decomposition"]["naming_error_fraction"],
                "tp_accuracy": c["tp_conditioned"]["top1_accuracy"],
                "tp_balanced_recall": c["tp_conditioned"]["balanced_recall"],
                "n_clips": g6["n_clips"]}
        for p in sorted(glob.glob(os.path.join(ART, f"eval_{ds}_*.json"))):
            e = json.load(open(p))
            s = e["summary"]
            if not s:
                continue
            pc = e["per_clip"]
            d["configs"][e["tag"]] = {
                "variant": e["variant"], "scale_policy": e["scale_policy"],
                "horizon": e["horizon"], "s2_conf": e.get("s2_conf"),
                "n_anchors": e["n_anchors"],
                "binary_iou": s["binary_iou"], "binary_precision": s["binary_precision"],
                "binary_recall": s["binary_recall"], "ssc_miou": s["ssc_miou"],
                "binary_iou_mean_per_clip": s.get("binary_iou_mean_per_clip"),
                "coverage_miss": s["decomposition"]["coverage_miss_fraction"],
                "naming_error": s["decomposition"]["naming_error_fraction"],
                "correct": s["decomposition"]["correct_fraction"],
                "tp_accuracy": s["tp_conditioned"]["top1_accuracy"],
                "tp_balanced_recall": s["tp_conditioned"]["balanced_recall"],
                "tp_reconstruction_support":
                    s["tp_conditioned_reconstruction_support"]["top1_accuracy"],
                "tp_moge_only": s["tp_conditioned_dilation_only"]["top1_accuracy"],
                "n_moge_only_voxels":
                    s["tp_conditioned_dilation_only"]["n"],
                "seconds": e["seconds"], "peak_gpu_gib": e["peak_gpu_gib"],
                "median_seconds_per_anchor": e["fuse"]["median_seconds_per_anchor"],
                "median_window": e["fuse"]["median_window"],
                "volumes": {
                    "occupied": int(np.mean([r["map_occupied"] for r in pc])),
                    "free": int(np.mean([r["map_free"] for r in pc])),
                    "unknown": int(np.mean([r["map_unknown"] for r in pc])),
                    "provisional": int(np.mean([r["map_provisional"] for r in pc]))},
                "occ_source": {
                    "lingbot": int(sum(r["n_occ_lingbot"] for r in pc)),
                    "moge_only": int(sum(r["n_occ_moge_only"] for r in pc)),
                    "beyond_five_frames": int(sum(r["n_occ_beyond_five_frames"]
                                                  for r in pc)),
                    "unlabelled": int(sum(r.get("n_occupied_without_semantics", 0)
                                          for r in pc))},
                "map_bytes_mean": float(np.mean([r["map_bytes"] for r in pc])),
                "scale": {k: {kk: vv for kk, vv in v.items() if kk != "series"}
                          for k, v in list(e["scale_diagnostics"].items())[:3]},
            }
        # ---- paired comparisons
        g6_counts = os.path.join(G6ART, f"counts_{ds}_B-D.npz")
        for b_tag, a_tag, what in C.PAIRED_COMPARISONS:
            bp = os.path.join(ART, f"counts_{ds}_{_tagof(b_tag)}.npz")
            ap_ = (g6_counts if a_tag == "S0"
                   else os.path.join(ART, f"counts_{ds}_{_tagof(a_tag)}.npz"))
            if not (os.path.exists(bp) and os.path.exists(ap_)):
                continue
            A, B = load_counts(ap_), load_counts(bp)
            d["comparisons"][f"{b_tag} vs {a_tag}"] = dict(
                paired(ds, A, B, a.n_boot, C.BOOTSTRAP["seed"]) or {}, what=what)
        res["datasets"][ds] = d
    res["decision"] = decide(res)
    write_json(os.path.join(ART, "summary.json"), res)
    print(json.dumps(res["decision"], indent=1))
    return 0


def _tagof(spec: str) -> str:
    if spec == "S0":
        return "S0"
    var, pol, hz = spec.split("|")
    t = f"{var}_{pol}_h{hz}"
    if var == "S2":
        t += "_c0.5"
    return t


def decide(res: dict) -> dict:
    """The predeclared branches, evaluated mechanically on the measured numbers."""
    ds_list = list(res["datasets"])
    out = {"rules": {}, "selected": []}

    def cmp_flags(key):
        wins, regress = [], []
        for ds in ds_list:
            c = res["datasets"][ds]["comparisons"].get(key)
            if not c:
                continue
            bi, mi = c["binary_iou"], c["ssc_miou"]
            up = (bi["difference"] > 0 and bi["excludes_zero"]
                  and mi["difference"] > 0 and mi["excludes_zero"])
            down = ((bi["difference"] < 0 and bi["excludes_zero"])
                    or (mi["difference"] < 0 and mi["excludes_zero"]))
            wins.append(ds if up else None)
            regress.append(ds if down else None)
        return [x for x in wins if x], [x for x in regress if x]

    w, r = cmp_flags("S1|G-A|all vs S0")
    out["rules"]["TEMPORAL_ACCUMULATION_WORKS"] = {
        "improves_on": w, "regresses_on": r,
        "passes": len(w) >= 2 and len(r) == 0}
    w4, r4 = cmp_flags("S4|G-A|all vs S1|G-A|all")
    out["rules"]["COMPLEMENTARY_DEPTH_WORKS"] = {
        "improves_on": w4, "regresses_on": r4,
        "passes": len(w4) >= 2 and len(r4) == 0}
    w2, r2 = cmp_flags("S2|G-A|all vs S1|G-A|all")
    out["rules"]["RELAXED_GATE_IS_SUFFICIENT"] = {
        "improves_on": w2, "regresses_on": r2,
        "passes": len(w2) >= 2 and len(r2) == 0 and not out["rules"][
            "COMPLEMENTARY_DEPTH_WORKS"]["passes"]}
    wb, rb = cmp_flags("S1|G-B|all vs S1|G-A|all")
    out["rules"]["SCALE_POLICY_RUNNING_MEDIAN"] = {
        "improves_on": wb, "regresses_on": rb, "passes": len(wb) >= 2 and len(rb) == 0}
    out["rules"]["SCALE_POLICY_FIXED_ANCHOR"] = {
        "passes": not out["rules"]["SCALE_POLICY_RUNNING_MEDIAN"]["passes"]}
    out["rules"]["EVIDENCE_REMAINS_INSUFFICIENT"] = {
        "passes": not (out["rules"]["TEMPORAL_ACCUMULATION_WORKS"]["passes"]
                       or out["rules"]["COMPLEMENTARY_DEPTH_WORKS"]["passes"])}
    for k, v in out["rules"].items():
        if v.get("passes"):
            out["selected"].append(k)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
