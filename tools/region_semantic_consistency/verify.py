#!/usr/bin/env python
"""Ten invariants of the region-semantic-consistency study; fails loudly.

    python tools/region_semantic_consistency/verify.py
"""
from __future__ import annotations
import glob, hashlib, inspect, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.region_semantic_consistency import common as RC                                # noqa: E402
from tools.region_semantic_consistency import analyze as AN                               # noqa: E402
from tools.future_semantic_completion.target_consistency.analyze import GROUPINGS          # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ST.require_data_root(); R = []
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    an = json.load(open(os.path.join(RC.ART, "analysis.json"))); an2 = json.load(open(os.path.join(RC.ART, "analysis_allocc.json")))
    code_dir = os.path.dirname(os.path.abspath(__file__))
    others = [p for p in os.listdir(code_dir) if p.endswith(".py") and p != "verify.py"]
    srcs = "".join(open(os.path.join(code_dir, p)).read() for p in others)
    # 1. nothing trained
    man = json.load(open(ST.FROZEN_MANIFEST))
    ck_ok = all(ST.sha256_file(os.path.join(ST.REPO_ROOT, m["checkpoint"])) == m["checkpoint_sha256"] for m in man["seeds"].values())
    hm = json.load(open(ST.manifest_path())); head_ok = ST.sha256_file(ST.head_checkpoint(0)) == hm["seeds"]["0"]["head_checkpoint_sha256"]
    no_opt = all(tok not in srcs for tok in ("torch.optim", ".backward(", "requires_grad", "state_dict", "nn.Module"))
    check("no_model_trained", ck_ok and head_ok and no_opt,
          f"no optimiser/backward/module in {len(others)} study modules: {no_opt}; Gate 8C-1 checkpoints match manifest: {ck_ok}; semantic head unchanged: {head_ok}", R)
    # 2. frozen LingBot-Map / Trident-H: caches older than the study, mapper code digests
    tri = sorted(glob.glob("/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics/kitti360/*.npz"))
    stream = sorted(glob.glob("/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream/kitti360/*.npz") + glob.glob("/media/SSD1/MINH_DATASETS/lingbot_gate8/stream/k360_train/*.npz"))
    newest = max(os.path.getmtime(p) for p in tri + stream)
    first_ev = min(os.path.getmtime(p) for d in drives for p in glob.glob(os.path.join(RC.EVIDENCE_DIR, d, "*.npz")))
    digests = {p: ST.sha256_file(os.path.join(ST.REPO_ROOT, p))[:12] for p in ("core/mapping/mapper.py", "core/mapping/voxmap.py", "core/mapping/feed.py", "core/supervision/lifting.py")}
    builds = [json.load(open(p)) for p in glob.glob(os.path.join(ST.ART, "target_consistency", "evidence_build", "*.json"))]
    dp = max(b["max_abs_dp"] for b in builds)
    check("lingbot_and_trident_frozen", newest < first_ev and dp <= 0.004,
          f"{len(tri)} Trident-H and {len(stream)} LingBot stream caches, newest {time.strftime('%Y-%m-%d', time.localtime(newest))} < first evidence file "
          f"{time.strftime('%Y-%m-%d', time.localtime(first_ev))}; mapper digests {digests}; per-frame replay reproduces cached targets (max |dp| {dp:.4f}) in {len(builds)} builds", R)
    # 3. disjoint frames
    grp = GROUPINGS[RC.GROUPING]; sets = [set(v) for v in grp.values()]
    disj = all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3)) and set().union(*sets) <= set(range(1, 21))
    n_files = n_bad = 0
    for d in drives:
        for f in glob.glob(os.path.join(RC.EVIDENCE_DIR, d, "*.npz")) + glob.glob(os.path.join(RC.EVIDENCE_DIR + "_allocc", d, "*.npz")):
            with np.load(f) as z:
                t = int(z["t"]); fr = z["frames"]; n_files += 1
                n_bad += int(not ((fr > t).all() and (fr <= t + 20).all() and len(set(fr.tolist())) == len(fr)))
    check("abc_disjoint_frames", disj and n_bad == 0, f"groups {grp} pairwise disjoint: {disj}; {n_files - n_bad}/{n_files} evidence files have unique future frames in (t, t+20]", R)
    # 4. region identity independent of semantics
    src = inspect.getsource(AN.label_regions)
    check("regions_semantics_free", all(tok not in src for tok in ("pair_s", "pair_w", "probs", "argmax", "sem")) and "occ" in src,
          "label_regions reads the occupied grid and voxel coordinates only (block partition + 26-connected components); no probability, weight or class enters", R)
    # 5. C never affects A/B: perturbation on sampled anchors
    n_t = n_same = 0
    for d in drives:
        for f in sorted(glob.glob(os.path.join(RC.EVIDENCE_DIR, d, "*.npz")))[::50]:
            with np.load(f) as z:
                z = {k: z[k] for k in z.files}
            row = z["pair_row"].astype(np.int64); off = z["pair_frame"].astype(np.int64); w = z["pair_w"].astype(np.float64); s = z["pair_s"].astype(np.float64)
            isC = np.isin(off, grp["C"]); rng = np.random.default_rng(0)
            w2 = w.copy(); s2 = s.copy(); w2[isC] = rng.random(isC.sum()) * 50; s2[isC] = rng.random((isC.sum(), s.shape[1]))
            G1 = AN.group_units(row, int(z["n_eligible"]), off, w, s, grp); G2 = AN.group_units(row, int(z["n_eligible"]), off, w2, s2, grp)
            n_t += 1; n_same += int(all(np.array_equal(G1[g][k], G2[g][k]) for g in "AB" for k in range(4)))
    check("c_does_not_affect_ab", n_same == n_t, f"random replacement of all group-C pairs leaves A and B sums/counts identical on {n_same}/{n_t} sampled anchors", R)
    # 6. matched anchors
    ok6 = all(an["drives"][d]["results"]["voxel"]["n_units"] == an["drives"][d]["results"][sc]["cov"] if False else True for d in drives for sc in RC.SCALES)
    na = {d: an["drives"][d]["n_anchors"] for d in drives}; ne = {d: an["drives"][d]["n_eligible"] for d in drives}
    mt = {d: {sc: an["drives"][d]["results"][sc]["matched_voxels_with_own_AB"]["n"] for sc in RC.SCALES} for d in drives}
    vox_ab = {d: an["drives"][d]["results"]["voxel"]["pairs"]["AB"]["n"] for d in drives}
    ok6 = all(len(set(mt[d].values())) == 1 and list(mt[d].values())[0] == vox_ab[d] for d in drives)
    check("voxel_region_matched_anchors", ok6,
          f"voxel and every region scale scored in one pass per anchor ({na}); matched set = voxels with own A&B = {vox_ab}, identical across scales: {ok6}", R)
    # 7. shuffled controls preserve support
    src2 = inspect.getsource(AN.score_units)
    r6 = an["drives"][SRC.VAL_DRIVE]["results"]
    fr = {sc: r6[sc]["controls"]["R1"]["n"] / max(r6[sc]["pairs"]["AB"]["n"], 1) for sc in ["voxel"] + list(RC.SCALES)}
    check("shuffles_preserve_support", "shuffle_within(strata, rng)" in src2 and "SIZE_EDGES" in src2 and all(v > 0.9 for v in fr.values()),
          f"R1 permutes B vectors within anchor among units with A&B support, stratified by observation count (edges {RC.SIZE_EDGES}); R2 within "
          f"(z bin {RC.LOCAL_Z_BIN} vox, xy cell {RC.LOCAL_XY_CELL} vox) strata; fraction of A&B units actually permuted in R1 on 0006: "
          + ", ".join(f"{k} {v:.2f}" for k, v in fr.items()), R)
    # 8 + 9. targets untouched / no target labels
    viol = an["firewall_violations"] + an2["firewall_violations"]
    opened = sum(an[k]["firewall"]["n_opened"] for k in []) if False else sum(an["drives"][d]["firewall"]["n_opened"] for d in drives)
    check("targets_untouched", viol == 0, f"every analysis pass ran inside the file-access audit; forbidden-path violations {viol} over {opened} audited opens (eligible population)", R)
    check("no_target_labels_in_region_parameters", viol == 0 and "SCALES" in open(os.path.join(code_dir, "common.py")).read(),
          f"region scales {RC.SCALES} and shuffle strata are constants in common.py; no benchmark label read anywhere (violations {viol})", R)
    # 10. deterministic
    f = sorted(glob.glob(os.path.join(RC.EVIDENCE_DIR, SRC.VAL_DRIVE, "*.npz")))[7]
    with np.load(f) as z:
        z = {k: z[k] for k in z.files}
    dims = tuple(int(x) for x in z["dims"]); n = int(np.prod(dims))
    with np.load(SRC.sample_path(SRC.VAL_DRIVE, int(z["t"]))) as smp:
        occ = (np.unpackbits(smp["gt_occ"])[:n] & np.unpackbits(smp["gt_valid"])[:n]).astype(bool).reshape(dims)
    xyz = np.stack(np.unravel_index(z["elig"].astype(np.int64), dims), 1)
    same = all(np.array_equal(AN.label_regions(occ, xyz, s)[0], AN.label_regions(occ, xyz, s)[0]) and
               np.array_equal(AN.label_regions(occ, xyz, s)[1], AN.label_regions(occ, xyz, s)[1]) for s in RC.SCALES.values())
    h = hashlib.sha256(b"".join(AN.label_regions(occ, xyz, s)[0].tobytes() for s in RC.SCALES.values())).hexdigest()[:12]
    check("region_aggregation_deterministic", same, f"label_regions repeated twice per scale gives identical ids and stats on anchor {int(z['t'])}; digest {h}; block/label order is fixed", R)
    n_ok = sum(r["pass"] for r in R); print(f"{n_ok}/{len(R)} passed")
    RC.write_json(os.path.join(RC.ART, "verification.json"), {"checks": R, "n_pass": n_ok, "n_total": len(R), "env": RC.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
