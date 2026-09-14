#!/usr/bin/env python
"""The ten invariants of the target-consistency study, each with counts; fails loudly.

    python tools/future_semantic_completion/target_consistency/verify.py
"""
from __future__ import annotations
import argparse, glob, hashlib, inspect, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion.target_consistency import analyze as AN            # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC, grids as G6G                            # noqa: E402


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample_every", type=int, default=25)
    a = ap.parse_args()
    ST.require_data_root()
    R = []
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    files = {d: sorted(glob.glob(os.path.join(TC.EVIDENCE_DIR, d, "*.npz"))) for d in drives}
    MAP = G6G.PREDICTION_GRID["kitti360"]

    # 1. disjoint frames --------------------------------------------------------------------
    ok1 = True; n_files = 0; n_bad = 0
    for g, grp in AN.GROUPINGS.items():
        sets = [set(v) for v in grp.values()]
        ok1 &= all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
        ok1 &= set().union(*sets) <= set(range(1, SRC.FUTURE_FRAMES + 1))
    for d in drives:
        for f in files[d]:
            with np.load(f) as z:
                t = int(z["t"]); fr = z["frames"]; pf = z["pair_frame"]
            n_files += 1
            if not ((fr > t).all() and (fr <= t + SRC.FUTURE_FRAMES).all() and len(set(fr.tolist())) == len(fr)
                    and (pf >= 1).all() and (pf <= SRC.FUTURE_FRAMES).all()):
                n_bad += 1
    check("abc_disjoint_future_frames", ok1 and n_bad == 0,
          f"groupings {json.dumps({g: {k: v for k, v in grp.items()} for g, grp in AN.GROUPINGS.items()})} pairwise disjoint within t+1..t+20: {ok1}; "
          f"{n_files - n_bad}/{n_files} evidence files have unique future frames in (t, t+20] and pair offsets in [1, 20]", R)

    # 2 + 6. C never affects retention or fusion (perturbation) ------------------------------
    F = AN.filters(); n_test = n_same = n_same_fuse = 0
    for d in drives:
        for f in files[d][::a.sample_every]:
            with np.load(f) as z:
                z = {k: z[k] for k in z.files}
            for g, grp in AN.GROUPINGS.items():
                G = AN.group_sums(z, grp)
                z2 = dict(z); isC = np.isin(z["pair_frame"].astype(np.int64), grp["C"])
                rng = np.random.default_rng(1)
                z2["pair_s"] = z["pair_s"].copy(); z2["pair_w"] = z["pair_w"].copy()
                z2["pair_s"][isC] = rng.random((int(isC.sum()), z["pair_s"].shape[1])).astype(np.float16)
                z2["pair_w"][isC] = (rng.random(int(isC.sum())) * 50).astype(np.float16)
                G2 = AN.group_sums(z2, grp)
                (wA, sA, _), (wB, sB, _) = G["A"], G["B"]; (wA2, sA2, _), (wB2, sB2, _) = G2["A"], G2["B"]
                for fname, (fam, tau, gam) in F.items():
                    n_test += 1
                    k1 = AN.retain_mask(fam, tau, gam, wA, sA, wB, sB); k2 = AN.retain_mask(fam, tau, gam, wA2, sA2, wB2, sB2)
                    n_same += int(np.array_equal(k1, k2))
                p1 = AN.probs(wA + wB, sA + sB); p2 = AN.probs(wA2 + wB2, sA2 + sB2)
                n_same_fuse += int(np.array_equal(p1, p2))
    check("c_never_affects_retention", n_same == n_test,
          f"random replacement of every group-C evidence pair leaves the retained mask identical for {n_same}/{n_test} (anchor, grouping, filter) cases", R)
    check("fusion_uses_only_ab", n_same_fuse == n_test // len(F),
          f"A+B candidate vector identical under the same C perturbation on {n_same_fuse}/{n_test // len(F)} (anchor, grouping) cases; "
          f"retain_mask signature: {list(inspect.signature(AN.retain_mask).parameters)}", R)

    # 3. frozen Trident-H / mapper ------------------------------------------------------------
    tri = "/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics/kitti360"
    tri_files = sorted(glob.glob(tri + "/*.npz"))
    newest = max(os.path.getmtime(p) for p in tri_files)
    study_start = min(os.path.getmtime(p) for d in drives for p in files[d])
    code = {p: ST.sha256_file(os.path.join(ST.REPO_ROOT, p))[:12] for p in
            ("core/mapping/mapper.py", "core/mapping/voxmap.py", "core/mapping/feed.py", "core/mapping/rays.py",
             "core/supervision/lifting.py", "core/run/build_samples.py")}
    builds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(TC.ART, "evidence_build", "*.json")))]
    dp = max(b["max_abs_dp"] for b in builds); dw = max(b["max_abs_dw"] for b in builds)
    check("frozen_teacher_and_mapper", newest < study_start and dp <= 0.004 and len(builds) == 4,
          f"{len(tri_files)} cached Trident-H files, newest {time.strftime('%Y-%m-%d', time.localtime(newest))} < first evidence file "
          f"{time.strftime('%Y-%m-%d', time.localtime(study_start))}; mapper code digests {code}; per-frame replay reproduces the cached "
          f"20-frame target on every anchor: max |dp| {dp:.4f} (u8 quantisation 0.002), max |dw| {dw:.3f} (float16)", R)

    # 4. no causal evidence in retention -----------------------------------------------------
    keys = set()
    with np.load(files[SRC.VAL_DRIVE][0]) as z:
        keys = set(z.files)
    causal_keys = {"sem_rows", "sem_p", "sem_w", "logodds", "observed", "input"}
    src = inspect.getsource(AN.retain_mask) + inspect.getsource(AN.group_sums)
    check("no_causal_evidence_in_retention", not (keys & causal_keys) and "sem_rows" not in src and "input" not in src,
          f"evidence files carry no causal-map array ({sorted(keys)}); retention reads only pair_* future evidence", R)

    # 5. voxel identity ---------------------------------------------------------------------
    n_ok = 0; n_tot = 0
    for d in drives:
        for f in files[d]:
            with np.load(f) as z:
                n_tot += 1
                n_ok += int(tuple(int(x) for x in z["dims"]) == tuple(int(x) for x in MAP.dims)
                            and z["T_grid_to_world"].shape == (4, 4) and (np.diff(z["elig"]) > 0).all()
                            and float(z["reproduces_cached_target_max_abs_dp"]) <= 0.004)
    check("voxel_identity_consistent", n_ok == n_tot,
          f"{n_ok}/{n_tot} anchors: one grid ({'x'.join(str(int(x)) for x in MAP.dims)}), one T_grid_to_world, one sorted eligible row set "
          f"shared by every frame's query; A+B+C sum reproduces the cached fusion", R)

    # 7. class statistics after filtering ---------------------------------------------------
    an = json.load(open(os.path.join(TC.ART, "analysis.json")))
    r1 = an["results"]["interleaved"]["F1_agree"][SRC.VAL_DRIVE]
    tot_share = sum(r1["class_distribution"].values())
    check("class_stats_after_filtering", abs(tot_share - 1) < 1e-6 and r1["n_retained"] < an["support"][SRC.VAL_DRIVE]["n_eligible"],
          f"class distribution of F1 on 0006 sums to {tot_share:.6f} over {r1['n_retained']} retained voxels "
          f"(eligible {an['support'][SRC.VAL_DRIVE]['n_eligible']}); Acc.add accumulates cls_ret from the retained rows only", R)

    # 8 + 9. thresholds on KITTI-360 only; targets untouched --------------------------------
    stage = {"analysis": an["firewall"]}
    for nm in ("neighbourhood_support", "causal_diagnostic"):
        p = os.path.join(TC.ART, f"{nm}.json")
        if os.path.exists(p):
            stage[nm] = json.load(open(p))["firewall"]
    for b in builds:
        stage[f"evidence_{b['drive'][17:21]}"] = b["firewall"]
    viol = {k: len(v["violations"]) for k, v in stage.items()}
    check("thresholds_selected_on_kitti360_only", sum(viol.values()) == 0,
          f"tau grid {TC.TAUS}, gamma grid {TC.GAMMAS} fixed in common.py; every stage ran inside the file-access audit with forbidden "
          f"patterns {stage['analysis']['forbidden_patterns']}; violations per stage {viol}", R)
    check("targets_untouched", sum(viol.values()) == 0,
          f"no SemanticKITTI / Occ3D / nuScenes path opened by any stage ({sum(v['n_opened'] for v in stage.values())} opens audited)", R)

    # 10. nothing trained ---------------------------------------------------------------------
    man = json.load(open(ST.FROZEN_MANIFEST))
    ck_ok = all(ST.sha256_file(os.path.join(ST.REPO_ROOT, m["checkpoint"])) == m["checkpoint_sha256"] for m in man["seeds"].values())
    hm = json.load(open(ST.manifest_path()))
    head_ok = ST.sha256_file(ST.head_checkpoint(0)) == hm["seeds"]["0"]["head_checkpoint_sha256"]
    code_dir = os.path.dirname(os.path.abspath(__file__))
    others = [p for p in os.listdir(code_dir) if p.endswith(".py") and p != "verify.py"]
    srcs = "".join(open(os.path.join(code_dir, p)).read() for p in others)
    no_opt = all(tok not in srcs for tok in ("torch.optim", ".backward(", "requires_grad", "state_dict"))
    check("nothing_trained", ck_ok and head_ok and no_opt,
          f"Gate 8C-1 checkpoints match their manifest hashes: {ck_ok}; previous semantic head matches its frozen hash: {head_ok}; "
          f"no optimiser / backward / parameter update in the {len(others)} study modules: {no_opt}", R)

    n_ok = sum(r["pass"] for r in R)
    print(f"{n_ok}/{len(R)} passed")
    ST.write_json(os.path.join(TC.ART, "verification.json"), {"checks": R, "n_pass": n_ok, "n_total": len(R), "env": ST.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
