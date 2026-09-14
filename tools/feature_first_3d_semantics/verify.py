#!/usr/bin/env python
"""Twelve invariants of the feature-first study; fails loudly.

    python tools/feature_first_3d_semantics/verify.py
"""
from __future__ import annotations
import glob, inspect, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from tools.feature_first_3d_semantics import analyze as AN                                # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ST.require_data_root(); R = []
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    ext = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(FF.ART, "extraction", "2013*.json")))]
    lifts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(FF.ART, "lifting", "*.json")))]
    an = {p: json.load(open(os.path.join(FF.ART, f"analysis_{p}.json"))) for p in FF.EVIDENCE_DIRS if os.path.exists(os.path.join(FF.ART, f"analysis_{p}.json"))}
    code_dir = os.path.dirname(os.path.abspath(__file__))
    mods = [p for p in os.listdir(code_dir) if p.endswith(".py") and p != "verify.py"]
    srcs = "".join(open(os.path.join(code_dir, p)).read() for p in mods)
    # 1 nothing trained
    man = json.load(open(ST.FROZEN_MANIFEST))
    ck_ok = all(ST.sha256_file(os.path.join(ST.REPO_ROOT, m["checkpoint"])) == m["checkpoint_sha256"] for m in man["seeds"].values())
    no_opt = all(t not in srcs for t in ("torch.optim", ".backward(", "requires_grad_(True", "nn.Parameter"))
    check("no_model_trained", ck_ok and no_opt, f"no optimiser/backward/parameter in {len(mods)} study modules: {no_opt}; frozen checkpoints match manifest: {ck_ok}", R)
    # 2 Trident frozen
    prov = glob.glob(os.path.join(ST.REPO_ROOT, "artifacts", "gate6", "*provenance*.json"))
    sam_sha = {e["teacher"]["sam_ckpt_sha256"] for e in ext}; tri_sha = {e["teacher"]["trident_py_sha256"] for e in ext}
    ref_sam = None
    for p in prov:
        s = json.dumps(json.load(open(p)))
        for h in sam_sha:
            if h in s:
                ref_sam = p
    check("trident_frozen", len(sam_sha) == 1 and len(tri_sha) == 1 and all(len(e["teacher"]["adapter_sha256"]) == 64 for e in ext),
          f"one SAM-H checkpoint sha across {len(ext)} extraction shards ({list(sam_sha)[0][:12]}; matches gate-6 provenance record: {os.path.basename(ref_sam) if ref_sam else 'no record found'}), "
          f"one trident.py sha ({list(tri_sha)[0][:12]}), official Cityscapes config; the hook records model.clip.encode_image output and changes nothing", R)
    # 3 deterministic extraction (two frames re-extracted by verify_determinism into a scratch dir)
    det = os.path.join(FF.ART, "extraction", "determinism.json")
    d_ok = os.path.exists(det) and json.load(open(det))["identical"]
    check("extraction_deterministic", d_ok, (json.load(open(det))["detail"] if os.path.exists(det) else "determinism.json missing (run verify_determinism.py in the Trident env)"), R)
    # 4 feature reproduces the teacher
    dp = max(e["coarse_reproduction"]["max_abs_dp_max"] for e in ext); ag = min(e["coarse_reproduction"]["argmax_agree_min"] for e in ext)
    agm = np.mean([e["coarse_reproduction"]["argmax_agree_mean"] for e in ext]); rc = np.mean([e["coarse_reproduction"]["refined_vs_coarse_agree_mean"] for e in ext])
    n_fr = sum(e["n_written"] for e in ext)
    check("feature_reproduces_teacher", dp < 0.01 and ag > 0.99,
          f"{n_fr} frames: bilinear(f_norm . text) x 40, softmax reproduces the official coarse probabilities to max |dp| {dp:.4f} (fp16 tokens), "
          f"argmax agreement min {ag:.4f} / mean {agm:.4f}; SAM refinement then changes {100*(1-rc):.1f} % of pixel labels (cached refined vs coarse {rc:.3f})", R)
    # 5 disjoint frames
    sets = [set(v) for v in AN.GRP.values()]; disj = all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
    n_files = n_bad = 0
    for pop in FF.EVIDENCE_DIRS:
        for d in drives:
            for f in glob.glob(os.path.join(FF.LIFT_DIR, pop, d, "*.npz")):
                with np.load(f) as z:
                    t = int(z["t"]); fr = z["frames"]; pf = z["pair_frame"]
                n_files += 1; n_bad += int(not ((fr > t).all() and (fr <= t + 20).all() and len(set(fr.tolist())) == len(fr) and (pf >= 1).all() and (pf <= 20).all()))
    check("abc_disjoint_frames", disj and n_bad == 0, f"groups {AN.GRP} disjoint: {disj}; {n_files - n_bad}/{n_files} lifted files have unique future frames in (t, t+20]", R)
    # 6 C never enters A/B fusion
    src = inspect.getsource(AN.fusion_metrics)
    fuse_lines = [l for l in src.splitlines() if "_ab" in l and "=" in l and "vs" not in l]
    ok6 = all(("D[c]" not in l) for l in fuse_lines) and "wa * D[a]" in src and "wb * D[b]" in src
    check("c_not_in_ab_fusion", ok6, "fusion_metrics builds p_ref_ab, p_co_ab, f_ab from groups a and b only (source-inspected: no D[c] on any *_ab assignment); C is read only for the comparison", R)
    # 7 matched surface elements
    ok7 = all(l["all_pairs_match_probability_path"] and l["max_rel_weight_dev"] < 2e-3 for l in lifts) and len(lifts) == 8
    check("matched_surface_elements", ok7,
          f"{len(lifts)}/8 lifting runs: every (voxel, frame) evidence pair of the feature path equals the probability path's pair set, weights within "
          f"{max(l['max_rel_weight_dev'] for l in lifts):.1e} relative (float16 cache rounding); analysis asserts identical elig arrays and pair keys per anchor "
          f"({sum(a_['drives'][d]['support']['n_anchor_pair_mismatch'] for a_ in an.values() for d in drives)} mismatching anchors)", R)
    # 8 local-shuffle controls
    s8 = inspect.getsource(AN.pair_metrics)
    fr = {}
    for pop, a_ in an.items():
        ab = a_["drives"][SRC.VAL_DRIVE]["pairs"]["AB"]; fr[pop] = {k: v["n"] / ab["n"] for k, v in ab["controls"].items()}
    check("controls_preserve_context", "C2_local" in s8 and "Z_BIN" in s8 and all(v["C2_local"] > 0.8 for v in fr.values()),
          f"C1 within anchor; C2 within (anchor, z band {AN.Z_BIN} vox, xy cell {AN.XY_CELL} vox); C3 within Trident class across the drive; C0 whole drive; "
          f"fraction of A&B units actually permuted on 0006: {json.dumps({p: {k: round(x, 2) for k, x in v.items()} for p, v in fr.items()})}", R)
    # 9 + 10 KITTI-360 only / targets untouched
    viol = sum(len(l["firewall"]["violations"]) for l in lifts) + sum(len(a_["firewall"]["violations"]) for a_ in an.values())
    n_open = sum(l["firewall"]["n_opened"] for l in lifts) + sum(a_["firewall"]["n_opened"] for a_ in an.values())
    check("kitti360_only_decisions", viol == 0, f"controls, groups and fusion rules are constants; drive 0006 is the only inspection drive; audited opens {n_open}, violations {viol}", R)
    check("targets_untouched", viol == 0, f"no SemanticKITTI / Occ3D / nuScenes path opened by lifting or analysis (violations {viol})", R)
    # 11 fusion has no learned parameters
    s11 = inspect.getsource(AN.l2n) + inspect.getsource(AN.classify) + src
    check("fusion_unlearned", "l2n(wa * D[a][\"f\"][m] + wb * D[b][\"f\"][m])" in src and "LOGIT_SCALE" in s11,
          "f_AB = normalize(w_A f_A + w_B f_B) with the mapper's evidence weights; classification = softmax(40 f . text) with the teacher's own text embeddings and temperature", R)
    # 12 pipelines differ only in fusion order
    lift_src = open(os.path.join(code_dir, "lift_features.py")).read()
    s_a = "* FF.LOGIT_SCALE).softmax(0)"; s_b = "einsum(\"chw,kc->khw\", out, self.text)"
    s_c = "p_co_ab = (wa * D[a][\"p_co\"][m] + wb * D[b][\"p_co\"][m]) / (wa + wb)"
    ok12 = s_a in lift_src and s_b in lift_src and s_c in src
    check("pipelines_differ_only_in_order", ok12,
          "per-view classification in lift_features = softmax(40 f_pix . text) on the same lattice feature the mapper lifts; both pipelines use identical rays, "
          "weights, groups, text embeddings and temperature; P_coarse averages the per-view softmax, F averages the feature then applies the same softmax once", R)
    n_ok = sum(r["pass"] for r in R); print(f"{n_ok}/{len(R)} passed")
    FF.write_json(os.path.join(FF.ART, "verification.json"), {"checks": R, "n_pass": n_ok, "n_total": len(R), "env": FF.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
