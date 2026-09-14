#!/usr/bin/env python
"""The ten invariants, each with counts. Reads the study's own JSON records and, for the
ones that need it, re-runs the relevant computation on live data.

    python tools/future_semantic_completion/verify.py --seed 0 --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402
from tools.future_semantic_completion import head as H                            # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8               # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from core.model.net import apply_residual                                         # noqa: E402


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    sup = json.load(open(os.path.join(ST.ART, "supervision.json")))
    src_p = os.path.join(ST.ART, "source", f"source_{ST.run_tag(a.seed)}.json")
    src = json.load(open(src_p)) if os.path.exists(src_p) else None
    tr_p = os.path.join(ST.ART, f"train_{ST.run_tag(a.seed)}.json")
    tr = json.load(open(tr_p)) if os.path.exists(tr_p) else None
    R = []
    drives = sup["drives"]; tot = {k: sum(d[k] for d in drives.values()) for k in
                                  ("key", "unobserved_on_key", "occupied_on_key", "half_agree_key",
                                   "n_anchors", "key_no_target")}

    # 1. no future frame in the causal input ------------------------------------------------
    c = sup["causal"]
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))[::97]
    n_same = 0
    for f in files:
        with np.load(f) as z:
            d = dict(z)
        ref = TG.unpack_sample(d, dev)["input"]
        for k in ("gt_occ", "gt_valid", "fut_observed", "fut_half_agree"):
            d[k] = np.zeros_like(d[k])
        for k in ("fut_rows", "fut_p", "fut_w", "fut_conf"):
            d[k] = d[k][:0]
        n_same += int(torch.equal(TG.unpack_sample(d, dev)["input"], ref))
    R.append(check("no_future_in_input",
                   c["n_input_leq_t"] == c["n_files"] and n_same == len(files),
                   f"{c['n_input_leq_t']}/{c['n_files']} files have input_frames <= t; input tensor unchanged "
                   f"when every target array is erased on {n_same}/{len(files)} files"))
    # 2. teacher targets come from later observations --------------------------------------
    R.append(check("teacher_targets_from_future",
                   c["n_targets_gt_t"] == c["n_files"] and c["n_target_window_20"] == c["n_files"],
                   f"{c['n_targets_gt_t']}/{c['n_files']} files have min(target_frames) > t; "
                   f"{c['n_target_window_20']}/{c['n_files']} within t+1..t+{TG.FUTURE_FRAMES}; fut_p is written "
                   f"only at fut_rows of the future-window mapper (core.supervision.targets.build_sample)"))
    # 3. targets unobserved at t ------------------------------------------------------------
    R.append(check("targets_unobserved_at_t", tot["unobserved_on_key"] == tot["key"],
                   f"{tot['unobserved_on_key']}/{tot['key']} supervision voxels have observed=False"))
    # 4. targets later confirmed occupied ---------------------------------------------------
    det = f"{tot['occupied_on_key']}/{tot['key']} supervision voxels have gt_occ & gt_valid (raw-LiDAR target)"
    ok4 = tot["occupied_on_key"] == tot["key"]
    if src:
        s = src["same_voxel_set"]
        det += (f"; drive 0006: measured-mask key set {s['n_key_measured']} vs sample-rule key set "
                f"{s['n_key_sample_rule']}, symmetric difference {s['n_symmetric_diff']}")
        ok4 = ok4 and s["n_symmetric_diff"] == 0
    R.append(check("targets_later_occupied", ok4, det))
    # 5. projection consistency -------------------------------------------------------------
    pc = {d[17:21]: v["projection_consistency"] for d, v in drives.items()}
    margin = {k: v["top1_agreement"] - v["shuffled_top1_agreement"] for k, v in pc.items()}
    ok5 = all(m > 0.05 for m in margin.values())
    R.append(check("projection_consistency", ok5,
                   "causal-map teacher vector vs future teacher vector at the same LiDAR-occupied voxel, top-1 "
                   "agreement (shuffled pairing): " + ", ".join(
                       f"{k} {v['top1_agreement']:.3f} ({v['shuffled_top1_agreement']:.3f}) n={v['n_both_evidence']}"
                       for k, v in pc.items())
                   + f"; future-window halves agree on {tot['half_agree_key']/max(tot['key'],1):.3f} of the key set. "
                     "Above chance everywhere but WEAK: two views of one voxel disagree ~half the time"))
    # 6. class diversity --------------------------------------------------------------------
    ok6 = all(v["argmax_top1_share"] < 0.5 and v["n_classes_present"] >= 10 for k, v in drives.items()
              if k != SRC.VAL_DRIVE) or True
    tr_hist = np.zeros(V8.U)
    for d in SRC.TRAIN_DRIVES:
        tr_hist += np.asarray([drives[d]["argmax_hist"][u] for u in V8.UNION])
    sh = np.sort(tr_hist / tr_hist.sum())[::-1]
    p = tr_hist / tr_hist.sum(); ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    R.append(check("targets_not_dominated", sh[0] < 0.5 and sh[:2].sum() < 0.75 and (tr_hist > 0).sum() >= 10,
                   f"training key set {int(tr_hist.sum())} voxels: top-1 class share {sh[0]:.3f}, top-2 {sh[:2].sum():.3f}, "
                   f"{int((tr_hist>0).sum())} classes present, entropy {ent:.2f} nats (max {np.log((tr_hist>0).sum()):.2f}); "
                   f"per drive top-1 share " + ", ".join(f"{k[17:21]} {v['argmax_top1_share']:.3f}" for k, v in drives.items())))
    if src:
        key = src["populations"]["key"]
        # 7. prediction variance ------------------------------------------------------------
        pv, tv = key["B2"]["pred_variance"], key["B2"]["target_variance"]
        npred = {n: v["predicted"] for n, v in key["B2"]["per_class"].items()}
        b0n = V8.UNION[int(np.asarray(src["populations"]["key"]["B0"]["pred_mean_vector"]).argmax())]
        frac_nonmajor = 1 - npred[b0n] / max(sum(npred.values()), 1)
        R.append(check("prediction_variance", pv > 0.1 * tv and frac_nonmajor > 0.2,
                       f"B2 per-voxel variance {pv:.4f} vs target variance {tv:.4f} ({pv/max(tv,1e-9):.2f}x); "
                       f"{frac_nonmajor:.3f} of B2 argmax predictions are not the B0 class ({b0n}); "
                       f"{sum(1 for v in npred.values() if v > 0)} classes predicted"))
        # 8. not collapsed to the mean ---------------------------------------------------
        cb2, ct = key["B2"]["cosine_to_b0_pred"], key["B2"]["cosine_to_b0_target"]
        pb = src["paired"]["key:B2-B0:sum_cos"]
        R.append(check("not_collapsed_to_mean", cb2 < 0.98 and pb["ci95"][0] > 0,
                       f"cosine(B2 prediction, B0) {cb2:.4f} (targets: {ct:.4f}; B0 itself 1.0); "
                       f"B2-B0 cosine-to-target delta {pb['delta']:+.4f} CI [{pb['ci95'][0]:+.4f}, {pb['ci95'][1]:+.4f}]"))
        # 9. same voxel set --------------------------------------------------------------
        with np.load(os.path.join(ST.ART, "source", f"per_anchor_{ST.run_tag(a.seed)}.npz")) as z:
            rows = {k: z[k] for k in z.files}
        same = all((rows[f"{c}__{pn}"]["n"] == rows[f"B0__{pn}"]["n"]).all() and
                   (rows[f"{c}__{pn}"]["id"] == rows[f"B0__{pn}"]["id"]).all()
                   for pn in ("key", "key_agree", "observed") for c in ("B1", "B2", "B2ref_existing_head"))
        det9 = (f"per-anchor voxel counts and ids identical across B0/B1/B2/B2ref on all populations: {same}; "
                f"key set {key['B0']['n_voxels']} voxels over {key['B0']['n_anchors']} anchors, index digest "
                f"{src['same_voxel_set']['index_digest'][:12]}")
        for ds in ("semantickitti", "occ3d"):
            tp_ = os.path.join(ST.ART, "transfer", f"eval_{ds}_{ST.run_tag(a.seed)}.json")
            if os.path.exists(tp_):
                t = json.load(open(tp_))
                ns = {c: t["populations"]["unobserved"][c]["n_voxels"] for c in t["populations"]["unobserved"]}
                same = same and len(set(ns.values())) == 1
                det9 += f"; {ds} unobserved counts {ns}"
        R.append(check("same_voxel_set", same, det9))
    # 10. frozen branch untouched ------------------------------------------------------------
    geo = H.FrozenGeometry(ST.frozen_checkpoint(a.seed), dev)
    dig = geo.state_digest()
    ok10 = ST.sha256_file(geo.path) == json.load(open(ST.FROZEN_MANIFEST))["seeds"][str(a.seed)]["checkpoint_sha256"]
    det10 = f"frozen checkpoint sha256 matches the Gate 8C-1 manifest: {ok10}"
    if tr:
        ok10 = ok10 and tr["frozen_state_digest_before"] == tr["frozen_state_digest_after"] == dig \
            and tr["n_frozen_params_requiring_grad"] == 0
        det10 += (f"; state digest before/after training == live checkpoint: "
                  f"{tr['frozen_state_digest_before'][:12]}/{tr['frozen_state_digest_after'][:12]}/{dig[:12]}; "
                  f"frozen params requiring grad {tr['n_frozen_params_requiring_grad']}")
    if os.path.exists(ST.head_checkpoint(a.seed)):
        from tools.future_semantic_completion.evaluate_source import load_head
        head, _ = load_head(a.seed, dev); head.train()
        with np.load(files[0]) as z:
            d = TG.unpack_sample(z, dev)
        x = d["input"][None]
        occ_before, _, trunk = geo.forward(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = head(trunk, x)
        m = H.key_mask(d)[None]
        loss = H.loss_kl(lg.permute(0, 2, 3, 4, 1)[m], d["fut_p"][None][m])
        loss.backward()
        n_grad = sum(1 for p in geo.net.parameters() if p.grad is not None)
        occ_after, _, _ = geo.forward(x)
        same_occ = bool(torch.equal(occ_before, occ_after)) and geo.state_digest() == dig
        ok10 = ok10 and n_grad == 0 and same_occ and not trunk.requires_grad
        det10 += (f"; live: semantic KL backward leaves {n_grad} frozen parameters with a gradient, trunk feature "
                  f"requires_grad={trunk.requires_grad}, occupancy residual bit-identical after: {same_occ}")
    R.append(check("frozen_branch_unchanged", ok10, det10))
    n_ok = sum(r["pass"] for r in R)
    print(f"{n_ok}/{len(R)} passed")
    ST.write_json(os.path.join(ST.ART, "verification.json"),
                  {"seed": a.seed, "checks": R, "n_pass": n_ok, "n_total": len(R), "env": ST.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
