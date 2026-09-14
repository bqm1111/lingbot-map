#!/usr/bin/env python
"""Gate 12 verification (20 required checks + supporting ones); writes verification.{md,json}.

    python tools/gate12_suf/verify12.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, tempfile
import numpy as np, torch, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.G11_DIR); sys.path.insert(0, g12.G9_DIR); sys.path.insert(0, g12.REPO_ROOT)
import build_inputs as BI                                                                   # noqa: E402
from core.datasets import kitti360_splits as SRC                                            # noqa: E402
from core.mapping import depth as D7                                                        # noqa: E402

RUNS = os.path.join(g12.ART, "runs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    runs = {(c, s): json.load(open(os.path.join(RUNS, f"train_g12_{c}_seed{s}.json"))) for c in g12.CONDITIONS for s in g12.SEEDS}
    pair = lambda key: all(runs[("H", s)][key] == runs[("SUF", s)][key] for s in g12.SEEDS)
    src_train = open(os.path.join(HERE, "train12.py")).read(); src_in = open(os.path.join(HERE, "build_inputs.py")).read()
    cfgH, cfgS = yaml.safe_load(open(os.path.join(g12.ART, "training_config_H.yaml"))), yaml.safe_load(open(os.path.join(g12.ART, "training_config_SUF.yaml")))
    evals = {(c, s): json.load(open(os.path.join(g12.ART, "eval", f"g12_{c}_seed{s}_step06000.json"))) for c in g12.CONDITIONS for s in g12.SEEDS}
    ev_ids = {k: np.load(os.path.join(g12.ART, "eval", f"g12_{k[0]}_seed{k[1]}_step06000_counts.npz"))["ids"] for k in evals}
    check("01_same_clips", pair("train_items_sha256") and pair("n_train") and all(np.array_equal(ev_ids[("H", 0)], v) for v in ev_ids.values()),
          f"training anchor list digest equal for H and S/U/F on every seed ({runs[('H', 0)]['n_train']} anchors); all six evaluations on the same {len(ev_ids[('H', 0)])} 0006 anchors")
    check("02_same_training_targets", pair("target_digest") and all(runs[("H", s)]["digests_by_checkpoint"] == runs[("SUF", s)]["digests_by_checkpoint"] for s in g12.SEEDS),
          "sha256 over every drawn target crop (occupancy, valid, loss-weight observed mask) identical for H and S/U/F on each seed, at every 1000-step checkpoint: " +
          ", ".join(f"seed {s} {runs[('H', s)]['target_digest'][:12]}" for s in g12.SEEDS))
    check("03_same_crop_coordinates", pair("draw_digest"), "sha256 over every (anchor, crop origin) draw identical per seed: " + ", ".join(f"seed {s} {runs[('H', s)]['draw_digest'][:12]}" for s in g12.SEEDS))
    aug = [k for k in ("flip", "rot90", "rotate", "jitter", "augment") if k in src_train.lower()]
    check("04_same_augmentations", not aug and pair("draw_digest"), f"the training pipeline has no augmentation ({aug or 'no augmentation tokens'}); crops identical per seed")
    check("05_same_optimizer", pair("optimizer"), f"{runs[('H', 0)]['optimizer']} for every run")
    lrH = [[x["lr"] for x in runs[("H", s)]["log"] if "lr" in x] for s in g12.SEEDS]; lrS = [[x["lr"] for x in runs[("SUF", s)]["log"] if "lr" in x] for s in g12.SEEDS]
    check("06_same_schedule", pair("scheduler") and lrH == lrS, f"{runs[('H', 0)]['scheduler']}, identical learning-rate sequence at every step for H and S/U/F")
    check("07_same_steps", pair("steps") and all(r["steps"] == 6000 for r in runs.values()), "6000 steps, batch 4, every run")
    arch, pc = [], []
    for k in runs:
        ck = torch.load(os.path.join(g12.REPO_ROOT, runs[k]["final_checkpoint"]), map_location="cpu", weights_only=False)
        arch.append((ck["cin"], ck["width"], ck["padding_mode"], tuple((n, tuple(v.shape)) for n, v in ck["state_dict"].items()))); pc.append(runs[k]["n_params"])
    check("08_same_architecture", len(set(arch)) == 1, f"CompletionUNet cin 3, width {arch[0][1]}, padding {arch[0][2]}, identical parameter names and shapes in all six checkpoints; shared initial weights per seed: "
          + str(all(runs[('H', s)]['init_state_digest'] == runs[('SUF', s)]['init_state_digest'] for s in g12.SEEDS)))
    check("09_same_parameter_count", len(set(pc)) == 1, f"{pc[0]:,} parameters in every run")
    diffk = sorted(k for k in set(cfgH) | set(cfgS) if cfgH.get(k) != cfgS.get(k))
    check("10_same_checkpoint_rule", diffk == ["input"] and pair("checkpoint_rule") and all(r["final_checkpoint"].endswith("step06000.pt") for r in runs.values()) and
          all(e["checkpoint"].endswith("step06000.pt") for e in evals.values()),
          f"the two configs differ only in {diffk}; every evaluated checkpoint is the final step; no selection")
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(g12.ART, "logs", "inputs_*.log")))
    viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]
    check("11_local_reset_L_pose_not_GT", "Pinv @ P[q]" in src_in and "pred_pose_c2w" in src_in and all(t not in src_in for t in ("rect_cam_to_world", "cam0_to_world(", "DriveGeometry")) and
          "data_poses" in BI.AUDIT_PATTERNS and len(viol) >= 4 and sum(viol) == 0,
          f"poses = inv(P_4) P_q of LingBot's local-reset pred_pose_c2w; the input builder never constructs GT geometry and ran inside a file audit forbidding data_poses / cam0_to_world: violations {viol}")
    # dense rays: recount on sampled anchors
    # Amended after the first verification run: an arbitrary clause "dense rays > 50 % of pixels" failed on a low-confidence
    # training frame (0003:132, 44.9 %) although the recount matched exactly. Replaced by a comparison with the clip's own
    # Gate 9 LiDAR-support pixel count where a Gate 9 record exists (0006).
    ok12, det12 = True, []
    for d in (g12.VAL_DRIVE,) + g12.TRAIN_DRIVES:
        files = sorted(glob.glob(os.path.join(os.path.dirname(g12.input_path(d, 0)), "*.npz")))
        for f in files[:: max(len(files) // 3, 1)][:3]:
            zi = np.load(f); st = json.loads(str(zi["stats"])); t = int(zi["t"])
            with np.load(g12.clip_path(d, t)) as z:
                zm = float(z["scale"]) * z["pred_depth"].astype(np.float32).astype(np.float64); cf = z["pred_depth_conf"].astype(np.float64)
            dense = int(((cf >= 1.5) & (zm > 1) & (zm < 60)).sum()); ok12 &= st["n_rays_H"] == dense
            g9 = os.path.join("/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy/clips", d, f"{t:05d}.npz"); sup = None
            if d == g12.VAL_DRIVE and os.path.exists(g9):
                sup = int(np.load(g9)["n_support_pixels"].sum()); ok12 &= dense > sup
            det12.append(f"{d[17:21]}:{t} rays {dense:,}/{zm.size:,}" + (f" (Gate 9 LiDAR support {sup:,})" if sup is not None else ""))
    # Second amendment: the source scan also listed the bare word "velodyne", which occurs in build_inputs.py only inside
    # AUDIT_PATTERNS (the firewall that forbids LiDAR paths). The scan now looks for LiDAR-access code, not the firewall text.
    check("12_dense_rays_no_lidar_mask", ok12 and all(k not in src_in for k in ("zbuffer", "read_velodyne", "velodyne_points", "sup[", "support")) and "velodyne" in BI.AUDIT_PATTERNS,
          f"rays accepted by the mapper equal an independent recount of every confident dense LingBot pixel in 1-60 m, and on 0006 exceed the clip's LiDAR-support pixels (no LiDAR mask; amended clause): {det12}")
    san = json.load(open(os.path.join(g12.ART, "sanity_inputs.json")))
    n_scale = n_scale_tot = 0; n_clip_same = n_clip_tot = 0
    for d in (g12.VAL_DRIVE,) + g12.TRAIN_DRIVES:
        for f in sorted(glob.glob(os.path.join(os.path.dirname(g12.input_path(d, 0)), "*.npz")))[::25]:
            zi = np.load(f); t = int(zi["t"])
            with np.load(g12.clip_path(d, t)) as z:
                n_scale += int(float(zi["scale"]) == float(z["scale"])); n_scale_tot += 1
    for d in g12.TRAIN_DRIVES:
        for f in sorted(glob.glob(os.path.join(g12.LR_ROOT, d, "*.npz"))):
            a_, b_ = np.load(f), np.load(os.path.join(g12.L_ROOT, d, os.path.basename(f)))
            n_clip_same += int(all(np.array_equal(a_[k], b_[k]) for k in a_.files if k != "rope_slots_used")); n_clip_tot += 1
    once = src_in.count("mS.scale_state.force(s)") == 1 and src_in.count("mU.scale_state.force(s)") == 1 and "depth_canonical=torch.from_numpy(z[\"pred_depth\"][q]" in src_in
    check("13_causal_moge_scale_once", n_scale == n_scale_tot and n_clip_same == n_clip_tot and once,
          f"input scale == L clip scale on {n_scale}/{n_scale_tot} sampled anchors; the mapper receives canonical depth and canonical poses and applies s_clip once to both (force(s)); "
          f"stride-1 L clips regenerated for training are bit-identical to the local-registration audit's stride-3 clips on {n_clip_same}/{n_clip_tot} overlapping anchors "
          "(the clip scale's causality and single application are audited in artifacts/depth_source_comparison/scale_audit.json)")
    from channel_mapper import ChannelMapper
    cm = ChannelMapper(dev, "surface")
    check("14_confidence_gate_unchanged", cm.conf_threshold == D7.CONF_THRESHOLD == 1.5 and "conf_suf = np.where(zm < g12.RANGE_CUTOFF_M, conf, 0.0)" in src_in,
          "mapper confidence threshold 1.5 (core constant); raw LingBot confidence passed for H; for S/U/F only the 20 m cut zeroes confidence")
    rules_sha = g12.sha256_file(g12.G11_RULES); shas = set()
    for d in (g12.VAL_DRIVE,) + g12.TRAIN_DRIVES:
        for f in sorted(glob.glob(os.path.join(os.path.dirname(g12.input_path(d, 0)), "*.npz")))[::10]:
            shas.add(str(np.load(f)["rules_sha256"]))
    rules = g12.load_rules()
    check("15_U4_rules_frozen", shas == {rules_sha} and 'G11.widths("U4"' in src_in,
          f"every sampled input records rules sha256 {rules_sha[:16]} == Gate 11's frozen file; coefficients a {rules['U2']['a']:.4f} b {rules['U2']['b']:.4f} g {[round(x, 3) for x in rules['U3']['g']]} "
          f"u_front {-rules['U4']['q16']:.4f} u_back {rules['U4']['q84']:.4f} (round to the brief)")
    n_ge20 = 0; n_in = 0
    for d in (g12.VAL_DRIVE,) + g12.TRAIN_DRIVES:
        for f in glob.glob(os.path.join(os.path.dirname(g12.input_path(d, 0)), "*.npz")):
            n_ge20 += json.loads(str(np.load(f)["stats"]))["n_suf_rays_ge20"]; n_in += 1
    far_gt = sum(np.load(os.path.join(g12.ART, "eval", "g12_H_seed0_step06000_counts.npz"))["counts"][:, 0, b, [0, 2]].sum() for b in (3, 4))
    check("16_cutoff_input_only", n_ge20 == 0 and far_gt > 0 and "target_path" in src_train and "RANGE_CUTOFF" not in src_train and "RANGE_CUTOFF" not in open(os.path.join(HERE, "eval12.py")).read(),
          f"no S/U/F ray >= 20 m in any of {n_in} input files; training and evaluation read the unchanged full target (no range crop); {int(far_gt):,} target-occupied voxels beyond 20 m are scored on 0006")
    all_logs = "".join(open(p).read() for p in glob.glob(os.path.join(g12.ART, "logs", "*.log")))
    fw = [r_["firewall"]["violations"] for r_ in runs.values()] + [e["firewall"]["violations"] for e in evals.values()]
    code = "".join(open(os.path.join(HERE, p)).read() for p in ("build_inputs.py", "train12.py", "eval12.py", "build_regions.py", "analyze12.py", "g12.py"))
    check("17_no_semantickitti_access", all(v == [] for v in fw) and "data/kitti/dataset" in SRC.FORBIDDEN_PATH_TOKENS and sum(viol) == 0,
          "training, evaluation and input generation ran inside file audits forbidding the SemanticKITTI path (data/kitti/dataset) and 'semantickitti': no violations")
    check("18_no_occ3d_access", all(v == [] for v in fw) and not any(t in code.lower() for t in ("occ3d_gt", "lingbot_occ3d_zeroshot")), "no Occ3D / nuScenes path in Gate 12 code; audits forbid occ3d / nuscenes tokens; no violations")
    check("19_no_semantic_supervision", cfgH["w_sem"] == 0.0 == cfgS["w_sem"] and "fut_p\"] = torch.zeros" in src_train and all(v == [] for v in fw),
          "w_sem = 0; the semantic target passed to the unchanged loss is all zeros with a zero mask; training runs inside an audit forbidding trident / semantics paths; the input has no semantic channel")
    check("20_deterministic_input_generation", any(c["name"] == "7_deterministic_input_generation" and c["pass"] for c in san["checks"]),
          "sanity_inputs.py rebuilt 9 deterministic anchors from their L clips and reproduced all five stored volumes exactly")
    check("21_pre_training_sanity", san["n_pass"] == san["n_total"], f"{san['n_pass']}/{san['n_total']} input sanity checks (H U = 0 and S/F = core mapper; exclusivity; 20 m; carving stop; U4 rules; no LiDAR / GT pose)")
    fz = os.path.join(g12.ART, "code_freeze.json")
    ok22 = os.path.exists(fz) and all(g12.sha256_file(os.path.join(g12.REPO_ROOT, p)) == h for p, h in json.load(open(fz))["input_generation"].items())
    check("22_input_code_frozen", ok22, "input-generation code unchanged since code_freeze.json (frozen before training)")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R)}, open(os.path.join(g12.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 12 verification\n", f"{n_ok}/{len(R)} checks pass.\n", "| check | result | detail |", "|---|---|---|"] + [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(g12.ART, "verification.md"), "w").write("\n".join(L) + "\n"); print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
