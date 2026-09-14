#!/usr/bin/env python
"""Gate 13 verification (22 required checks); writes verification.{md,json}.

    python tools/gate13_oracle/verify13.py --device cuda:0
"""
from __future__ import annotations
import argparse, datetime, glob, json, os, re, sys
import numpy as np, torch, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
from g13 import g12                                                                          # noqa: E402
sys.path.insert(0, g13.G9_DIR); sys.path.insert(0, g13.REPO_ROOT)
import build_oracle as BO                                                                    # noqa: E402
from variant_mapper import VariantMapper, table_state                                        # noqa: E402  (Gate 9)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.mapping.mapper import IncrementalMapper                                            # noqa: E402
from core.mapping.feed import CachedFeed                                                     # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    runs = {(c, s): json.load(open(os.path.join(g13.ART, "runs", f"train_g13_{c}_seed{s}.json"))) for c in g13.CONDITIONS for s in g13.SEEDS}
    g12r = {s: json.load(open(os.path.join(g13.G12_ART, "runs", f"train_g12_H_seed{s}.json"))) for s in g13.SEEDS}
    pair = lambda k: all(runs[("M", s)][k] == runs[("O", s)][k] for s in g13.SEEDS)
    an = json.load(open(os.path.join(g13.ART, "analysis.json"))); iq = json.load(open(os.path.join(g13.ART, "input_quality.json")))
    fz = json.load(open(os.path.join(g13.G12_ART, "code_freeze.json"))); t_fz12 = datetime.datetime.fromisoformat(fz["frozen_at"]).timestamp()
    m_files = [g13.m_path(d, t) for d, t in [tuple(x) for x in json.loads(json.dumps([[d, t] for d in g13.TRAIN_DRIVES for t in sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(g13.m_path(d, 0))))]))]]
    m_old = all(os.path.getmtime(f) < t_fz12 for f in m_files)
    check("01_M_is_gate12_H_input", m_old and all(runs[("M", s)]["draw_digest"] == g12r[s]["draw_digest"] and runs[("M", s)]["target_digest"] == g12r[s]["target_digest"] and
                                                runs[("M", s)]["train_items_sha256"] == g12r[s]["train_items_sha256"] for s in g13.SEEDS) and "g13.m_path" in open(os.path.join(HERE, "train13.py")).read(),
          f"M reads the Gate 12 H cache (S_H, F_H) unchanged since Gate 12's code freeze (all {len(m_files)} training inputs older than the freeze); M's anchor list, draws and target / loss-mask digests equal Gate 12 H's on every seed")
    mr = an["m_reproduction"]
    check("02_M_reproduces_gate12_H_validation", mr["pass"], f"0006 full IoU Gate 12 H {[round(x, 4) for x in mr['gate12_H']]} vs Gate 13 M {[round(x, 4) for x in mr['gate13_M']]}, diff {[round(x, 4) for x in mr['diff']]} (tolerance per seed {mr['tolerance']['per_seed_abs']}, mean {mr['tolerance']['mean_abs']}, fixed before training)")
    ev_ids = [np.load(os.path.join(g13.ART, "eval", f"g13_{c}_seed{s}_step06000_counts.npz"))["ids"] for c in g13.CONDITIONS for s in g13.SEEDS]
    check("03_identical_anchors", pair("train_items_sha256") and pair("n_train") and all(np.array_equal(ev_ids[0], x) for x in ev_ids), f"{runs[('M', 0)]['n_train']} training anchors (Gate 12's), identical per seed; all evaluations on the same {len(ev_ids[0])} 0006 anchors")
    check("04_same_target_tensor", pair("target_digest") and all(runs[("M", s)]["digests_by_checkpoint"] == runs[("O", s)]["digests_by_checkpoint"] for s in g13.SEEDS),
          "sha256 over every drawn target crop (occupancy, valid, loss-weight mask) identical for M and O at every 1000-step checkpoint")
    check("05_same_crops", pair("draw_digest"), "sha256 over every (anchor, crop origin) identical per seed: " + ", ".join(runs[("M", s)]["draw_digest"][:12] for s in g13.SEEDS))
    src_t = open(os.path.join(HERE, "train13.py")).read() + open(os.path.join(g13.G12_DIR, "train12.py")).read()
    check("06_same_augmentations", not any(k in src_t.lower() for k in ("flip", "rot90", "jitter", "augment")) and pair("draw_digest"), "no augmentation exists in the training path; crop RNG draws identical")
    check("07_same_initial_weights", all(runs[("M", s)]["init_state_digest"] == runs[("O", s)]["init_state_digest"] == g12r[s]["init_state_digest"] for s in g13.SEEDS),
          "per seed M, O and Gate 12 H load the same saved initial weights (identical state digests)")
    check("08_same_optimizer", pair("optimizer") and pair("scheduler"), f"{runs[('M', 0)]['optimizer']}, {runs[('M', 0)]['scheduler']}")
    lr = lambda r: [x["lr"] for x in r["log"] if "lr" in x]
    check("09_same_lr_sequence", all(lr(runs[("M", s)]) == lr(runs[("O", s)]) for s in g13.SEEDS), "identical learning-rate value at every step for M and O")
    arch, pc = set(), set()
    for r in runs.values():
        ck = torch.load(os.path.join(REPO_ROOT, r["final_checkpoint"]), map_location="cpu", weights_only=False)
        arch.add((ck["cin"], ck["width"], ck["padding_mode"], tuple((k, tuple(v.shape)) for k, v in ck["state_dict"].items()))); pc.add(r["n_params"])
    check("10_same_architecture", len(arch) == 1, f"CompletionUNet cin 3 width 24 replicate padding; identical parameter names and shapes in all six checkpoints")
    check("11_same_parameter_count", len(pc) == 1 and pc == {967322}, f"{sorted(pc)} parameters")
    check("12_same_training_steps", pair("steps") and all(r["steps"] == 6000 for r in runs.values()), "6000 steps, batch 4")
    cM, cO = yaml.safe_load(open(os.path.join(g13.ART, "training_config_M.yaml"))), yaml.safe_load(open(os.path.join(g13.ART, "training_config_O.yaml")))
    cH = yaml.safe_load(open(os.path.join(g13.G12_ART, "training_config_H.yaml")))
    dk = sorted(k for k in set(cM) | set(cO) if cM.get(k) != cO.get(k)); dk12 = sorted(k for k in set(cM) | set(cH) if cM.get(k) != cH.get(k))
    evs = [json.load(open(os.path.join(g13.ART, "eval", f"g13_{c}_seed{s}_step06000.json"))) for c in g13.CONDITIONS for s in g13.SEEDS]
    check("13_same_checkpoint_rule", dk == ["input"] and dk12 == ["input"] and all(e["checkpoint"].endswith("step06000.pt") for e in evs),
          f"configs M / O differ only in {dk} and M differs from Gate 12 H only in {dk12}; every evaluated checkpoint is the final step")
    o_files = sorted(glob.glob(os.path.join(g13.O_ROOT, "*", "*.npz"))); n_ok14 = 0; max_future = -99
    natmap = {d: [int(f.order) for f in [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0].frames] for d in (g13.VAL_DRIVE,) + g13.TRAIN_DRIVES}
    for f in o_files:
        z = np.load(f); t = int(z["t"]); d = f.split("/")[-2]; si = z["stream_indices"].tolist()
        n_ok14 += int(si == [t - 4, t - 3, t - 2, t - 1, t] and z["natives"].tolist() == [natmap[d][j] for j in si]); max_future = max(max_future, max(si) - t)
    n_m = sum(len(os.listdir(os.path.dirname(g13.m_path(d, 0)))) for d in (g13.VAL_DRIVE,) + g13.TRAIN_DRIVES)
    check("14_O_only_sweeps_t-4_to_t", n_ok14 == len(o_files) == n_m, f"all {len(o_files)} oracle inputs (one per Gate 12 anchor, {n_m}) record stream indices t-4..t and exactly those frames' native sweeps")
    src_o = open(os.path.join(HERE, "build_oracle.py")).read()
    # Amended after the first verification run: the bare-word scan for "future" matched `from __future__ import annotations`
    # and the docstring. The scan now looks for future-sweep identifiers in code.
    fut_ids = [k for k in ("future_natives", "future_stream_indices", "fut_", "t + 1", "FUTURE_FRAMES") if k in src_o]
    check("15_O_no_future_sweeps", max_future == 0 and "idx = [t - 4, t - 3, t - 2, t - 1, t]" in src_o and not fut_ids,
          f"largest recorded input stream index minus anchor = {max_future}; the builder's only sweep list is t-4..t")
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(g13.ART, "logs", "oracle_*.log"))); viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]
    check("16_O_no_target_voxels", len(viol) >= 5 and sum(viol) == 0 and all(k not in src_o for k in ("load_target", "rawtarget", "target_path", "gt_occ")) and "lingbot_k360_7drives/targets" in BO.AUDIT_PATTERNS,
          f"the oracle builder never reads a target, evaluation mask, region or label file (strict file audit, violations {viol}); its only geometry is the five sweeps' camera z-buffer through the mapper")
    n_pose = 0; smp = o_files[::97]
    for f in smp:
        z = np.load(f); d = f.split("/")[-2]; t = int(z["t"]); geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Gt = geo.rect_cam_to_world(natmap[d][t])
        n_pose += int(all(np.allclose(z["T_j_to_t"][q], TF.inv(Gt) @ geo.rect_cam_to_world(natmap[d][j]), atol=1e-12) for q, j in enumerate(z["stream_indices"])))
    check("17_O_GT_relative_poses_input_only", n_pose == len(smp) and "rect_cam_to_world" not in open(os.path.join(g13.G12_DIR, "build_inputs.py")).read(),
          f"oracle T_j->t equal GT inv(rect_cam_to_world(t)) rect_cam_to_world(j) on {n_pose}/{len(smp)} sampled inputs; GT poses enter only the oracle input builder (M inputs contain none)")
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == g13.VAL_DRIVE][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = IncrementalMapper(dev, sem_into=None, n_teacher=0); m1 = VariantMapper(dev, 0.2, 0.2, True)
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr = feed.frame(j); m0.step(fr); m1.step(fr)
    s0, s1 = table_state(m0), table_state(m1); eqm = all(np.array_equal(s0[k], s1[k]) for k in s0)
    check("18_O_same_hard_mapper", eqm and src_o.count("VariantMapper(dev, 0.2, 0.2, True)") == 1 and iq["O_equals_gate9_LD_map"][0] == iq["O_equals_gate9_LD_map"][1] == 590,
          f"O uses VariantMapper(band 0.2, margin 0.2, carving) == core IncrementalMapper bit for bit (the mapper of M's S_H/F_H); O's occupied map equals Gate 9's LD map on {iq['O_equals_gate9_LD_map']} 0006 anchors")
    check("19_same_grid", TF.GRID.dims == (256, 256, 32) and float(TF.GRID.voxel_size) == 0.2 and "TF.GRID" in src_o, "both inputs queried on the SSCBench-KITTI-360 anchor grid 256x256x32 at 0.2 m (TF.GRID)")
    fw = [r["firewall"]["violations"] for r in runs.values()] + [e["firewall"]["violations"] for e in evs]
    check("20_no_semantic_information", cM["w_sem"] == 0.0 == cO["w_sem"] and "fut_p\"] = torch.zeros" in src_t and "semantics" in BO.AUDIT_PATTERNS and all(v == [] for v in fw),
          "w_sem = 0, zero semantic target and mask, no semantic input channel; training and oracle building audited against trident / semantics paths")
    check("21_no_target_dataset_access", all(v == [] for v in fw) and sum(viol) == 0, "training, evaluation and oracle input generation ran inside file audits forbidding SemanticKITTI / Occ3D / nuScenes paths: no violations")
    geo = TF.DriveGeometry(g13.VAL_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kl = BO.lattice_K(np.asarray(geo.calib.K, np.float64))
    v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); Tgw = np.linalg.inv(np.asarray(geo.rect_cam_to_velo, np.float64)); same = []
    for t in (4, 766, 1747):
        S_O, F_O, _, _ = BO.build_one(dev, geo, natmap[g13.VAL_DRIVE], t, Kl, v2c, Tgw); z = np.load(g13.o_path(g13.VAL_DRIVE, t)); N = S_O.size
        same.append(bool(np.array_equal(np.packbits(S_O), z["S_O"]) and np.array_equal(np.packbits(F_O), z["F_O"])))
    check("22_deterministic_input_generation", all(same), f"rebuilding oracle inputs for 0006 anchors 4 / 766 / 1747 reproduces the stored S_O and F_O exactly: {same}")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R)}, open(os.path.join(g13.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 13 verification\n", f"{n_ok}/{len(R)} checks pass.\n", "| check | result | detail |", "|---|---|---|"] + [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(g13.ART, "verification.md"), "w").write("\n".join(L) + "\n"); print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
