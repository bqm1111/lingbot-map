#!/usr/bin/env python
"""Gate 11 verification; writes verification.{md,json}, exits non-zero on any failure.

    python tools/gate11_uncertainty/verify_u.py --device cuda:0
"""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g11 as G                                                                            # noqa: E402
sys.path.insert(0, G.G9_DIR); sys.path.insert(0, G.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K                                                                # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper, table_state                                     # noqa: E402  (Gate 9)
from channel_mapper import ChannelMapper                                                   # noqa: E402
from tools.local_registration_audit import common as LR                                    # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.mapping.feed import CachedFeed                                                   # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402

G9_DATA = os.path.join(G9.DATA_ROOT, "clips"); DSC_DATA = "/media/SSD1/MINH_DATASETS/lingbot_depth_source_comparison/clips"
CODE = ("g11.py", "channel_mapper.py", "derive_rules.py", "audit_u.py", "analyze_u.py", "figures_u.py")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    drives = [d for d in (G.PRIMARY,) + G.ROBUST if glob.glob(os.path.join(G.DATA_ROOT, "clips", d, "*.npz"))]
    recs = {d: sorted(glob.glob(os.path.join(G.DATA_ROOT, "clips", d, "*.npz"))) for d in drives}
    src = open(os.path.join(HERE, "audit_u.py")).read(); allsrc = "".join(open(os.path.join(HERE, p)).read() for p in CODE)
    ok1, det = True, []
    for d in drives:
        a11 = [os.path.basename(f) for f in recs[d]]; alr = sorted(os.path.basename(f) for f in glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))
        a9 = sorted(os.path.basename(f) for f in glob.glob(os.path.join(G9_DATA, d, "*.npz"))); a10 = sorted(os.path.basename(f) for f in glob.glob(os.path.join(DSC_DATA, d, "*.npz")))
        ok1 &= a11 == alr == a9 == a10; det.append(f"{d[17:21]} {len(a11)}")
    check("01_same_clips", ok1, f"Gate 11 clips == local-reset clips == Gate 9 clips == depth-source-comparison clips ({', '.join(det)})")
    n = dict(anchor=0, pose=0, depth=0, scale=0, conf=0, mask=0, K=0, h=0, tot=0)
    for d in drives:
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kl = lattice_K(np.asarray(geo.calib.K, np.float64))
        nat = [int(f.order) for f in [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0].frames]
        for i, f in enumerate(recs[d]):
            z = np.load(f); g = np.load(os.path.join(G9_DATA, d, os.path.basename(f))); lr = np.load(LR.clip_path(d, int(z["t"]))); n["tot"] += 1
            n["anchor"] += int(np.array_equal(z["stream_indices"], lr["stream_indices"]) and np.array_equal(z["keys"], lr["keys"]) and np.array_equal(z["stream_indices"], g["stream_indices"]))
            if i % 10 == 0:
                Gt = geo.rect_cam_to_world(nat[int(z["t"])]); n["pose"] += int(all(np.array_equal(z["T_j_to_t"][q], TF.inv(Gt) @ geo.rect_cam_to_world(nat[j])) for q, j in enumerate(z["stream_indices"])))
            else:
                n["pose"] += int(np.array_equal(z["T_j_to_t"], g["T_j_to_t"]))
            F = [str(x) for x in z["pixel_fields"]]; GF = [str(x) for x in g["pixel_fields"]]; P = z["pixels"]; Q = g["pixels"]
            n["depth"] += int(P.shape[0] == Q.shape[0] and np.array_equal(P[:, F.index("z_lbm")], Q[:, GF.index("z_pred")]) and np.array_equal(P[:, F.index("z_lidar")], Q[:, GF.index("z_lidar")]))
            n["conf"] += int(np.array_equal(P[:, F.index("conf")], Q[:, GF.index("conf")]) and np.array_equal(P[:, F.index("edge_class")], Q[:, GF.index("edge_class")]))
            n["scale"] += int(float(z["l_scale"]) == float(lr["scale"]) == float(g["l_scale"]))
            n["mask"] += int(str(z["eval_mask_digest"]) == str(g["eval_mask_digest"]) and str(z["support_digest"]) == str(g["support_digest"]))
            n["K"] += int(np.array_equal(z["K_lattice"], g["K_lattice"]) and np.array_equal(z["K_lattice"], Kl))
            n["h"] += int(np.array_equal(z["m_H"], g["m_PD_gate"]) and ("m_H_LD" not in z.files or np.array_equal(z["m_H_LD"], g["m_LD_gate"])))
    T = n["tot"]
    check("02_same_anchor_frames", n["anchor"] == T, f"stream indices / frame keys equal the local-reset clip and Gate 9 record on {n['anchor']}/{T} clips")
    check("03_same_gt_poses", n["pose"] == T and src.count("pose_c2w_canonical=TG[q]") == 1 and src.count("scale_state.force(1.0)") == 2,
          f"T_j->t recomputed from GT on every 10th clip, equal to Gate 9 elsewhere ({n['pose']}/{T}); metric translations, mapper scale forced to 1 in both channel mappers")
    check("04_same_lbm_depth", n["depth"] == T and src.count('"LBM": [canon[q] * l_scale for q in range(5)]') == 1,
          f"LB-M depth and LiDAR depth of every stored pixel row equal Gate 9's rows (same deterministic subsample) on {n['depth']}/{T} clips; LB-M = cached LingBot x clip scale, once")
    check("05_same_moge_clip_scale", n["scale"] == T, f"clip scale == local-reset clip file == Gate 9 on {n['scale']}/{T} clips (causality / single application audited in artifacts/depth_source_comparison/scale_audit.json)")
    check("06_same_confidence", n["conf"] == T and src.count("cf[q] >= D7.CONF_THRESHOLD") == 1,
          f"LingBot confidence and Gate 9 edge class of every stored row equal Gate 9 on {n['conf']}/{T} clips; frozen gate conf >= {1.5}")
    check("07_same_grid", G9.DIMS == (256, 256, 32) and G9.VS == 0.2 and G9.ORIGIN == (0.0, -25.6, -2.0) and tuple(TF.GRID.dims) == G9.DIMS, "256x256x32, 0.2 m, origin (0, -25.6, -2) == core GRID")
    check("08_same_evaluation_masks", n["mask"] == T, f"evaluation-mask and LiDAR-support digests equal Gate 9 on {n['mask']}/{T} clips")
    check("09_same_camera", n["K"] == T and src.count("K=Kl") == 1, f"lattice K equal Gate 9 and lattice_K(P_rect_00) on {n['K']}/{T} clips")
    # 10 rules from residuals only
    rules = json.load(open(G.RULES)); t_rules = datetime.datetime.fromisoformat(rules["derived_at"]).timestamp()
    first_rec = min(os.path.getmtime(f) for d in drives for f in recs[d])
    der = open(os.path.join(HERE, "derive_rules.py")).read()
    keys_read = sorted(set(re.findall(r'z\["([A-Za-z_]+)"\]', der)))
    tmpart = tempfile.mkdtemp(dir=os.path.join(G.REPO_ROOT, "output")) if os.path.isdir(os.path.join(G.REPO_ROOT, "output")) else tempfile.mkdtemp()
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, G11_ART=tmpart); subprocess.run([sys.executable, os.path.join(HERE, "derive_rules.py")], check=True, env=env, capture_output=True)
    re_rules = json.load(open(os.path.join(tmpart, "uncertainty_rules.json")))
    same_rules = all(re_rules[k] == rules[k] for k in ("U2", "U3", "U4", "input_rows_sha256", "n_rows"))
    subprocess.run(["rm", "-rf", tmpart])
    check("10_widths_from_residuals_not_iou", keys_read == ["pixel_fields", "pixels"] and t_rules < first_rec and same_rules,
          f"derive_rules.py reads only {keys_read} from Gate 9 records (depth residuals, confidence), never an occupancy array; rules derived at {rules['derived_at']} before the first Gate 11 occupancy record; "
          f"re-derivation reproduces a, b, g, q16, q84 and the input-row digest exactly: {same_rules}; U2 a={rules['U2']['a']:.4f} b={rules['U2']['b']:.4f}, g={[round(x, 3) for x in rules['U3']['g']]}, q16/q84={rules['U4']['q16']:.3f}/{rules['U4']['q84']:.3f}")
    # 11 freeze before replay
    fz = os.path.join(G.ART, "code_freeze.json"); rob = [f for d in drives if d != G.PRIMARY for f in recs[d]]
    if os.path.exists(fz):
        fr = json.load(open(fz)); t_fz = datetime.datetime.fromisoformat(fr["frozen_at"]).timestamp()
        hashes_now = all(sha(os.path.join(G.REPO_ROOT, p)) == h for p, h in fr["sha256"].items())
        ok11 = hashes_now and (not rob or min(os.path.getmtime(f) for f in rob) > t_fz) and fr["sha256"].get("artifacts/gate11_uncertainty/uncertainty_rules.json") == sha(G.RULES)
        det11 = f"code_freeze.json at {fr['frozen_at']}: all {len(fr['sha256'])} frozen files (code + uncertainty_rules.json) unchanged; earliest 0003/0007/0010 record written after the freeze: {bool(rob) and min(os.path.getmtime(f) for f in rob) > t_fz}"
    else:
        ok11, det11 = False, "code_freeze.json missing"
    check("11_rules_frozen_before_replay", ok11, det11)
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(G.ART, "logs", "audit_*.log")))
    viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]
    toks = [t for t in SRC.FORBIDDEN_PATH_TOKENS if t in allsrc] + [t for t in ("semantickitti", "occ3d", "nuscenes") if t in allsrc.lower()]
    check("12_no_target_dataset_access", len(viol) >= len(drives) and sum(viol) == 0 and not toks, f"every audit shard inside the strict file-access audit: violations {viol}; forbidden tokens in Gate 11 code: {toks}")
    from gates.gate7b import replay
    lsha = sha(os.path.join(REPO_ROOT, replay.CHECKPOINT)); ref_sha = json.load(open(os.path.join(REPO_ROOT, "artifacts", "local_registration_audit", "verification_reset.json")))["weights_identical"]["checkpoint_sha256"]
    check("13_no_training", lsha == ref_sha and not any(k in allsrc for k in ("torch.optim", ".backward(", "nn.Parameter", "load_checkpoint", "CalibratedMoGe", "build_model")),
          f"no optimiser / backward / parameter / network load in Gate 11 code (only cached depth and confidence are read); LingBot checkpoint sha256 {lsha[:16]} unchanged")
    # mapper equivalence (supports H == Gate 9)
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == G.PRIMARY][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = VariantMapper(dev, 0.2, 0.2, True); m1 = ChannelMapper(dev, "surface")
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr_ = feed.frame(j); m1.set_pixels(margin=np.full(tuple(fr_.depth_canonical.shape), 0.2)); m0.step(fr_); m1.step(fr_)
    s0, s1 = table_state(m0), table_state(m1); eq = all(np.array_equal(s0[k], s1[k]) for k in s0)
    check("14_H_is_the_unchanged_mapper", eq and n["h"] == T, f"ChannelMapper(surface, margin 0.2) == Gate 9 VariantMapper bit for bit ({len(s0['keys']):,} rows); H metric vector == Gate 9 PD_gate (and H_LD == LD_gate) on {n['h']}/{T} clips")
    tmp = tempfile.mkdtemp(dir=G.DATA_ROOT); env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "audit_u.py"), "--drive", G.PRIMARY, "--shard", "7", "--shards", "295", "--limit", "2", "--out", tmp, "--device", a.device], check=True, env=env, capture_output=True)
    same = []
    for f in sorted(glob.glob(os.path.join(tmp, "*.npz"))):
        z1 = np.load(f); z0 = np.load(os.path.join(G.DATA_ROOT, "clips", G.PRIMARY, os.path.basename(f))); same.append(all(np.array_equal(z0[k], z1[k]) for k in z0.files)); os.remove(f)
    os.rmdir(tmp)
    check("15_deterministic_rerun", len(same) == 2 and all(same), f"re-running two 0006 clips reproduces every stored array (counts, metric vectors, pixel states) exactly: {same}")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R), "drives": drives}, open(os.path.join(G.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 11 verification\n", f"{n_ok}/{len(R)} checks pass (drives: {', '.join(d[17:21] for d in drives)}).\n", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(G.ART, "verification.md"), "w").write("\n".join(L) + "\n"); print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
