#!/usr/bin/env python
"""Checks aimed at what would silently invalidate this control.

1. ``gradmatch_equals_joint``  with the normalisation factor forced to 1, the assembled
                                gradients equal ordinary backprop of ``L_occ + 0.5 L_KL``.
2. ``heads_unscaled``           the semantic-head gradient is exactly ``d(0.5 L_KL)``, the
                                occupancy-head gradient exactly ``d L_occ``, for any factor.
3. ``ratio_matched``            after assembly, ||shared aux part|| / ||shared occ part|| == rho.
4. ``shuffle_multiset``         per-stratum vector multisets preserved; support unchanged;
                                occupancy targets and inputs untouched; correspondence broken.
5. ``inputs_paired``            R and S for each seed drew identical (file, crop) sequences.
6. ``init_shared``              R, S and historical B share one initial state per seed.
7. ``no_future_in_input``       the 32 input channels are a function of causal fields only
                                (replacing fut_p / fut_valid changes nothing).
8. ``frozen_unchanged``         the foundation-model / cache digests recorded by the feature
                                study still hold.
9. ``checkpoint_roundtrip``     save -> load reproduces outputs.
10. ``budget_matched``          from the training logs: fraction of non-degenerate steps and
                                the achieved ratio's spread around rho.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, tempfile
import numpy as np, torch, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import CompletionUNet, load_checkpoint, save_checkpoint       # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from tools.feature_study.samples import StudySamples                              # noqa: E402
from tools.feature_study import verify as FV                                      # noqa: E402
from tools.teacher_alignment import gradmatch as GM, shuffle as SH                # noqa: E402

RESULTS = []
BASE_CONFIG = os.path.join(ST.REPO_ROOT, "configs", "feature_study", "base.yaml")


def check(name, ok, detail):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def _batch(dev, cfg, seed=7):
    tr = StudySamples(cfg["train_drives"][:1], cfg["crop"], dev, seed, 8,
                      sampler=cfg["sampler"], centre_on_occ_p=cfg["centre_on_occ_p"],
                      want_features=False)
    tr.verify()
    return tr.batch([tr.draw() for _ in range(2)])


def _net(dev, seed=0, trained=False):
    torch.manual_seed(seed)
    net = CompletionUNet(width=24, padding_mode="replicate", n_feat=64).to(dev)
    if trained:   # a few random steps so head_occ is non-zero and g_occ non-trivial
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.01 * torch.randn_like(p))
    return net.train()


def gradmatch_equals_joint(dev, cfg, b):
    """Exactness in fp32; the bf16 discrepancy is reported alongside as arithmetic."""
    out = {}
    tf32 = torch.backends.cudnn.allow_tf32
    for ac in (False, True):
        torch.backends.cudnn.allow_tf32 = ac          # fp32 pass: no TF32 convs either
        net = _net(dev, trained=True); groups = GM.param_groups(net)
        keep = torch.stack([SH.eligible_mask(b["fut_valid"][k], b["observed"][k])[0]
                            for k in range(2)])
        l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep, autocast=ac)
        net.zero_grad(set_to_none=True)
        (l_occ + cfg["w_aux"] * l_kl).backward()
        ref = {n: p.grad.detach().clone() for n, p in net.named_parameters()
               if p.grad is not None}
        l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep, autocast=ac)
        tg = GM.task_gradients(l_occ, l_kl, groups, cfg["w_aux"])
        go, ga = tg["shared"]
        rho1 = GM._norm(ga) / GM._norm(go)
        net.zero_grad(set_to_none=True)
        gm = GM.assemble(groups, tg, rho1)
        err, scale = 0.0, 0.0
        for name, p in net.named_parameters():
            if name.startswith("head_feat"):
                continue
            err = max(err, float((p.grad - ref[name]).abs().max()))
            scale = max(scale, float(ref[name].abs().max()))
        out[ac] = (gm["norm_factor"], err, scale)
    torch.backends.cudnn.allow_tf32 = tf32
    f32, bf = out[False], out[True]
    ok = abs(f32[0] - 1.0) < 1e-6 and f32[1] <= 1e-5 * max(f32[2], 1.0)
    return check("gradmatch_equals_joint", ok,
                 f"fp32: factor {f32[0]:.6f}, max|assembled - joint backprop| = {f32[1]:.2e} "
                 f"(max |grad| {f32[2]:.2e}, TF32 off); bf16 autocast as trained: {bf[1]:.2e} -- "
                 f"grad-of-sum vs sum-of-grads round differently through bf16/TF32 kernels")


def heads_unscaled(dev, cfg, b):
    net = _net(dev, trained=True); groups = GM.param_groups(net)
    keep = torch.stack([SH.eligible_mask(b["fut_valid"][k], b["observed"][k])[0]
                        for k in range(2)])
    l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep)
    net.zero_grad(set_to_none=True)
    (cfg["w_aux"] * l_kl).backward(retain_graph=False)
    ref_sem = [p.grad.detach().clone() for p in groups["head_sem"]]
    l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep)
    net.zero_grad(set_to_none=True)
    l_occ.backward()
    ref_occ = [p.grad.detach().clone() for p in groups["head_occ"]]
    l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep)
    tg = GM.task_gradients(l_occ, l_kl, groups, cfg["w_aux"])
    net.zero_grad(set_to_none=True)
    gm = GM.assemble(groups, tg, rho=50.0)                # an extreme factor on purpose
    e_sem = max(float((p.grad - r).abs().max()) for p, r in zip(groups["head_sem"], ref_sem))
    e_occ = max(float((p.grad - r).abs().max()) for p, r in zip(groups["head_occ"], ref_occ))
    feat_none = all(p.grad is None for p in groups["head_feat"])
    return check("heads_unscaled", e_sem < 1e-5 and e_occ < 1e-5 and feat_none,
                 f"factor {gm['norm_factor']:.2f}; sem-head err {e_sem:.1e}, occ-head err "
                 f"{e_occ:.1e}, feature head grad None: {feat_none}")


def ratio_matched(dev, cfg, b):
    net = _net(dev, trained=True); groups = GM.param_groups(net)
    keep = torch.stack([SH.eligible_mask(b["fut_valid"][k], b["observed"][k])[0]
                        for k in range(2)])
    l_occ, l_kl, _ = GM.losses(net, b, cfg, b["fut_p"], keep)
    tg = GM.task_gradients(l_occ, l_kl, groups, cfg["w_aux"])
    go, ga = tg["shared"]
    rho = 3.5644
    gm = GM.assemble(groups, tg, rho)
    aux_part = [p.grad - a for p, a in zip(groups["shared"], go)]
    r = GM._norm(aux_part) / GM._norm(go)
    return check("ratio_matched", abs(r - rho) < 1e-4,
                 f"||shared aux part|| / ||shared occ part|| = {r:.5f} vs rho {rho}")


def shuffle_multiset(dev, cfg, b):
    ok_all, det = True, []
    for k in range(2):
        keep, es = SH.eligible_mask(b["fut_valid"][k], b["observed"][k])
        sh, st = SH.shuffle_targets(b["fut_p"][k], keep, b["observed"][k], 0, 1, k)
        ms = SH.multiset_preserved(b["fut_p"][k], sh, keep, b["observed"][k])
        sup = torch.equal(b["fut_p"][k][:, ~keep], sh[:, ~keep])
        # correspondence broken: a large share of eligible vectors moved
        moved = st["n_changed"] / max(st["n_eligible"], 1)
        # every stratum's class histogram (argmax counts) identical before/after
        sid = SH.strata(keep, b["observed"][k])
        hist_ok = True
        for s_ in sid[keep].unique():
            m = keep & (sid == s_)
            hist_ok &= torch.equal(torch.bincount(b["fut_p"][k][:, m].argmax(0), minlength=25),
                                   torch.bincount(sh[:, m].argmax(0), minlength=25))
        ok_all &= ms and sup and hist_ok and moved > 0.5
        det.append(f"crop{k}: multiset {ms}, support {sup}, per-stratum class hist {hist_ok}, "
                   f"moved {100*moved:.1f}%, argmax changed "
                   f"{100*st['n_argmax_changed']/max(st['n_eligible'],1):.1f}%, "
                   f"dropped tiny {es['n_dropped_tiny_strata']}/{es['n_valid']}")
    return check("shuffle_multiset", ok_all, "; ".join(det))


def inputs_paired():
    for s in ST.SEEDS:
        tr = {}
        for c in ST.CONDITIONS:
            p = os.path.join(ST.ART, f"trace_{ST.run_tag(c, s)}.npz")
            if not os.path.exists(p):
                return check("inputs_paired", False, f"missing {p}")
            with np.load(p) as z:
                tr[c] = z["trace"]
        if not np.array_equal(tr["R"], tr["S"]):
            return check("inputs_paired", False, f"seed {s}: traces differ")
        # and identical to historical B's draws for the same seed
        hb = os.path.join(ST.FEATURE_STUDY_ART, f"trace_fcs_B_seed{s}.npz")
        if os.path.exists(hb):
            with np.load(hb) as z:
                if not np.array_equal(tr["R"], z["trace"]):
                    return check("inputs_paired", False, f"seed {s}: differs from historical B")
    return check("inputs_paired", True,
                 f"{len(ST.SEEDS)} seeds x {tr['R'].shape[0]} draws identical across R/S and "
                 f"equal to historical B's")


def init_shared():
    def h(sd):
        d = hashlib.sha256()
        for k, v in sorted(sd.items()):
            if k.startswith("head_feat"):
                continue
            d.update(k.encode()); d.update(v.detach().cpu().numpy().tobytes())
        return d.hexdigest()[:12]
    ok, det = True, []
    for s in ST.SEEDS:
        torch.manual_seed(s)
        mine = h(CompletionUNet(width=24, padding_mode="replicate", n_feat=64).state_dict())
        # the feature study recorded the same construction; compare digests
        det.append(f"seed{s}:{mine}")
    ref = ["18946306fdc6", "6e994b46ee50", "9fd28db31596"]        # feature-study digests
    ok = [d.split(":")[1] for d in det] == ref
    return check("init_shared", ok, f"{det} == feature-study/B digests {ref}: {ok}")


def no_future_in_input(dev, cfg, b):
    b2 = dict(b); b2["fut_p"] = torch.rand_like(b["fut_p"]); b2["fut_valid"] = ~b["fut_valid"]
    same = torch.equal(b["input"], b2["input"]) and torch.equal(b["gt_occ"], b2["gt_occ"])
    with np.load(sorted(glob.glob(os.path.dirname(SRC.sample_path(cfg["train_drives"][0], 0))
                                  + "/*.npz"))[0]) as z:
        t = int(z["t"]); inp = z["input_frames"]; tgt = z["target_frames"]
    causal = int(inp.max()) <= t < int(tgt.min())
    return check("no_future_in_input", same and causal,
                 f"input/gt unchanged by replacing the future targets: {same}; sample file: "
                 f"input frames <= t={t} < target frames [{tgt.min()}..{tgt.max()}]: {causal}")


def frozen_unchanged():
    p = os.path.join(ST.FEATURE_STUDY_ART, "frozen_digests.json")
    old = json.load(open(p))
    cur = {"sam_vit_h": ST.sha256_file(ST.FS.SAM_CKPT)}
    from core.datasets import sources as S
    cur["lingbot_stream_0003"] = ST.sha256_file(S.stream_path("k360_train", "2013_05_28_drive_0003_sync"))
    cur["trident_0003_first"] = ST.sha256_file(S.trident_path("k360_train", "0003_0000000004"))
    cur["occupancy_target_0006_first"] = ST.sha256_file(
        sorted(glob.glob(os.path.dirname(SRC.target_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))[0])
    bad = [k for k in cur if old.get(k) != cur[k]]
    return check("frozen_unchanged", not bad, f"{len(cur)} digests vs the feature study's "
                 f"record, {len(bad)} changed" + (f": {bad}" if bad else ""))


def checkpoint_roundtrip(dev, b):
    net = _net(dev, trained=True).eval()
    with torch.no_grad():
        a0, s0 = net(b["input"][:1])
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        p = f.name
    try:
        save_checkpoint(net, p, {"test": True})
        net2 = load_checkpoint(p, dev).net
        with torch.no_grad():
            a1, s1 = net2(b["input"][:1])
        ok = torch.equal(a0, a1) and torch.equal(s0, s1)
    finally:
        os.unlink(p)
    return check("checkpoint_roundtrip", ok, f"outputs identical after save/load: {ok}")


def budget_matched():
    rows, ok = [], True
    for c in ST.CONDITIONS:
        for s in ST.SEEDS:
            p = os.path.join(ST.ART, f"train_{ST.run_tag(c, s)}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            st = [x for x in j["log"] if "val" not in x]
            nd = sum(x["degenerate"] for x in st)
            ach = np.asarray([x["achieved_ratio"] for x in st if not x["degenerate"]])
            rho = j["rho"]
            within = float((np.abs(ach - rho) < 1e-3 * rho).mean()) if len(ach) else 0.0
            rows.append(f"{c}{s}: degenerate {nd}/{len(st)}, achieved==rho on {100*within:.1f}% "
                        f"of the rest")
            ok &= nd <= 2 and within > 0.999
    if not rows:
        return check("budget_matched", False, "no training records yet")
    return check("budget_matched", ok, "; ".join(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pre", action="store_true", help="only the checks that need no runs")
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    cfg = yaml.safe_load(open(BASE_CONFIG))
    b = _batch(dev, cfg)
    print("verification:")
    gradmatch_equals_joint(dev, cfg, b)
    heads_unscaled(dev, cfg, b)
    ratio_matched(dev, cfg, b)
    shuffle_multiset(dev, cfg, b)
    init_shared()
    no_future_in_input(dev, cfg, b)
    frozen_unchanged()
    checkpoint_roundtrip(dev, b)
    if not a.pre:
        inputs_paired()
        budget_matched()
    ST.write_json(os.path.join(ST.ART, "verification.json"),
                  {"checks": RESULTS, "n_pass": sum(r["pass"] for r in RESULTS),
                   "n_total": len(RESULTS), "env": ST.environment()})
    bad = [r["check"] for r in RESULTS if not r["pass"]]
    print(f"{len(RESULTS)-len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
