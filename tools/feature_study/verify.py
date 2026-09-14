#!/usr/bin/env python
"""Implementation checks aimed at the failure modes that would silently invalidate the study.

Each check prints PASS/FAIL and the evidence it used; the exit status is non-zero if any
fails. These are about *this* study's real risks, not a general test suite:

1. ``inputs_paired``      A/B/C for one seed draw the same files at the same crops.
2. ``init_shared``        the three conditions start from bit-identical shared parameters.
3. ``aux_reaches_trunk``  the B and C auxiliary losses produce non-zero gradient at the
                          first encoder convolution -- i.e. they are not sidecars.
4. ``frozen_unchanged``   SAM-H and the frozen cache files hash to what they hashed before.
5. ``alignment``          feature/image/voxel alignment survives preprocessing and cropping:
                          the sparse crop gather equals a dense crop, and a voxel's target
                          feature matches the SAM feature at its own projected pixel far
                          better than at a shuffled one.
6. ``gt_not_in_input``    replacing the ground truth changes no network input or output.
7. ``head_sparse_dense``  the sparse feature read equals the dense 1x1x1 convolution.
8. ``checkpoint_roundtrip`` save -> load reproduces predictions exactly.
9. ``unused_heads_frozen`` a head with no loss term is bit-identical to its initialisation
   in the trained checkpoint.

    python tools/feature_study/verify.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, tempfile
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets import grids as G6G, kitti360_splits as SRC, union_vocab as V8  # noqa: E402
from core.datasets import kitti360_transforms as TF                               # noqa: E402
from core.datasets import sources as S                                            # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.model import losses as L                                                # noqa: E402
from core.model.net import CompletionUNet, apply_residual, load_checkpoint, save_checkpoint  # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from tools.feature_study import feature_targets as FT, study as ST                # noqa: E402
from tools.feature_study.losses import feature_cosine, features_at                # noqa: E402
from tools.feature_study.samples import StudySamples                              # noqa: E402
from tools.feature_study.sam_teacher import lattice_grid, to_lattice              # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
def inputs_paired():
    for s in ST.SEEDS:
        tr = {}
        for c in ST.CONDITIONS:
            p = os.path.join(ST.ART, f"trace_{ST.run_tag(c, s)}.npz")
            if not os.path.exists(p):
                return check("inputs_paired", False, f"missing trace {p}")
            with np.load(p) as z:
                tr[c] = z["trace"]
        if not (np.array_equal(tr["A"], tr["B"]) and np.array_equal(tr["A"], tr["C"])):
            d = int((tr["A"] != tr["B"]).sum() + (tr["A"] != tr["C"]).sum())
            return check("inputs_paired", False, f"seed {s}: {d} differing trace entries")
    n = tr["A"].shape[0]
    return check("inputs_paired", True,
                 f"{len(ST.SEEDS)} seeds x {n} (file, crop) draws identical across A/B/C")


def init_shared():
    def h(net):
        d = hashlib.sha256()
        for k, v in sorted(net.state_dict().items()):
            if k.startswith("head_feat"):
                continue
            d.update(k.encode()); d.update(v.detach().cpu().numpy().tobytes())
        return d.hexdigest()
    ok, ds = True, []
    for s in ST.SEEDS:
        hs = []
        for _ in ST.CONDITIONS:
            torch.manual_seed(s)
            hs.append(h(CompletionUNet(width=24, padding_mode="replicate",
                                       n_feat=ST.FEATURE_DIM)))
        ok &= len(set(hs)) == 1
        ds.append(hs[0][:12])
    return check("init_shared", ok, f"shared-parameter digests per seed: {ds}")


def aux_reaches_trunk(dev, sample):
    d = sample
    out = {}
    for cond in ("B", "C"):
        torch.manual_seed(0)
        net = CompletionUNet(width=24, padding_mode="replicate",
                             n_feat=ST.FEATURE_DIM).to(dev)
        occ_res, sem_log, trunk = net(d["input"][None], return_trunk=True)
        if cond == "B":
            l = L.semantic_kl(sem_log.float(), d["fut_p_b"], d["fut_valid_b"])
        else:
            l = feature_cosine(features_at(net, trunk, d["aux_index"]), d["aux_target"])
        g = torch.autograd.grad(l, net.enc1[0].weight, retain_graph=False)[0]
        out[cond] = float(g.norm())
    ok = out["B"] > 0 and out["C"] > 0
    return check("aux_reaches_trunk", ok,
                 f"||d L_aux / d enc1.weight||: B {out['B']:.3e}, C {out['C']:.3e} "
                 f"(both must be > 0; the trunk is shared, not bypassed)")


def unused_heads_frozen(dev):
    """A head that contributes no loss must not have moved.

    ``opt.zero_grad(set_to_none=True)`` leaves those parameters with ``grad=None`` and AdamW
    skips them, so weight decay never reaches them either -- but that is a claim about the
    optimizer, and the point of a check is not to take the claim on trust. The feature head
    must be at initialisation in A and B, and the 25-class head must be at initialisation in
    A and C.
    """
    expect = {"A": ["head_feat", "head_sem"], "B": ["head_feat"], "C": ["head_sem"]}
    moved, checked = [], 0
    for c in ST.CONDITIONS:
        for sd in ST.SEEDS:
            p_ = ST.checkpoint_path(c, sd)
            if not os.path.exists(p_):
                continue
            torch.manual_seed(sd)
            ref = CompletionUNet(width=24, padding_mode="replicate",
                                 n_feat=ST.FEATURE_DIM).state_dict()
            got = torch.load(p_, map_location="cpu", weights_only=False)["state_dict"]
            for pre in expect[c]:
                for k in [k for k in ref if k.startswith(pre)]:
                    checked += 1
                    if not torch.equal(ref[k].cpu(), got[k].cpu()):
                        moved.append(f"{c}_seed{sd}:{k}")
    return check("unused_heads_frozen", not moved and checked > 0,
                 f"{checked} untrained-head tensors compared against their initialisation, "
                 f"{len(moved)} moved" + (f": {moved[:4]}" if moved else ""))


def frozen_unchanged():
    p = os.path.join(ST.ART, "frozen_digests.json")
    cur = {"sam_vit_h": ST.sha256_file(ST.SAM_CKPT)}
    # a deterministic sample of the frozen caches the study consumes read-only
    for name, path in (("lingbot_stream_0003",
                        S.stream_path("k360_train", "2013_05_28_drive_0003_sync")),
                       ("trident_0003_first",
                        S.trident_path("k360_train", "0003_0000000004")),
                       ("occupancy_target_0006_first",
                        next(iter(sorted(glob.glob(os.path.dirname(
                            SRC.target_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))), "")),
                       ("feature_target_0006_first",
                        next(iter(sorted(glob.glob(os.path.dirname(
                            ST.feature_target_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))), "")),
                       ("pca_basis", ST.pca_path())):
        if os.path.exists(path):
            cur[name] = ST.sha256_file(path)
    if not os.path.exists(p):
        ST.write_json(p, cur)
        return check("frozen_unchanged", True,
                     f"recorded {len(cur)} digests for the first time: {list(cur)}")
    old = json.load(open(p))
    bad = [k for k in old if k in cur and old[k] != cur[k]]
    return check("frozen_unchanged", not bad,
                 f"{len(set(old) & set(cur))} digests compared, {len(bad)} changed"
                 + (f": {bad}" if bad else ""))


def alignment(dev, drive="2013_05_28_drive_0003_sync"):
    """Sparse-vs-dense crop gather, and a projected-pixel positive control."""
    files = sorted(glob.glob(os.path.dirname(ST.feature_target_path(drive, 0)) + "/*.npz"))
    if not files:
        return check("alignment", False, "no feature targets built")
    idx = int(os.path.basename(files[0])[:-4])
    a = FT.load(files[0])
    X, Y, Z = a["dims"]
    # -- (a) the crop gather the trainer uses equals a dense crop of the same volume
    cx, cy, cz = 128, 128, 32
    rng = np.random.default_rng(0)
    x0, y0, z0 = int(rng.integers(X - cx + 1)), int(rng.integers(Y - cy + 1)), 0
    dense = np.zeros((X, Y, Z, ST.FEATURE_DIM), np.float32)
    dense.reshape(-1, ST.FEATURE_DIM)[a["feat_rows"]] = a["feat"]
    sub = dense[x0:x0 + cx, y0:y0 + cy, z0:z0 + cz].reshape(-1, ST.FEATURE_DIM)
    r = a["feat_rows"]
    xx = r // (Y * Z); yy = (r // Z) % Y; zz = r % Z
    keep = ((xx >= x0) & (xx < x0 + cx) & (yy >= y0) & (yy < y0 + cy)
            & (zz >= z0) & (zz < z0 + cz))
    local = (((xx[keep] - x0) * cy) + (yy[keep] - y0)) * cz + (zz[keep] - z0)
    sparse_ok = np.array_equal(sub[local], a["feat"][keep]) and \
        int((np.abs(sub).sum(1) > 0).sum()) == int(keep.sum())
    # -- (b) positive control: a voxel's target must match the SAM feature at its own
    #        projected pixel much better than at a shuffled pixel
    src = SRC.source_of(drive)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == drive][0]
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    with np.load(SRC.sample_path(drive, idx)) as z:
        scale = float(z["scale"])
    MAP = G6G.PREDICTION_GRID["kitti360"]
    proj = FT.Projection(dev)
    j = idx + 1                                            # the first future frame
    with np.load(ST.sam_feature_path(src, seg.frames[j].key)) as z:
        feat = torch.from_numpy(z["feat"].astype(np.float32)).to(dev)
        nhw = tuple(int(v) for v in z["native_hw"]); ihw = tuple(int(v) for v in z["input_hw"])
    with np.load(S.stream_path(src, drive)) as z:
        K = z["pred_K"][j].astype(np.float64)
        pose = z["pred_pose_c2w"][j].astype(np.float64)
        hp, wp = z["pred_depth"].shape[1], z["pred_depth"].shape[2]
    lat = to_lattice(proj.apply(feat), lattice_grid(nhw, ihw, (hp, wp), dev))   # [64, hp, wp]
    # voxel centres of the anchor grid -> world (scaled) -> future camera -> lattice pixel
    sel = a["feat_rows"][:: max(1, len(a["feat_rows"]) // 4000)]
    xv = sel // (Y * Z); yv = (sel // Z) % Y; zv = sel % Z
    c = (np.stack([xv, yv, zv], 1) + 0.5) * MAP.voxel_size + np.asarray(MAP.origin)
    P = pose.copy(); P[:3, 3] *= scale                     # future camera -> world
    T_cam_to_grid = np.asarray(geo.rect_cam_to_velo, np.float64)
    # grid -> world with the ANCHOR pose (the grid is anchored there), then world -> the
    # FUTURE camera with its own pose: the same chain the fusion walked, run backwards.
    with np.load(S.stream_path(src, drive)) as z:
        Pa = z["pred_pose_c2w"][idx].astype(np.float64).copy()
    Pa[:3, 3] *= scale
    T_grid_to_world = Pa @ np.linalg.inv(T_cam_to_grid)
    w = c @ T_grid_to_world[:3, :3].T + T_grid_to_world[:3, 3]
    cam = (w - P[:3, 3]) @ P[:3, :3]                        # world -> future camera
    front = cam[:, 2] > 0.5
    uv = (cam[front] @ K.T)
    u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
    ins = (u >= 0) & (u < wp - 1) & (v >= 0) & (v < hp - 1)
    ui = np.clip(u[ins].astype(int), 0, wp - 1); vi = np.clip(v[ins].astype(int), 0, hp - 1)
    tgt = torch.from_numpy(a["feat"][:: max(1, len(a["feat_rows"]) // 4000)][front][ins]).to(dev)
    pix = lat[:, vi, ui].T
    pix = pix / pix.norm(dim=1, keepdim=True).clamp_min(1e-9)
    true_cos = float((pix * tgt).sum(1).mean())
    perm = torch.randperm(pix.shape[0], device=dev)
    shuf_cos = float((pix[perm] * tgt).sum(1).mean())
    ok = sparse_ok and (true_cos > shuf_cos + 0.05) and tgt.shape[0] > 200
    return check("alignment", ok,
                 f"sparse crop gather exact: {sparse_ok}; projected-pixel cosine "
                 f"{true_cos:.4f} vs shuffled {shuf_cos:.4f} on {tgt.shape[0]} voxels")


def gt_not_in_input(dev, path):
    with np.load(path) as z:
        d0 = TG.unpack_sample(z, dev)
        raw = {k: z[k] for k in z.files}
    raw["gt_occ"] = np.packbits(np.ones(int(np.prod([int(x) for x in raw["dims"]])), bool))
    raw["gt_valid"] = np.packbits(np.zeros(int(np.prod([int(x) for x in raw["dims"]])), bool))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, **raw); tmp = f.name
    try:
        with np.load(tmp) as z:
            d1 = TG.unpack_sample(z, dev)
        same_in = torch.equal(d0["input"], d1["input"])
        torch.manual_seed(0)
        net = CompletionUNet(width=24, padding_mode="replicate", n_feat=ST.FEATURE_DIM).to(dev)
        with torch.no_grad():
            o0 = net(d0["input"][None])[0]; o1 = net(d1["input"][None])[0]
        same_out = torch.equal(o0, o1)
    finally:
        os.unlink(tmp)
    return check("gt_not_in_input", same_in and same_out,
                 f"input tensor identical: {same_in}; network output identical: {same_out} "
                 f"after replacing gt_occ with all-ones and gt_valid with all-zeros")


def head_sparse_dense(dev):
    torch.manual_seed(0)
    net = CompletionUNet(width=24, padding_mode="replicate", n_feat=ST.FEATURE_DIM).to(dev)
    t = torch.randn(2, 24, 8, 8, 4, device=dev)
    dense = net.predict_features(t).permute(0, 2, 3, 4, 1).reshape(-1, ST.FEATURE_DIM)
    ix = torch.randperm(dense.shape[0], device=dev)[:64]
    sparse = features_at(net, t, ix)
    err = float((dense[ix] - sparse).abs().max())
    return check("head_sparse_dense", err < 1e-4,
                 f"max |sparse - dense| over 64 voxels = {err:.3e}")


def checkpoint_roundtrip(dev, sample):
    torch.manual_seed(0)
    net = CompletionUNet(width=24, padding_mode="replicate", n_feat=ST.FEATURE_DIM).to(dev)
    with torch.no_grad():
        a0, s0, t0 = net(sample["input"][None], return_trunk=True)
        f0 = net.predict_features(t0)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        p = f.name
    try:
        save_checkpoint(net, p, {"test": True})
        net2 = load_checkpoint(p, dev).net
        with torch.no_grad():
            a1, s1, t1 = net2(sample["input"][None], return_trunk=True)
            f1 = net2.predict_features(t1)
        ok = torch.equal(a0, a1) and torch.equal(s0, s1) and torch.equal(f0, f1)
    finally:
        os.unlink(p)
    return check("checkpoint_roundtrip", ok,
                 f"occupancy, semantic and feature outputs identical after save/load: {ok}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-traces", action="store_true",
                    help="skip the A/B/C pairing check (needs all nine runs)")
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    drive = "2013_05_28_drive_0003_sync"
    path = sorted(glob.glob(os.path.dirname(SRC.sample_path(drive, 0)) + "/*.npz"))[0]
    with np.load(path) as z:
        d = TG.unpack_sample(z, dev)
    ft = FT.load(ST.feature_target_path(drive, int(os.path.basename(path)[:-4])))
    n = d["gt_occ"].numel()
    sample = {"input": d["input"],
              "fut_p_b": d["fut_p"].reshape(-1, V8.U).T.reshape(1, V8.U, *d["gt_occ"].shape),
              "fut_valid_b": d["fut_valid"][None].float(),
              "aux_index": torch.from_numpy(ft["feat_rows"]).to(dev),
              "aux_target": torch.from_numpy(ft["feat"]).to(dev)}
    print("verification:")
    if not a.skip_traces:
        inputs_paired()
    init_shared()
    unused_heads_frozen(dev)
    aux_reaches_trunk(dev, sample)
    frozen_unchanged()
    alignment(dev, drive)
    gt_not_in_input(dev, path)
    head_sparse_dense(dev)
    checkpoint_roundtrip(dev, sample)
    ST.write_json(os.path.join(ST.ART, "verification.json"),
                  {"checks": RESULTS, "n_pass": sum(r["pass"] for r in RESULTS),
                   "n_total": len(RESULTS), "env": ST.environment()})
    bad = [r["check"] for r in RESULTS if not r["pass"]]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
