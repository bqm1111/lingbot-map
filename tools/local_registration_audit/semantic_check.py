#!/usr/bin/env python
"""Step 11 (diagnostic only): does corrected registration restore same-voxel semantic agreement?

Uses the verified physical correspondences of the direct audit on drive 0006 whose source
frame A is a clip anchor t and whose partner B is t-1 .. t-4 (so both frames are in the clip).
For every landmark X (GT position in the anchor camera) and condition c, the 0.2 m voxel of
X in the anchor camera is taken, and the pixels of frame t and of frame B whose endpoint band
(-1, 0, +1 voxel along the ray, the mapper's rule) lands in that voxel under condition c's
depth and pose are collected -- the observations condition c would fuse into "the same
voxel". Reported: support (both frames have >= 1 pixel in the voxel), top-1 agreement of the
mean Trident probabilities, cosine of the mean dense features, and the same two statistics
against a local control (a different landmark 0.3-1.5 m away, same set).

    python tools/local_registration_audit/semantic_check.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

H, W = 376, 1408; LH, LW = 140, 518; VS = 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max-landmarks", type=int, default=300); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = LR.PRIMARY; src = "kitti360"
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]; sidx = {n: i for i, n in enumerate(nat)}
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K, np.float64)
    with np.load(S.stream_path(src, d)) as z:
        s_dep = z["pred_depth"]; s_conf = z["pred_depth_conf"]; s_pose = z["pred_pose_c2w"].astype(np.float64); s_K = z["pred_K"].astype(np.float64)
    with np.load(S.scale_path(src, d)) as z:
        s_scale = float(np.exp(np.median(z["log_s"][:5])))
    with np.load(DC.TEXT_PATH) as z:
        text = torch.from_numpy(z["text"].astype(np.float32)).to(dev)
    clips = {int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz")}
    stats = {c: {"n": 0, "support": 0, "agree": 0, "cos": 0.0, "ctrl_n": 0, "ctrl_agree": 0, "ctrl_cos": 0.0} for c in ("S", "L", "Lp", "G", "GLd")}
    rng = np.random.default_rng(0); t0 = time.time(); n_clips = 0
    for pf in sorted(glob.glob(os.path.join(DC.PAIR_DIR, d, "*.npz"))):
        with np.load(pf) as z:
            fa = int(z["native_A"]); t = sidx.get(fa, -1)
            if t not in clips:
                continue
            P = {k: z[k] for k in z.files}
        R = P["rows"]; R = R[(R[:, 6] > 0) & (R[:, 9] < 0) & (R[:, 9] >= -4)]
        if not len(R):
            continue
        with np.load(LR.clip_path(d, t)) as z:
            Lz = {k: z[k] for k in z.files}
        l_scale = float(Lz["scale"]); Gt = geo.rect_cam_to_world(nat[t]); Twc_t = TF.inv(Gt)
        tok = lambda n: torch.from_numpy(np.load(os.path.join(DC.TOKEN_DIR, src, f"{seg.frames[sidx[n]].key}.npz"))["tokens"].astype(np.float32)).to(dev).reshape(34, 128, -1).permute(2, 0, 1)[None]
        prob = lambda n: torch.from_numpy(np.load(S.trident_path(src, seg.frames[sidx[n]].key))["probs"].astype(np.float32)).to(dev)[None]
        n_clips += 1
        for fb in np.unique(R[:, 1]).astype(int):
            Rb = R[R[:, 1] == fb]; li = Rb[:, 0].astype(int)
            if len(li) > a.max_landmarks:
                li = np.sort(rng.choice(li, a.max_landmarks, replace=False))
            X = P["X_world"][li] @ Twc_t[:3, :3].T + Twc_t[:3, 3]                          # landmark in the anchor camera
            vox = np.floor(X / VS).astype(np.int64); tree = cKDTree(X)
            b = sidx[fb]; q = b - t + 4; assert 0 <= q < 4
            TG = TF.inv(Gt) @ geo.rect_cam_to_world(fb); TS = np.linalg.inv(s_pose[t]) @ s_pose[b]; TS[:3, 3] *= s_scale
            TL = np.linalg.inv(Lz["pred_pose_c2w"][4].astype(np.float64)) @ Lz["pred_pose_c2w"][q].astype(np.float64); TL[:3, 3] *= l_scale
            I4 = np.eye(4)
            feats = {"t": (tok(nat[t]), prob(nat[t])), "b": (tok(fb), prob(fb))}
            conds = {"S": ((s_dep[t], s_conf[t], s_K[t], s_scale, I4), (s_dep[b], s_conf[b], s_K[b], s_scale, TS)),
                     "L": ((Lz["pred_depth"][4], Lz["pred_depth_conf"][4], Lz["pred_K"][4], l_scale, I4), (Lz["pred_depth"][q], Lz["pred_depth_conf"][q], Lz["pred_K"][q], l_scale, TL)),
                     "Lp": ((s_dep[t], s_conf[t], s_K[t], s_scale, I4), (s_dep[b], s_conf[b], s_K[b], s_scale, TL)),
                     "G": ((s_dep[t], s_conf[t], s_K[t], s_scale, I4), (s_dep[b], s_conf[b], s_K[b], s_scale, TG)),
                     "GLd": ((Lz["pred_depth"][4], Lz["pred_depth_conf"][4], Lz["pred_K"][4], l_scale, I4), (Lz["pred_depth"][q], Lz["pred_depth_conf"][q], Lz["pred_K"][q], l_scale, TG))}
            # nearby control landmark per X
            ctrl = np.full(len(X), -1)
            for i, cand in enumerate(tree.query_ball_point(X, 1.5)):
                cand = [j for j in cand if j != i and np.linalg.norm(X[j] - X[i]) >= 0.3]
                if cand: ctrl[i] = cand[rng.integers(len(cand))]
            for c, (ft, fbb) in conds.items():
                pix = {}
                for which, (dep, conf, Kl, sc, T) in (("t", ft), ("b", fbb)):
                    dd = torch.from_numpy(dep.astype(np.float64)).to(dev) * sc; cf = torch.from_numpy(conf.astype(np.float64)).to(dev)
                    ok = (cf >= D7.CONF_THRESHOLD) & torch.isfinite(dd) & (dd > D7.MIN_DEPTH_M) & (dd < D7.MAX_DEPTH_M)
                    sel = ok.reshape(-1).nonzero(as_tuple=True)[0]; vv, uu = sel // LW, sel % LW
                    Kt = torch.as_tensor(np.asarray(Kl, np.float64), device=dev); dirs = torch.stack([(uu - Kt[0, 2]) / Kt[0, 0], (vv - Kt[1, 2]) / Kt[1, 1], torch.ones_like(uu, dtype=torch.float64)], 1)
                    rl = dirs.norm(dim=1); Tt = torch.as_tensor(T, device=dev); keys = []
                    for o in (-1, 0, 1):
                        p = dirs * (dd.reshape(-1)[sel] + o * VS / rl).unsqueeze(1); keys.append(torch.floor((p @ Tt[:3, :3].T + Tt[:3, 3]) / VS).to(torch.int64))
                    kk = torch.stack(keys)                                                 # [3, n, 3]
                    kid = (kk[..., 0] + 5000) * 100_000_000 + (kk[..., 1] + 5000) * 10_000 + (kk[..., 2] + 5000)
                    vid = torch.from_numpy((vox[:, 0] + 5000) * 100_000_000 + (vox[:, 1] + 5000) * 10_000 + (vox[:, 2] + 5000)).to(dev)
                    match = (kid.unsqueeze(-1) == vid.view(1, 1, -1)).any(0)                   # [n_pix, n_landmarks]
                    un = torch.stack([(uu.double() + 0.5) * W / LW - 0.5, (vv.double() + 0.5) * H / LH - 0.5], 1)
                    pix[which] = (match, un)
                sT = stats[c]
                def mean_repr(which, lm):
                    match, un = pix[which]; m_ = match[:, lm]
                    if not m_.any():
                        return None
                    uvs = un[m_]; tk, pr = feats[which]
                    g = torch.stack([(uvs[:, 0] + 0.5) / W * 2 - 1, (uvs[:, 1] + 0.5) / H * 2 - 1], -1).float()[None, None]
                    fv = F.grid_sample(tk, g, mode="bilinear", align_corners=False)[0, :, 0].T.mean(0)
                    pv = F.grid_sample(pr, g, mode="bilinear", align_corners=False)[0, :, 0].T.mean(0)
                    return fv, pv
                for i in range(len(X)):
                    sT["n"] += 1
                    rt_, rb_ = mean_repr("t", i), mean_repr("b", i)
                    if rt_ is None or rb_ is None:
                        continue
                    sT["support"] += 1; sT["agree"] += int(rt_[1].argmax() == rb_[1].argmax()); sT["cos"] += float(F.cosine_similarity(rt_[0], rb_[0], dim=0))
                    if ctrl[i] >= 0:
                        rc = mean_repr("b", ctrl[i])
                        if rc is not None:
                            sT["ctrl_n"] += 1; sT["ctrl_agree"] += int(rt_[1].argmax() == rc[1].argmax()); sT["ctrl_cos"] += float(F.cosine_similarity(rt_[0], rc[0], dim=0))
        if n_clips % 10 == 0:
            print(f"  {n_clips} clips, {(time.time()-t0)/n_clips:.1f}s each; " + " ".join(f"{c}: sup {v['support']/max(v['n'],1):.3f} agr {v['agree']/max(v['support'],1):.3f}" for c, v in stats.items()), flush=True)
    out = {"drive": d, "n_clips": n_clips, "definition": __doc__.split("\n\n")[1], "results": {}}
    for c, v in stats.items():
        out["results"][c] = {"landmarks": v["n"], "support_fraction": v["support"] / max(v["n"], 1), "class_agreement": v["agree"] / max(v["support"], 1),
                             "feature_cosine": v["cos"] / max(v["support"], 1), "local_control_n": v["ctrl_n"], "local_control_agreement": v["ctrl_agree"] / max(v["ctrl_n"], 1),
                             "local_control_cosine": v["ctrl_cos"] / max(v["ctrl_n"], 1),
                             "agreement_margin_over_local": v["agree"] / max(v["support"], 1) - v["ctrl_agree"] / max(v["ctrl_n"], 1),
                             "cosine_margin_over_local": v["cos"] / max(v["support"], 1) - v["ctrl_cos"] / max(v["ctrl_n"], 1)}
    C0.write_json(os.path.join(LR.ART, "semantic_check.json"), out)
    print(json.dumps(out["results"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
