#!/usr/bin/env python
"""Step 11: what does the mapper associate with the same physical point?

For a subset of verified pairs (A, B, X) on drive 0006 the mapper's own association is
replayed from its frozen inputs (LingBot depth, LingBot poses, the frozen five-frame
scale): X is placed in the mapper's world through frame A's LingBot pose (using X's GT
camera-frame coordinates at A, exactly the chain the mapper trusts), its 0.2 m world voxel
is taken, and the pixels of frame B whose LingBot endpoint band lands in that voxel -- the
observations the mapper would have fused into "the same voxel" -- are collected. Their
image distance to the direct projection u_B, the depth error of LingBot at u_B, the pure
pose error of the LingBot relative pose, and the mapper-style feature/class similarity are
recorded per pair. The mapper's semantic tables are never consulted.

    python tools/direct_correspondence_audit/mapper_compare.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from tools.direct_correspondence_audit.sample_features import Frames, sample, classify, cos  # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping import depth as D7, rays as RY                                          # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = DC.NATIVE_HW; LH, LW = DC.LATTICE_HW; VS = 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default=SRC.VAL_DRIVE); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-offset", type=int, default=3)
    a = ap.parse_args()
    DC.require_data_root(); dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    natives = [int(f.key.split("_")[-1]) for f in seg.frames]; sidx = {n: i for i, n in enumerate(natives)}; key_of = {n: f.key for n, f in zip(natives, seg.frames)}
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kn = np.asarray(geo.calib.K)
    with np.load(S.stream_path(src, a.drive)) as z:
        dep = z["pred_depth"]; conf = z["pred_depth_conf"]; pose = z["pred_pose_c2w"].astype(np.float64); Kl = z["pred_K"].astype(np.float64)
    with np.load(S.scale_path(src, a.drive)) as z:
        scale = float(np.exp(np.median(z["log_s"][:5])))
    with np.load(DC.TEXT_PATH) as z:
        text = torch.from_numpy(z["text"].astype(np.float32)).to(dev)
    frames = Frames(src, key_of, dev)
    def lingbot_endpoints(i):
        """world endpoints (LingBot frame) of accepted lattice pixels, band offsets -1/0/+1: keys [3, n], pix [n, 2] (u, v lattice), depth [n]."""
        d = torch.from_numpy(dep[i].astype(np.float64)).to(dev) * scale; c = torch.from_numpy(conf[i].astype(np.float64)).to(dev)
        acc = (c >= D7.CONF_THRESHOLD) & torch.isfinite(d) & (d > D7.MIN_DEPTH_M) & (d < D7.MAX_DEPTH_M)
        T = pose[i].copy(); T[:3, 3] *= scale; Tw = torch.as_tensor(T, device=dev)
        dirs = RY.pixel_rays(Kl[i], d.shape, dev); sel = acc.reshape(-1).nonzero(as_tuple=True)[0]
        dv = dirs.reshape(-1, 3)[sel]; dd = d.reshape(-1)[sel]; rl = dv.norm(dim=1)
        keys = []
        for o in (-1, 0, 1):
            dz = dd + o * VS / rl; p = dv * dz.unsqueeze(1); pw = p @ Tw[:3, :3].T + Tw[:3, 3]
            keys.append(torch.floor(pw / VS).to(torch.int64))
        vv, uu = sel // LW, sel % LW
        return torch.stack(keys), torch.stack([uu, vv], 1).double(), dd, Tw, dirs
    def key_id(k):
        return (k[..., 0] + 100000) * 4_000_000_000 + (k[..., 1] + 100000) * 40000 + (k[..., 2] + 100000)
    files = sorted(glob.glob(os.path.join(DC.PAIR_DIR, a.drive, "*.npz")))[::a.every]
    rows = []; t0 = time.time(); n_done = 0
    fields = ["stream_index_A", "native_A", "native_B", "landmark", "offset", "baseline_m", "angle_deg", "z_B", "u_B", "v_B",
              "n_mapper_pix_B", "mapper_pix_dist_B_px", "mapper_pix_dist_B_min_px", "n_mapper_pix_A", "mapper_pix_dist_A_px",
              "lingbot_depth_err_uB_m", "lingbot_depth_rel_err_uB", "pose_reproj_err_px", "endpoint_3d_err_m",
              "cos_direct", "top1_direct", "cos_mapper", "top1_mapper", "cos_direct_vs_mapper_B"]
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                P = {k: z[k] for k in z.files}
            fa = int(P["native_A"]); ia = sidx[fa]; R = P["rows"]; R = R[(R[:, 6] > 0) & (np.abs(R[:, 9]) <= a.max_offset)]
            if len(R) == 0:
                continue
            Xw = P["X_world"]; Twc_A = TF.inv(geo.rect_cam_to_world(fa)); XcA = Xw @ Twc_A[:3, :3].T + Twc_A[:3, 3]      # GT camera-frame coords at A
            kA, pixA, dA, TwA, _ = lingbot_endpoints(ia)
            TA = pose[ia].copy(); TA[:3, 3] *= scale
            pwX = XcA @ TA[:3, :3].T + TA[:3, 3]                                                     # X in the mapper's world via A
            keyX = key_id(torch.floor(torch.from_numpy(pwX).to(dev) / VS).to(torch.int64))
            idA = key_id(kA)                                                                        # [3, nA]
            tokA, _ = frames.get(fa); uA = torch.from_numpy(P["u_A"]).to(dev); vA = torch.from_numpy(P["v_A"]).to(dev)
            for fb in np.unique(R[:, 1]).astype(int):
                ib = sidx[fb]; Rb = R[R[:, 1] == fb]; li = Rb[:, 0].astype(int); n = len(li)
                kB, pixB, dB, TwB, dirsB = lingbot_endpoints(ib); idB = key_id(kB)
                tokB, _ = frames.get(fb)
                uB = torch.from_numpy(Rb[:, 2]).to(dev); vB = torch.from_numpy(Rb[:, 3]).to(dev)
                fA_d = sample(tokA, uA[li].float(), vA[li].float()); fB_d = sample(tokB, uB.float(), vB.float())
                cA_d, cB_d = classify(fA_d, text).argmax(1), classify(fB_d, text).argmax(1)
                # pose-only reprojection error: X_camA -> camB through LingBot relative pose vs GT projection
                TB = pose[ib].copy(); TB[:3, 3] *= scale; rel = np.linalg.inv(TB) @ TA
                XcB_l = XcA[li] @ rel[:3, :3].T + rel[:3, 3]
                u_l = Kn[0, 0] * XcB_l[:, 0] / XcB_l[:, 2] + Kn[0, 2]; v_l = Kn[1, 1] * XcB_l[:, 1] / XcB_l[:, 2] + Kn[1, 2]
                pose_err = np.sqrt((u_l - Rb[:, 2]) ** 2 + (v_l - Rb[:, 3]) ** 2)
                # LingBot depth at the direct pixel (lattice) vs GT depth z_B
                ul = np.clip(np.round((Rb[:, 2] + 0.5) * LW / W - 0.5).astype(int), 0, LW - 1); vl = np.clip(np.round((Rb[:, 3] + 0.5) * LH / H - 0.5).astype(int), 0, LH - 1)
                d_l = dep[ib][vl, ul].astype(np.float64) * scale; zB = Rb[:, 4]
                depth_err = d_l - zB
                # endpoint 3D error in the mapper's world: endpoint of direct pixel vs X
                dirs_np = dirsB[vl, ul].cpu().numpy(); ep = dirs_np * d_l[:, None]; TwB_np = TwB.cpu().numpy(); ep_w = ep @ TwB_np[:3, :3].T + TwB_np[:3, 3]
                ep_err = np.linalg.norm(ep_w - pwX[li], axis=1)
                # which B pixels does the mapper put into voxel(X)?
                kx = keyX[li]
                matchB = (idB[:, None, :] == kx[:, None]).any(0)                                    # [n, nB] (memory: n<=1500 x nB<=70k bool ok)
                matchA = (idA[:, None, :] == kx[:, None]).any(0)
                nB = matchB.sum(1); nA = matchA.sum(1)
                pixB_nat = torch.stack([(pixB[:, 0] + 0.5) * W / LW - 0.5, (pixB[:, 1] + 0.5) * H / LH - 0.5], 1)
                pixA_nat = torch.stack([(pixA[:, 0] + 0.5) * W / LW - 0.5, (pixA[:, 1] + 0.5) * H / LH - 0.5], 1)
                distB = torch.full((n,), float("nan"), device=dev); distBmin = distB.clone(); distA = distB.clone()
                cos_m = distB.clone(); top_m = distB.clone(); cos_dm = distB.clone()
                for i in range(n):
                    if nB[i] > 0:
                        pb = pixB_nat[matchB[i]]; dd_ = ((pb[:, 0] - uB[i]) ** 2 + (pb[:, 1] - vB[i]) ** 2).sqrt()
                        distB[i] = dd_.mean(); distBmin[i] = dd_.min()
                        fBm = sample(tokB, pb[:, 0].float(), pb[:, 1].float()).mean(0, keepdim=True)
                        cos_dm[i] = cos(fB_d[i:i + 1], fBm)[0]
                        if nA[i] > 0:
                            pa = pixA_nat[matchA[i]]; distA[i] = ((pa[:, 0] - uA[li[i]]) ** 2 + (pa[:, 1] - vA[li[i]]) ** 2).sqrt().mean()
                            fAm = sample(tokA, pa[:, 0].float(), pa[:, 1].float()).mean(0, keepdim=True)
                            cos_m[i] = cos(fAm, fBm)[0]; top_m[i] = float(classify(fAm, text).argmax(1)[0] == classify(fBm, text).argmax(1)[0])
                cols = [np.full(n, ia), np.full(n, fa), np.full(n, fb), li, Rb[:, 9], Rb[:, 7], Rb[:, 8], zB, Rb[:, 2], Rb[:, 3], nB.cpu().numpy(), distB.cpu().numpy(), distBmin.cpu().numpy(),
                        nA.cpu().numpy(), distA.cpu().numpy(), depth_err, depth_err / zB, pose_err, ep_err,
                        cos(fA_d, fB_d).cpu().numpy(), (cA_d == cB_d).float().cpu().numpy(), cos_m.cpu().numpy(), top_m.cpu().numpy(), cos_dm.cpu().numpy()]
                rows.append(np.stack([np.asarray(c, np.float64) for c in cols], 1).astype(np.float32))
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done} source frames, {sum(len(r) for r in rows):,} rows, {(time.time()-t0)/n_done:.1f}s each", flush=True)
    arr = np.concatenate(rows)
    os.makedirs(os.path.join(DC.DATA_ROOT, "mapper"), exist_ok=True)
    np.savez_compressed(os.path.join(DC.DATA_ROOT, "mapper", f"{a.drive}.npz"), rows=arr, fields=np.asarray(fields))
    Fi = {k: i for i, k in enumerate(fields)}; g = lambda k: arr[:, Fi[k]]
    hasB = g("n_mapper_pix_B") > 0; hasA = g("n_mapper_pix_A") > 0; both = hasA & hasB
    pd = g("mapper_pix_dist_B_min_px")[hasB]
    out = {"drive": a.drive, "n_pairs": int(len(arr)), "every": a.every, "max_offset": a.max_offset, "scale": scale,
           "mapper_has_B_observation_in_voxel_of_X": float(hasB.mean()), "mapper_has_A_observation_in_voxel_of_X": float(hasA.mean()), "both": float(both.mean()),
           "mapper_pixel_offset_B_px": {"median": float(np.median(pd)), "p25": float(np.percentile(pd, 25)), "p75": float(np.percentile(pd, 75)), "p90": float(np.percentile(pd, 90)),
                                        "frac_lt_5px": float((pd < 5).mean()), "frac_lt_11px": float((pd < 11).mean()), "frac_gt_30px": float((pd > 30).mean())},
           "lingbot_depth_error_at_uB_m": {"median_abs": float(np.median(np.abs(g("lingbot_depth_err_uB_m")))), "p90_abs": float(np.percentile(np.abs(g("lingbot_depth_err_uB_m")), 90)),
                                          "median_rel": float(np.median(np.abs(g("lingbot_depth_rel_err_uB")))), "frac_rel_gt_10pct": float((np.abs(g("lingbot_depth_rel_err_uB")) > 0.1).mean())},
           "pose_reprojection_error_px": {"median": float(np.median(g("pose_reproj_err_px"))), "p90": float(np.percentile(g("pose_reproj_err_px"), 90))},
           "endpoint_3d_error_m": {"median": float(np.median(g("endpoint_3d_err_m"))), "p90": float(np.percentile(g("endpoint_3d_err_m"), 90)), "frac_gt_0p2m": float((g("endpoint_3d_err_m") > 0.2).mean())},
           "similarity": {"direct_cos_all": float(g("cos_direct").mean()), "direct_top1_all": float(g("top1_direct").mean()),
                          "direct_cos_on_both": float(g("cos_direct")[both].mean()), "direct_top1_on_both": float(g("top1_direct")[both].mean()),
                          "mapper_cos_on_both": float(np.nanmean(g("cos_mapper")[both])), "mapper_top1_on_both": float(np.nanmean(g("top1_mapper")[both])),
                          "cos_direct_pixel_vs_mapper_pixels_B": float(np.nanmean(g("cos_direct_vs_mapper_B")[hasB])), "n_both": int(both.sum())},
           "seconds": time.time() - t0, "firewall": audit.summary()}
    # by stream position of A (LingBot pose drift grows along the stream)
    out["by_stream_position"] = {}
    si = g("stream_index_A")
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000)):
        m = (si >= lo) & (si < hi)
        if m.sum() < 100:
            continue
        mb = m & both; hb = m & hasB
        out["by_stream_position"][f"{lo}-{hi}"] = {
            "n": int(m.sum()), "mapper_has_B_obs": float(hasB[m].mean()), "pose_reproj_err_px_median": float(np.median(g("pose_reproj_err_px")[m])),
            "endpoint_3d_err_m_median": float(np.median(g("endpoint_3d_err_m")[m])), "depth_rel_err_median": float(np.median(np.abs(g("lingbot_depth_rel_err_uB")[m]))),
            "mapper_pix_offset_px_median": float(np.median(g("mapper_pix_dist_B_min_px")[hb])) if hb.any() else None,
            "direct_top1": float(g("top1_direct")[m].mean()), "direct_top1_on_both": float(g("top1_direct")[mb].mean()) if mb.any() else None,
            "mapper_top1_on_both": float(np.nanmean(g("top1_mapper")[mb])) if mb.any() else None, "n_both": int(mb.sum())}
    DC.write_json(os.path.join(DC.ART, "mapper_comparison.json"), out)
    print(out["similarity"], out["mapper_pixel_offset_B_px"], out["lingbot_depth_error_at_uB_m"], out["pose_reprojection_error_px"], out["endpoint_3d_error_m"], "hasB", out["mapper_has_B_observation_in_voxel_of_X"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
