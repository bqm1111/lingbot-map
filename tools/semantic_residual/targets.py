"""B5 supervision: the frozen Trident-H FEATURE of a later observation of the same surface.

The frozen delayed targets (``tools/delayed_semantic/targets.py``) store ``prob [18]`` -- a distribution over the 18
phrases -- which cannot supply a cosine feature loss, and must never be turned back into a feature vector. This module
rebuilds the same correspondences but keeps the 1024-d unit token.

What is reused unchanged, so the correspondences are the audited ones:
  * ``camera_view``  LiDAR sweep -> rectified camera -> z-buffer visibility -> bilinear Trident-H token
  * ``_voxel_of``    floor binning into the anchor's velodyne grid (the evaluator's rule)
  * ``rawtarget``    the free/occupied conflict measurement
  * the tier rule    tier A ``hit_hist & n_fut>=1 & static``; tier B ``~hit_hist & n_fut>=2 & static``;
                     ``static = span >= persistence_min_span_s & ~conflict``

What differs, and only this:
  * the future window is ``FUTURE_OFFSETS = (+1, +5)`` stream frames (0.52 s, 2.61 s), the two separations the frozen
    registration gate passed on every drive including the development drive;
  * each target voxel stores the unit teacher feature of its EARLIEST valid future view, not a class distribution.

``camera_view`` discards the feature after taking cosines, so the feature is recomputed here from the same returns and
the same token map -- and the recomputation is checked against the frozen cosines voxel for voxel (``_CHECK_TOL``).
A mismatch is a hard error: a silently different teacher would make every downstream number meaningless.

    PYTHONPATH=$PWD python -m tools.semantic_residual.targets --drive 2013_05_28_drive_0003_sync --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from core.supervision import rawtarget as RT
from tools.delayed_semantic import config as C
from tools.delayed_semantic.stream import DriveStream, StreamFrame
from tools.delayed_semantic.targets import NONE, ViewCache, _voxel_of
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_residual import config as B5

_CHECK_TOL = 2e-3          # fp16 token storage + bilinear sampling; a real teacher mismatch is orders of magnitude larger


def visible_features(geo: TF.DriveGeometry, frame: StreamFrame, cv, device) -> torch.Tensor:
    """``[Nv, 1024]`` unit teacher features at the camera-visible returns of ``cv``, in ``cv.vis_index`` order.

    Recomputed from the same returns and the same token map that ``camera_view`` used; the caller checks it against
    ``cv.cos``."""
    T = torch.as_tensor(np.asarray(geo.calib.velo_to_rect_cam, np.float64), device=device)
    cam = cv.pts_velo[cv.vis_index] @ T[:3, :3].T + T[:3, 3]
    K = geo.calib.K; H, W = geo.calib.native_hw
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    tok = TR.load_token_map(frame.token_path, device)
    g = torch.stack([(u + 0.5) / W * 2 - 1, (v + 0.5) / H * 2 - 1], -1).float()[None, None]
    return F.normalize(F.grid_sample(tok[None], g, mode="bilinear", align_corners=False)[0, :, 0].T, dim=-1)


def future_frames(stream: DriveStream, t: int) -> List[StreamFrame]:
    """Exactly the declared offsets, when they exist on this drive."""
    return [stream.frames[t + o] for o in B5.FUTURE_OFFSETS if t + o < len(stream.frames)]


@torch.no_grad()
def build_anchor(stream: DriveStream, geo: TF.DriveGeometry, t: int, views: ViewCache, evidence_mask: torch.Tensor,
                 text: torch.Tensor, cfg: C.TargetConfig, device) -> Dict:
    n = C.N_VOXELS
    anchor = stream.frames[t]; t0 = anchor.time_s
    hist = [stream.frames[t + o] for o in C.HISTORY_OFFSETS]
    fut = future_frames(stream, t)
    frames = [f for f in hist + fut if f.native in geo.cam0_to_world]

    hit_hist = torch.zeros(n, dtype=torch.bool, device=device)
    n_fut = torch.zeros(n, dtype=torch.int16, device=device)
    tmin = torch.full((n,), float("inf"), dtype=torch.float64, device=device); tmax = torch.full_like(tmin, -float("inf"))
    n_views = torch.zeros(n, dtype=torch.int16, device=device)
    h_views = torch.zeros(n, dtype=torch.int16, device=device)
    feat0 = torch.zeros(n, B5.ResidualConfig().teacher_dim, device=device)       # earliest valid future view
    dt0 = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    top0 = torch.full((n,), NONE, dtype=torch.uint8, device=device)
    max_cos_gap = 0.0

    for f in frames:
        future = f.index > t
        cv = views.get(f)
        T = np.eye(4) if f.native == anchor.native else TF.inv(geo.velo_to_world(anchor.native)) @ geo.velo_to_world(f.native)
        Tt = torch.as_tensor(T, device=device)
        flat = _voxel_of(cv.pts_velo @ Tt[:3, :3].T + Tt[:3, 3])
        hit = torch.zeros(n, dtype=torch.bool, device=device); hit[flat[flat >= 0]] = True
        ts = torch.tensor(f.time_s, dtype=torch.float64, device=device)
        tmin = torch.where(hit, torch.minimum(tmin, ts), tmin); tmax = torch.where(hit, torch.maximum(tmax, ts), tmax)
        if future:
            n_fut += hit.to(torch.int16)
        else:
            hit_hist |= hit

        vflat = flat[cv.vis_index]; keep = vflat >= 0
        if not keep.any():
            continue
        uq, inv = torch.unique(vflat[keep], return_inverse=True)
        if future:
            fe = visible_features(geo, f, cv, device)[keep]
            # the recomputed feature must reproduce the frozen cosines, or the teacher is not the frozen teacher
            max_cos_gap = max(max_cos_gap, float((fe @ text.T - cv.cos[keep]).abs().max()))
            fsum = torch.zeros(len(uq), fe.shape[1], device=device).index_add_(0, inv, fe)
            cnt = torch.bincount(inv, minlength=len(uq)).float()
            fmean = F.normalize(fsum / cnt[:, None], dim=-1)
            first = n_views[uq] == 0                                    # earliest valid future observation wins
            feat0[uq[first]] = fmean[first]
            dt0[uq[first]] = float(f.time_s - t0)
            top0[uq[first]] = (fmean[first] @ text.T).argmax(1).to(torch.uint8)
            n_views[uq] += 1
        else:
            h_views[uq] += 1

    if max_cos_gap > _CHECK_TOL:
        raise RuntimeError(f"recomputed teacher features disagree with the frozen cosines by {max_cos_gap:.2e} "
                           f"(tolerance {_CHECK_TOL}); the token map or calibration is not the frozen one")

    meas = RT.build(geo, anchor.native, [f.native for f in frames], device)
    conflict = torch.from_numpy((meas.n_obs > 0) & (meas.n_free > 0)).to(device)
    static = ((tmax - tmin) >= cfg.persistence_min_span_s) & ~conflict
    tierA = hit_hist & (n_fut >= 1) & static
    tierB = ~hit_hist & (n_fut >= 2) & static
    target = (tierA | tierB) & (n_views >= 1)
    idx = target.nonzero(as_tuple=True)[0]

    E = evidence_mask.to(device)
    arrays = {"flat_index": idx.to(torch.int32).cpu().numpy(),
              "feature": feat0[idx].half().cpu().numpy(),
              "first_dt_s": dt0[idx].cpu().numpy().astype(np.float16),
              "view0_top1": top0[idx].cpu().numpy(),
              "n_views": n_views[idx].clamp(max=255).to(torch.uint8).cpu().numpy(),
              "tier": torch.where(tierA[idx], 1, 2).to(torch.uint8).cpu().numpy(),
              "in_evidence": E[idx].cpu().numpy(),
              "hist_seen": (h_views[idx] > 0).cpu().numpy(),
              "n_future_sweeps": n_fut[idx].clamp(max=255).to(torch.uint8).cpu().numpy(),
              "t": np.int64(t)}
    stats = {"t": t, "native": anchor.native, "n_future_frames": len(fut),
             "future_dt_s": [round(f.time_s - t0, 3) for f in fut],
             "n_target": int(len(idx)), "n_tierA": int(tierA.sum()), "n_tierB": int(tierB.sum()),
             "n_D": int((~E[idx]).sum()),                                  # targets that are NOT student evidence
             "n_D_unobserved": int((~E[idx] & (h_views[idx] == 0)).sum()),  # and never seen by an H_t camera
             "n_evidence": int(E.sum()), "max_cos_gap": max_cos_gap}
    arrays["stats_json"] = json.dumps(stats)
    return {"arrays": arrays, "stats": stats}


def build_drive(drive: str, device: str, limit: int | None = None, overwrite: bool = False) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    stream = DriveStream(drive)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    cfg = C.TargetConfig()
    views = ViewCache(geo, text, cfg, dev)
    g15 = C.Paths().g15
    out_dir = os.path.join(B5.TARGET_ROOT, drive); os.makedirs(out_dir, exist_ok=True)
    ids = list(G15B.anchor_ids(drive, g15))[:limit]
    written = skipped = 0; agg = []; t0 = time.time()
    for i, t in enumerate(ids):
        path = B5.target_path(drive, t)
        if os.path.exists(path) and not overwrite:
            skipped += 1; continue
        with np.load(g15.evidence_path(drive, t)) as z:
            ev_flat = torch.from_numpy(z["flat_index"].astype(np.int64))
        mask = torch.zeros(C.N_VOXELS, dtype=torch.bool); mask[ev_flat] = True
        rec = build_anchor(stream, geo, t, views, mask, text, cfg, dev)
        tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **rec["arrays"]); os.replace(tmp, path)
        written += 1; agg.append(rec["stats"])
        if written % 25 == 0:
            print(f"  [{drive[-9:-5]}] {i + 1}/{len(ids)} {(time.time() - t0) / written:.2f}s each", flush=True)
    summary = {"drive": drive, "n_anchors": len(ids), "written": written, "skipped": skipped,
               "seconds": time.time() - t0, "future_offsets": list(B5.FUTURE_OFFSETS)}
    if agg:
        for k in ("n_target", "n_D", "n_D_unobserved", "n_tierA", "n_tierB"):
            summary[f"mean_{k}"] = float(np.mean([a[k] for a in agg]))
        summary["max_cos_gap"] = max(a["max_cos_gap"] for a in agg)
    os.makedirs(B5.ARTIFACT_DIR, exist_ok=True)
    json.dump(summary, open(os.path.join(B5.ARTIFACT_DIR, f"targets_{drive[-9:-5]}.json"), "w"), indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_drive(a.drive, a.device, a.limit, a.overwrite), indent=1))


if __name__ == "__main__":
    main()
