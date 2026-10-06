"""Delayed semantic targets: later, physically synchronised observations of the scene at time t.

TARGET CONSTRUCTION ONLY. Reads future frames, raw LiDAR, GT poses and calibration; never human labels.
Nothing in the student prediction path imports this module (tests/delayed_semantic checks it).

Per future stream frame j (``camera_view``):
    LiDAR sweep of native frame j (velodyne frame j) -> rectified camera j (calibration) -> pixel (u, v)
    camera-visible  := z >= 1 m, inside 1408x376, z <= zmin(4 px cell) + max(0.5 m, 0.05 zmin)
    teacher view    := bilinear Trident-H token of image j at (u, v), unit norm, cosine to 18 text embeddings
Image and LiDAR timestamps of a stream frame differ by < 5 ms, so a visible return and its pixel observe
the same surface at the same instant; the association residual is calibration only.

Per anchor t (``build_anchor``), in the velodyne grid of t (GT poses, ``velo_to_velo(j, t)``):
    hit_hist[v]      a return from an H_t sweep (t-4..t) landed in v
    n_fut_sweeps[v]  future sweeps with a return in v;   span[v] = latest - earliest return time (s)
    conflict[v]      occupied in one sweep, traversed free by another (frozen rawtarget measurement)
    view prob        per (v, j): softmax(40 * mean cosine of j's visible returns in v)
    tier A           hit_hist & n_fut_sweeps >= 1 & span >= 1 s & ~conflict     (occupied at t)
    tier B           ~hit_hist & n_fut_sweeps >= 2 & span >= 1 s & ~conflict    (earlier occupancy uncertain)
    target           (A | B) & >= 1 future view;  prob = mean of its future view distributions

Stored per target voxel: flat_index int32, prob [18] fp16, n_views, first_dt_s / second_dt_s (time of the
first two views after t), view0_top1 / view1_top1, tier (1 = A, 2 = B), in_evidence (student evidence
voxel at t), hist_seen (an H_t camera saw a return there -- privileged, diagnostics only), hist_top1.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from core.datasets.audit import FileAudit
from core.supervision import rawtarget as RT
from tools.delayed_semantic import config as C
from tools.delayed_semantic.stream import DriveStream, StreamFrame
from tools.semantic_completion.evidence import build as G15B, trident as TR

NONE = 255


@dataclass
class CameraView:
    native: int
    time_s: float
    pts_velo: torch.Tensor      # [N, 3] float64, every return beyond the rawtarget near range
    vis_index: torch.Tensor     # [Nv] int64 rows of pts_velo that are camera-visible
    cos: torch.Tensor           # [Nv, 18] float32 teacher cosine to the text embeddings
    n_in_image: int
    #: rows of ``pts_velo`` inside the image at z >= min depth, BEFORE the z-buffer occlusion test.
    #: Additive, used only by the Gate 20 frustum diagnostic; ``vis_index`` is a subset of it.
    in_image_index: torch.Tensor = None


def camera_view(geo: TF.DriveGeometry, frame: StreamFrame, text: torch.Tensor, cfg: C.TargetConfig, device) -> CameraView:
    pts = torch.as_tensor(geo.read_velodyne(frame.native), device=device)
    pts = pts[pts.norm(dim=1) > RT.CARVE_NEAR_M]
    T = torch.as_tensor(np.asarray(geo.calib.velo_to_rect_cam, np.float64), device=device)
    cam = pts @ T[:3, :3].T + T[:3, 3]
    K = geo.calib.K; H, W = geo.calib.native_hw
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]; v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    inimg = (z >= cfg.min_camera_depth_m) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    idx = inimg.nonzero(as_tuple=True)[0]
    cell = cfg.zbuffer_cell_px; cw = (W + cell - 1) // cell; ch = (H + cell - 1) // cell
    cid = (torch.floor((v[idx] + 0.5) / cell).long() * cw + torch.floor((u[idx] + 0.5) / cell).long())
    zz = z[idx]
    zmin = torch.full((cw * ch,), float("inf"), dtype=zz.dtype, device=device).scatter_reduce_(0, cid, zz, "amin")[cid]
    vi = idx[zz <= zmin + torch.clamp(zmin * cfg.zbuffer_tol_rel, min=cfg.zbuffer_tol_m)]
    tok = TR.load_token_map(frame.token_path, device)
    g = torch.stack([(u[vi] + 0.5) / W * 2 - 1, (v[vi] + 0.5) / H * 2 - 1], -1).float()[None, None]
    feat = F.normalize(F.grid_sample(tok[None], g, mode="bilinear", align_corners=False)[0, :, 0].T, dim=-1)
    return CameraView(frame.native, frame.time_s, pts, vi, (feat @ text.T).float(), int(len(idx)), idx)


class ViewCache:
    def __init__(self, geo, text, cfg: C.TargetConfig, device):
        self.geo, self.text, self.cfg, self.device, self.d = geo, text, cfg, device, OrderedDict()

    def get(self, f: StreamFrame) -> CameraView:
        if f.index in self.d:
            self.d.move_to_end(f.index); return self.d[f.index]
        v = camera_view(self.geo, f, self.text, self.cfg, self.device); self.d[f.index] = v
        while len(self.d) > self.cfg.lru_frames:
            self.d.popitem(last=False)
        return v


def _voxel_of(pts_grid: torch.Tensor) -> torch.Tensor:
    """[N] flat index, -1 outside the grid (floor binning, the evaluator's rule)."""
    origin = torch.as_tensor(C.GRID_ORIGIN_M, dtype=pts_grid.dtype, device=pts_grid.device)
    idx = torch.floor((pts_grid - origin) / C.VOXEL_SIZE_M).long()
    dims = torch.as_tensor(C.GRID_DIMS, device=pts_grid.device)
    ok = ((idx >= 0) & (idx < dims)).all(1)
    flat = (idx[:, 0] * C.GRID_DIMS[1] + idx[:, 1]) * C.GRID_DIMS[2] + idx[:, 2]
    return torch.where(ok, flat, torch.full_like(flat, -1))


def build_anchor(stream: DriveStream, geo: TF.DriveGeometry, t: int, views: ViewCache, evidence_mask: torch.Tensor,
                 cfg: C.TargetConfig, device) -> Dict:
    n = C.N_VOXELS; anchor = stream.frames[t]; t0 = anchor.time_s
    hist = [stream.frames[t + o] for o in C.HISTORY_OFFSETS]
    fut = stream.future(t, cfg)
    frames = [f for f in hist + fut if f.native in geo.cam0_to_world]
    hit_hist = torch.zeros(n, dtype=torch.bool, device=device)
    n_fut = torch.zeros(n, dtype=torch.int16, device=device)
    tmin = torch.full((n,), float("inf"), dtype=torch.float64, device=device); tmax = torch.full_like(tmin, -float("inf"))
    psum = torch.zeros(n, C.N_CLASSES, device=device); hsum = torch.zeros(n, C.N_CLASSES, device=device)
    n_views = torch.zeros(n, dtype=torch.int16, device=device); h_views = torch.zeros(n, dtype=torch.int16, device=device)
    top = [torch.full((n,), NONE, dtype=torch.uint8, device=device) for _ in range(2)]
    dts = [torch.full((n,), float("nan"), dtype=torch.float32, device=device) for _ in range(2)]
    frame_stats = []
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
        uq, inv = torch.unique(vflat[keep], return_inverse=True)
        if len(uq):
            csum = torch.zeros(len(uq), C.N_CLASSES, device=device).index_add_(0, inv, cv.cos[keep])
            cnt = torch.bincount(inv, minlength=len(uq)).float()
            prob = torch.softmax(C.LOGIT_SCALE * csum / cnt[:, None], -1)
            if future:
                psum[uq] += prob
                am = prob.argmax(1).to(torch.uint8)
                for r in range(2):
                    sel = n_views[uq] == r
                    top[r][uq[sel]] = am[sel]; dts[r][uq[sel]] = float(f.time_s - t0)
                n_views[uq] += 1
            else:
                hsum[uq] += prob; h_views[uq] += 1
        frame_stats.append({"j": f.index, "dt_s": round(f.time_s - t0, 3), "future": future, "returns": int(len(cv.pts_velo)),
                            "in_image": cv.n_in_image, "visible": int(len(cv.vis_index)), "visible_in_grid": int(keep.sum())})
    meas = RT.build(geo, anchor.native, [f.native for f in frames], device)
    conflict = torch.from_numpy((meas.n_obs > 0) & (meas.n_free > 0)).to(device)
    static = ((tmax - tmin) >= cfg.persistence_min_span_s) & ~conflict
    tierA = hit_hist & (n_fut >= 1) & static
    tierB = ~hit_hist & (n_fut >= 2) & static
    target = (tierA | tierB) & (n_views >= 1)
    idx = target.nonzero(as_tuple=True)[0]
    E = evidence_mask.to(device)
    hist_top = torch.where(h_views > 0, hsum.argmax(1).to(torch.uint8), torch.full((n,), NONE, dtype=torch.uint8, device=device))
    arrays = {"flat_index": idx.to(torch.int32), "prob": (psum[idx] / n_views[idx, None].float()).half(),
              "n_views": n_views[idx].clamp(max=255).to(torch.uint8), "first_dt_s": dts[0][idx].half(), "second_dt_s": dts[1][idx].half(),
              "view0_top1": top[0][idx], "view1_top1": top[1][idx], "tier": torch.where(tierA[idx], 1, 2).to(torch.uint8),
              "in_evidence": E[idx], "hist_seen": h_views[idx] > 0, "hist_top1": hist_top[idx], "n_future_sweeps": n_fut[idx].clamp(max=255).to(torch.uint8)}
    fut_hit = n_fut >= 1
    stats = {"t": t, "native": anchor.native, "n_future_frames": len([f for f in frames if f.index > t]), "horizon_dt_s": round(fut[-1].time_s - t0, 3) if fut else 0.0,
             "n_future_hit": int(fut_hit.sum()), "n_conflict": int(conflict.sum()), "n_future_hit_not_static": int((fut_hit & ~static).sum()),
             "n_tierA": int(tierA.sum()), "n_tierB": int(tierB.sum()), "n_target": int(len(idx)),
             "n_D": int((~E[idx]).sum()), "n_D_A": int((~E[idx] & tierA[idx]).sum()), "n_D_B": int((~E[idx] & tierB[idx]).sum()),
             "n_D_hist_seen": int((~E[idx] & (h_views[idx] > 0)).sum()), "n_evidence": int(E.sum()),
             "n_evidence_reobserved": int((E[idx]).sum()), "frames": frame_stats}
    return {"arrays": {k: v.cpu().numpy() for k, v in arrays.items()}, "stats": stats}


def build_drive(drive: str, device: str, shard: int = 0, shards: int = 1, limit: int | None = None,
                cfg: C.TargetConfig = C.TargetConfig(), paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    ids = G15B.anchor_ids(drive, paths.g15)[shard::shards][:limit]
    rows, t0 = [], time.time()
    with FileAudit(patterns=C.TARGET_FORBIDDEN, strict=True) as audit:
        stream = DriveStream(drive)
        geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
        views = ViewCache(geo, text, cfg, dev)
        for k, t in enumerate(ids):
            out = paths.target_path(drive, t)
            if os.path.exists(out):
                continue
            with np.load(paths.g15.evidence_path(drive, t)) as z:
                E = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); E[torch.from_numpy(z["flat_index"].astype(np.int64)).to(dev)] = True
            r = build_anchor(stream, geo, t, views, E, cfg, dev)
            os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp.npz"
            st = dict(r["stats"]); st.pop("frames")
            np.savez(tmp, **r["arrays"], t=np.int64(t), stats_json=np.asarray(__import__("json").dumps(r["stats"])))
            os.replace(tmp, out); rows.append(st)
            if (k + 1) % 50 == 0:
                print(f"  [targets {drive[17:21]} s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    return {"drive": drive, "shard": shard, "shards": shards, "n_anchors": len(ids), "rows": rows, "seconds": time.time() - t0,
            "config": cfg.__dict__, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}


__all__ = ["CameraView", "camera_view", "ViewCache", "build_anchor", "build_drive"]
