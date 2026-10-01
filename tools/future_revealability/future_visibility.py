"""When, if ever, does a voxel of the anchor grid become observable again?

Per anchor `t`, every future stream frame of the frozen privileged window (`t+1..t+20`, <= 10 s) is
projected into the grid of `t` and three nested reveal levels are recorded per voxel:

    geometric   a LiDAR return of a future sweep falls in the voxel
    in_frustum  that return also projects inside the future image at z >= 1 m (no occlusion test) -- a
                diagnostic that separates the camera's field of view from occlusion
    visible     that return is camera-visible under the frozen z-buffer rule AND carries a Trident-H token
    deployable  `visible` AND the frozen delayed-target selector keeps the voxel (read, not recomputed)

`camera_view`, `ViewCache`, `_voxel_of` and the pose composition are imported from
`tools.delayed_semantic.targets` and used unchanged -- this module adds no coordinate-transform, occlusion
or lifting logic of its own. `validate_against_frozen_targets` requires the `visible` level to reproduce
the frozen delayed-target cache's own `first_dt_s`, `n_views` and `prob` on the cache's voxels.

Future information enters nothing but this record. The caller computes the causal state first, under a file
audit that forbids poses, LiDAR and future frames, and `build_targets.assert_no_feedback` re-checks that
the causal state is byte-identical afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from core.datasets import kitti360_transforms as TF
from tools.delayed_semantic import config as C
from tools.delayed_semantic.stream import DriveStream
from tools.delayed_semantic.targets import ViewCache, _voxel_of
from tools.future_revealability import config as G

NONE = 255
INF = float("inf")


@dataclass
class RevealRecord:
    """Rows are the voxels selected by ``rows_mask``; columns indexed by ``G.HORIZONS_S``."""
    flat_index: np.ndarray        # [R] int32
    geo_first_dt: np.ndarray      # [R] float32, inf = never a future return
    img_first_dt: np.ndarray      # [R] float32, inf = never inside a future image (frustum diagnostic)
    vis_first_dt: np.ndarray      # [R] float32, inf = never a camera-visible future return
    vis_n_views: np.ndarray       # [R, H] int16   views with dt <= horizon
    mean_top1: np.ndarray         # [R, H] uint8   argmax of the mean future probability, 255 if no view
    last_top1: np.ndarray         # [R, H] uint8   argmax of the latest view, 255 if no view
    first_top1: np.ndarray        # [R] uint8      argmax of the first view, 255 if never
    frames: List[Dict]
    #: Additive, only filled when ``reveal_record(..., want_mean_prob=True)``: the mean future probability
    #: itself at the LAST horizon, ``[R, 18]`` float32, all zeros where the voxel is never visible. Its
    #: argmax is exactly ``mean_top1[:, -1]`` -- Gate 21 distils the vector whose argmax Gate 20 reported,
    #: and ``tools.future_semantic_prior.targets`` asserts that agreement voxel for voxel.
    mean_prob: Optional[np.ndarray] = None


@torch.no_grad()
def reveal_record(stream: DriveStream, geo: TF.DriveGeometry, t: int, views: ViewCache,
                  rows_mask: Optional[torch.Tensor], cfg: C.TargetConfig, device,
                  want_mean_prob: bool = False) -> RevealRecord:
    n = C.N_VOXELS
    anchor = stream.frames[t]
    t0 = anchor.time_s
    fut = [f for f in stream.future(t, cfg) if f.native in geo.cam0_to_world]
    horizons = list(G.HORIZONS_S)

    geo_first = torch.full((n,), INF, dtype=torch.float32, device=device)
    img_first = torch.full((n,), INF, dtype=torch.float32, device=device)
    vis_first = torch.full((n,), INF, dtype=torch.float32, device=device)
    first_top1 = torch.full((n,), NONE, dtype=torch.uint8, device=device)
    last_top1 = torch.full((n,), NONE, dtype=torch.uint8, device=device)
    psum = torch.zeros(n, C.N_CLASSES, dtype=torch.float32, device=device)
    n_views = torch.zeros(n, dtype=torch.int16, device=device)

    snap_n = torch.zeros(len(horizons), n, dtype=torch.int16, device=device)
    snap_mean = torch.full((len(horizons), n), NONE, dtype=torch.uint8, device=device)
    snap_last = torch.full((len(horizons), n), NONE, dtype=torch.uint8, device=device)

    Tw = TF.inv(geo.velo_to_world(anchor.native))
    frames, dt_prev, h = [], -INF, 0
    for f in fut:
        dt = float(f.time_s - t0)
        assert dt > dt_prev, f"future frames must be time-ordered: {dt} after {dt_prev}"
        while h < len(horizons) and horizons[h] < dt:            # close every horizon this frame leaves behind
            snap_n[h] = n_views
            seen = n_views > 0
            snap_mean[h][seen] = psum[seen].argmax(1).to(torch.uint8)
            snap_last[h] = last_top1
            h += 1
        dt_prev = dt
        cv = views.get(f)
        Tt = torch.as_tensor(Tw @ geo.velo_to_world(f.native), device=device)
        flat = _voxel_of(cv.pts_velo @ Tt[:3, :3].T + Tt[:3, 3])
        ok = flat >= 0
        hit = flat[ok]
        geo_first[hit] = torch.minimum(geo_first[hit], torch.full_like(geo_first[hit], dt))
        iflat = flat[cv.in_image_index]
        ihit = iflat[iflat >= 0]
        img_first[ihit] = torch.minimum(img_first[ihit], torch.full_like(img_first[ihit], dt))
        vflat = flat[cv.vis_index]
        keep = vflat >= 0
        uq, inv = torch.unique(vflat[keep], return_inverse=True)
        if len(uq):
            csum = torch.zeros(len(uq), C.N_CLASSES, device=device).index_add_(0, inv, cv.cos[keep])
            cnt = torch.bincount(inv, minlength=len(uq)).float()
            prob = torch.softmax(C.LOGIT_SCALE * csum / cnt[:, None], -1)
            am = prob.argmax(1).to(torch.uint8)
            fresh = n_views[uq] == 0
            vis_first[uq[fresh]] = dt
            first_top1[uq[fresh]] = am[fresh]
            psum[uq] += prob
            last_top1[uq] = am
            n_views[uq] += 1
        frames.append({"j": f.index, "dt_s": round(dt, 3), "returns": int(len(cv.pts_velo)),
                       "in_image": cv.n_in_image, "visible": int(len(cv.vis_index)),
                       "returns_in_grid": int(ok.sum()), "in_image_in_grid": int((iflat >= 0).sum()),
                       "visible_in_grid": int(keep.sum())})
    while h < len(horizons):                                      # horizons beyond the last future frame
        snap_n[h] = n_views
        seen = n_views > 0
        snap_mean[h][seen] = psum[seen].argmax(1).to(torch.uint8)
        snap_last[h] = last_top1
        h += 1

    rows = (rows_mask if rows_mask is not None else torch.ones(n, dtype=torch.bool, device=device)).nonzero(as_tuple=True)[0]
    return RevealRecord(flat_index=rows.to(torch.int32).cpu().numpy(),
                        geo_first_dt=geo_first[rows].cpu().numpy(),
                        img_first_dt=img_first[rows].cpu().numpy(),
                        vis_first_dt=vis_first[rows].cpu().numpy(),
                        vis_n_views=snap_n[:, rows].T.cpu().numpy(),
                        mean_top1=snap_mean[:, rows].T.cpu().numpy(),
                        last_top1=snap_last[:, rows].T.cpu().numpy(),
                        first_top1=first_top1[rows].cpu().numpy(),
                        frames=frames,
                        mean_prob=(psum[rows] / n_views[rows].clamp(min=1)[:, None]).cpu().numpy()
                        if want_mean_prob else None)


def nested_levels_ok(r: RevealRecord) -> bool:
    """geometric <= in_frustum <= visible: a camera-visible return is in the image, and is a return."""
    return bool(np.all(r.geo_first_dt <= r.img_first_dt + 1e-6) and np.all(r.img_first_dt <= r.vis_first_dt + 1e-6))


@torch.no_grad()
def validate_against_frozen_targets(drive: str, ids: Sequence[int], device, cfg: C.TargetConfig = C.TargetConfig(),
                                    paths: C.Paths = C.Paths()) -> Dict:
    """The `visible` level must reproduce the frozen delayed-target cache on the cache's own voxels.

    Checks, per anchor: identical first-view times, identical view counts, and the same mean-probability
    argmax at the 10 s horizon."""
    from tools.semantic_completion.evidence import trident as TR
    from core.datasets import kitti360_splits as SRC
    dev = torch.device(device)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    stream = DriveStream(drive)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    views = ViewCache(geo, text, cfg, dev)
    hi = len(G.HORIZONS_S) - 1
    rows = []
    for t in ids:
        with np.load(paths.target_path(drive, int(t))) as z:
            idx = z["flat_index"].astype(np.int64)
            first = z["first_dt_s"].astype(np.float32)
            nv = z["n_views"].astype(np.int64)
            prob_top1 = z["prob"].astype(np.float32).argmax(1).astype(np.uint8)
        m = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)
        m[torch.from_numpy(idx).to(dev)] = True
        r = reveal_record(stream, geo, int(t), views, m, cfg, dev)
        assert np.array_equal(r.flat_index.astype(np.int64), np.sort(idx)), "row order"
        order = np.argsort(idx)
        rows.append({"t": int(t), "n": int(len(idx)),
                     "first_dt_max_abs_diff": float(np.abs(r.vis_first_dt - first[order]).max()),
                     "n_views_equal": bool(np.array_equal(r.vis_n_views[:, hi].astype(np.int64), nv[order])),
                     "mean_top1_agree": float((r.mean_top1[:, hi] == prob_top1[order]).mean()),
                     "all_have_a_view": bool((r.vis_n_views[:, hi] >= 1).all()),
                     "levels_nested": nested_levels_ok(r)})
    ok = all(x["first_dt_max_abs_diff"] <= 1e-2 and x["n_views_equal"] and x["mean_top1_agree"] >= 0.999
             and x["all_have_a_view"] and x["levels_nested"] for x in rows)
    return {"drive": drive, "n_checked": len(rows), "reproduces_frozen_targets": ok, "rows": rows}


__all__ = ["RevealRecord", "reveal_record", "nested_levels_ok", "validate_against_frozen_targets"]
