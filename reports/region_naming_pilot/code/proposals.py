"""Causal, class-agnostic region proposals: SAM masks -> lifted 3D support -> cross-frame regions.

Nothing here reads a class prediction, a text embedding or a human label. The proposer is frozen SAM
ViT-H automatic mask generation, the same checkpoint the frozen teacher uses, run on each of the five
causal frames independently.

Occlusion is handled by construction, not by a test afterwards. A mask pixel becomes 3D support only if
it is a ray the frozen mapper **accepts** in that same frame, and its 3D point is that frame's own
visible surface (``p = dir * s*z``). A pixel is therefore never attributed to a surface behind what the
frame actually sees, and a mask is never extruded along the viewing direction.

Ownership is deterministic twice over:
  * overlapping SAM masks are painted largest-area-first, so the smallest mask containing a pixel owns it;
  * a world key touched by several regions goes to the one with the most accepted rays (ties: lower id).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import legacy as L
import paths as P

SETTINGS = {
    "proposer": "SAM ViT-H SamAutomaticMaskGenerator (vendored in third_party/Trident)",
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 0,
    "min_mask_region_area": 200,          # native pixels; below this a mask cannot carry 3D support
    "overlap_ownership": "paint largest area first; smallest mask wins a pixel",
    "association": "3D world-key IoU over lifted visible support, all frame pairs in the clip",
    "assoc_iou_min": 0.15,                # calibration anchors only
    "assoc_rgb_max": 0.22,                # mean-RGB L2 in [0,1] units, appearance guard against false links
    "min_support_voxels": 3,              # a region needs this many evidence voxels to seed a name
}


# ---- SAM -------------------------------------------------------------------------------
def load_sam(device: str):
    import sys
    if P.SAM_PKG not in sys.path:
        sys.path.insert(0, P.SAM_PKG)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=P.SAM_CKPT).to(device=device).eval()
    return SamAutomaticMaskGenerator(
        sam, points_per_side=SETTINGS["points_per_side"], pred_iou_thresh=SETTINGS["pred_iou_thresh"],
        stability_score_thresh=SETTINGS["stability_score_thresh"], crop_n_layers=SETTINGS["crop_n_layers"],
        min_mask_region_area=SETTINGS["min_mask_region_area"])


def paint_masks(records: List[dict], hw: Tuple[int, int]) -> np.ndarray:
    """``[H, W] int16`` mask id per native pixel, -1 where SAM proposed nothing.

    Masks are painted in descending area so that the smallest mask containing a pixel owns it.
    Ids index ``records`` after the area sort, which is also the order stored on disk.
    """
    order = np.argsort([-int(r["area"]) for r in records], kind="stable")
    lab = np.full(hw, -1, np.int16)
    for new_id, k in enumerate(order):
        lab[records[k]["segmentation"]] = new_id
    return lab, [records[k] for k in order]


def to_lattice(label_native: np.ndarray, lattice_hw: Tuple[int, int], device) -> torch.Tensor:
    """Nearest-resample the native mask-id image onto the LingBot lattice.

    Uses the same normalised-coordinate convention as the frozen teacher sampler
    (``evidence.trident.sample_at_lattice``): the lattice is a full-frame resize of the native image.
    """
    LH, LW = (int(x) for x in lattice_hw)
    ys = (torch.arange(LH, device=device, dtype=torch.float32) + 0.5) / LH * 2 - 1
    xs = (torch.arange(LW, device=device, dtype=torch.float32) + 0.5) / LW * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None]
    src = torch.from_numpy(label_native.astype(np.float32)).to(device)[None, None]
    return F.grid_sample(src, grid, mode="nearest", align_corners=False)[0, 0].round().to(torch.int64)


# ---- cross-frame association -----------------------------------------------------------
@dataclass
class FrameMask:
    frame: int                      # clip slot 0..4
    mask_id: int                    # id within the frame
    keys: torch.Tensor              # [n] int64 world keys of its accepted rays (unique)
    votes: torch.Tensor             # [n] int32 accepted rays per key
    area_native: int                # owned pixels, after the overlap painting
    mean_rgb: np.ndarray            # [3] in [0, 1], over the owned pixels
    bbox_native: Tuple[int, int, int, int]   # x0, y0, w, h of the owned pixels


def key_iou(a: FrameMask, b: FrameMask) -> float:
    inter = torch.isin(a.keys, b.keys).sum().item()
    union = len(a.keys) + len(b.keys) - inter
    return inter / union if union else 0.0


class Union:
    def __init__(self, n): self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i
    def join(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b: self.p[max(a, b)] = min(a, b)          # deterministic: the lower index becomes the root


def associate(fms: List[FrameMask]) -> Tuple[List[int], Dict]:
    """Link frame masks across frames by 3D support overlap plus an appearance guard. Returns region ids."""
    u = Union(len(fms))
    links, rejected_rgb = 0, 0
    for i in range(len(fms)):
        for j in range(i + 1, len(fms)):
            if fms[i].frame == fms[j].frame:
                continue
            if key_iou(fms[i], fms[j]) < SETTINGS["assoc_iou_min"]:
                continue
            if float(np.linalg.norm(fms[i].mean_rgb - fms[j].mean_rgb)) > SETTINGS["assoc_rgb_max"]:
                rejected_rgb += 1; continue
            u.join(i, j); links += 1
    roots = sorted({u.find(i) for i in range(len(fms))})
    remap = {r: k for k, r in enumerate(roots)}
    rid = [remap[u.find(i)] for i in range(len(fms))]
    # diagnostics: a region holding two masks of one frame was merged; non-contiguous frames are inconsistent
    per_region_frames: Dict[int, List[int]] = {}
    for k, r in enumerate(rid):
        per_region_frames.setdefault(r, []).append(fms[k].frame)
    merged = sum(1 for fr in per_region_frames.values() if len(fr) != len(set(fr)))
    noncontig = sum(1 for fr in per_region_frames.values()
                    if len(set(fr)) > 1 and sorted(set(fr)) != list(range(min(fr), max(fr) + 1)))
    return rid, {"n_frame_masks": len(fms), "n_regions": len(roots), "n_links": links,
                 "n_links_rejected_by_rgb": rejected_rgb,
                 "n_regions_merging_two_masks_of_one_frame": merged,
                 "n_regions_with_non_contiguous_frames": noncontig,
                 "n_single_frame_regions": sum(1 for fr in per_region_frames.values() if len(set(fr)) == 1)}


# ---- the per-anchor build --------------------------------------------------------------
def native_image(key: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(os.path.join(P.RGB_ROOT, f"{key}.png")).convert("RGB"), np.uint8)


def build_anchor(t: int, amg, device, T_cam_to_grid: np.ndarray) -> Dict:
    """Region proposals of one anchor. Everything returned is derived from the five causal frames only."""
    clip = L.LocalClip.load(L.C.Paths().clip_path(P.DRIVE, t))
    S, _ = L.load_hard_map_flat(t, device)
    lattice_hw = clip.depth_canonical.shape[1:]
    endpoints = L.LF.ray_endpoints(clip, device)                 # per frame: (world keys [n], lattice pixel [n])

    fms: List[FrameMask] = []
    per_frame, painted = [], []
    for q, key in enumerate(clip.keys):
        img = native_image(str(key))
        recs = amg.generate(img)
        lab_native, recs = paint_masks(recs, img.shape[:2])
        painted.append(lab_native)
        lab_lat = to_lattice(lab_native, lattice_hw, device).reshape(-1)
        keys_q, pix_q = endpoints[q]
        mid = lab_lat[pix_q]                                     # mask id per accepted ray
        rgb = img.astype(np.float32) / 255.0
        n_masked_rays = int((mid >= 0).sum())
        for m in range(len(recs)):
            sel = (mid == m).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            uq, cnt = torch.unique(keys_q[sel], return_counts=True)
            own = lab_native == m                                # the pixels this mask actually owns
            ys, xs = own.nonzero()
            if len(ys) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            fms.append(FrameMask(q, m, uq, cnt.to(torch.int32), int(own.sum()),
                                 rgb[own].mean(0), bbox))
        per_frame.append({"frame": q, "key": str(key), "native_hw": list(img.shape[:2]),
                          "n_sam_masks": len(recs), "n_accepted_rays": int(len(pix_q)),
                          "n_accepted_rays_in_a_mask": n_masked_rays,
                          "ray_mask_coverage": n_masked_rays / max(len(pix_q), 1),
                          "pixel_mask_coverage": float((lab_native >= 0).mean())})

    rid, assoc = associate(fms)

    # Every accepted ray defines the voxel set (that is the frozen evidence rule); only rays that fell
    # inside a SAM mask vote on ownership, so a voxel seen exclusively through unmasked pixels stays -1.
    n_regions = assoc["n_regions"]
    uk = torch.unique(torch.cat([k for k, _ in endpoints]))
    if fms and n_regions:
        allk = torch.cat([f.keys for f in fms])
        allv = torch.cat([f.votes for f in fms]).to(torch.float32)
        allr = torch.cat([torch.full_like(f.keys, rid[k]) for k, f in enumerate(fms)])
        row = torch.searchsorted(uk, allk)
        assert bool((uk[row] == allk).all()), "a mask ray key is not an accepted-ray key"
        tally = torch.zeros(len(uk) * n_regions, dtype=torch.float32, device=device)
        tally.index_add_(0, row * n_regions + allr, allv)
        tally = tally.reshape(len(uk), n_regions)
        owner = torch.where(tally.sum(1) > 0, tally.argmax(1), torch.full((len(uk),), -1, dtype=torch.int64, device=device))
    else:
        owner = torch.full((len(uk),), -1, dtype=torch.int64, device=device)

    gk = L.LF.grid_world_keys(np.linalg.inv(np.asarray(T_cam_to_grid, np.float64)), device)
    pos = torch.searchsorted(uk, gk).clamp_(max=max(len(uk) - 1, 0))
    hit = (uk[pos] == gk) if len(uk) else torch.zeros_like(gk, dtype=torch.bool)
    voxel_region = torch.full((L.N_VOXELS,), -1, dtype=torch.int64, device=device)
    voxel_region[hit] = owner[pos[hit]]
    support = hit & S                                            # == the frozen evidence set
    voxel_region[~support] = -1
    n_unowned_keys = int((owner < 0).sum())

    ev = L.LF.load_evidence(L.C.Paths().evidence_path(P.DRIVE, t), device)
    E = ev.dense_mask()
    assert bool((support == E).all()), "lifted support is not the frozen evidence set"

    regions = []
    for r in range(n_regions):
        members = [k for k in range(len(fms)) if rid[k] == r]
        vox = (voxel_region == r).nonzero(as_tuple=True)[0]
        ranked = sorted(members, key=lambda k: (-fms[k].area_native, fms[k].frame))
        best = ranked[0]
        alt = next((k for k in ranked[1:] if fms[k].frame != fms[best].frame), None)
        regions.append({"region_id": r, "frames": sorted({fms[k].frame for k in members}),
                        "frame_masks": [[fms[k].frame, fms[k].mask_id, fms[k].area_native] for k in members],
                        "n_evidence_voxels": int(len(vox)), "n_lifted_keys": int(sum(len(fms[k].keys) for k in members)),
                        "best_frame": fms[best].frame, "best_mask_id": fms[best].mask_id,
                        "best_key": str(clip.keys[fms[best].frame]),
                        "best_bbox": list(fms[best].bbox_native), "best_area": fms[best].area_native,
                        "mean_rgb": [round(float(v), 4) for v in fms[best].mean_rgb],
                        "alt": None if alt is None else {"frame": fms[alt].frame, "mask_id": fms[alt].mask_id,
                                                         "key": str(clip.keys[fms[alt].frame]),
                                                         "bbox": list(fms[alt].bbox_native)}})

    n_no_ev = sum(1 for r in regions if r["n_evidence_voxels"] == 0)
    stats = {"t": t, "keys": [str(k) for k in clip.keys], "lattice_hw": list(lattice_hw),
             "per_frame": per_frame, **assoc,
             "n_regions_with_rgb_support_no_evidence": n_no_ev,
             "n_regions_seeding": sum(1 for r in regions if r["n_evidence_voxels"] >= SETTINGS["min_support_voxels"]),
             "n_evidence_voxels": int(E.sum()),
             "n_evidence_voxels_owned": int((voxel_region >= 0).sum()),
             "n_accepted_ray_keys": int(len(uk)), "n_keys_with_no_mask_vote": n_unowned_keys,
             "evidence_ownership_fraction": float((voxel_region >= 0).sum() / max(int(E.sum()), 1))}
    return {"stats": stats, "regions": regions, "voxel_region": voxel_region.cpu().numpy().astype(np.int32),
            "evidence_flat": ev.flat_index.cpu().numpy().astype(np.int32),
            "label_native": np.stack(painted).astype(np.int16)}


def save(t: int, out: Dict) -> str:
    path = P.under_experiment(os.path.join(P.PROPOSALS, f"{t:05d}.npz"))
    np.savez_compressed(path + ".tmp.npz", voxel_region=out["voxel_region"], evidence_flat=out["evidence_flat"],
                        label_native=out["label_native"],
                        stats=json.dumps(out["stats"]), regions=json.dumps(out["regions"]))
    os.replace(path + ".tmp.npz", path)
    return path


def load(t: int) -> Dict:
    with np.load(os.path.join(P.PROPOSALS, f"{t:05d}.npz"), allow_pickle=False) as z:
        return {"voxel_region": z["voxel_region"], "evidence_flat": z["evidence_flat"],
                "label_native": z["label_native"],
                "stats": json.loads(str(z["stats"])), "regions": json.loads(str(z["regions"]))}
