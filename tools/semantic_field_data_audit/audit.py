"""B4 training-data audit: read-only, CPU only, no model execution.

Answers what supervision B4 actually used and what extra KITTI-360 data is accessible. Every quantity is read from
the existing caches, run records and manifests; the sampler is replayed exactly (same torch CPU generator calls as
``tools/semantic_field/training/trainer.py``) and checked against each run's ``draw_digest`` and logged per-step
target counts. Teacher classes are ``argmax cos(f, text)`` of the frozen Trident-H bank -- pseudo-labels, never GT.
No human semantic label file is opened.

Grid: SSCBench-KITTI-360 velodyne grid of anchor t, 256x256x32 @ 0.2 m, flat C order.
World positions: ``data_poses/<drive>/poses.txt`` translation (KITTI-360 shared world frame, metres).

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python -m tools.semantic_field_data_audit.audit index|replay|curves|geography|dev_mix|cost
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from core.datasets import frames as G6F, kitti360 as K3, kitti360_splits as SRC
from tools.delayed_semantic import config as PILOT
from tools.semantic_completion.dataset import hidden_block_mask
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_field.config.schema import ARTIFACT_DIR as B4_ART, DATA_ROOT as B4_DATA, DEV_DRIVE, REPO_ROOT, SEEDS, TRAIN_DRIVES, MaskConfig, TrainConfig
from tools.semantic_field.training.trainer import to_crop

GRID = PILOT.GRID_DIMS
OUT = os.path.join(REPO_ROOT, "artifacts", "semantic_field_data_audit")
CACHE = os.path.join(B4_DATA, "data_audit_cache")            # small per-drive index files (tens of MB)
K7 = os.path.join(REPO_ROOT, "artifacts", "kitti360_data_scaling_7drives")
NEW_DRIVES = tuple(f"2013_05_28_drive_{d}_sync" for d in ("0000", "0002", "0004", "0005"))
TEMPORAL_BLOCK = 20                                             # anchors (= stream frames ~ 10 s), the 7-drive manifest convention
CELL_M = 51.2                                                   # world XY cell = grid forward extent
GRID_REACH_M = math.hypot(51.2, 25.6)                           # farthest grid point from the sensor
CLASS_BLOCK_MIN = 100                                           # voxel-draws for a block to "contribute" a class (also reported at >= 1)


def _drive_tag(drive: str) -> str:
    return drive[17:21]


# ----------------------------------------------------------------------------------------------------- index
@dataclass
class DriveIndex:
    """Per drive, CSR over anchors. ``ev_*`` rows = evidence voxels; ``d_*`` rows = delayed targets B4 kept."""
    ids: np.ndarray          # [A] stream index t
    native: np.ndarray       # [A] native frame of t
    ev_off: np.ndarray       # [A+1]
    ev_flat: np.ndarray      # [E] int32
    ev_cls: np.ndarray       # [E] uint8 teacher class 0..17
    d_off: np.ndarray        # [A+1]
    d_flat: np.ndarray       # [D] int32
    d_cls: np.ndarray        # [D] uint8 argmax of the stored view-averaged distribution
    d_pmax: np.ndarray       # [D] float16
    d_nviews: np.ndarray     # [D] uint8
    d_v0: np.ndarray         # [D] uint8 first-view top-1
    d_v1: np.ndarray         # [D] uint8 second-view top-1 (255 if a single view)
    d_tier: np.ndarray       # [D] uint8 1 = A, 2 = B
    funnel: Dict[str, int]

    @staticmethod
    def path(drive: str) -> str:
        return os.path.join(CACHE, f"index_{drive}.npz")

    @classmethod
    def load(cls, drive: str) -> "DriveIndex":
        with np.load(cls.path(drive), allow_pickle=False) as z:
            kw = {k: z[k] for k in z.files if k != "funnel"}
            kw["funnel"] = json.loads(str(z["funnel"]))
        return cls(**kw)


def build_index(drive: str, with_targets: bool) -> DriveIndex:
    text, _, _ = TR.load_text_embeddings()
    tiers = json.load(open(os.path.join(PILOT.Paths().reliability, "reliability.json")))["gate"]["delayed_loss_tiers"]
    g15, pp = PILOT.Paths().g15, PILOT.Paths()
    ids = G15B.anchor_ids(drive, g15)
    native, ev_off, ev_flat, ev_cls = [], [0], [], []
    d = {k: [] for k in ("flat", "cls", "pmax", "nviews", "v0", "v1", "tier")}; d_off = [0]
    funnel = {k: 0 for k in ("n_future_hit", "n_conflict", "n_future_hit_not_static", "n_tierA", "n_tierB", "n_target", "n_D", "n_evidence_reobserved",
                             "kept_by_B4", "anchors_with_kept")}
    for t in ids:
        with np.load(g15.evidence_path(drive, t)) as z:
            assert int(z["t"]) == t and int(z["stream_indices"][-1]) == t
            native.append(int(str(z["keys"][-1]).split("_")[-1]))     # "0007_0000000502" (train source) or "0000000704" (0006 source)
            feat = torch.from_numpy(z["feature"].astype(np.float32)); ev_flat.append(z["flat_index"].astype(np.int32))
        ev_cls.append((F.normalize(feat, dim=-1) @ text.T).argmax(1).numpy().astype(np.uint8)); ev_off.append(ev_off[-1] + len(ev_flat[-1]))
        if with_targets:
            with np.load(pp.target_path(drive, t)) as z:
                keep = ~z["in_evidence"] & np.isin(z["tier"], tiers)          # the trainer's filter
                d["flat"].append(z["flat_index"][keep].astype(np.int32)); d["cls"].append(z["prob"][keep].astype(np.float32).argmax(1).astype(np.uint8))
                d["pmax"].append(z["prob"][keep].astype(np.float32).max(1).astype(np.float16)); d["nviews"].append(z["n_views"][keep])
                d["v0"].append(z["view0_top1"][keep]); d["v1"].append(z["view1_top1"][keep]); d["tier"].append(z["tier"][keep])
                st = json.loads(str(z["stats_json"]))
            for k in list(funnel)[:8]:
                funnel[k] += int(st[k])
            funnel["kept_by_B4"] += int(keep.sum()); funnel["anchors_with_kept"] += int(keep.any())
            d_off.append(d_off[-1] + int(keep.sum()))
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    idx = DriveIndex(np.asarray(ids), np.asarray(native), np.asarray(ev_off, np.int64), cat(ev_flat, np.int32), cat(ev_cls, np.uint8),
                     np.asarray(d_off, np.int64), cat(d["flat"], np.int32), cat(d["cls"], np.uint8), cat(d["pmax"], np.float16),
                     cat(d["nviews"], np.uint8), cat(d["v0"], np.uint8), cat(d["v1"], np.uint8), cat(d["tier"], np.uint8), funnel)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(DriveIndex.path(drive), **{k: v for k, v in idx.__dict__.items() if k != "funnel"}, funnel=np.asarray(json.dumps(funnel)))
    return idx


# ----------------------------------------------------------------------------------------------------- replay
def replay_seed(seed: int, drives: Dict[str, DriveIndex], cfg: TrainConfig = TrainConfig(), mask: MaskConfig = MaskConfig()) -> Dict:
    """Exact CPU replay of the trainer's data draws. Returns per-step / per-anchor counts and verification flags."""
    run = json.load(open(os.path.join(B4_ART, "runs", f"train_B4_seed{seed}.json")))
    anchors = [(dr, a) for dr in TRAIN_DRIVES for a in range(len(drives[dr].ids))]            # trainer order
    ev_base = np.cumsum([0] + [len(drives[dr].ev_flat) for dr in TRAIN_DRIVES]); d_base = np.cumsum([0] + [len(drives[dr].d_flat) for dr in TRAIN_DRIVES])
    base = {dr: (int(ev_base[i]), int(d_base[i])) for i, dr in enumerate(TRAIN_DRIVES)}
    A, K = len(anchors), 18
    gen, gen_d = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)
    dig = hashlib.sha256()
    draws = np.zeros(A, np.int64); m_cls = np.zeros((A, K), np.int64); d_cls = np.zeros((A, K), np.int64)
    m_seen = np.zeros(int(ev_base[-1]), bool); d_seen = np.zeros(int(d_base[-1]), bool)
    step = {k: np.zeros(cfg.steps, np.float64) for k in ("n_masked", "n_delayed", "masked_majority_share", "delayed_majority_share")}
    per_draw = {k: [] for k in ("hidden_in_crop", "delayed_in_crop", "delayed_avail", "hidden_avail")}
    for it in range(cfg.steps):
        picks = torch.randint(A, (cfg.anchors_per_step,), generator=gen).tolist()
        hm, hd = np.zeros(K, np.int64), np.zeros(K, np.int64)
        for i in picks:
            dr, a = anchors[i]; ix = drives[dr]; t = int(ix.ids[a]); draws[i] += 1
            origin = tuple(int(torch.randint(0, g - c + 1, (1,), generator=gen)) for g, c in zip(GRID, cfg.crop))
            e0, e1 = int(ix.ev_off[a]), int(ix.ev_off[a + 1]); flat = torch.from_numpy(ix.ev_flat[e0:e1].astype(np.int64))
            hide = hidden_block_mask(flat, GRID, mask.block_voxels, mask.hide_probability, gen)
            _, vloc = to_crop(flat[~hide], origin, cfg.crop)
            hin, hloc = to_crop(flat[hide], origin, cfg.crop)
            pick = torch.randperm(len(hloc), generator=gen)[:cfg.max_masked_targets]
            rows = hide.nonzero(as_tuple=True)[0][hin][pick].numpy()
            p0, p1 = int(ix.d_off[a]), int(ix.d_off[a + 1]); dflat = torch.from_numpy(ix.d_flat[p0:p1].astype(np.int64))
            din, dloc = to_crop(dflat, origin, cfg.crop)
            pick_d = torch.randperm(len(dloc), generator=gen_d)[:cfg.max_delayed_targets]
            drows = din.nonzero(as_tuple=True)[0][pick_d].numpy()
            dig.update(f"{_drive_tag(dr)}:{t}:{origin}:{len(vloc)}:{len(hloc)};".encode())
            c = np.bincount(ix.ev_cls[e0:e1][rows], minlength=K); m_cls[i] += c; hm += c
            c = np.bincount(ix.d_cls[p0:p1][drows], minlength=K); d_cls[i] += c; hd += c
            m_seen[base[dr][0] + e0 + rows] = True; d_seen[base[dr][1] + p0 + drows] = True
            per_draw["hidden_in_crop"].append(len(hloc)); per_draw["delayed_in_crop"].append(len(dloc))
            per_draw["hidden_avail"].append(int(hide.sum())); per_draw["delayed_avail"].append(p1 - p0)
        step["n_masked"][it], step["n_delayed"][it] = hm.sum(), hd.sum()
        step["masked_majority_share"][it] = hm.max() / max(hm.sum(), 1); step["delayed_majority_share"][it] = hd.max() / max(hd.sum(), 1)
    logged_m = np.asarray([r["n_masked"] for r in run["log"]]); logged_d = np.asarray([r["n_delayed"] for r in run["log"]])
    return {"seed": seed, "digest_match": dig.hexdigest() == run["draw_digest"], "n_masked_match": bool(np.array_equal(logged_m, step["n_masked"])),
            "n_delayed_match": bool(np.array_equal(logged_d, step["n_delayed"])), "draws": draws, "m_cls": m_cls, "d_cls": d_cls,
            "m_seen": m_seen, "d_seen": d_seen, "step": step, "per_draw": {k: np.asarray(v) for k, v in per_draw.items()}}


def _blocks(drives: Dict[str, DriveIndex], positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """[A] temporal block id (drive, rank // 20) and [A] spatial cell id (51.2 m world XY, revisit-aware) in trainer order."""
    tb, sc = [], []
    for dr in TRAIN_DRIVES:
        n = len(drives[dr].ids); tb += [f"{_drive_tag(dr)}:{r // TEMPORAL_BLOCK}" for r in range(n)]
        cell = np.floor(positions[dr][:, :2] / CELL_M).astype(np.int64); sc += [f"{x}:{y}" for x, y in cell]
    return np.unique(tb, return_inverse=True)[1], np.unique(sc, return_inverse=True)[1]


def _group_count(values: np.ndarray, groups: np.ndarray, minimum: int) -> np.ndarray:
    """[K] number of groups whose summed per-anchor count of class k is >= minimum."""
    g = np.zeros((groups.max() + 1, values.shape[1]), np.int64); np.add.at(g, groups, values)
    return (g >= minimum).sum(0)


# ----------------------------------------------------------------------------------------------------- geography
def stream_natives(drive: str) -> List[int]:
    """The k360_train stream definition (core/datasets/kitti360_folds.py): every 5th SSCBench index with an image."""
    frames = K3.pose_frames(os.path.join(SRC.SSCBENCH_ROOT, "data_poses", drive, "poses.txt"))
    root = os.path.join(G6F.KITTI360_ROOT, "data_2d_raw", drive, "image_00", "data_rect")
    out = []
    for idx in range(0, len(frames) - 1, 5):
        n = int(K3.sscbench_to_native(idx, frames))
        if os.path.isfile(os.path.join(root, f"{n:010d}.png")):
            out.append(n)
    return out


def pose_positions(drive: str) -> Dict[int, np.ndarray]:
    a = np.loadtxt(os.path.join(SRC.SSCBENCH_ROOT, "data_poses", drive, "poses.txt"))
    return {int(r[0]): r[1:].reshape(3, 4)[:, 3] for r in a}


def geography(drives: Dict[str, DriveIndex]) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    full = json.load(open(os.path.join(K7, "manifests", "full.json")))["drives"]
    cov = json.load(open(os.path.join(K7, "manifests", "coverage.json")))["per_drive"]
    anchor_pos, checks = {}, {}
    for dr in TRAIN_DRIVES + NEW_DRIVES + (DEV_DRIVE,):
        pos = pose_positions(dr)
        if dr in drives:
            natives = drives[dr].native
            if dr != DEV_DRIVE:                       # the manifest's anchor definition reproduces B4's anchors on the current drives
                sn = stream_natives(dr); checks[dr] = bool(list(full[dr]) == list(drives[dr].ids) and [sn[i] for i in full[dr]] == list(natives))
        else:
            sn = stream_natives(dr); natives = np.asarray([sn[i] for i in full[dr]])
            checks[dr] = len(sn) == cov[dr]["stream_frames"]
        anchor_pos[dr] = np.stack([pos[int(n)] for n in natives])
    dev_all = np.stack(list(pose_positions(DEV_DRIVE).values()))[:, :2]
    dev_tree = cKDTree(dev_all); cur_tree = cKDTree(np.concatenate([anchor_pos[d][:, :2] for d in TRAIN_DRIVES]))
    cur_cells = {tuple(c) for d in TRAIN_DRIVES for c in np.floor(anchor_pos[d][:, :2] / CELL_M).astype(int)}
    dev_cells = {tuple(c) for c in np.floor(dev_all / CELL_M).astype(int)}
    rows = []
    for dr in TRAIN_DRIVES + NEW_DRIVES:
        p = anchor_pos[dr][:, :2]; dd, _ = dev_tree.query(p); dc, _ = cur_tree.query(p)
        cells = {tuple(c) for c in np.floor(p / CELL_M).astype(int)}
        clear = dd > 2 * GRID_REACH_M
        rows.append({"drive": _drive_tag(dr), "pool": "current" if dr in TRAIN_DRIVES else "candidate", "anchors": len(p),
                     "stream_definition_check": checks[dr], "temporal_blocks_20": int(math.ceil(len(p) / TEMPORAL_BLOCK)),
                     "path_m_at_anchors": round(float(np.linalg.norm(np.diff(anchor_pos[dr], axis=0), axis=1).sum()), 0),
                     "cells_51m": len(cells), "cells_new_vs_current": len(cells - cur_cells) if dr not in TRAIN_DRIVES else len(cells),
                     "cells_shared_with_0006_trajectory": len(cells & dev_cells),
                     "frac_anchors_within_25m_of_0006": round(float((dd <= 25).mean()), 4),
                     "frac_anchors_within_57m_of_0006": round(float((dd <= GRID_REACH_M).mean()), 4),
                     "frac_anchors_within_114m_of_0006": round(float((dd <= 2 * GRID_REACH_M).mean()), 4),
                     "anchors_beyond_114m_of_0006": int(clear.sum()),
                     "temporal_blocks_beyond_114m": int(len({r // TEMPORAL_BLOCK for r in np.nonzero(clear)[0]})),
                     "frac_anchors_within_57m_of_current_train": round(float((dc <= GRID_REACH_M).mean()), 4) if dr not in TRAIN_DRIVES else None})
    return rows, anchor_pos


# ----------------------------------------------------------------------------------------------------- helpers
def _write_csv(name: str, rows: Sequence[Dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _phrases() -> List[str]:
    return [str(p) for p in TR.load_text_embeddings()[1]]


def run_curves() -> List[Dict]:
    rows = []
    for s in SEEDS:
        log = json.load(open(os.path.join(B4_ART, "runs", f"train_B4_seed{s}.json")))["log"]
        for w0 in list(range(0, 6000, 500)):
            win = log[w0:w0 + 500]; mid = w0 + 250
            r = {"seed": s, "steps": f"{w0 + 1}-{w0 + 500}", "lr_mid": round(1e-3 * 0.5 * (1 + math.cos(math.pi * mid / 6000)), 6)}
            for k in ("loss", "masked", "masked_text", "masked_feature", "masked_teacher_agree", "delayed", "delayed_agree", "n_masked", "n_delayed"):
                v = np.asarray([x[k] for x in win if k in x], np.float64); r[k] = round(float(v.mean()), 4); r[k + "_sd"] = round(float(v.std()), 4)
            r["masked_agree_voxel_weighted"] = round(float(sum(x["masked_teacher_agree"] * x["n_masked"] for x in win) / sum(x["n_masked"] for x in win)), 4)
            r["steps_without_masked_term"] = sum("masked" not in x for x in win); r["steps_without_delayed_term"] = sum("delayed" not in x for x in win)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["index", "analyse"]); args = ap.parse_args()
    torch.set_num_threads(16)
    if args.stage == "index":
        for dr in TRAIN_DRIVES:
            build_index(dr, with_targets=True); print("indexed", dr, flush=True)
        build_index(DEV_DRIVE, with_targets=False); print("indexed", DEV_DRIVE, flush=True)
        return
    drives = {dr: DriveIndex.load(dr) for dr in TRAIN_DRIVES + (DEV_DRIVE,)}
    names = _phrases(); summary: Dict = {}
    geo_rows, anchor_pos = geography(drives); _write_csv("geography.csv", geo_rows); summary["geography"] = geo_rows
    curves = run_curves(); _write_csv("training_curves.csv", curves)
    tb, sc = _blocks(drives, anchor_pos)
    reps = [replay_seed(s, drives) for s in SEEDS]
    anchor_drive = np.concatenate([[_drive_tag(dr)] * len(drives[dr].ids) for dr in TRAIN_DRIVES])
    seed_rows = []
    for r in reps:
        pd_ = r["per_draw"]; st = r["step"]
        row = {"seed": r["seed"], "draw_digest_match": r["digest_match"], "logged_n_masked_match": r["n_masked_match"], "logged_n_delayed_match": r["n_delayed_match"],
               "optimizer_updates": TrainConfig().steps, "anchor_draws": int(r["draws"].sum()), "unique_anchors": int((r["draws"] > 0).sum()),
               "draws_per_anchor_min": int(r["draws"].min()), "draws_per_anchor_median": float(np.median(r["draws"])), "draws_per_anchor_max": int(r["draws"].max()),
               **{f"draws_{d}": int(r["draws"][anchor_drive == d].sum()) for d in map(_drive_tag, TRAIN_DRIVES)},
               "draws_zero_hidden_in_crop": int((pd_["hidden_in_crop"] == 0).sum()), "draws_zero_delayed_in_crop": int((pd_["delayed_in_crop"] == 0).sum()),
               "draws_hidden_cap_hit": int((pd_["hidden_in_crop"] > TrainConfig().max_masked_targets).sum()),
               "draws_delayed_cap_hit": int((pd_["delayed_in_crop"] > TrainConfig().max_delayed_targets).sum()),
               "mean_hidden_in_grid": round(float(pd_["hidden_avail"].mean()), 1), "mean_hidden_in_crop": round(float(pd_["hidden_in_crop"].mean()), 1),
               "mean_delayed_in_grid": round(float(pd_["delayed_avail"].mean()), 1), "mean_delayed_in_crop": round(float(pd_["delayed_in_crop"].mean()), 1),
               "masked_voxel_draws": int(st["n_masked"].sum()), "delayed_voxel_draws": int(st["n_delayed"].sum()),
               "distinct_masked_anchor_voxels": int(r["m_seen"].sum()), "frac_evidence_ever_masked_target": round(float(r["m_seen"].mean()), 4),
               "distinct_delayed_anchor_voxels": int(r["d_seen"].sum()), "frac_kept_delayed_ever_sampled": round(float(r["d_seen"].mean()), 4),
               "steps_zero_masked": int((st["n_masked"] == 0).sum()), "steps_zero_delayed": int((st["n_delayed"] == 0).sum()),
               "median_step_delayed_to_masked_ratio": round(float(np.median(st["n_delayed"] / np.maximum(st["n_masked"], 1))), 3),
               "masked_majority_share_steps_1_200": round(float(st["masked_majority_share"][:200].mean()), 4),
               "masked_majority_share_steps_5001_6000": round(float(st["masked_majority_share"][5000:].mean()), 4)}
        seed_rows.append(row)
    _write_csv("sampling_by_seed.csv", seed_rows); summary["sampling"] = seed_rows
    union_m = np.logical_or.reduce([r["m_seen"] for r in reps]); union_d = np.logical_or.reduce([r["d_seen"] for r in reps])
    summary["union_over_seeds"] = {"distinct_masked_anchor_voxels": int(union_m.sum()), "distinct_delayed_anchor_voxels": int(union_d.sum())}
    # ---- per-drive pool ----
    pool = []
    for dr in TRAIN_DRIVES:
        ix = drives[dr]; n_ev = np.diff(ix.ev_off); n_d = np.diff(ix.d_off)
        pool.append({"drive": _drive_tag(dr), "anchors_cached": len(ix.ids), "evidence_voxels": int(n_ev.sum()), "evidence_per_anchor_mean": round(float(n_ev.mean()), 1),
                     "anchors_empty_evidence": int((n_ev == 0).sum()), **ix.funnel, "kept_delayed_per_anchor_mean": round(float(n_d.mean()), 1),
                     "kept_delayed_per_anchor_p10": int(np.percentile(n_d, 10)), "kept_delayed_per_anchor_p90": int(np.percentile(n_d, 90)),
                     "anchors_empty_kept_delayed": int((n_d == 0).sum())})
    _write_csv("training_pool.csv", pool); summary["pool"] = pool
    # ---- per class ----
    ev_cls = np.concatenate([drives[dr].ev_cls for dr in TRAIN_DRIVES]); d_all = {k: np.concatenate([getattr(drives[dr], k) for dr in TRAIN_DRIVES]) for k in ("d_cls", "d_pmax", "d_nviews", "d_v0", "d_v1")}
    ev_per_anchor = np.stack([np.bincount(drives[dr].ev_cls[drives[dr].ev_off[a]:drives[dr].ev_off[a + 1]], minlength=18) for dr in TRAIN_DRIVES for a in range(len(drives[dr].ids))])
    d_per_anchor = np.stack([np.bincount(drives[dr].d_cls[drives[dr].d_off[a]:drives[dr].d_off[a + 1]], minlength=18) for dr in TRAIN_DRIVES for a in range(len(drives[dr].ids))])
    m_mean = np.mean([r["m_cls"].sum(0) for r in reps], 0); dl_mean = np.mean([r["d_cls"].sum(0) for r in reps], 0)
    m_cls0, d_cls0 = reps[0]["m_cls"], reps[0]["d_cls"]
    two = d_all["d_nviews"] >= 2
    dev = dev_mix(drives[DEV_DRIVE])
    cls_rows = []
    for k, nm in enumerate(names):
        sel_two = two & (d_all["d_cls"] == k)
        cls_rows.append({"class": nm, "evidence_voxels_available": int((ev_cls == k).sum()), "evidence_share": round(float((ev_cls == k).mean()), 4),
                         "evidence_anchors": int((ev_per_anchor[:, k] > 0).sum()), "evidence_temporal_blocks_ge100": int(_group_count(ev_per_anchor, tb, CLASS_BLOCK_MIN)[k]),
                         "masked_voxel_draws_mean_seeds": int(m_mean[k]), "masked_share": round(float(m_mean[k] / m_mean.sum()), 4),
                         "masked_distinct_voxels_seed0": int((ev_cls[reps[0]["m_seen"]] == k).sum()),
                         "masked_temporal_blocks_ge1_seed0": int(_group_count(m_cls0, tb, 1)[k]), "masked_temporal_blocks_ge100_seed0": int(_group_count(m_cls0, tb, CLASS_BLOCK_MIN)[k]),
                         "masked_cells51_ge100_seed0": int(_group_count(m_cls0, sc, CLASS_BLOCK_MIN)[k]),
                         "delayed_voxels_available": int((d_all["d_cls"] == k).sum()), "delayed_share": round(float((d_all["d_cls"] == k).mean()), 4),
                         "delayed_voxel_draws_mean_seeds": int(dl_mean[k]), "delayed_sampled_share": round(float(dl_mean[k] / dl_mean.sum()), 4),
                         "delayed_distinct_voxels_seed0": int((d_all["d_cls"][reps[0]["d_seen"]] == k).sum()),
                         "delayed_temporal_blocks_ge1_seed0": int(_group_count(d_cls0, tb, 1)[k]), "delayed_temporal_blocks_ge100_seed0": int(_group_count(d_cls0, tb, CLASS_BLOCK_MIN)[k]),
                         "delayed_cells51_ge100_seed0": int(_group_count(d_cls0, sc, CLASS_BLOCK_MIN)[k]),
                         "delayed_mean_confidence": round(float(d_all["d_pmax"][d_all["d_cls"] == k].astype(np.float32).mean()), 3) if (d_all["d_cls"] == k).any() else None,
                         "delayed_two_view_pairs": int(sel_two.sum()),
                         "delayed_two_view_top1_agreement": round(float((d_all["d_v0"][sel_two] == d_all["d_v1"][sel_two]).mean()), 3) if sel_two.any() else None,
                         "dev0006_evidence_share": round(float(dev["evidence_share"][k]), 4), "dev0006_masked_hidden_share": round(float(dev["hidden_share"][k]), 4)})
    _write_csv("class_supervision.csv", cls_rows)
    summary["classes"] = {"n_temporal_blocks": int(tb.max() + 1), "n_cells51": int(sc.max() + 1), "dev_masked_diagnostic": {k: v for k, v in dev.items() if not isinstance(v, np.ndarray)},
                          "train_evidence_majority_share": round(float(np.bincount(ev_cls, minlength=18).max() / len(ev_cls)), 4),
                          "train_sampled_masked_majority_share": round(float(m_mean.max() / m_mean.sum()), 4)}
    summary["cost"] = cost_table()
    json.dump(summary, open(os.path.join(OUT, "audit_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


def dev_mix(ix: DriveIndex, mask: MaskConfig = MaskConfig()) -> Dict:
    """Teacher-class mix of 0006 evidence and of the hidden voxels of the frozen masked diagnostic (masks seeded by t)."""
    ev = np.bincount(ix.ev_cls, minlength=18).astype(np.float64); hid = np.zeros(18); n_hidden = 0
    for a, t in enumerate(ix.ids):
        e0, e1 = int(ix.ev_off[a]), int(ix.ev_off[a + 1]); flat = torch.from_numpy(ix.ev_flat[e0:e1].astype(np.int64))
        hide = hidden_block_mask(flat, GRID, mask.block_voxels, mask.hide_probability, torch.Generator().manual_seed(int(t))).numpy()
        if hide.any() and (~hide).any():
            hid += np.bincount(ix.ev_cls[e0:e1][hide], minlength=18); n_hidden += int(hide.sum())
    return {"evidence_share": ev / ev.sum(), "hidden_share": hid / hid.sum(), "hidden_voxels": n_hidden, "hidden_voxels_expected": 5264367,
            "hidden_majority_share": round(float(hid.max() / hid.sum()), 4), "evidence_majority_share": round(float(ev.max() / ev.sum()), 4)}


def cost_table() -> List[Dict]:
    """Measured per-anchor sizes and times on the current drives, extrapolated to the four candidate drives (labelled estimates)."""
    M = "/media/SSD1/MINH_DATASETS"
    cov = json.load(open(os.path.join(K7, "manifests", "coverage.json")))["per_drive"]
    new_anchors = sum(cov[d]["eligible"] for d in NEW_DRIVES); new_frames = sum(cov[d]["stream_frames"] for d in NEW_DRIVES)
    def size_mb(root: str, drive: str) -> float:
        d = os.path.join(root, drive); fs = [os.path.join(d, f) for f in os.listdir(d)]
        return sum(os.path.getsize(f) for f in fs) / len(fs) / 1e6
    def token_mb(drive: str) -> float:
        d = os.path.join(M, "lingbot_feature_first_3d/trident_tokens/k360_train"); fs = [os.path.join(d, f) for f in os.listdir(d) if f.startswith(_drive_tag(drive))]
        return sum(os.path.getsize(f) for f in fs) / len(fs) / 1e6
    logs = os.path.join(REPO_ROOT, "artifacts")
    def per_anchor_s(pattern: str, drive: str, field: str = "seconds", key: str = "n_anchors") -> float:
        r = json.load(open(os.path.join(logs, pattern.format(tag=_drive_tag(drive))))); return r[field] / r.get(key, r.get("written"))
    def text_rate(path: str, rx: str) -> List[float]:
        return [float(x) for x in re.findall(rx, open(os.path.join(logs, path)).read())]
    stages = {
        "trident_h_tokens (per stream frame)": ([token_mb(d) for d in TRAIN_DRIVES], text_rate("feature_first_3d_semantics/logs/extract_0007.log", r"([\d.]+)s/frame") + text_rate("feature_first_3d_semantics/logs/extract_0010_0003.log", r"([\d.]+)s/frame"), new_frames),
        "local_reset_lingbot_clips": ([size_mb(f"{M}/lingbot_gate12_suf/local_lingbot", d) for d in TRAIN_DRIVES], [60 * 0.5 / 193, 60 * 1.5 / 569, 60 * 1.5 / 596], new_anchors),
        "hard_map_inputs": ([size_mb(f"{M}/lingbot_gate12_suf/inputs", d) for d in TRAIN_DRIVES], text_rate("gate12_suf/logs/inputs_0007.log", r"([\d.]+)s each") + text_rate("gate12_suf/logs/inputs_0010.log", r"([\d.]+)s each"), new_anchors),
        "gate15_evidence": ([size_mb(f"{M}/lingbot_gate15_semantic/evidence", d) for d in TRAIN_DRIVES], [per_anchor_s("gate15_semantic_completion/logs/evidence_{tag}_s0.json", d) for d in TRAIN_DRIVES], new_anchors),
        "delayed_targets": ([size_mb(f"{M}/lingbot_delayed_semantic/delayed_targets", d) for d in TRAIN_DRIVES], [per_anchor_s("delayed_semantic_pilot/logs/targets_{tag}_s0of1.json", d) for d in TRAIN_DRIVES], new_anchors),
        "b3_train_cache (B3 arm only)": ([size_mb(f"{M}/lingbot_delayed_semantic/train_cache/student", d) + size_mb(f"{M}/lingbot_delayed_semantic/train_cache/query", d) for d in TRAIN_DRIVES],
                                         [per_anchor_s("delayed_semantic_pilot/logs/train_cache_{tag}_s0.json", d, key="written") for d in TRAIN_DRIVES], new_anchors),
    }
    rows = []
    for name, (mb, sec, n) in stages.items():
        rows.append({"stage": name, "units_needed_new_drives": n, "measured_mb_per_unit_min": round(min(mb), 3), "measured_mb_per_unit_max": round(max(mb), 3),
                     "measured_s_per_unit_min": round(min(sec), 3), "measured_s_per_unit_max": round(max(sec), 3),
                     "estimate_storage_gb_low": round(n * min(mb) / 1e3, 1), "estimate_storage_gb_high": round(n * max(mb) / 1e3, 1),
                     "estimate_single_gpu_hours_low": round(n * min(sec) / 3600, 2), "estimate_single_gpu_hours_high": round(n * max(sec) / 3600, 2)})
    _write_csv("expansion_cost.csv", rows)
    return rows


if __name__ == "__main__":
    main()
