"""Phase-1 experiment: parameter-matched probes on frozen LingBot representations.

Stages
------
1. deterministic chunk splits;
2. frozen inference -> per-chunk cache (LingBot representations, DINO teacher targets,
   predicted geometry, oracle geometry, evaluation-only labels);
3. correspondence sets on the held-out chunks, built once and shared by every
   representation so the comparison is exactly matched;
4. raw-token cross-view diagnostic (no training involved);
5. probe training for each (representation, teacher budget) pair;
6. held-out evaluation, ablations, machine-readable metrics and the gate verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from research.lingbot_semantic_memory.config import (
    ChunkSpec, Phase1Config, REPO_ROOT, RepresentationSpec, collect_provenance,
    set_seed, write_json,
)
from research.lingbot_semantic_memory.dataset_adapter import build_splits
from research.lingbot_semantic_memory.feature_cache import (
    CachedChunk, cache_size_bytes, ensure_cache, load_chunk,
)
from research.lingbot_semantic_memory.hooks import FrozenDinoTeacher, FrozenLingBot
from research.lingbot_semantic_memory.metrics import (
    aggregate, boundary_stats, boundary_stats_from_depth, cross_view_stats,
    feature_diversity, teacher_fidelity,
)
from research.lingbot_semantic_memory.probes import (
    build_matched_probe, cosine_distillation_loss,
)
from research.lingbot_semantic_memory.reprojection import (
    Correspondences, compute_correspondences, patch_center_pixels,
)

TEACHER_NAME = "external_dino_teacher"
BASELINE = "lingbot_encoder"
#: Pre-registered gate thresholds (README "Pre-registration").
CONSISTENCY_GAIN = 1.15
FIDELITY_RETENTION = 0.95
SPARSE_RETENTION = 0.90


# --------------------------------------------------------------------------- #
# Correspondence construction
# --------------------------------------------------------------------------- #
@dataclass
class PairSpec:
    chunk: int
    i: int
    j: int
    gap: int
    band: str  # "short" | "long"


def build_pairs(cfg: Phase1Config, n_chunks: int, chunk_len: int) -> List[PairSpec]:
    """Deterministic, evenly strided frame pairs per chunk and gap."""
    gaps = [(g, "short") for g in cfg.eval.short_gaps] + [(g, "long") for g in cfg.eval.long_gaps]
    per_gap = max(1, cfg.eval.max_pairs_per_chunk // max(len(gaps), 1))
    pairs: List[PairSpec] = []
    for c in range(n_chunks):
        for gap, band in gaps:
            hi = chunk_len - gap
            if hi <= 0:
                continue
            starts = np.linspace(0, hi - 1, num=min(per_gap, hi)).round().astype(int)
            for i in np.unique(starts):
                pairs.append(PairSpec(c, int(i), int(i) + gap, gap, band))
    return pairs


def patch_depth(depth: np.ndarray, grid_hw: Tuple[int, int]) -> torch.Tensor:
    """Depth at each patch centre, ``[S, gh*gw]``."""
    S, H, W = depth.shape
    uv = patch_center_pixels(grid_hw, (H, W))
    x = uv[:, 0].long().clamp(0, W - 1).numpy()
    y = uv[:, 1].long().clamp(0, H - 1).numpy()
    return torch.from_numpy(depth[:, y, x])


def patch_confidence(conf: np.ndarray, grid_hw: Tuple[int, int]) -> torch.Tensor:
    """Depth confidence at each patch centre, ``[S, gh*gw]``.

    ``depth_conf`` is produced with ``conf_activation="expp1"``, i.e. ``1 + exp(raw)``:
    it is a **confidence / precision** in ``[1, inf)`` where larger is better, not an
    uncertainty.  Verified empirically in the Phase-0 inventory.
    """
    S, H, W = conf.shape
    uv = patch_center_pixels(grid_hw, (H, W))
    x = uv[:, 0].long().clamp(0, W - 1).numpy()
    y = uv[:, 1].long().clamp(0, H - 1).numpy()
    return torch.from_numpy(conf[:, y, x])


def correspondence_sets(
    cfg: Phase1Config, chunks: List[CachedChunk], pairs: List[PairSpec],
    geometry: str, conf_filter: bool, device: torch.device,
) -> List[Optional[Correspondences]]:
    """Build one correspondence set per pair for a given geometry source."""
    out: List[Optional[Correspondences]] = []
    cache: Dict[int, Dict[str, torch.Tensor]] = {}
    for ci, ch in enumerate(chunks):
        if geometry == "predicted":
            depth = torch.from_numpy(ch.pred_depth).float()
            extr = torch.from_numpy(ch.pred_extrinsic).float()
            K = torch.from_numpy(ch.pred_intrinsic).float()
        else:
            if ch.oracle_depth is None:
                raise RuntimeError(f"{ch.name} has no oracle geometry cached")
            depth = torch.from_numpy(ch.oracle_depth).float()
            extr = torch.from_numpy(ch.oracle_extrinsic).float()
            K = torch.from_numpy(ch.oracle_intrinsic).float()
        valid = None
        if conf_filter:
            pc = patch_confidence(ch.pred_conf, ch.grid_hw)
            thr = torch.quantile(pc, cfg.eval.conf_percentile / 100.0, dim=1, keepdim=True)
            valid = (pc >= thr)
        cache[ci] = {"depth": depth.to(device), "extr": extr.to(device),
                     "K": K.to(device), "valid": None if valid is None else valid.to(device)}

    for p in pairs:
        c = cache[p.chunk]
        v = c["valid"]
        out.append(compute_correspondences(
            c["depth"][p.i], c["depth"][p.j], c["extr"][p.i], c["extr"][p.j], c["K"],
            chunks[p.chunk].grid_hw,
            occlusion_rel_tol=cfg.eval.occlusion_rel_tol,
            fb_tol_patches=cfg.eval.fb_tol_patches,
            valid_src=None if v is None else v[p.i],
            valid_dst=None if v is None else v[p.j],
        ))
    return out


@torch.no_grad()
def benchmark_frozen(cfg: Phase1Config, chunk: ChunkSpec, lingbot: FrozenLingBot,
                     teacher: FrozenDinoTeacher) -> Dict[str, float]:
    """Time both frozen models on one chunk, after a warm-up pass."""
    from research.lingbot_semantic_memory.dataset_adapter import load_chunk_images
    images = load_chunk_images(cfg.data, chunk)
    lingbot.run_chunk(images[:4], capture=False)
    teacher.encode(images[:4])
    torch.cuda.synchronize()

    t0 = time.time()
    lingbot.run_chunk(images, capture=True)
    torch.cuda.synchronize()
    t_ling = time.time() - t0

    t0 = time.time()
    teacher.encode(images)
    torch.cuda.synchronize()
    t_teach = time.time() - t0
    S = images.shape[0]
    return {"lingbot_ms_per_frame": 1000 * t_ling / S,
            "teacher_ms_per_frame": 1000 * t_teach / S,
            "benchmark_frames": S, "benchmark_chunk": chunk.name}


# --------------------------------------------------------------------------- #
# Probe training
# --------------------------------------------------------------------------- #
def teacher_frame_subset(n_frames: int, budget: float, seed: int) -> np.ndarray:
    """Deterministic uniform selection of the frames that carry a teacher target."""
    if budget >= 1.0:
        return np.arange(n_frames)
    k = max(1, int(round(n_frames * budget)))
    # Uniform stride keeps the subset spread over the sequence; `seed` shifts the
    # phase so budgets do not all start on frame 0.
    offset = seed % max(1, n_frames // k)
    idx = (np.linspace(0, n_frames, num=k, endpoint=False).astype(int) + offset) % n_frames
    return np.unique(idx)


def train_probe(
    cfg: Phase1Config, rep: RepresentationSpec, feats: torch.Tensor, targets: torch.Tensor,
    budget: float, device: torch.device, steps: Optional[int] = None,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Distil the frozen teacher into a parameter-matched probe.

    Args:
        feats: ``[F, P, C]`` frozen representation over all training frames.
        targets: ``[F, P, 768]`` frozen teacher features.
        budget: fraction of training frames that carry a teacher target.

    Returns:
        ``(probe, info)``.
    """
    set_seed(cfg.seed)
    steps = steps if steps is not None else cfg.probe.steps
    probe, hidden = build_matched_probe(rep.dim, cfg.teacher.dim,
                                        cfg.probe.target_params, cfg.probe.max_params)
    probe = probe.to(device).train()
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe.lr,
                            weight_decay=cfg.probe.weight_decay)
    sub = teacher_frame_subset(feats.shape[0], budget, cfg.seed)
    # Index on whatever device the cached features live on; they are moved to the
    # compute device per batch so a large cache need not fit in VRAM.
    sub_t = torch.from_numpy(sub).to(feats.device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def lr_at(s: int) -> float:
        if s < cfg.probe.warmup_steps:
            return cfg.probe.lr * (s + 1) / cfg.probe.warmup_steps
        p = (s - cfg.probe.warmup_steps) / max(1, steps - cfg.probe.warmup_steps)
        return cfg.probe.lr * 0.5 * (1 + math.cos(math.pi * p))

    t0 = time.time()
    losses: List[float] = []
    bf = min(cfg.probe.batch_frames, len(sub))
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        pick = sub_t[torch.randint(len(sub), (bf,), generator=gen).to(feats.device)]
        x = feats[pick].to(device, non_blocking=True).float().reshape(-1, rep.dim)
        y = targets[pick].to(device, non_blocking=True).float().reshape(-1, cfg.teacher.dim)
        loss = cosine_distillation_loss(probe(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.probe.grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()

    probe.eval()
    for p in probe.parameters():
        p.grad = None
    return probe, {
        "hidden": hidden, "params": probe.num_params, "steps": steps,
        "teacher_frames": int(len(sub)), "train_frames": int(feats.shape[0]),
        "final_loss": float(np.mean(losses[-50:])), "train_s": time.time() - t0,
    }


@torch.no_grad()
def probe_features(probe: torch.nn.Module, feats: torch.Tensor, device: torch.device,
                   batch: int = 16) -> torch.Tensor:
    """Apply a probe frame-by-frame, returning ``[S, P, 768]`` on CPU float32."""
    out = []
    for i in range(0, feats.shape[0], batch):
        x = feats[i: i + batch].to(device).float()
        out.append(probe(x).cpu())
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_cross_view(
    feats: List[torch.Tensor], pairs: List[PairSpec],
    corrs: Sequence[Optional[Correspondences]], cfg: Phase1Config, device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Cross-view statistics overall and split by temporal band."""
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    rows: List[Dict[str, float]] = []
    bands: Dict[str, List[Dict[str, float]]] = {"short": [], "long": []}
    by_gap: Dict[int, List[Dict[str, float]]] = {}
    for p, c in zip(pairs, corrs):
        if c is None or len(c) == 0:
            continue
        fs = feats[p.chunk][p.i].to(device)
        fd = feats[p.chunk][p.j].to(device)
        s = cross_view_stats(fs, fd, c, num_negatives=cfg.eval.num_negatives, generator=gen)
        if s is None:
            continue
        rows.append(s)
        bands[p.band].append(s)
        by_gap.setdefault(p.gap, []).append(s)
    out = {"all": aggregate(rows), "short": aggregate(bands["short"]), "long": aggregate(bands["long"])}
    for g, r in sorted(by_gap.items()):
        out[f"gap{g}"] = aggregate(r)
    return out


def eval_fidelity_and_boundary(
    feats: List[torch.Tensor], chunks: List[CachedChunk], device: torch.device,
    max_tokens: int = 400_000,
) -> Dict[str, float]:
    """Teacher fidelity and boundary preservation on held-out frames."""
    preds, teach = [], []
    for f, ch in zip(feats, chunks):
        preds.append(f.reshape(-1, f.shape[-1]))
        teach.append(torch.from_numpy(ch.teacher).float().reshape(-1, ch.teacher.shape[-1]))
    P = torch.cat(preds); T = torch.cat(teach)
    if P.shape[0] > max_tokens:
        idx = torch.randperm(P.shape[0], generator=torch.Generator().manual_seed(0))[:max_tokens]
        P, T = P[idx], T[idx]
    fid = teacher_fidelity(P.to(device), T.to(device))
    fid["diversity"] = feature_diversity(P.to(device))

    # Semantic-label boundaries (KITTI only) and depth-discontinuity boundaries
    # (any dataset with oracle depth) -- the latter is what makes datasets comparable.
    brows, drows = [], []
    for f, ch in zip(feats, chunks):
        lab = None if ch.labels is None else torch.from_numpy(ch.labels).reshape(ch.labels.shape[0], -1)
        dpatch = None
        if ch.oracle_depth is not None:
            dpatch = patch_depth(ch.oracle_depth, ch.grid_hw)
        for s in range(f.shape[0]):
            fs = f[s].to(device)
            if lab is not None:
                b = boundary_stats(fs, lab[s].to(device), ch.grid_hw)
                if b:
                    brows.append(b)
            if dpatch is not None:
                b = boundary_stats_from_depth(fs, dpatch[s].to(device), ch.grid_hw)
                if b:
                    drows.append(b)
    if brows:
        agg = aggregate(brows, weight_key="n_within")
        fid.update({f"boundary_{k}": v for k, v in agg.items() if k in ("within", "across", "margin")})
    if drows:
        agg = aggregate(drows, weight_key="n_within")
        fid.update({f"depth_boundary_{k}": v for k, v in agg.items() if k in ("within", "across", "margin")})
    return fid


def make_figures(
    cfg: Phase1Config, specs: List[ChunkSpec], chunks: List[CachedChunk],
    pairs: List[PairSpec], corr_sets: Dict[str, List[Optional[Correspondences]]],
    probe_out: Dict[str, List[torch.Tensor]], out_dir: str,
) -> List[str]:
    """Cross-view correspondence and similarity figures for a few held-out pairs."""
    from research.lingbot_semantic_memory.dataset_adapter import load_chunk_images
    from research.lingbot_semantic_memory.visualize import (
        correspondence_figure, similarity_figure,
    )
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    imgs = load_chunk_images(cfg.data, specs[0]).numpy()
    ch = chunks[0]
    gh, gw = ch.grid_hw

    # one representative pair per distinct gap in chunk 0
    seen, chosen = set(), []
    for k, p in enumerate(pairs):
        if p.chunk == 0 and p.gap not in seen:
            seen.add(p.gap); chosen.append((k, p))
    for k, p in chosen:
        for geom in ("predicted", "oracle"):
            c = corr_sets[geom][k]
            if c is None or len(c) == 0:
                continue
            f = os.path.join(out_dir, f"corr_gap{p.gap:02d}_{geom}.png")
            correspondence_figure(
                imgs[p.i], imgs[p.j], c, (gh, gw), ch.image_hw, f, n_show=12,
                title=f"{ch.name}  frames {p.i}->{p.j} (gap {p.gap}, {geom} geometry, "
                      f"{len(c)}/{c.num_candidates} patches valid)")
            written.append(f)

        c = corr_sets["predicted"][k]
        if c is None or len(c) == 0 or not probe_out:
            continue
        j = int(len(c) // 2)
        q, tgt = int(c.src_idx[j]), int(c.dst_idx[j])
        feats = {name: {"src": fo[0][p.i], "dst": fo[0][p.j]} for name, fo in probe_out.items()}
        feats[TEACHER_NAME] = {"src": torch.from_numpy(ch.teacher[p.i]).float(),
                               "dst": torch.from_numpy(ch.teacher[p.j]).float()}
        f = os.path.join(out_dir, f"similarity_gap{p.gap:02d}.png")
        similarity_figure(feats, q, (gh, gw), imgs[p.j], tgt, f,
                          title=f"query patch {q}, frames {p.i}->{p.j} (gap {p.gap}); "
                                f"red/cyan x = geometric match")
        written.append(f)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 geometry-conditioned semantic probe study")
    ap.add_argument("--config", default="research/lingbot_semantic_memory/configs/phase1.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: 2 short chunks, 200 probe steps, 1 budget")
    ap.add_argument("--force-cache", action="store_true")
    a = ap.parse_args()

    cfg = Phase1Config.load(os.path.join(REPO_ROOT, a.config))
    if a.device:
        cfg.device = a.device
    if a.seed is not None:
        cfg.seed = a.seed
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.smoke:
        cfg.data.chunk_length = 24
        cfg.data.train_sequences = cfg.data.train_sequences[:2]
        cfg.data.val_chunks = 1
        cfg.probe.steps = 200
        cfg.probe.warmup_steps = 20
        cfg.teacher_budgets = [1.0]
        cfg.eval.max_pairs_per_chunk = 24
        cfg.eval.long_gaps = [8]
        cfg.eval.short_gaps = [1, 3]
        cfg.output_dir = cfg.output_dir + "_smoke"
        cfg.cache_dir = cfg.cache_dir + "_smoke"

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = os.path.join(REPO_ROOT, cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    print(f"== Phase 1 == device={cfg.device} seed={cfg.seed} smoke={a.smoke}")
    train_chunks, val_chunks = build_splits(cfg.data, cfg.seed)
    print(f"[split] train {[c.name for c in train_chunks]}")
    print(f"[split] val   {[c.name for c in val_chunks]}")

    # -- stage 2: frozen inference cache ------------------------------------ #
    timings: Dict[str, float] = {}
    lingbot = FrozenLingBot(cfg.lingbot, cfg.representations, device)
    teacher = FrozenDinoTeacher(cfg.teacher, device)
    print(f"[frozen] lingbot {lingbot.num_params:,} params (0 trainable), "
          f"teacher {teacher.num_params:,} params (0 trainable)")
    ensure_cache(cfg, train_chunks, lingbot, teacher, False, False, timings, a.force_cache)
    ensure_cache(cfg, val_chunks, lingbot, teacher, True, True, timings, a.force_cache)
    # Measured on a fixed chunk regardless of cache hits, so the efficiency table is
    # never empty on a re-run.
    bench = benchmark_frozen(cfg, val_chunks[0], lingbot, teacher)
    print(f"[bench] lingbot {bench['lingbot_ms_per_frame']:.1f} ms/frame, "
          f"teacher {bench['teacher_ms_per_frame']:.1f} ms/frame")
    del lingbot, teacher
    torch.cuda.empty_cache()

    from research.lingbot_semantic_memory.feature_cache import cache_path
    tr = [load_chunk(cache_path(cfg, c)) for c in train_chunks]
    va = [load_chunk(cache_path(cfg, c)) for c in val_chunks]
    grid_hw = va[0].grid_hw
    print(f"[cache] grid={grid_hw} image={va[0].image_hw} size={cache_size_bytes(cfg)/2**30:.2f} GiB")

    # -- stage 3: shared correspondence sets -------------------------------- #
    pairs = build_pairs(cfg, len(va), va[0].pred_depth.shape[0])
    corr_sets: Dict[str, List[Optional[Correspondences]]] = {}
    for geom in ("predicted", "oracle"):
        for cf in (False, True):
            key = f"{geom}{'_conf' if cf else ''}"
            corr_sets[key] = correspondence_sets(cfg, va, pairs, geom, cf, device)
            n = sum(len(c) for c in corr_sets[key] if c)
            frac = float(np.mean([c.valid_fraction for c in corr_sets[key] if c]))
            print(f"[corr] {key:16s} pairs={len(pairs)} matches={n:,} valid_fraction={frac:.3f}")
    primary = corr_sets["predicted"]

    # -- stage 4: raw-token diagnostic -------------------------------------- #
    raw: Dict[str, Dict] = {}
    for rep in cfg.representations:
        feats = [torch.from_numpy(c.reps[rep.name]).float() for c in va]
        raw[rep.name] = {
            "cross_view": eval_cross_view(feats, pairs, primary, cfg, device),
            "diversity": feature_diversity(torch.cat([f.reshape(-1, f.shape[-1]) for f in feats]).to(device)),
        }
        print(f"[raw ] {rep.name:20s} pos={raw[rep.name]['cross_view']['all']['pos']:.4f} "
              f"margin={raw[rep.name]['cross_view']['all']['margin']:.4f} "
              f"recall@1={raw[rep.name]['cross_view']['all']['recall@1']:.4f}")
    tf = [torch.from_numpy(c.teacher).float() for c in va]
    raw[TEACHER_NAME] = {
        "cross_view": eval_cross_view(tf, pairs, primary, cfg, device),
        "diversity": feature_diversity(torch.cat([f.reshape(-1, f.shape[-1]) for f in tf]).to(device)),
    }
    print(f"[raw ] {TEACHER_NAME:20s} pos={raw[TEACHER_NAME]['cross_view']['all']['pos']:.4f} "
          f"margin={raw[TEACHER_NAME]['cross_view']['all']['margin']:.4f} "
          f"recall@1={raw[TEACHER_NAME]['cross_view']['all']['recall@1']:.4f}")

    # teacher's own fidelity row (a trivial 1.0) plus its boundary/diversity reference
    teacher_ref = eval_fidelity_and_boundary(tf, va, device)

    # -- stage 5/6: probes -------------------------------------------------- #
    targets = torch.from_numpy(np.concatenate([c.teacher for c in tr], axis=0))
    results: Dict[str, Dict] = {}
    probe_infer_s: Dict[str, float] = {}
    probe_out_full: Dict[str, List[torch.Tensor]] = {}
    for rep in cfg.representations:
        train_feats = torch.from_numpy(np.concatenate([c.reps[rep.name] for c in tr], axis=0))
        val_feats = [torch.from_numpy(c.reps[rep.name]) for c in va]
        for budget in cfg.teacher_budgets:
            key = f"{rep.name}@{budget:g}"
            probe, info = train_probe(cfg, rep, train_feats, targets, budget, device)
            t0 = time.time()
            pf = [probe_features(probe, f, device) for f in val_feats]
            torch.cuda.synchronize()
            probe_infer_s[key] = time.time() - t0

            if budget == max(cfg.teacher_budgets):
                probe_out_full[rep.name] = pf
            row: Dict[str, object] = {"representation": rep.name, "budget": budget, **info}
            row["fidelity"] = eval_fidelity_and_boundary(pf, va, device)
            row["cross_view"] = {k: eval_cross_view(pf, pairs, v, cfg, device)
                                 for k, v in corr_sets.items()}
            results[key] = row
            cv = row["cross_view"]["predicted"]["all"]
            print(f"[probe] {key:28s} params={info['params']:,} "
                  f"cos={row['fidelity']['cos_mean']:.4f} pos={cv['pos']:.4f} "
                  f"margin={cv['margin']:.4f} r@1={cv['recall@1']:.4f} ({info['train_s']:.0f}s)")
        del train_feats
    del targets

    # -- qualitative figures ------------------------------------------------ #
    fig_dir = os.path.join(out_dir, "figures")
    try:
        figures = make_figures(cfg, val_chunks, va, pairs, corr_sets, probe_out_full, fig_dir)
        print(f"[figs] wrote {len(figures)} figures to {fig_dir}")
    except Exception as exc:                       # figures must never sink the run
        import traceback; traceback.print_exc()
        figures = []
        print(f"[figs] skipped: {exc}")

    # -- verdict ------------------------------------------------------------ #
    verdict = decide(results, cfg)
    n_val = sum(c.pred_depth.shape[0] for c in va)
    probe_ms = {k: 1000 * v / n_val for k, v in probe_infer_s.items()}
    efficiency = {
        "lingbot_inference_ms_per_frame": bench["lingbot_ms_per_frame"],
        "teacher_inference_ms_per_frame": bench["teacher_ms_per_frame"],
        "probe_inference_ms_per_frame_mean": float(np.mean(list(probe_ms.values()))),
        "probe_inference_ms_per_frame": probe_ms,
        "probe_params": {k: r["params"] for k, r in results.items()},
        "cache_bytes": cache_size_bytes(cfg),
        "cache_bytes_per_frame": cache_size_bytes(cfg) / max(n_val + sum(c.pred_depth.shape[0] for c in tr), 1),
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
        "frames_total": n_val + sum(c.pred_depth.shape[0] for c in tr),
    }
    payload = {
        "provenance": collect_provenance(cfg, {"smoke": a.smoke}),
        "config": cfg.to_dict(),
        "splits": {"train": [c.__dict__ for c in train_chunks], "val": [c.__dict__ for c in val_chunks]},
        "grid_hw": list(grid_hw), "image_hw": list(va[0].image_hw),
        "n_pairs": len(pairs),
        "correspondence_counts": {k: int(sum(len(c) for c in v if c)) for k, v in corr_sets.items()},
        "correspondence_valid_fraction": {
            k: float(np.mean([c.valid_fraction for c in v if c])) for k, v in corr_sets.items()},
        "raw_token_diagnostic": raw,
        "teacher_reference": teacher_ref,
        "results": results,
        "efficiency": efficiency,
        "figures": figures,
        "text_query_evaluation": "unavailable: no text-to-DINO bridge exists in this repository",
        "verdict": verdict,
    }
    write_json(os.path.join(out_dir, "metrics.json"), payload)
    write_csv(os.path.join(out_dir, "results.csv"), results, corr_sets)
    print(json.dumps(verdict, indent=2))
    print(f"[done] wrote {out_dir}/metrics.json and results.csv")


def decide(results: Dict[str, Dict], cfg: Phase1Config) -> Dict[str, object]:
    """Apply the pre-registered gate. No threshold is reinterpreted here."""
    full = max(cfg.teacher_budgets)
    base_key = f"{BASELINE}@{full:g}"
    if base_key not in results:
        return {"verdict": "INCONCLUSIVE", "reason": "baseline probe missing"}
    base = results[base_key]
    base_pos = base["cross_view"]["predicted"]["all"]["pos"]
    base_cos = base["fidelity"]["cos_mean"]

    rows = []
    for rep in cfg.representations:
        if rep.name == BASELINE:
            continue
        k = f"{rep.name}@{full:g}"
        if k not in results:
            continue
        r = results[k]
        cv = r["cross_view"]["predicted"]["all"]
        rows.append({
            "representation": rep.name,
            "consistency_ratio": cv["pos"] / base_pos if base_pos else float("nan"),
            "fidelity_ratio": r["fidelity"]["cos_mean"] / base_cos if base_cos else float("nan"),
            "margin_ratio": (cv["margin"] / base["cross_view"]["predicted"]["all"]["margin"]
                             if base["cross_view"]["predicted"]["all"]["margin"] else float("nan")),
            "recall_ratio": (cv["recall@1"] / base["cross_view"]["predicted"]["all"]["recall@1"]
                             if base["cross_view"]["predicted"]["all"]["recall@1"] else float("nan")),
            "pos": cv["pos"], "cos": r["fidelity"]["cos_mean"],
        })

    passing = [r for r in rows if r["consistency_ratio"] >= CONSISTENCY_GAIN
               and r["fidelity_ratio"] >= FIDELITY_RETENTION]
    any_consistency = [r for r in rows if r["consistency_ratio"] >= CONSISTENCY_GAIN]
    collapsed = [r for r in passing if r["margin_ratio"] < 1.0 and r["recall_ratio"] < 1.0]

    if passing and len(collapsed) < len(passing):
        v = "PASS"
        reason = "at least one GCT representation met both pre-registered criteria without collapse"
    elif passing:
        v = "FAIL_CONSISTENCY"
        reason = ("the only representations meeting the raw cross-view threshold did so while "
                  "losing both margin and recall@1 -- the gain is feature collapse, "
                  "which the pre-registered anti-degeneracy guard excludes")
    elif any_consistency:
        v = "FAIL_SEMANTICS"
        reason = "cross-view threshold met but teacher fidelity fell below 95 % of the encoder baseline"
    else:
        v = "FAIL_CONSISTENCY"
        reason = "no GCT representation improved cross-view consistency by 15 % over the encoder"

    best = max(rows, key=lambda r: r["consistency_ratio"]) if rows else None
    sparse = None
    if best:
        lo = min(cfg.teacher_budgets)
        klo, khi = f"{best['representation']}@{lo:g}", f"{best['representation']}@{full:g}"
        if klo in results and khi in results:
            ratio = results[klo]["fidelity"]["cos_mean"] / results[khi]["fidelity"]["cos_mean"]
            sparse = {"representation": best["representation"], "budget": lo,
                      "retention": ratio, "SPARSE_PASS": bool(ratio >= SPARSE_RETENTION)}

    return {
        "verdict": v, "reason": reason,
        "thresholds": {"consistency_gain": CONSISTENCY_GAIN,
                       "fidelity_retention": FIDELITY_RETENTION,
                       "sparse_retention": SPARSE_RETENTION},
        "baseline": {"representation": BASELINE, "pos": base_pos, "cos_mean": base_cos},
        "candidates": rows, "sparse": sparse,
        "text_query_mIoU": "unavailable (no text-to-DINO bridge)",
    }


def write_csv(path: str, results: Dict[str, Dict], corr_sets: Dict[str, List]) -> None:
    """Per-(representation, budget, geometry, band) metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["representation", "budget", "params", "hidden", "teacher_frames", "train_frames",
            "geometry", "band", "cos_mean", "cos_median", "cos_centered", "recon_loss",
            "diversity", "boundary_within", "boundary_across", "boundary_margin",
            "cv_pos", "cv_neg", "cv_margin", "cv_variance", "cv_recall@1",
            "cv_valid_fraction", "cv_n_total", "train_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results.values():
            f = r["fidelity"]
            for geom in r["cross_view"]:
                for band, cv in r["cross_view"][geom].items():
                    if not cv:
                        continue
                    w.writerow({
                        "representation": r["representation"], "budget": r["budget"],
                        "params": r["params"], "hidden": r["hidden"],
                        "teacher_frames": r["teacher_frames"], "train_frames": r["train_frames"],
                        "geometry": geom, "band": band,
                        "cos_mean": f["cos_mean"], "cos_median": f["cos_median"],
                        "cos_centered": f["cos_centered"], "recon_loss": f["recon_loss"],
                        "diversity": f.get("diversity"),
                        "boundary_within": f.get("boundary_within"),
                        "boundary_across": f.get("boundary_across"),
                        "boundary_margin": f.get("boundary_margin"),
                        "cv_pos": cv.get("pos"), "cv_neg": cv.get("neg"),
                        "cv_margin": cv.get("margin"), "cv_variance": cv.get("variance"),
                        "cv_recall@1": cv.get("recall@1"),
                        "cv_valid_fraction": cv.get("valid_fraction"),
                        "cv_n_total": cv.get("n_total"), "train_s": r["train_s"],
                    })


if __name__ == "__main__":
    main()
