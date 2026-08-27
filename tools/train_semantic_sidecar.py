#!/usr/bin/env python
"""Train one semantic sidecar on cached frozen features (single GPU).

    python tools/train_semantic_sidecar.py --config configs/semantic_sidecar/consensus_mlp.yaml \
        --run_name mlp_consensus --teacher_density 1.0

LingBot is never loaded here: training reads the frozen token cache, so it is
impossible for a gradient to reach the geometry model.  ``--ridge`` solves the linear
sidecar in closed form from streaming covariance instead of running SGD.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

import _bootstrap  # noqa: F401

from semantic_sidecar.config import collect_provenance, config_to_dict, load_config, set_seed, write_json
from semantic_sidecar.consensus import ConsensusTargets
from semantic_sidecar.datasets import FrameBatchSampler, SceneBundle, collate_frame_batch, expand_scenes
from semantic_sidecar.lingbot_features import FeatureCacheReader, manifest_path, set_shard_cache_limit
from semantic_sidecar.losses import SidecarLoss, centered_cosine_loss
from semantic_sidecar.metrics import cross_view_consistency
from semantic_sidecar.semantic_map import embedding_diversity
from semantic_sidecar.models import RidgeAccumulator, build_sidecar, parameter_report
from semantic_sidecar.models.semantic_sidecar import LinearSidecarWrapper, format_parameter_report
from semantic_sidecar.tracks import TrackSet, tracks_are_cached

logger = logging.getLogger("train_sidecar")

LINGBOT_TOTAL_PARAMS = 1_157_943_540  # measured; see docs/semantic_sidecar_repo_audit.md


def load_bundles(cfg, tracks_root: str, roles=("train",), scene_filter=None) -> List[SceneBundle]:
    """Assemble every cached scene into a :class:`SceneBundle`."""
    from safetensors.torch import load_file

    bundles: List[SceneBundle] = []
    for spec in expand_scenes(cfg.scenes):
        if spec.role not in roles:
            continue
        if scene_filter and spec.name not in scene_filter:
            continue
        if not os.path.exists(manifest_path(cfg.paths.cache_root, spec.name)):
            continue
        if not tracks_are_cached(tracks_root, spec.name):
            logger.warning("%s has no tracks under %s, skipping", spec.name, tracks_root)
            continue
        reader = FeatureCacheReader(cfg.paths.cache_root, spec.name)
        tracks = TrackSet.load(tracks_root, spec.name)
        targets = ConsensusTargets.load(tracks_root, spec.name)
        obs = load_file(os.path.join(tracks_root, spec.name, "obs_features.safetensors"))
        bundles.append(
            SceneBundle(
                name=spec.name,
                reader=reader,
                tracks=tracks,
                obs_teacher=obs["obs_features"].float(),
                obs_has_teacher=obs["obs_has_teacher"].bool(),
                consensus=targets.features,
                consensus_valid=targets.valid,
                frame_teacher_mask=obs["frame_teacher_mask"].numpy().astype(bool),
            )
        )
    return bundles


def target_mean_direction(bundles: List[SceneBundle]) -> torch.Tensor:
    """Unit mean of the valid training targets, used only for centred diagnostics.

    Estimated on training targets alone — no evaluation data and no class names.
    """
    parts = [b.consensus[b.consensus_valid] for b in bundles if bool(b.consensus_valid.any())]
    if not parts:
        raise RuntimeError("no valid consensus targets to estimate a mean direction from")
    return torch.nn.functional.normalize(torch.cat(parts).mean(0), dim=-1)


def evaluate_batchwise(
    model, sampler, steps: int, tokens_per_frame: int, device, rng,
    mean_direction: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Diagnostics: raw and mean-removed cosine to targets, plus cross-view consistency.

    ``val_cos_centered`` is the score used for model selection: the raw cosine sits
    near 0.9 for a model that predicts nothing but the corpus mean direction, so it
    cannot distinguish a working sidecar from a collapsed one.
    """
    model.eval()
    cos_c, cos_p, cos_centered, cvc, div, n = [], [], [], [], [], 0
    with torch.no_grad():
        for _ in range(steps):
            bundle, frames = sampler.sample()
            try:
                batch = collate_frame_batch(bundle, frames, tokens_per_frame, rng, device)
            except RuntimeError:
                continue
            pred = model(batch["tokens"])
            cm, pm = batch["consensus_mask"], batch["pixel_mask"]
            if bool(cm.any()):
                cos_c.append(float(((pred * batch["consensus"]).sum(-1))[cm].mean()))
                if mean_direction is not None:
                    cos_centered.append(
                        1.0 - float(centered_cosine_loss(pred, batch["consensus"], mean_direction, cm))
                    )
                div.append(embedding_diversity(pred[cm].detach().cpu()))
            if bool(pm.any()):
                cos_p.append(float(((pred * batch["pixel"]).sum(-1))[pm].mean()))
            c, k = cross_view_consistency(pred, batch["track_id"])
            if k > 0 and not np.isnan(c):
                cvc.append(c)
            n += 1
    model.train()
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return {
        "val_cos_consensus": mean(cos_c),
        "val_cos_pixel": mean(cos_p),
        "val_cos_centered": mean(cos_centered),
        "val_cross_view": mean(cvc),
        "val_diversity": mean(div),
        "val_batches": n,
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    """Model-selection score: the mean-removed cosine when available.

    Falls back to the raw cosines only when no centred number exists, because the
    raw cosine cannot separate a working sidecar from one that predicts the corpus
    mean direction.
    """
    for key in ("val_cos_centered", "val_cos_consensus", "val_cos_pixel"):
        value = metrics.get(key, float("nan"))
        if not np.isnan(value):
            return value
    return float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--teacher_density", type=float, default=None)
    ap.add_argument("--ridge", type=float, default=None,
                    help="solve the linear sidecar in closed form with this ridge coefficient")
    ap.add_argument("--ridge_batches", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="resume from last.pt if present")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.train.seed)
    set_shard_cache_limit(cfg.train.shard_cache_limit)
    device = torch.device(cfg.train.device)

    density = args.teacher_density if args.teacher_density is not None else cfg.train.teacher_density
    tag = f"density{int(round(density * 100)):03d}"
    tracks_root = os.path.join(cfg.paths.tracks_root, tag)
    run_name = args.run_name or f"{cfg.name}_{cfg.model.arch}_{tag}"
    run_dir = os.path.join(cfg.paths.runs_root, run_name)
    if os.path.exists(os.path.join(run_dir, "best.pt")) and not (args.overwrite or args.resume):
        raise SystemExit(f"{run_dir} already has a checkpoint; pass --overwrite or --resume")
    os.makedirs(run_dir, exist_ok=True)

    bundles = load_bundles(cfg, tracks_root, roles=("train",))
    if not bundles:
        raise SystemExit(f"no training bundles under {tracks_root}")
    total_frames = sum(b.num_frames for b in bundles)
    total_tracks = sum(b.tracks.num_tracks for b in bundles)
    logger.info(
        "%d scenes, %d frames, %d tracks, teacher density %.2f",
        len(bundles), total_frames, total_tracks, density,
    )

    num_layers = len(bundles[0].reader.manifest["layers"])
    model = build_sidecar(cfg.model, num_layers=num_layers).to(device)
    report = parameter_report(model, LINGBOT_TOTAL_PARAMS)
    logger.info("parameter budget:\n%s", format_parameter_report(report))
    if report["sidecar_trainable"] > 20_000_000:
        logger.warning("sidecar has %d trainable params, above the 20M target", report["sidecar_trainable"])

    rng = np.random.default_rng(cfg.train.seed)
    sampler = FrameBatchSampler(
        bundles, cfg.train.frames_per_batch, window=cfg.train.window,
        seed=cfg.train.seed, resample_every=cfg.train.resample_every,
    )
    val_sampler = FrameBatchSampler(
        bundles, cfg.train.frames_per_batch, window=cfg.train.window,
        seed=cfg.train.seed + 7919, resample_every=cfg.train.resample_every,
    )
    mean_direction = target_mean_direction(bundles).to(device)
    criterion = SidecarLoss(cfg.loss, mean_direction=mean_direction)
    provenance = collect_provenance(cfg, tool="train_semantic_sidecar", run_name=run_name,
                                    teacher_density=density)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    t_start = time.time()

    # ── closed-form ridge path ──────────────────────────────────────────── #
    if args.ridge is not None:
        if not isinstance(model, LinearSidecarWrapper):
            raise SystemExit("--ridge only applies to model.arch=linear")
        in_dim = num_layers * model.in_channels
        acc = RidgeAccumulator(in_dim, cfg.model.out_dim, device=str(device))
        use_consensus = cfg.loss.lambda_consensus > 0
        for _ in range(args.ridge_batches):
            bundle, frames = sampler.sample()
            try:
                batch = collate_frame_batch(bundle, frames, cfg.train.tokens_per_frame, rng, device)
            except RuntimeError:
                continue
            mask = batch["consensus_mask"] if use_consensus else batch["pixel_mask"]
            target = batch["consensus"] if use_consensus else batch["pixel"]
            if not bool(mask.any()):
                continue
            acc.add(batch["tokens"][mask].flatten(-2), target[mask])
        weight = acc.solve(args.ridge)
        model.linear.load_ridge(weight.to(device))
        logger.info("ridge solved on %d samples (lambda=%g)", acc.n, args.ridge)
        metrics = evaluate_batchwise(model, val_sampler, 20, cfg.train.tokens_per_frame, device, rng,
                                     mean_direction)
        logger.info("ridge diagnostics: %s", metrics)
        torch.save(
            {"model": model.state_dict(), "config": config_to_dict(cfg), "provenance": provenance,
             "parameter_report": report, "metrics": metrics, "method": "ridge", "ridge": args.ridge},
            os.path.join(run_dir, "best.pt"),
        )
        write_json(os.path.join(run_dir, "summary.json"),
                   {"run": run_name, "method": "ridge", "metrics": metrics,
                    "parameter_report": report, "training_seconds": time.time() - t_start,
                    "provenance": provenance, "teacher_density": density})
        return

    # ── gradient training ───────────────────────────────────────────────── #
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: min(1.0, (s + 1) / max(cfg.train.warmup_steps, 1))
        * 0.5 * (1 + np.cos(np.pi * min(s / max(cfg.train.steps, 1), 1.0))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(cfg.train.seed)

    start_step, best = 0, float("-inf")
    last_path = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step, best = state["step"], state.get("best", float("-inf"))
        logger.info("resumed from step %d", start_step)

    model.train()
    peak_vram = 0.0
    log_file = open(log_path, "a")
    for step in range(start_step, cfg.train.steps):
        bundle, frames = sampler.sample()
        try:
            batch = collate_frame_batch(bundle, frames, cfg.train.tokens_per_frame, rng, device)
        except RuntimeError as exc:
            logger.debug("skipping batch: %s", exc)
            continue

        with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
            pred = model(batch["tokens"])
            terms = criterion(
                pred.float(),
                pixel_target=batch["pixel"],
                pixel_mask=batch["pixel_mask"],
                consensus_target=batch["consensus"],
                consensus_mask=batch["consensus_mask"],
                track_id=batch["track_id"],
                generator=gen,
            )
        optim.zero_grad(set_to_none=True)
        scaler.scale(terms["total"]).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optim)
        scaler.update()
        sched.step()

        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

        if step % cfg.train.log_every == 0:
            record = {
                "step": step,
                "lr": sched.get_last_lr()[0],
                "scene": bundle.name,
                "tokens": int(batch["tokens"].shape[0]),
                **{k: float(v) for k, v in terms.items()},
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            logger.info(
                "step %5d | %s | tokens %5d | %s",
                step, bundle.name, record["tokens"],
                " ".join(f"{k}={float(v):.4f}" for k, v in terms.items()),
            )

        if step > 0 and step % cfg.train.val_every == 0:
            metrics = evaluate_batchwise(model, val_sampler, 12, cfg.train.tokens_per_frame, device, rng, mean_direction)
            score = _selection_score(metrics)
            log_file.write(json.dumps({"step": step, **metrics}) + "\n")
            log_file.flush()
            logger.info("step %5d | val %s", step, {k: round(v, 4) for k, v in metrics.items()})
            if score > best:
                best = score
                torch.save(
                    {"model": model.state_dict(), "step": step, "config": config_to_dict(cfg),
                     "provenance": provenance, "parameter_report": report, "metrics": metrics,
                     "method": "sgd", "target_mean_direction": mean_direction.cpu()},
                    os.path.join(run_dir, "best.pt"),
                )
            torch.save(
                {"model": model.state_dict(), "optimizer": optim.state_dict(),
                 "scheduler": sched.state_dict(), "step": step, "best": best,
                 "config": config_to_dict(cfg), "provenance": provenance},
                last_path,
            )

    metrics = evaluate_batchwise(model, val_sampler, 24, cfg.train.tokens_per_frame, device, rng, mean_direction)
    score = _selection_score(metrics)
    if score > best or not os.path.exists(os.path.join(run_dir, "best.pt")):
        best = score
        torch.save(
            {"model": model.state_dict(), "step": cfg.train.steps, "config": config_to_dict(cfg),
             "provenance": provenance, "parameter_report": report, "metrics": metrics,
             "method": "sgd", "target_mean_direction": mean_direction.cpu()},
            os.path.join(run_dir, "best.pt"),
        )
    torch.save(
        {"model": model.state_dict(), "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
         "step": cfg.train.steps, "best": best, "config": config_to_dict(cfg), "provenance": provenance},
        last_path,
    )
    log_file.close()

    elapsed = time.time() - t_start
    write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "run": run_name, "method": "sgd", "steps": cfg.train.steps, "metrics": metrics,
            "best_score": best, "parameter_report": report, "training_seconds": elapsed,
            "training_gpu_hours": elapsed / 3600.0, "peak_vram_gb": peak_vram,
            "teacher_density": density, "num_scenes": len(bundles), "num_frames": total_frames,
            "num_tracks": total_tracks, "loss": config_to_dict(cfg)["loss"], "provenance": provenance,
        },
    )
    logger.info("done in %.1fs (peak VRAM %.2f GB) -> %s", elapsed, peak_vram, run_dir)


if __name__ == "__main__":
    main()
