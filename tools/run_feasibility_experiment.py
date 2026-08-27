#!/usr/bin/env python
"""Run the whole feasibility matrix and emit the comparison table.

    python tools/run_feasibility_experiment.py --config configs/semantic_sidecar/feasibility.yaml

Stages (each is skipped when its artefacts already exist unless ``--overwrite``):

  1. cache frozen LingBot tokens + geometry
  2. cache frozen teacher features and fit the shared PCA basis
  3. build 3D tracks and consensus targets at every teacher density
  4. train the sidecars
  5. build semantic maps for every model *and* the two training-free baselines
  6. evaluate all of them on the identical geometry
  7. write the feasibility table (markdown + JSON)

Every model in the table consumes exactly the same cached geometry, so any difference
in the numbers is semantic, never geometric.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import _bootstrap  # noqa: F401

from semantic_sidecar.config import collect_provenance, load_config, write_json
from semantic_sidecar.metrics import format_table

logger = logging.getLogger("feasibility")
PYTHON = sys.executable
TOOLS = os.path.dirname(os.path.abspath(__file__))


def run(cmd: List[str], dry: bool = False) -> None:
    logger.info("$ %s", " ".join(cmd))
    if dry:
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=os.path.dirname(TOOLS))
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    logger.info("  ... %.1fs", time.time() - t0)


#: (run name, model arch, loss overrides, extra training flags)
TRAINING_MATRIX = [
    ("linear_pixel", "linear", {"lambda_pixel": 1.0, "lambda_consensus": 0.0}, []),
    ("mlp_pixel", "mlp", {"lambda_pixel": 1.0, "lambda_consensus": 0.0}, []),
    ("linear_consensus", "linear", {"lambda_pixel": 0.0, "lambda_consensus": 1.0}, []),
    ("mlp_consensus", "mlp", {"lambda_pixel": 0.0, "lambda_consensus": 1.0}, []),
    ("mlp_consensus_mv", "mlp", {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25}, []),
    ("mlp_full", "mlp",
     {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1}, []),
    # Diagnostic beyond the eight required baselines: adds the mean-removed cosine,
    # which tests whether the plain cosine's anisotropy (not adapter capacity) is
    # what limits the sidecar.
    ("mlp_full_center", "mlp",
     {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1,
      "lambda_center": 1.0}, []),
]


def loss_overrides(spec: Dict[str, float]) -> List[str]:
    full = {"lambda_pixel": 0.0, "lambda_consensus": 0.0, "lambda_mv": 0.0,
            "lambda_rel": 0.0, "lambda_center": 0.0}
    full.update(spec)
    return [f"loss.{k}={v}" for k, v in full.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--densities", nargs="*", type=float, default=[1.0, 0.25],
                    help="teacher frame densities to run the matrix at")
    ap.add_argument("--primary_density", type=float, default=1.0)
    ap.add_argument("--stages", nargs="*", default=["cache", "teacher", "tracks", "train", "infer", "eval", "table"])
    ap.add_argument("--class_ply_for", nargs="*",
                    default=["teacher_direct", "lingbot_raw", "mlp_full_d100"],
                    help="models to export class-coloured / per-class PLYs for")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    overwrite = ["--overwrite"] if args.overwrite else []
    sets = ["--set", *args.set] if args.set else []
    tool = lambda name: os.path.join(TOOLS, name)  # noqa: E731
    t_start = time.time()

    if "cache" in args.stages:
        run([PYTHON, tool("cache_lingbot_features.py"), "--config", args.config, *sets, *overwrite], args.dry_run)
    if "teacher" in args.stages:
        run([PYTHON, tool("cache_teacher_features.py"), "--config", args.config, *sets, *overwrite], args.dry_run)
    if "tracks" in args.stages:
        for d in args.densities:
            run([PYTHON, tool("build_semantic_tracks.py"), "--config", args.config,
                 "--teacher_density", str(d), *sets, *overwrite], args.dry_run)

    runs: List[Dict[str, object]] = []
    for density in args.densities:
        tag = f"d{int(round(density * 100)):03d}"
        for name, arch, losses, extra in TRAINING_MATRIX:
            run_name = f"{name}_{tag}"
            runs.append({"run_name": run_name, "arch": arch, "density": density,
                         "losses": losses, "extra": extra, "base": name})

    if "train" in args.stages:
        for r in runs:
            run([PYTHON, tool("train_semantic_sidecar.py"), "--config", args.config,
                 "--run_name", str(r["run_name"]), "--teacher_density", str(r["density"]),
                 "--set", f"model.arch={r['arch']}", *loss_overrides(r["losses"]), *args.set,
                 *r["extra"], *overwrite], args.dry_run)

    baselines = [
        ("teacher_direct", "--baseline", "teacher_direct"),
        ("teacher_fullres", "--baseline", "teacher_fullres"),
        ("lingbot_raw", "--baseline", "lingbot_raw"),
    ]
    primary_tag = f"density{int(round(args.primary_density * 100)):03d}"

    if "infer" in args.stages:
        for name, flag, value in baselines:
            run([PYTHON, tool("infer_semantic_sidecar.py"), "--config", args.config, flag, value,
                 "--roles", "eval", "qualitative", "--tracks_tag", primary_tag,
                 "--export_ply", *sets, *overwrite], args.dry_run)
        for r in runs:
            run_dir = os.path.join(cfg.paths.runs_root, str(r["run_name"]))
            if not args.dry_run and not os.path.exists(os.path.join(run_dir, "best.pt")):
                logger.warning("%s has no checkpoint; skipping inference for it", r["run_name"])
                continue
            run([PYTHON, tool("infer_semantic_sidecar.py"), "--config", args.config,
                 "--run_dir", run_dir,
                 "--roles", "eval", "qualitative", "--tracks_tag", primary_tag,
                 "--export_ply", *sets, *overwrite], args.dry_run)

    names = [b[0] for b in baselines] + [str(r["run_name"]) for r in runs]
    if "eval" in args.stages:
        for i, name in enumerate(names):
            maps_dir = os.path.join(cfg.paths.maps_root, name)
            if not args.dry_run and not os.path.exists(os.path.join(maps_dir, "inference_summary.json")):
                logger.warning("%s has no maps; skipping evaluation for it", name)
                continue
            cmd = [PYTHON, tool("evaluate_semantic_reconstruction.py"), "--config", args.config,
                   "--maps", maps_dir, *sets]
            if i == 0:
                cmd += ["--report_geometry"]
            if name in args.class_ply_for:
                cmd += ["--export_class_ply"]
            run(cmd, args.dry_run)

    if "table" in args.stages and not args.dry_run:
        rows, raw = [], {}
        for name in names:
            path = os.path.join(cfg.paths.maps_root, name, "eval.json")
            if not os.path.exists(path):
                logger.warning("missing %s", path)
                continue
            with open(path) as fh:
                res = json.load(fh)
            raw[name] = res
            train = (res.get("inference_summary") or {}).get("training_summary") or {}
            params = ((res.get("inference_summary") or {}).get("parameter_report") or {})
            rows.append({
                "model": name,
                "teacher@infer": "yes" if res.get("needs_teacher_at_inference") else "no",
                "3D mIoU": res["end_to_end_miou"],
                "matched mIoU": res["matched_miou"],
                "mean acc": res["mean_accuracy"],
                "coverage %": res["coverage"],
                "cross-view": res.get("cross_view_consistency", float("nan")),
                "cv held-out": res.get("cross_view_consistency_heldout", float("nan")),
                "diversity": res.get("embedding_diversity", float("nan")),
                "params": int(params.get("sidecar_trainable", 0)),
                "train s": float(train.get("training_seconds", float("nan"))),
                "peak GB": float((res.get("inference_summary") or {}).get("peak_vram_gb", float("nan"))),
            })
        # Requirement: LingBot geometry must be byte-identical across every model.
        fingerprints = {name: res.get("geometry_fingerprint") for name, res in raw.items()}
        unique = {f for f in fingerprints.values() if f}
        geometry_identical = len(unique) == 1
        if geometry_identical:
            logger.info("geometry fingerprint identical across %d models: %s",
                        len(fingerprints), next(iter(unique))[:16])
        else:
            logger.error("GEOMETRY DIFFERS between models: %s", fingerprints)

        table = format_table(rows)
        out_dir = cfg.paths.reports_root
        write_json(os.path.join(out_dir, "feasibility_results.json"),
                   {"rows": rows, "raw": raw, "densities": args.densities,
                    "geometry_identical_across_models": geometry_identical,
                    "geometry_fingerprints": fingerprints,
                    "wall_seconds": time.time() - t_start,
                    "provenance": collect_provenance(cfg, tool="run_feasibility_experiment")})
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "feasibility_table.md"), "w") as fh:
            fh.write(table + "\n")
        print("\n" + table + "\n")
        logger.info("wrote %s", os.path.join(out_dir, "feasibility_table.md"))

    logger.info("feasibility pipeline finished in %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
