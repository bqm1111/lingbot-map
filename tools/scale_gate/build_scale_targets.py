#!/usr/bin/env python
"""Build per-clip oracle scale targets: depth-derived, pose-derived and joint.

    python tools/scale_gate/build_scale_targets.py --config <cfg> --split val

Writes one CSV row per clip with every estimate, its sample counts, dispersion and the
depth/pose agreement error, plus a JSON summary. Clips whose targets are unreliable are
flagged with a reason and never silently dropped.
"""
from __future__ import annotations

import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import agreement_error, depth_scale, joint_scale, pose_scale

FIELDS = ["clip_id", "sequence", "split", "n_lidar_px", "n_pose_pairs",
          "s_depth", "s_pose", "s_joint", "s_depth_uniform_w",
          "log_s_depth", "log_s_pose", "log_s_joint",
          "depth_mad_log", "pose_mad_log", "per_frame_log_s_std",
          "per_frame_s_min", "per_frame_s_max", "agreement_error_log",
          "gt_motion_m", "reliable", "flags"]


def clip_targets(cfg, rec, cache_root):
    cid = rec["clip_id"]
    lp = os.path.join(cache_root, "lingbot", f"{cid}.npz")
    dp = os.path.join(cache_root, "lidar_depth", f"{cid}.npz")
    if not (os.path.exists(lp) and os.path.exists(dp)):
        return None
    L, D = np.load(lp, allow_pickle=False), np.load(dp, allow_pickle=False)
    dep = L["pred_depth"].astype(np.float32)
    conf = L["pred_depth_conf"].astype(np.float32)
    gtd, val = D["depth"].astype(np.float32), D["valid"]
    if dep.shape != gtd.shape:
        return {"clip_id": cid, "sequence": rec["sequence"], "reliable": 0,
                "flags": f"shape_mismatch {dep.shape} vs {gtd.shape}"}

    m = val & (dep > 0) & np.isfinite(dep)
    fi = np.broadcast_to(np.arange(dep.shape[0])[:, None, None], dep.shape)
    trim = tuple(cfg.scale.trim_log_ratio_quantiles)
    ds = depth_scale(dep[m], gtd[m], conf[m] if cfg.scale.use_confidence_weights else None,
                     fi[m], trim, cfg.scale.minimum_valid_lidar_pixels)
    # Ablation required by the plan: the same estimate with uniform weights.
    ds_u = depth_scale(dep[m], gtd[m], None, None, trim, cfg.scale.minimum_valid_lidar_pixels)

    pc = L["pred_pose_c2w"].astype(np.float64)
    gc = L["gt_pose_c2w"].astype(np.float64)
    tp = pc[1:, :3, 3] - pc[:-1, :3, 3]
    tg = gc[1:, :3, 3] - gc[:-1, :3, 3]
    ps = pose_scale(tp, tg, cfg.scale.minimum_translation_m)
    js = joint_scale(ds, ps, cfg.scale.joint_depth_weight)

    ae = agreement_error(ds, ps)
    flags = []
    if not ds.valid:
        flags.append(f"depth:{ds.reason}")
    if not ps.valid:
        flags.append(f"pose:{ps.reason}")
    if np.isfinite(ae) and ae > cfg.scale.max_agreement_error_log:
        flags.append(f"agreement {ae:.3f} > {cfg.scale.max_agreement_error_log}")
    dd = ds.to_dict()
    return {
        "clip_id": cid, "sequence": rec["sequence"], "split": "",
        "n_lidar_px": ds.n_raw, "n_pose_pairs": ps.n_raw,
        "s_depth": ds.s, "s_pose": ps.s, "s_joint": js.s, "s_depth_uniform_w": ds_u.s,
        "log_s_depth": ds.log_s, "log_s_pose": ps.log_s, "log_s_joint": js.log_s,
        "depth_mad_log": ds.mad_log, "pose_mad_log": ps.mad_log,
        "per_frame_log_s_std": dd.get("per_frame_log_s_std", float("nan")),
        "per_frame_s_min": dd.get("per_frame_s_min", float("nan")),
        "per_frame_s_max": dd.get("per_frame_s_max", float("nan")),
        "agreement_error_log": ae,
        "gt_motion_m": float(np.linalg.norm(tg, axis=1).sum()),
        "reliable": int(ds.valid and ps.valid and np.isfinite(ae)
                        and ae <= cfg.scale.max_agreement_error_log),
        "flags": ";".join(flags),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="*", default=["val", "train"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    cache_root = os.path.join(REPO_ROOT, cfg.cache.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    summary = {}
    for split in a.splits:
        recs = read_manifest(os.path.join(out_dir, "manifests", f"{split}.jsonl"))
        rows, missing = [], 0
        for rec in recs:
            r = clip_targets(cfg, rec, cache_root)
            if r is None:
                missing += 1
                continue
            r["split"] = split
            rows.append(r)
        p = os.path.join(out_dir, f"scale_targets_{split}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})

        good = [r for r in rows if r.get("reliable")]
        ae = np.array([r["agreement_error_log"] for r in rows
                       if np.isfinite(r.get("agreement_error_log", np.nan))])
        ls = np.array([r["log_s_joint"] for r in good])
        summary[split] = {
            "n_clips": len(recs), "n_with_cache": len(rows), "n_missing_cache": missing,
            "n_reliable": len(good),
            "excluded": {r["clip_id"]: r["flags"] for r in rows if not r.get("reliable")},
            "agreement_error_log": {
                "median": float(np.median(ae)) if ae.size else float("nan"),
                "p90": float(np.percentile(ae, 90)) if ae.size else float("nan"),
                "max": float(ae.max()) if ae.size else float("nan"),
                "frac_above_threshold": float((ae > cfg.scale.max_agreement_error_log).mean())
                if ae.size else float("nan")},
            "log_s_joint": {"mean": float(ls.mean()) if ls.size else float("nan"),
                            "std": float(ls.std()) if ls.size else float("nan"),
                            "s_median": float(np.exp(np.median(ls))) if ls.size else float("nan"),
                            "s_min": float(np.exp(ls.min())) if ls.size else float("nan"),
                            "s_max": float(np.exp(ls.max())) if ls.size else float("nan")},
            "csv": os.path.relpath(p, REPO_ROOT),
        }
        s = summary[split]
        print(f"{split}: {len(rows)}/{len(recs)} cached, {len(good)} reliable | "
              f"agreement median {s['agreement_error_log']['median']:.4f} "
              f"p90 {s['agreement_error_log']['p90']:.4f} | "
              f"s range {s['log_s_joint']['s_min']:.1f}-{s['log_s_joint']['s_max']:.1f} "
              f"(median {s['log_s_joint']['s_median']:.1f})")

    write_json(os.path.join(out_dir, "scale_targets_summary.json"),
               {"provenance": collect_provenance(cfg, "build_scale_targets"),
                "splits": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
