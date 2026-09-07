#!/usr/bin/env python
"""Project LiDAR into every clip frame -> sparse metric depth at LingBot resolution.

    python tools/scale_gate/project_lidar.py --config <cfg> --split val
    python tools/scale_gate/project_lidar.py --config <cfg> --qa-only

Stores, per clip, ``depth[T,H,W] float16`` and ``valid[T,H,W] bool`` at the **processed**
resolution, plus the native-resolution intrinsics and the preprocessing record, so the
two coordinate systems are never confused.
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import (
    Preprocess, parse_calibration, project_lidar_to_depth, read_manifest, read_velodyne,
    sequence_dir, validate_sequence,
)


def clip_out(cache_root: str, clip_id: str) -> str:
    return os.path.join(cache_root, "lidar_depth", f"{clip_id}.npz")


def project_clip(cfg, rec, root: str, calib_cache: dict, pre_cache: dict):
    seq = rec["sequence"]
    if seq not in calib_cache:
        calib_cache[seq] = parse_calibration(os.path.join(root, rec["calibration_path"]))
        inv = validate_sequence(root, seq, cfg.dataset.camera)
        pre_cache[seq] = Preprocess.build(inv.image_hw, cfg.lingbot.inference_resolution,
                                          cfg.lingbot.patch_size)
    calib, pre = calib_cache[seq], pre_cache[seq]
    K_proc = pre.scale_intrinsics(calib.K)
    H, W = pre.proc_hw

    depths = np.zeros((len(rec["frame_ids"]), H, W), np.float16)
    valids = np.zeros((len(rec["frame_ids"]), H, W), bool)
    for i, lp in enumerate(rec["lidar_paths"]):
        pts = read_velodyne(os.path.join(root, lp))[:, :3]
        d, v = project_lidar_to_depth(pts, calib, (H, W), K_proc,
                                      cfg.scale.lidar_min_depth_m, cfg.scale.lidar_max_depth_m)
        depths[i], valids[i] = d.astype(np.float16), v
    return depths, valids, calib, pre, K_proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "smoke"])
    ap.add_argument("--qa-only", action="store_true", help="only render QA overlays")
    ap.add_argument("--qa-frames", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache_root = os.path.join(REPO_ROOT, cfg.cache.root)
    man = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "manifests", f"{a.split}.jsonl")
    recs = read_manifest(man)
    calib_cache, pre_cache = {}, {}

    if a.qa_only:
        return qa(cfg, recs, root, calib_cache, pre_cache, a.qa_frames)

    os.makedirs(os.path.join(cache_root, "lidar_depth"), exist_ok=True)
    t0, n_done, n_skip = time.time(), 0, 0
    stats = []
    for k, rec in enumerate(recs):
        out = clip_out(cache_root, rec["clip_id"])
        if os.path.exists(out) and not (a.overwrite or cfg.cache.overwrite):
            n_skip += 1
            continue
        d, v, calib, pre, K_proc = project_clip(cfg, rec, root, calib_cache, pre_cache)
        np.savez_compressed(
            out + ".tmp.npz", depth=d, valid=v, frame_ids=np.asarray(rec["frame_ids"]),
            K_processed=K_proc.astype(np.float32), K_native=calib.K.astype(np.float32),
            proc_hw=np.asarray(pre.proc_hw), orig_hw=np.asarray(pre.orig_hw))
        os.replace(out + ".tmp.npz", out)
        stats.append({"clip_id": rec["clip_id"], "sequence": rec["sequence"],
                      "valid_per_frame": [int(x) for x in v.reshape(len(v), -1).sum(1)],
                      "median_depth_m": float(np.median(d[v])) if v.any() else float("nan")})
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(recs)} clips ({time.time()-t0:.0f}s)", flush=True)
    print(f"{a.split}: projected {n_done}, skipped {n_skip}, {time.time()-t0:.1f}s")
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir,
                            f"lidar_projection_{a.split}.json"),
               {"provenance": collect_provenance(cfg, "project_lidar"),
                "split": a.split, "n_clips": len(recs), "n_projected": n_done,
                "clips": stats[:200]})
    return 0


def qa(cfg, recs, root, calib_cache, pre_cache, n_frames):
    """Overlay projected LiDAR on RGB at both resolutions, for visual calibration QA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "qa",
                           "semantickitti_projection")
    os.makedirs(out_dir, exist_ok=True)
    # Deterministic spread across sequences and clips.
    by_seq = {}
    for r in recs:
        by_seq.setdefault(r["sequence"], []).append(r)
    picks = []
    seqs = sorted(by_seq)
    per = max(1, n_frames // max(len(seqs), 1))
    for s in seqs:
        rs = by_seq[s]
        picks += [rs[i] for i in np.linspace(0, len(rs) - 1, per).round().astype(int)]
    picks = picks[:n_frames]

    rows = []
    for rec in picks:
        seq = rec["sequence"]
        d, v, calib, pre, K_proc = project_clip(cfg, rec, root, calib_cache, pre_cache)
        fi = 0
        img = np.array(Image.open(os.path.join(root, rec["image_paths"][fi])).convert("RGB"))
        # native-resolution projection for the same frame
        pts = read_velodyne(os.path.join(root, rec["lidar_paths"][fi]))[:, :3]
        dn, vn = project_lidar_to_depth(pts, calib, pre.orig_hw, calib.K,
                                        cfg.scale.lidar_min_depth_m, cfg.scale.lidar_max_depth_m)
        imgp = np.array(Image.fromarray(img).resize((pre.proc_hw[1], pre.proc_hw[0]),
                                                    Image.BICUBIC))

        fig, ax = plt.subplots(2, 1, figsize=(13, 6.4), constrained_layout=True)
        for axis, im, dd, vv, name in ((ax[0], img, dn, vn, f"native {pre.orig_hw[1]}x{pre.orig_hw[0]}"),
                                       (ax[1], imgp, d[fi].astype(np.float32), v[fi],
                                        f"processed {pre.proc_hw[1]}x{pre.proc_hw[0]}")):
            axis.imshow(im)
            yy, xx = np.nonzero(vv)
            sc = axis.scatter(xx, yy, c=dd[vv], s=2.2, cmap="turbo", vmin=1, vmax=60)
            axis.set_title(f"seq {seq} frame {rec['frame_ids'][fi]} — {name} — "
                           f"{int(vv.sum()):,} pts, {dd[vv].min():.1f}-{dd[vv].max():.1f} m",
                           fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.7, label="metric depth (m)")
        p = os.path.join(out_dir, f"proj_{seq}_{rec['frame_ids'][fi]:06d}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        rows.append({"sequence": seq, "frame": rec["frame_ids"][fi],
                     "n_valid_native": int(vn.sum()), "n_valid_processed": int(v[fi].sum()),
                     "depth_min_m": float(dn[vn].min()), "depth_max_m": float(dn[vn].max()),
                     "png": os.path.relpath(p, REPO_ROOT)})
        print(f"  {os.path.relpath(p, REPO_ROOT)}  native {int(vn.sum()):6,}  "
              f"processed {int(v[fi].sum()):5,}  {dn[vn].min():.1f}-{dn[vn].max():.1f} m")
    write_json(os.path.join(out_dir, "qa_summary.json"),
               {"provenance": collect_provenance(cfg, "project_lidar:qa"), "frames": rows})
    print(f"\n{len(rows)} QA overlays -> {os.path.relpath(out_dir, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
