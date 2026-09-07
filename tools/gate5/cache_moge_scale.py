#!/usr/bin/env python
"""Gate 5 step 1 — frozen MoGe-2 inference and the per-clip metric-gauge scale.

Runs MoGe-2 on the *identical* RGB LingBot consumed (same geometric resize, converted back
to [0,1] without LingBot normalisation), so teacher and student pixels correspond with no
interpolation. Computes one scalar per clip with the predeclared estimator.

No Occ3D label, camera mask or LiDAR file is opened anywhere in this tool.

    python tools/gate5/cache_moge_scale.py --dataset occ3d
    python tools/gate5/cache_moge_scale.py --dataset kitti
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from moge_gauge.adapter import FrozenMoGe
from moge_gauge.estimator import estimate_clip_scale


def occ3d_clips(cfg, g4):
    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, cfg.datasets.occ3d.manifest))]
    for r in recs:
        p = os.path.join(cfg.datasets.occ3d.lingbot_cache, r["clip_id"] + ".npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        paths = [os.path.join(g4.data.nuscenes_root, q) for q in r["image_paths"]]
        rgb = load_and_preprocess_images(
            paths, mode=g4.frozen_values.lingbot_preprocess_mode,
            image_size=int(g4.frozen_values.lingbot_image_size),
            patch_size=int(g4.frozen_values.lingbot_patch_size)).numpy()
        yield {"clip_id": r["clip_id"], "group": r["scene"], "rgb": rgb,
               "dep": d["pred_depth"].astype(np.float32),
               "conf": d["pred_depth_conf"].astype(np.float32)}


def kitti_clips(cfg):
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    recs = read_manifest(os.path.join(REPO_ROOT, cfg.datasets.kitti.manifest))
    for r in recs:
        lp = os.path.join(REPO_ROOT, cfg.datasets.kitti.lingbot_cache,
                          r["clip_id"] + ".npz")
        rp = os.path.join(REPO_ROOT, cfg.datasets.kitti.rgb_cache, r["clip_id"] + ".npz")
        if not (os.path.exists(lp) and os.path.exists(rp)):
            continue
        d = np.load(lp)
        rgb = np.load(rp)["rgb"].astype(np.float32) / 255.0     # exactly the LingBot RGB
        yield {"clip_id": r["clip_id"], "group": r["sequence"], "rgb": rgb,
               "dep": d["pred_depth"].astype(np.float32),
               "conf": d["pred_depth_conf"].astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5/moge_metric_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--store-depth", action="store_true", default=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    est = cfg.estimator
    dev = torch.device(cfg.moge.device if torch.cuda.is_available() else "cpu")

    moge = FrozenMoGe(dev, moge_src=cfg.moge.src_dir, hf_repo=cfg.moge.hf_repo,
                      revision=cfg.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == cfg.moge.hf_revision
    print(f"MoGe-2 {prov['hf_repo']}@{prov['hf_revision'][:12]}  "
          f"{prov['n_params']:,} params  weights {str(prov['weight_sha256'])[:16]}")

    out_root = os.path.join(cfg.moge.cache_root, a.dataset)
    os.makedirs(out_root, exist_ok=True)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    os.makedirs(art, exist_ok=True)

    gen = occ3d_clips(cfg, g4) if a.dataset == "occ3d" else kitti_clips(cfg)
    rows, frames, t0, n_fail = [], [], time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for i, c in enumerate(gen):
        if a.limit and i >= a.limit:
            break
        T = c["rgb"].shape[0]
        md = np.empty_like(c["dep"], dtype=np.float32)
        mm = np.empty(c["dep"].shape, dtype=bool)
        fovs, Ks = [], []
        for t in range(T):
            o = moge.infer(c["rgb"][t])
            assert o.depth_z.shape == c["dep"][t].shape, (
                f"{c['clip_id']} frame {t}: MoGe {o.depth_z.shape} vs LingBot "
                f"{c['dep'][t].shape} -- pixel correspondence broken")
            md[t], mm[t] = o.depth_z, o.mask
            fovs.append(o.fov_x_deg); Ks.append(o.intrinsics)
        r = estimate_clip_scale(md, mm, c["dep"], c["conf"],
                                float(est.conf_threshold), float(est.min_depth_m),
                                float(est.max_depth_m), int(est.min_valid_pixels_per_clip))
        if not r["ok"]:
            n_fail += 1
        if a.store_depth:
            np.savez_compressed(os.path.join(out_root, c["clip_id"] + ".npz"),
                                moge_depth=md.astype(np.float16),
                                moge_mask=np.packbits(mm.reshape(-1)),
                                mask_shape=np.asarray(mm.shape, np.int32),
                                fov_x_deg=np.asarray(fovs, np.float32),
                                intrinsics=np.stack(Ks).astype(np.float32))
        rows.append({"clip_id": c["clip_id"], "group": c["group"],
                     "s_moge": r["s_moge"], "log_s_moge": r["log_s_moge"],
                     "ok": int(r["ok"]), "n_valid_total": r["n_valid_total"],
                     "mad_log_clip": r["mad_log_clip"],
                     "per_frame_log_scale_std": r["per_frame_log_scale_std"],
                     "per_frame_scale_ratio_max_min": r["per_frame_scale_ratio_max_min"],
                     "fov_x_deg_median": float(np.median(fovs)),
                     "fov_x_deg_std": float(np.std(fovs))})
        for pf in r["per_frame"]:
            frames.append({"clip_id": c["clip_id"], **pf,
                           "fov_x_deg": fovs[pf["frame"]]})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}  {time.time()-t0:.0f}s "
                  f"({1000*(time.time()-t0)/(i+1):.0f} ms/clip)", flush=True)

    with open(os.path.join(art, f"moge_scale_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, f"moge_scale_frames_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    ok = [r for r in rows if r["ok"]]
    S = np.array([r["s_moge"] for r in ok])
    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, f"moge_scale_{a.dataset}.json"), {
        "dataset": a.dataset, "moge": prov, "estimator": dict(est),
        "n_clips": len(rows), "n_ok": len(ok), "n_failed": n_fail,
        "failed_clip_ids": [r["clip_id"] for r in rows if not r["ok"]],
        "s_moge": {"median": float(np.median(S)),
                   **{f"p{int(p*100):02d}": float(np.quantile(S, p))
                      for p in (0.05, 0.25, 0.75, 0.95)}},
        "n_valid_total": {"median": float(np.median([r["n_valid_total"] for r in rows])),
                          "min": int(min(r["n_valid_total"] for r in rows))},
        "mad_log_clip_median": float(np.nanmedian([r["mad_log_clip"] for r in ok])),
        "per_frame_log_scale_std_median": float(
            np.nanmedian([r["per_frame_log_scale_std"] for r in ok])),
        "fov_x_deg": {"median": float(np.median([r["fov_x_deg_median"] for r in rows])),
                      "p05": float(np.quantile([r["fov_x_deg_median"] for r in rows], .05)),
                      "p95": float(np.quantile([r["fov_x_deg_median"] for r in rows], .95))},
        "elapsed_s": time.time() - t0, "peak_gpu_gib": peak,
        "ms_per_clip": 1000 * (time.time() - t0) / max(len(rows), 1),
        "cache_root": out_root})
    print(f"\n{a.dataset}: {len(rows)} clips, ok {len(ok)}, failed {n_fail}")
    print(f"  s_moge median {np.median(S):.3f}  p05 {np.quantile(S,.05):.3f}  "
          f"p95 {np.quantile(S,.95):.3f}")
    print(f"  MoGe FOV_x median {np.median([r['fov_x_deg_median'] for r in rows]):.2f} deg")
    print(f"  {time.time()-t0:.0f}s, peak {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
