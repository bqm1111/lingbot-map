#!/usr/bin/env python
"""Gate 5.2 step 2 — frozen LingBot outputs for every eligible KITTI-360 clip.

One independent direct pass per clip, KV cache cleared between clips: exactly the Gate-0
protocol, with only the dataset changed. No target, ``.invalid`` or LiDAR file is opened.

    python tools/gate5_2/cache_lingbot.py
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "scale_gate")))

from gates.scale_gate.config import REPO_ROOT, file_sha256, load_config, set_seed, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from sscbench_kitti360.adapter import SEQUENCE, parse_calibration
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def build_model(cfg, device):
    from lingbot_map.models.gct_stream import GCTStream
    m = GCTStream(img_size=cfg.lingbot.inference_resolution, patch_size=cfg.lingbot.patch_size,
                  enable_3d_rope=True, max_frame_num=1024,
                  kv_cache_sliding_window=cfg.lingbot.kv_cache_sliding_window,
                  kv_cache_scale_frames=cfg.lingbot.num_scale_frames,
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(os.path.join(REPO_ROOT, cfg.lingbot.checkpoint), map_location="cpu",
                    weights_only=False)
    missing, unexpected = m.load_state_dict(ck.get("model", ck), strict=False)
    if missing or unexpected:
        print(f"  checkpoint: {len(missing)} missing / {len(unexpected)} unexpected keys")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if [n for n, p in m.named_parameters() if p.requires_grad]:
        raise RuntimeError("LingBot parameters are trainable")
    return m


def validate(d, n_frames, native_frames) -> str:
    if len(d["pred_pose_c2w"]) != n_frames:
        return f"frame count {len(d['pred_pose_c2w'])} != {n_frames}"
    if not np.array_equal(d["native_frames"], np.asarray(native_frames)):
        return "native_frames mismatch"
    R = d["pred_pose_c2w"][:, :3, :3].astype(np.float64)
    err = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    if err > 1e-3:
        return f"rotations not orthonormal (max dev {err:.2e})"
    if not np.isfinite(d["pred_depth"]).all() or not np.isfinite(d["pred_depth_conf"]).all():
        return "non-finite depth or confidence"
    if (d["pred_depth"] <= 0).all():
        return "no positive depth anywhere"
    if not np.isfinite(d["pred_K"]).all():
        return "non-finite intrinsics"
    return ""


@torch.inference_mode()
def cache_clip(cfg, model, rec, device, calib, pre):
    root = os.path.join(cfg.dataset.kitti360_root, "data_2d_raw", SEQUENCE)
    paths = [os.path.join(root, p) for p in rec["image_paths"]]
    images = load_and_preprocess_images(paths, mode="crop",
                                        image_size=cfg.lingbot.inference_resolution,
                                        patch_size=cfg.lingbot.patch_size).to(device)
    if tuple(images.shape[-2:]) != tuple(pre.proc_hw):
        raise RuntimeError(f"preprocessing gave {tuple(images.shape[-2:])}, "
                           f"expected {pre.proc_hw}")
    model.clean_kv_cache()
    with torch.amp.autocast("cuda", dtype=getattr(torch, cfg.lingbot.autocast_dtype)):
        preds = model.inference_streaming(images,
                                          num_scale_frames=cfg.lingbot.num_scale_frames,
                                          keyframe_interval=1)
    pose_enc = preds["pose_enc"][0].float().cpu()
    extr, intr = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), images.shape[-2:])
    c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
    c2w[:, :3, :4] = extr[0].double().numpy()
    return {
        "native_frames": np.asarray(rec["native_frames"], np.int32),
        "anchor": np.int32(rec["anchor"]),
        "pred_pose_c2w": c2w.astype(np.float32),          # camera_to_world, canonical scale
        "pred_K": intr[0].float().cpu().numpy().astype(np.float32),
        "pred_depth": preds["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
        "pred_depth_conf": preds["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        "pose_enc": pose_enc.numpy().astype(np.float32),
        "K_native": calib.K.astype(np.float64),
        "K_processed": pre.scale_intrinsics(calib.K).astype(np.float64),
        "proc_hw": np.asarray(pre.proc_hw), "orig_hw": np.asarray(pre.orig_hw),
        "rect_cam_to_velo": calib.rect_cam_to_velo.astype(np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg.experiment.seed)
    device = torch.device(cfg.lingbot.device)
    torch.cuda.set_device(device)
    cache = cfg.lingbot.cache_root
    os.makedirs(cache, exist_ok=True)

    recs = read_manifest(os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    calib = parse_calibration(os.path.join(cfg.dataset.root, "calibration"))
    pre = Preprocess.build(calib.native_hw, cfg.lingbot.inference_resolution,
                           cfg.lingbot.patch_size)
    print(f"native {calib.native_hw} -> processed {pre.proc_hw}  "
          f"(sx {pre.sx:.6f}, sy {pre.sy:.6f})")
    stamp = {"checkpoint_sha256": file_sha256(os.path.join(REPO_ROOT,
                                                          cfg.lingbot.checkpoint)),
             "config_hash": cfg.hash, "clip_length": int(cfg.dataset.clip_length),
             "proc_hw": list(pre.proc_hw)}

    todo = []
    for r in recs:
        p = os.path.join(cache, r["clip_id"] + ".npz")
        if os.path.exists(p) and not a.overwrite:
            try:
                z = np.load(p, allow_pickle=False)
                if json.loads(str(z["stamp"])) == stamp and not validate(
                        {k: z[k] for k in z.files if k != "stamp"},
                        int(cfg.dataset.clip_length), r["native_frames"]):
                    continue
            except Exception:                                     # noqa: BLE001
                pass
        todo.append(r)
    print(f"{len(recs)} clips, {len(todo)} to cache")
    if not todo:
        return 0

    model = build_model(cfg, device)
    torch.cuda.reset_peak_memory_stats(device)
    index, t0 = [], time.time()
    for i, rec in enumerate(todo):
        try:
            d = cache_clip(cfg, model, rec, device, calib, pre)
            why = validate(d, int(cfg.dataset.clip_length), rec["native_frames"])
            if why:
                index.append({"clip_id": rec["clip_id"], "ok": False, "reason": why})
                print(f"  FAIL {rec['clip_id']}: {why}")
                continue
            p = os.path.join(cache, rec["clip_id"] + ".npz")
            np.savez_compressed(p + ".tmp.npz", stamp=np.asarray(json.dumps(stamp)), **d)
            os.replace(p + ".tmp.npz", p)
            index.append({"clip_id": rec["clip_id"], "ok": True, "reason": ""})
        except Exception as exc:                                  # noqa: BLE001
            index.append({"clip_id": rec["clip_id"], "ok": False, "reason": repr(exc)})
            print(f"  ERROR {rec['clip_id']}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    dt, n_ok = time.time() - t0, sum(1 for r in index if r["ok"])
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"{n_ok}/{len(todo)} cached in {dt:.0f}s, peak {peak:.2f} GiB")
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "cache_index.json"),
               {"stamp": stamp, "n_clips": len(recs), "n_ok": n_ok, "seconds": dt,
                "peak_gpu_gib": peak, "proc_hw": list(pre.proc_hw),
                "K_processed": pre.scale_intrinsics(calib.K).tolist(),
                "failures": [r for r in index if not r["ok"]]})
    return 0 if n_ok == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
