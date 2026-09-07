#!/usr/bin/env python
"""Gate 4 step 2 — frozen LingBot inference on the Occ3D-nuScenes val clips.

One independent direct pass per clip with the KV cache cleared between clips, using the
identical construction, preprocessing and call that produced every SemanticKITTI cache in
Gates 0-3.1. No target label, mask or occupancy array is opened anywhere in this file.

    python tools/occ3d_zeroshot/cache_lingbot.py
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames


def build_model(cfg, scfg, device):
    """Identical to tools/scale_gate/cache_lingbot.py:build_model."""
    from lingbot_map.models.gct_stream import GCTStream
    m = GCTStream(img_size=int(cfg.frozen_values.lingbot_image_size),
                  patch_size=int(cfg.frozen_values.lingbot_patch_size),
                  enable_3d_rope=True, max_frame_num=1024,
                  kv_cache_sliding_window=int(cfg.lingbot.kv_cache_sliding_window),
                  kv_cache_scale_frames=int(cfg.lingbot.num_scale_frames),
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(os.path.join(REPO_ROOT, cfg.frozen.lingbot), map_location="cpu",
                    weights_only=False)
    missing, unexpected = m.load_state_dict(ck.get("model", ck), strict=False)
    if missing or unexpected:
        print(f"  checkpoint: {len(missing)} missing / {len(unexpected)} unexpected keys")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(f"{len(trainable)} LingBot parameters are trainable")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    scfg = load_config(cfg.frozen.gate0_config)
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")

    ns_root, occ3d, cam = cfg.data.nuscenes_root, cfg.data.occ3d_root, cfg.data.camera
    out = cfg.data.cache_root
    os.makedirs(out, exist_ok=True)

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    if a.limit:
        recs = recs[: a.limit]

    ann = load_annotations(occ3d)
    frames_by_scene = {}
    model = build_model(cfg, scfg, dev)
    assert not any(p.requires_grad for p in model.parameters())
    print(f"{len(recs)} clips -> {out}")

    t0, n_done, n_skip = time.time(), 0, 0
    for i, r in enumerate(recs):
        dst = os.path.join(out, f"{r['clip_id']}.npz")
        if os.path.exists(dst) and not a.overwrite:
            n_skip += 1
            continue
        if r["scene"] not in frames_by_scene:
            frames_by_scene[r["scene"]] = {
                f.token: f for f in scene_frames(ann, r["scene"], cam, ns_root)}
        fmap = frames_by_scene[r["scene"]]
        fr = [fmap[t] for t in r["sample_tokens"]]

        paths = [f.image_path for f in fr]
        images = load_and_preprocess_images(
            paths, mode=cfg.frozen_values.lingbot_preprocess_mode,
            image_size=int(cfg.frozen_values.lingbot_image_size),
            patch_size=int(cfg.frozen_values.lingbot_patch_size)).to(dev)

        model.clean_kv_cache()
        with torch.amp.autocast("cuda", dtype=getattr(torch, scfg.lingbot.autocast_dtype)):
            preds = model.inference_streaming(
                images, num_scale_frames=int(cfg.lingbot.num_scale_frames),
                keyframe_interval=1)
        pose_enc = preds["pose_enc"][0].float().cpu()
        # camera-to-world, canonical (non-metric) translation units -- verified in Gate 0
        extr, intr = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), images.shape[-2:])
        c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
        c2w[:, :3, :4] = extr[0].double().numpy()

        # native -> processed intrinsics scaling, from the actual preprocessing result
        H_native, W_native = 900, 1600
        Hp, Wp = int(images.shape[-2]), int(images.shape[-1])
        np.savez_compressed(
            dst + ".partial.npz",
            pred_pose_c2w=c2w.astype(np.float32),
            pred_K=intr[0].float().cpu().numpy().astype(np.float32),
            pred_depth=preds["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
            pred_depth_conf=preds["depth_conf"][0].float().cpu().numpy().astype(np.float16),
            pose_enc=pose_enc.numpy().astype(np.float32),
            K_native=np.stack([f.K for f in fr]).astype(np.float64),
            T_camera_to_ego=np.stack([f.T_camera_to_ego for f in fr]).astype(np.float64),
            T_camera_to_ego_cam=np.stack([f.T_camera_to_ego_cam for f in fr]).astype(np.float64),
            timestamps_ns=np.asarray(r["timestamps_ns"], np.int64),
            proc_hw=np.asarray([Hp, Wp], np.int64),
            native_hw=np.asarray([H_native, W_native], np.int64))
        os.replace(dst + ".partial.npz", dst)
        n_done += 1
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(recs)}  {el:.0f}s  "
                  f"({1000*el/max(n_done,1):.0f} ms/clip)", flush=True)

    el = time.time() - t0
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "cache_lingbot.json"),
               {"n_clips": len(recs), "n_cached": n_done, "n_skipped": n_skip,
                "cache_root": out, "elapsed_s": el,
                "ms_per_clip": 1000 * el / max(n_done, 1),
                "processed_hw": [int(Hp), int(Wp)] if n_done else None,
                "device": str(dev)})
    print(f"\ncached {n_done} ({n_skip} already present) in {el:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
