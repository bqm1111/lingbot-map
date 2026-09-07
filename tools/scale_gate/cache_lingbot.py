#!/usr/bin/env python
"""Cache frozen LingBot outputs for every clip, one independent direct pass per clip.

    python tools/scale_gate/cache_lingbot.py --config <cfg> --split smoke

Contract (plan 2.1): every stored pose key names its direction explicitly, the
checkpoint and config hashes are stored so stale caches invalidate themselves, and
resuming skips only clips that already validate.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import (
    REPO_ROOT, collect_provenance, file_sha256, load_config, set_seed, write_json,
)
from gates.scale_gate.kitti import Preprocess, parse_calibration, load_poses_cam2_c2w, read_manifest, validate_sequence
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def clip_out(cache_root: str, clip_id: str) -> str:
    return os.path.join(cache_root, "lingbot", f"{clip_id}.npz")


def build_model(cfg, device):
    from lingbot_map.models.gct_stream import GCTStream
    m = GCTStream(img_size=cfg.lingbot.inference_resolution, patch_size=cfg.lingbot.patch_size,
                  enable_3d_rope=True, max_frame_num=1024,
                  kv_cache_sliding_window=cfg.lingbot.kv_cache_sliding_window,
                  kv_cache_scale_frames=cfg.lingbot.num_scale_frames,
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(os.path.join(REPO_ROOT, cfg.lingbot.checkpoint),
                    map_location="cpu", weights_only=False)
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


def validate_arrays(d: dict, n_frames: int, frame_ids) -> str:
    """Return '' when the cache is sound, else the reason it is not."""
    if len(d["pred_pose_c2w"]) != n_frames:
        return f"frame count {len(d['pred_pose_c2w'])} != {n_frames}"
    if not np.array_equal(d["frame_ids"], np.asarray(frame_ids)):
        return "frame_ids mismatch"
    R = d["pred_pose_c2w"][:, :3, :3].astype(np.float64)
    err = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    if err > 1e-3:
        return f"rotations not orthonormal (max dev {err:.2e})"
    dep, conf = d["pred_depth"].astype(np.float32), d["pred_depth_conf"].astype(np.float32)
    if not np.isfinite(dep).all() or not np.isfinite(conf).all():
        return "non-finite depth or confidence"
    if (dep <= 0).all():
        return "no positive depth anywhere"
    if not np.isfinite(d["pred_K"]).all():
        return "non-finite intrinsics"
    return ""


class PooledEncoderFeature:
    """Mean-pool LingBot's pre-GCT DINO tokens, read with a forward hook.

    The hook returns ``None`` so it cannot alter geometry. Pre-GCT tokens are used
    because the previous study found post-GCT tokens unstable across views; the plan
    forbids making them the only input to the scale predictor.
    """

    def __init__(self, model):
        self.model = model
        self.buf = []
        self._h = None

    def __enter__(self):
        def hook(_m, _i, out):
            t = out["x_norm_patchtokens"] if isinstance(out, dict) else out
            self.buf.append(t.detach().float().mean(dim=1).cpu())   # [B*S, C]
            return None
        self._h = self.model.aggregator.patch_embed.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._h.remove(); self._h = None
        return False

    def stack(self, n_frames):
        if not self.buf:
            return None
        f = torch.cat(self.buf, 0)
        return f[:n_frames].numpy().astype(np.float16) if len(f) >= n_frames else None


@torch.inference_mode()
def cache_clip(cfg, model, rec, root, device, calib_cache, pre_cache):
    seq = rec["sequence"]
    if seq not in calib_cache:
        calib_cache[seq] = parse_calibration(os.path.join(root, rec["calibration_path"]))
        inv = validate_sequence(root, seq, cfg.dataset.camera)
        pre_cache[seq] = Preprocess.build(inv.image_hw, cfg.lingbot.inference_resolution,
                                          cfg.lingbot.patch_size)
    calib, pre = calib_cache[seq], pre_cache[seq]

    paths = [os.path.join(root, p) for p in rec["image_paths"]]
    images = load_and_preprocess_images(paths, mode="crop",
                                        image_size=cfg.lingbot.inference_resolution,
                                        patch_size=cfg.lingbot.patch_size).to(device)
    if tuple(images.shape[-2:]) != tuple(pre.proc_hw):
        raise RuntimeError(f"{seq}: preprocessing gave {tuple(images.shape[-2:])}, "
                           f"expected {pre.proc_hw}")
    model.clean_kv_cache()                                   # no state carried across clips
    with PooledEncoderFeature(model) as feat, \
            torch.amp.autocast("cuda", dtype=getattr(torch, cfg.lingbot.autocast_dtype)):
        preds = model.inference_streaming(images,
                                          num_scale_frames=cfg.lingbot.num_scale_frames,
                                          keyframe_interval=1)
    pooled = feat.stack(len(rec["frame_ids"]))
    pose_enc = preds["pose_enc"][0].float().cpu()
    # `pose_encoding_to_extri_intri` returns **camera-to-world** despite demo.py's
    # "convert w2c to c2w" comment. Verified by warping: used as-is the relative
    # transforms give 0.57 deg rotation error and +1.000 translation-direction cosine
    # against GT; inverting gives 2.67 deg and -0.999 (i.e. exactly backwards).
    extr, intr = pose_encoding_to_extri_intri(pose_enc.unsqueeze(0), images.shape[-2:])
    c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
    c2w[:, :3, :4] = extr[0].double().numpy()

    gt_c2w = load_poses_cam2_c2w(os.path.join(root, rec["poses_path"]), calib)[rec["frame_ids"]]
    return {
        "frame_ids": np.asarray(rec["frame_ids"], np.int32),
        "pred_pose_c2w": c2w.astype(np.float32),             # camera_to_world, canonical scale
        "pred_K": intr[0].float().cpu().numpy().astype(np.float32),
        "pred_depth": preds["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
        "pred_depth_conf": preds["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        "pose_enc": pose_enc.numpy().astype(np.float32),
        "gt_pose_c2w": gt_c2w.astype(np.float64),            # camera_to_world, METRIC metres
        "gt_K_native": calib.K.astype(np.float64),
        "K_processed": pre.scale_intrinsics(calib.K).astype(np.float64),
        "proc_hw": np.asarray(pre.proc_hw), "orig_hw": np.asarray(pre.orig_hw),
        # frame-independent pre-GCT encoder feature, mean-pooled over patches
        "pooled_pre_gct": pooled if pooled is not None
        else np.zeros((len(rec["frame_ids"]), 1), np.float16),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="smoke", choices=["train", "val", "smoke"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg.experiment.seed)
    device = torch.device(a.device or cfg.lingbot.device)
    torch.cuda.set_device(device)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache_root = os.path.join(REPO_ROOT, cfg.cache.root)
    os.makedirs(os.path.join(cache_root, "lingbot"), exist_ok=True)

    recs = read_manifest(os.path.join(REPO_ROOT, cfg.experiment.output_dir,
                                      "manifests", f"{a.split}.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    ck_hash = file_sha256(os.path.join(REPO_ROOT, cfg.lingbot.checkpoint))
    stamp = {"checkpoint_sha256": ck_hash, "config_hash": cfg.hash,
             "mode": cfg.lingbot.mode, "clip_length": cfg.lingbot.clip_length,
             "frame_stride": cfg.lingbot.frame_stride}

    todo = []
    for r in recs:
        p = clip_out(cache_root, r["clip_id"])
        if os.path.exists(p) and not (a.overwrite or cfg.cache.overwrite):
            try:
                z = np.load(p, allow_pickle=False)
                if json.loads(str(z["stamp"])) == stamp and not validate_arrays(
                        {k: z[k] for k in z.files if k != "stamp"},
                        cfg.lingbot.clip_length, r["frame_ids"]):
                    continue                                  # valid and current: skip
            except Exception:                                 # noqa: BLE001
                pass
        todo.append(r)
    print(f"{a.split}: {len(recs)} clips, {len(todo)} to cache")
    if not todo:
        return 0

    model = build_model(cfg, device)
    torch.cuda.reset_peak_memory_stats(device)
    calib_cache, pre_cache, index, t0 = {}, {}, [], time.time()
    for i, rec in enumerate(todo):
        try:
            d = cache_clip(cfg, model, rec, root, device, calib_cache, pre_cache)
            why = validate_arrays(d, cfg.lingbot.clip_length, rec["frame_ids"])
            if why:
                index.append({"clip_id": rec["clip_id"], "ok": False, "reason": why})
                print(f"  FAIL {rec['clip_id']}: {why}")
                continue
            p = clip_out(cache_root, rec["clip_id"])
            np.savez_compressed(p + ".tmp.npz", stamp=np.asarray(json.dumps(stamp)), **d)
            os.replace(p + ".tmp.npz", p)
            index.append({"clip_id": rec["clip_id"], "ok": True, "reason": ""})
        except Exception as exc:                              # noqa: BLE001
            index.append({"clip_id": rec["clip_id"], "ok": False, "reason": repr(exc)})
            print(f"  ERROR {rec['clip_id']}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    n_ok = sum(1 for r in index if r["ok"])
    print(f"{a.split}: {n_ok}/{len(todo)} cached in {dt:.1f}s "
          f"({1000*dt/max(len(todo)*cfg.lingbot.clip_length,1):.1f} ms/frame), "
          f"peak {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB")
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"cache_index_{a.split}.json"),
               {"provenance": collect_provenance(cfg, "cache_lingbot"), "stamp": stamp,
                "split": a.split, "n_clips": len(recs), "n_ok": n_ok,
                "seconds": dt, "peak_gpu_gib": torch.cuda.max_memory_allocated(device)/2**30,
                "clips": index})
    return 0 if n_ok == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
