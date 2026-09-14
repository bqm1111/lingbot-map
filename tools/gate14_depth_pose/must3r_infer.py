#!/usr/bin/env python
"""MUSt3R-512 depth for every Gate 14 clip, with the OFFICIAL naver/must3r code (run in occany_env).

    PYTHONPATH=<must3r>:<must3r>/dust3r:<must3r>/dust3r/croco <occany_env python> tools/gate14_depth_pose/must3r_infer.py \
        --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n] [--limit k] [--out dir]

This file imports nothing from the lingbot-map repository. Per clip (manifest entry): the five images of
stream frames t-4..t only, official preprocessing (must3r.tools.image.get_resize_function at 512: centre
crop to the 16:5 bucket, resize to 512x160, ImgNorm), official must3r.model.load_model (strict weight
loading), official must3r.engine.inference.inference with mem_batches = [2, 1, 1, 1] (memory built frame by
frame in temporal order, as the official sequential mode) followed by rendering all five views against the
memory, official postprocess (compute_cam=True). Saved per view: z-depth of pts3d_local (the view's own
camera frame; scale-ambiguous), confidence, MUSt3R focal and c2w (not used by Gate 14), and the
crop/resize matrices from rescaled to original pixel coordinates.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, torch
import PIL.Image

MEM_BATCHES = [2, 1, 1, 1]
IMAGE_SIZE = 512


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    ap.add_argument("--data_root", default="/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r")
    ap.add_argument("--checkpoint", default="/media/SSD1/MINH_DATASETS/occany_checkpoints/MUSt3R_512.pth")
    a = ap.parse_args()
    from must3r.model import load_model, get_pointmaps_activation
    from must3r.engine.inference import inference, postprocess
    from must3r.tools.image import get_resize_function
    from dust3r.datasets import ImgNorm
    import must3r
    src = os.path.dirname(os.path.dirname(os.path.abspath(must3r.__file__)))
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    ck_md5 = md5(a.checkpoint); assert ck_md5 == "8854f948a8674fb1740258c1872f80dc", f"checkpoint MD5 {ck_md5} != official MUSt3R_512.pth"
    encoder, decoder = load_model(a.checkpoint, device=dev, img_size=IMAGE_SIZE, verbose=False)
    for m in (encoder, decoder):
        for p in m.parameters():
            p.requires_grad_(False)
    act = get_pointmaps_activation(decoder, verbose=False)
    man = json.load(open(os.path.join(a.data_root, "manifests", f"{a.drive}.json")))
    clips = man["clips"][a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.join(a.data_root, "must3r_depth", a.drive); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); nw = 0
    for c in clips:
        t = c["t"]; op = os.path.join(out_dir, f"{t:05d}.npz")
        if os.path.exists(op):
            continue
        assert c["stream_indices"] == [t - 4, t - 3, t - 2, t - 1, t]
        imgs, shapes, to_orig, to_resc = [], [], None, None
        for path in c["image_paths"]:
            im = PIL.Image.open(path).convert("RGB"); im.load(); W, H = im.size
            f, resc, orig = get_resize_function(IMAGE_SIZE, encoder.patch_size, H, W)
            x = f(ImgNorm(im)); imgs.append(x); shapes.append(torch.tensor([x.shape[-2], x.shape[-1]], dtype=torch.int32))
            to_orig, to_resc, native_hw = orig, resc, (H, W)
        imgs = torch.stack(imgs)[None].to(dev); true_shape = torch.stack(shapes)[None].to(dev)
        with torch.no_grad():
            _, pm = inference(encoder, decoder, imgs, true_shape, MEM_BATCHES, verbose=False)
            out = postprocess(pm, pointmaps_activation=act, compute_cam=True)
        z = out["pts3d_local"][0, ..., 2].float().cpu().numpy()                                 # [5, 160, 512]
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, t=np.int64(t), stream_indices=np.asarray(c["stream_indices"], np.int64), keys=np.asarray(c["keys"]),
                            depth_z=z.astype(np.float32), conf=out["conf"][0].float().cpu().numpy().astype(np.float32),
                            focal=out["focal"][0].float().cpu().numpy(), c2w=out["c2w"][0].float().cpu().numpy(),
                            to_orig=np.asarray(to_orig, np.float64), to_rescaled=np.asarray(to_resc, np.float64), native_hw=np.asarray(native_hw),
                            true_shape=true_shape[0].cpu().numpy(), mem_batches=np.asarray(MEM_BATCHES), checkpoint_md5=ck_md5, must3r_commit=commit, dtype="float32")
        os.replace(tmp, op); nw += 1
        if nw % 50 == 0:
            print(f"  [{a.drive[17:21]} s{a.shard}] {nw}/{len(clips)} {(time.time()-t0)/nw:.2f}s each", flush=True)
    print(f"[{a.drive} s{a.shard}] {nw} MUSt3R clips, {(time.time()-t0)/60:.1f} min, checkpoint md5 {ck_md5}, must3r commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
