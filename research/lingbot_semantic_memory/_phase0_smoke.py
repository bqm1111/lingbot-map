"""Phase 0 smoke test: checkpoint loads, hooks fire, outputs finite, determinism."""
import argparse, json, os, sys, time
import numpy as np, torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images


def build(device, model_path):
    m = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                  kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default="checkpoints/lingbot-map/204754b/lingbot-map.pt")
    ap.add_argument("--image-folder", default="data/kitti/dataset/sequences/08/image_2")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--output-dir", default="research/lingbot_semantic_memory/reports")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    print(f"[env] torch={torch.__version__} device={torch.cuda.get_device_name(dev)}")

    files = sorted(os.listdir(os.path.join(REPO, a.image_folder)))[: a.num_frames]
    paths = [os.path.join(REPO, a.image_folder, f) for f in files]
    imgs = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14).to(dev)
    print(f"[data] images {tuple(imgs.shape)} dtype={imgs.dtype} range=[{imgs.min():.3f},{imgs.max():.3f}]")

    model = build(dev, os.path.join(REPO, a.model_path))
    n_par = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_par:,} trainable={n_train:,}")
    agg = model.aggregator
    print(f"[model] depth={agg.depth} aa_order={agg.aa_order} aa_block_size={agg.aa_block_size} "
          f"aa_block_num={agg.aa_block_num} embed_dim={agg.embed_dim} patch_start_idx={agg.patch_start_idx}")
    print(f"[model] n frame_blocks={len(agg.frame_blocks)} n global_blocks={len(agg.global_blocks)}")
    print(f"[model] patch_embed={type(agg.patch_embed).__name__}")

    # ---- hooks: patch_embed (pre-GCT) + frame/global blocks ----
    caught = {}
    handles = []

    def mk(name):
        def h(_m, _i, out):
            o = out["x_norm_patchtokens"] if isinstance(out, dict) else out
            caught[name] = o.detach()
        return h

    handles.append(agg.patch_embed.register_forward_hook(mk("patch_embed")))
    for b in (0, 4, 11, 17, 23):
        handles.append(agg.frame_blocks[b].register_forward_hook(mk(f"frame_{b}")))
        handles.append(agg.global_blocks[b].register_forward_hook(mk(f"global_{b}")))

    def run():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.clean_kv_cache()
            return model.inference_streaming(imgs, num_scale_frames=8, keyframe_interval=1)

    t0 = time.time(); preds = run(); torch.cuda.synchronize(); dt = time.time() - t0
    print(f"[infer] {a.num_frames} frames in {dt:.2f}s ({dt/a.num_frames*1000:.0f} ms/frame)")
    print("[preds] keys:", sorted(preds.keys()))
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f"   {k:20s} {str(tuple(v.shape)):28s} {str(v.dtype):16s} finite={bool(torch.isfinite(v).all())}")

    print("[hooks]")
    for k in sorted(caught, key=lambda s: (s.split('_')[0], int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else -1)):
        v = caught[k]
        print(f"   {k:14s} {str(tuple(v.shape)):28s} {str(v.dtype):16s} finite={bool(torch.isfinite(v).all())}")

    d, c = preds["depth"], preds["depth_conf"]
    print(f"[depth] shape={tuple(d.shape)} min={d.min():.4f} max={d.max():.4f} median={d.median():.4f}")
    print(f"[conf]  shape={tuple(c.shape)} min={c.min():.4f} max={c.max():.4f} median={c.median():.4f}")
    print(f"[conf]  min>1 (expp1 => confidence/precision, higher=better): {bool((c>1).all())}")

    # ---- determinism ----
    ref_tokens = {k: v.float().clone() for k, v in caught.items()}
    ref_depth = preds["depth"].float().clone(); ref_pose = preds["pose_enc"].float().clone()
    caught.clear()
    preds2 = run()
    dd = (preds2["depth"].float() - ref_depth).abs().max().item()
    dp = (preds2["pose_enc"].float() - ref_pose).abs().max().item()
    dt_tok = max((caught[k].float() - ref_tokens[k]).abs().max().item() for k in ref_tokens)
    print(f"[determinism] max|Δdepth|={dd:.3e} max|Δpose|={dp:.3e} max|Δtokens|={dt_tok:.3e}")

    for h in handles:
        h.remove()

    # ---- hook is non-invasive ----
    caught.clear()
    preds3 = run()
    print(f"[non-invasive] depth bit-exact without hooks: {torch.equal(preds3['depth'], preds2['depth'])}")
    print(f"[non-invasive] pose  bit-exact without hooks: {torch.equal(preds3['pose_enc'], preds2['pose_enc'])}")
    print(f"[non-invasive] hooks silent after removal: {len(caught) == 0}")
    print(f"[mem] peak alloc = {torch.cuda.max_memory_allocated(dev)/2**30:.2f} GiB")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
