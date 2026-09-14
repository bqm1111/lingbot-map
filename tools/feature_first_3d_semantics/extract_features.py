#!/usr/bin/env python
"""Cache Trident-H's dense language-aligned token feature for every frame of a KITTI-360
drive, with the per-frame proof that it reproduces the teacher's coarse probabilities.

Runs in the Trident environment (``third_party/trident_env``), from the Trident root, with
the byte-identical frozen stack of ``gates/gate6/trident_adapter.TridentTeacher``. The only
addition is a recording wrapper around ``model.clip.encode_image`` that keeps the tensor
the official code normalises and projects; the official prediction is unchanged.

    PYTHONPATH=$REPO:$TRI trident_env/bin/python tools/feature_first_3d_semantics/extract_features.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher, OFFICIAL_CITYSCAPES, sha256_file  # noqa: E402
from core.datasets import vocab, sources as S, kitti360_splits as SRC                     # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

TOKEN_HW = (34, 128)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    src = SRC.source_of(a.drive); v = vocab.load("kitti360")
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    frames = seg.frames[a.shard::a.num_shards]
    if a.limit:
        frames = frames[:a.limit]
    out_dir = os.path.join(FF.TOKEN_DIR, src); os.makedirs(out_dir, exist_ok=True)
    T = TridentTeacher(v.phrases, sam_ckpt=FF.SAM_CKPT, device=a.device)
    q = T.model.query_features.float()
    assert float(T.model.logit_scale) == FF.LOGIT_SCALE and q.shape == (len(v), FF.FEAT_DIM)
    rec = {}; orig = T.model.clip.encode_image
    def hook(*args, **kw):
        out = orig(*args, **kw); rec["feat"] = out.detach(); return out
    T.model.clip.encode_image = hook
    if not os.path.exists(FF.TEXT_PATH):
        os.makedirs(os.path.dirname(FF.TEXT_PATH), exist_ok=True)
        np.savez(FF.TEXT_PATH, text=q.cpu().numpy().astype(np.float32), phrases=np.asarray(list(v.phrases)),
                 labels=np.asarray(list(v.labels)), logit_scale=np.float32(FF.LOGIT_SCALE),
                 note="Trident.query_features: CLIP ViT-H-14 laion2b text embeddings, mean over openai_imagenet_template, unit norm")
    t0, done, skipped, stats = time.time(), 0, 0, []
    for f in frames:
        dst = FF.token_path(src, f.key)
        if os.path.exists(dst):
            skipped += 1; continue
        rec.clear()
        out = T.predict_frame(f.path)
        feat = rec["feat"]; assert feat.shape[0] == 1 and feat.shape[-1] == FF.FEAT_DIM, feat.shape
        fn = F.normalize(feat.float(), dim=-1)[0]
        h, w = TOKEN_HW; assert fn.shape[0] == h * w, fn.shape
        # proof: the official coarse probabilities are bilinear(f_norm @ text^T) * 40, softmax
        H, W = out.coarse_probs.shape[-2:]
        lg = F.interpolate((fn @ q.T).T.reshape(1, -1, h, w), size=(H, W), mode="bilinear")
        p = (lg[0] * FF.LOGIT_SCALE).softmax(0); cp = out.coarse_probs.to(p.device)
        dp = float((p - cp).abs().max()); agree = float((p.argmax(0) == cp.argmax(0)).float().mean())
        ref_vs_coarse = float((out.probs.argmax(0) == cp.argmax(0).cpu()).float().mean())
        stats.append((dp, agree, ref_vs_coarse))
        tmp = dst + ".tmp.npz"
        np.savez(tmp, tokens=fn.cpu().numpy().astype(np.float16), grid_hw=np.asarray(TOKEN_HW, np.int32),
                 native_hw=np.asarray(out.lattice_hw, np.int32), coarse_max_abs_dp=np.float32(dp),
                 coarse_argmax_agree=np.float32(agree), refined_vs_coarse_agree=np.float32(ref_vs_coarse),
                 image_path=np.asarray(f.path), key=np.asarray(f.key))
        os.replace(tmp, dst); done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[{a.drive[17:21]} s{a.shard}] {done}/{len(frames)-skipped} {el/done:.2f}s/frame, max|dp| {max(s[0] for s in stats):.4f}, "
                  f"min argmax agree {min(s[1] for s in stats):.4f}", flush=True)
    st = np.asarray(stats).reshape(-1, 3)
    FF.write_json(os.path.join(FF.ART, "extraction", f"{a.drive}_s{a.shard}.json"),
                  {"drive": a.drive, "source": src, "shard": a.shard, "num_shards": a.num_shards, "n_written": done, "n_skipped": skipped,
                   "token_grid_hw": TOKEN_HW, "feat_dim": FF.FEAT_DIM, "dtype": "float16 (unit-normalised)",
                   "bytes_per_frame": int(TOKEN_HW[0] * TOKEN_HW[1] * FF.FEAT_DIM * 2),
                   "coarse_reproduction": {"max_abs_dp_max": float(st[:, 0].max()) if len(st) else None,
                                           "argmax_agree_min": float(st[:, 1].min()) if len(st) else None,
                                           "argmax_agree_mean": float(st[:, 1].mean()) if len(st) else None,
                                           "refined_vs_coarse_agree_mean": float(st[:, 2].mean()) if len(st) else None},
                   "teacher": {"config": OFFICIAL_CITYSCAPES, "sam_ckpt": FF.SAM_CKPT, "sam_ckpt_sha256": sha256_file(FF.SAM_CKPT),
                               "clip": "open_clip ViT-H-14 laion2b_s32b_b79k (fp16)", "vocabulary": list(v.phrases),
                               "trident_py_sha256": sha256_file(os.path.join(FF.TRIDENT_ROOT, "trident.py")),
                               "adapter_sha256": sha256_file(os.path.join(REPO_ROOT, "gates", "gate6", "trident_adapter.py")),
                               "hook": "model.clip.encode_image output, L2-normalised (the tensor the official code projects onto text)"},
                   "seconds": time.time() - t0, "torch": torch.__version__})
    print(f"[{a.drive}] wrote {done}, skipped {skipped}, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
