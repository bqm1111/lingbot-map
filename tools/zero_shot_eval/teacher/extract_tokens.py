#!/usr/bin/env python
"""Frozen Trident-H dense token features for target frames (runs in ``third_party/trident_env``, from the Trident root).

Identical teacher and hook as ``tools/feature_first_3d_semantics/extract_features.py`` (KITTI-360): the output of
``model.clip.encode_image`` inside the official Cityscapes configuration, L2-normalised. Only the image size differs, so
the token grid is derived from the official keep-ratio resize and asserted against the token count; the image encoder
must run exactly once per frame (no sliding window), otherwise the frame is refused. Per frame the official coarse
probabilities are reproduced from the tokens and the dataset's frozen text embeddings as a proof.

The teacher vocabulary is the benchmark's frozen Gate 6 phrase list (``core/datasets/vocab.py``); Trident's own
``query_features`` for those phrases are saved once as the readout text embeddings.

    cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$REPO:$PWD ../trident_env/bin/python \
        $REPO/tools/zero_shot_eval/teacher/extract_tokens.py --dataset semantickitti --shard 0 --shards 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from core.datasets import vocab                                                          # noqa: E402
from gates.gate6.trident_adapter import OFFICIAL_CITYSCAPES, TridentTeacher, keep_ratio_size  # noqa: E402
from tools.zero_shot_eval import config as C                                             # noqa: E402

FEAT_DIM = 1024
LOGIT_SCALE = 40.0
SAM_CKPT = os.environ.get("GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")


def token_grid(sam_calls, n_tokens: int):
    """Token grid (h, w): Trident reshapes the CLIP tokens to the SAM-valid grid; in the official Cityscapes mode the frame is
    split into two halves and the grid is (valid_h of the first half, 2 x valid_w of the first half) (trident.py:get_trident_seg)."""
    assert len(sam_calls) == 2, f"expected the two Cityscapes half-images, got {len(sam_calls)} SAM calls"
    h, w = int(sam_calls[0][3]), 2 * int(sam_calls[0][4])
    assert h * w == n_tokens, f"SAM-valid grid {h}x{w} does not explain {n_tokens} tokens"
    return h, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    frames = json.load(open(C.FRAME_LIST.format(dataset=a.dataset)))["frames"][a.shard::a.shards][:a.limit]
    v = vocab.load(a.dataset)
    teacher = TridentTeacher(v.phrases, sam_ckpt=SAM_CKPT, device=a.device)
    q = teacher.model.query_features.float()
    assert float(teacher.model.logit_scale) == LOGIT_SCALE and q.shape == (len(v), FEAT_DIM)
    text_path = C.TEXT_EMBEDDINGS.format(dataset=a.dataset)
    if not os.path.exists(text_path):
        np.savez(text_path, text=q.cpu().numpy().astype(np.float32), phrases=np.asarray(list(v.phrases)), labels=np.asarray(list(v.labels)),
                 names=np.asarray(list(v.names)), logit_scale=np.float32(LOGIT_SCALE),
                 note="Trident.query_features of the frozen Gate 6 phrases: CLIP ViT-H-14 laion2b text, mean over openai_imagenet_template, unit norm")
    calls = []; original = teacher.model.clip.encode_image

    def hook(*args, **kw):
        out = original(*args, **kw); calls.append(out.detach()); return out
    teacher.model.clip.encode_image = hook
    sam_calls = []; original_sam = teacher.model.get_sam_feat

    def sam_hook(*args, **kw):
        out = original_sam(*args, **kw); sam_calls.append(out); return out
    teacher.model.get_sam_feat = sam_hook
    os.makedirs(os.path.join(C.TOKEN_ROOT, a.dataset), exist_ok=True)
    stats, done, skipped, t0 = [], 0, 0, time.time()
    for f in frames:
        dst = C.token_path(a.dataset, f["key"])
        if os.path.exists(dst):
            skipped += 1; continue
        calls.clear(); sam_calls.clear()
        out = teacher.predict_frame(f["path"])
        assert len(calls) == 1 and calls[0].shape[0] == 1 and calls[0].shape[-1] == FEAT_DIM, f"{f['key']}: {len(calls)} encoder calls"
        tokens = F.normalize(calls[0].float(), dim=-1)[0]
        h, w = token_grid(sam_calls, tokens.shape[0])
        H, W = out.coarse_probs.shape[-2:]
        logits = F.interpolate((tokens @ q.T).T.reshape(1, -1, h, w), size=(H, W), mode="bilinear")
        p = (logits[0] * LOGIT_SCALE).softmax(0); cp = out.coarse_probs.to(p.device)
        dp = float((p - cp).abs().max()); agree = float((p.argmax(0) == cp.argmax(0)).float().mean())
        tmp = dst + ".tmp.npz"
        np.savez(tmp, tokens=tokens.cpu().numpy().astype(np.float16), grid_hw=np.asarray((h, w), np.int32), native_hw=np.asarray(out.lattice_hw, np.int32),
                 coarse_max_abs_dp=np.float32(dp), coarse_argmax_agree=np.float32(agree), image_path=np.asarray(f["path"]), key=np.asarray(f["key"]))
        os.replace(tmp, dst); done += 1; stats.append((dp, agree))
        if done % 50 == 0:
            print(f"[{a.dataset} s{a.shard}] {done}/{len(frames) - skipped} {(time.time() - t0) / done:.2f}s/frame min agree {min(s[1] for s in stats):.4f}", flush=True)
    st = np.asarray(stats).reshape(-1, 2)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    json.dump({"dataset": a.dataset, "shard": a.shard, "shards": a.shards, "written": done, "skipped": skipped, "seconds": time.time() - t0,
               "coarse_reproduction": {"max_abs_dp": float(st[:, 0].max()) if len(st) else None, "argmax_agree_min": float(st[:, 1].min()) if len(st) else None},
               "teacher": {"config": OFFICIAL_CITYSCAPES, "vocabulary": list(v.phrases), "sam_ckpt": SAM_CKPT}},
              open(os.path.join(C.LOG_DIR, f"tokens_{a.dataset}_s{a.shard}of{a.shards}.json"), "w"), indent=1)
    print(f"[{a.dataset} s{a.shard}] wrote {done}, skipped {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
