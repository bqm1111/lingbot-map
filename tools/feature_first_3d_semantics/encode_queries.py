#!/usr/bin/env python
"""Free-text query embeddings for the open-vocabulary diagnostic, encoded exactly as Trident
encodes its vocabulary (CLIP ViT-H-14 laion2b text tower, mean over the OpenAI ImageNet
templates, unit norm). Runs in the Trident environment from the Trident root. Also
re-encodes the 18 vocabulary phrases and checks them against the cached text embeddings.

    PYTHONPATH=$REPO:$TRI trident_env/bin/python tools/feature_first_3d_semantics/encode_queries.py --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402

#: free-form queries with an unambiguous vocabulary class for the agreement check
QUERIES = {"a tree": "vegetation", "leaves and branches": "vegetation", "a bush": "vegetation", "a house": "building",
           "a facade of a building": "building", "asphalt": "road", "a street": "road", "a pavement": "sidewalk",
           "an automobile": "car", "a parked car": "car", "a pedestrian": "person", "a wooden fence": "fence",
           "a lamp post": "pole", "a lawn": "terrain", "a truck": "truck", "a traffic sign": "traffic sign"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    sys.path.insert(0, FF.TRIDENT_ROOT)
    from open_clip import create_model, tokenizer                                         # noqa: E402
    from prompts.imagenet_template import openai_imagenet_template                         # noqa: E402
    clip = create_model("ViT-H-14", pretrained="laion2b_s32b_b79k", precision="fp16").to(a.device).eval()
    def enc(words):
        out = []
        with torch.no_grad():
            for qw in words:
                q = tokenizer.tokenize([t(qw) for t in openai_imagenet_template]).to(a.device)
                f = clip.encode_text(q); f = f / f.norm(dim=-1, keepdim=True); f = f.mean(0); f = f / f.norm()
                out.append(f.float().cpu().numpy())
        return np.stack(out)
    with np.load(FF.TEXT_PATH) as z:
        text, phrases = z["text"], [str(p) for p in z["phrases"]]
    re_enc = enc(phrases)
    dev = float(np.abs(re_enc - text).max())
    qe = enc(list(QUERIES))
    np.savez(os.path.join(FF.DATA_ROOT, "text_queries.npz"), queries=np.asarray(list(QUERIES)), target_class=np.asarray(list(QUERIES.values())),
             embeddings=qe.astype(np.float32), vocab_reencode_max_abs_dev=np.float32(dev))
    print(f"vocabulary re-encoded, max |dev| vs cached text embeddings {dev:.2e}; {len(QUERIES)} queries saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
