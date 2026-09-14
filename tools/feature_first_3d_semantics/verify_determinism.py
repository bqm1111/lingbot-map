#!/usr/bin/env python
"""Re-extract two frames with a fresh teacher into a scratch directory and compare the
tokens bit-for-bit with the cache. Trident environment.

    PYTHONPATH=$REPO:$TRI trident_env/bin/python tools/feature_first_3d_semantics/verify_determinism.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, tempfile
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from gates.gate6.trident_adapter import TridentTeacher                                    # noqa: E402
from core.datasets import vocab                                                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    v = vocab.load("kitti360"); T = TridentTeacher(v.phrases, sam_ckpt=FF.SAM_CKPT, device=a.device)
    rec = {}; orig = T.model.clip.encode_image
    def hook(*args, **kw):
        out = orig(*args, **kw); rec["feat"] = out.detach(); return out
    T.model.clip.encode_image = hook
    files = [sorted(glob.glob(os.path.join(FF.TOKEN_DIR, "kitti360", "*.npz")))[100], sorted(glob.glob(os.path.join(FF.TOKEN_DIR, "k360_train", "*.npz")))[100]]
    details = []; ident = True
    for f in files:
        with np.load(f) as z:
            cached = z["tokens"]; img = str(z["image_path"])
        rec.clear(); T.predict_frame(img)
        fn = F.normalize(rec["feat"].float(), dim=-1)[0].cpu().numpy().astype(np.float16)
        same = np.array_equal(fn, cached); ident &= same
        details.append(f"{os.path.basename(f)}: bit-identical {same}, max|diff| {float(np.abs(fn.astype(np.float32) - cached.astype(np.float32)).max()):.2e}")
    FF.write_json(os.path.join(FF.ART, "extraction", "determinism.json"), {"identical": bool(ident), "detail": "; ".join(details), "torch": torch.__version__})
    print(ident, details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
