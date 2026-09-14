#!/usr/bin/env python
"""Reset and stream-reproduction checks for the LingBot conditions.

1. no leakage into L: clip B run on a fresh model == clip B run right after clip A (bit-identical
   depth, confidence, pose), and == the cached L output of clip B;
2. weights identical: parameter digest of the model used for L == digest of a fresh load;
3. S reproduces the stream: re-running the unchanged streaming replay on the first 60 frames of
   0006 with the stream's keyframe interval reproduces the cached depth/pose of those frames
   (the stream is causal, so a prefix run is the stream up to that frame).

    python tools/local_registration_audit/verify_reset.py --device cuda:0
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from gates.gate7b import replay                                                           # noqa: E402
from core.datasets import sources as S, kitti360_splits as SRC                            # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402


def digest(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = LR.PRIMARY; out = {}
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0]
    m1, _ = replay.build_model(dev); dg1 = digest(m1)
    clips = [x.stream_index for x in SRC.anchors(d, REPO_ROOT)[::LR.ANCHOR_STRIDE]]
    A, B = clips[200], clips[201]
    imgs = lambda t: replay.load_images([seg.frames[t + o].path for o in LR.OFFSETS])
    fresh = replay.replay_segment(m1, imgs(B), 1, dev)
    m2, _ = replay.build_model(dev); dg2 = digest(m2)
    replay.replay_segment(m2, imgs(A), 1, dev); after = replay.replay_segment(m2, imgs(B), 1, dev)
    keys = ("pred_depth", "pred_depth_conf", "pred_pose_c2w")
    same = {k: bool(np.array_equal(fresh[k], after[k])) for k in keys}
    with np.load(LR.clip_path(d, B)) as z:
        cached = {k: bool(np.array_equal(z[k], fresh[k])) for k in keys}
    out["no_leakage"] = {"clip_A": A, "clip_B": B, "fresh_vs_after_A": same, "cached_vs_fresh": cached, "pass": all(same.values()) and all(cached.values())}
    out["weights_identical"] = {"digest_first_load": dg1[:16], "digest_second_load": dg2[:16], "checkpoint": replay.CHECKPOINT,
                                "checkpoint_sha256": C0.sha256_file(os.path.join(REPO_ROOT, replay.CHECKPOINT)), "pass": dg1 == dg2}
    del m2; torch.cuda.empty_cache()
    with np.load(S.stream_path("kitti360", d)) as z:
        k_stream = int(z["keyframe_interval"]); cd = z["pred_depth"][:60].astype(np.float32); cp = z["pred_pose_c2w"][:60].astype(np.float64)
    pre = replay.replay_segment(m1, replay.load_images([f.path for f in seg.frames[:60]]), k_stream, dev)
    dd = np.abs(pre["pred_depth"].astype(np.float32) - cd) / np.maximum(cd, 1e-6); dp = np.abs(pre["pred_pose_c2w"].astype(np.float64) - cp)
    out["stream_prefix_reproduction"] = {"frames": 60, "keyframe_interval": k_stream, "depth_rel_max": float(dd.max()), "depth_rel_median": float(np.median(dd)),
                                         "pose_abs_max": float(dp.max()), "bit_identical_depth": bool(np.array_equal(pre["pred_depth"], z_ := cd.astype(np.float16))),
                                         "pass": bool(np.median(dd) < 1e-3 and dp.max() < 1e-2)}
    C0.write_json(os.path.join(LR.ART, "verification_reset.json"), out)
    print(json.dumps(out, indent=1))
    return 0 if all(v["pass"] for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
