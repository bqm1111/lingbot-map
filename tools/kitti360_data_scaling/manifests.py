#!/usr/bin/env python
"""Freeze the training manifests (section 5): the full eligible pool and a nested 25 %
subset of temporal blocks, with coverage statistics from geometry/time metadata only.

    python tools/kitti360_data_scaling/manifests.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.kitti360_data_scaling import common as C                                       # noqa: E402
os.environ["G8C1_DATA_ROOT"] = C.DATA_ROOT
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402


def main() -> int:
    if os.path.exists(os.path.join(C.ART, "manifests", "full.json")):
        raise SystemExit("manifests already frozen")
    man = {"full": {}, "small": {}}; stats = {}
    for d in SRC.TRAIN_DRIVES:
        anchors = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(C.DATA_ROOT, "samples", d, "*.npz")))
        elig = {a.stream_index for a in SRC.anchors(d, REPO_ROOT)}
        assert set(anchors) <= elig, "sample without eligibility"
        blocks = [anchors[i:i + C.BLOCK_ANCHORS] for i in range(0, len(anchors), C.BLOCK_ANCHORS)]
        small_blocks = [b for k, b in enumerate(blocks) if k % C.SUBSET_EVERY_BLOCK == 0]
        man["full"][d] = anchors; man["small"][d] = sorted(a for b in small_blocks for a in b)
        # coverage from time / GT trajectory only
        seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; natives = [int(f.key.split("_")[-1]) for f in seg.frames]
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        pos = np.stack([geo.rect_cam_to_world(natives[i])[:3, 3] for i in range(len(natives))])
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(step)])
        def cover(sel):
            sel = sorted(sel); frames_all_past = set(range(0, max(sel) + 1)); frames_win20 = set(j for t in sel for j in range(max(0, t - 19), t + 1))
            fut = set(j for t in sel for j in range(t + 1, min(t + 21, len(natives))))
            traj = sum(cum[min(t + 1, len(cum) - 1)] - cum[t] for t in sel); span = (min(sel), max(sel))
            return {"n_anchors": len(sel), "unique_anchor_frames": len(set(sel)), "unique_input_frames_all_past": len(frames_all_past),
                    "unique_input_frames_last20": len(frames_win20), "unique_future_target_frames": len(fut),
                    "trajectory_m_within_anchor_steps": float(traj), "drive_trajectory_m": float(cum[-1]), "stream_span": span}
        stats[d] = {"n_eligible_by_rule": len(elig), "n_samples": len(anchors), "n_blocks": len(blocks), "n_small_blocks": len(small_blocks),
                    "full": cover(anchors), "small": cover(man["small"][d]),
                    "shared_context": "inputs are all-past causal maps, so every block shares its past with earlier blocks; selected 25 % blocks are separated by 3 blocks (60 stream frames > the 20-frame future horizon), so no future-target frame is shared between selected blocks"}
    for run in ("full", "small"):
        C.write_json(os.path.join(C.ART, "manifests", f"{run}.json"), {"run": run, "drives": man[run], "n_anchors": sum(len(v) for v in man[run].values()),
                     "block_anchors": C.BLOCK_ANCHORS, "subset_every_block": C.SUBSET_EVERY_BLOCK, "nested": all(set(man["small"][d]) <= set(man["full"][d]) for d in man["full"]),
                     "sha256": C.sha256_json(man[run]), "env": C.environment()})
    tot = {r: sum(len(v) for v in man[r].values()) for r in man}
    C.write_json(os.path.join(C.ART, "manifests", "coverage.json"), {"per_drive": stats, "totals": tot, "small_fraction": tot["small"] / tot["full"]})
    for d, s in stats.items():
        print(f"{d[17:21]}: eligible-by-rule {s['n_eligible_by_rule']} samples {s['n_samples']} | full {s['full']} | small {s['small']}")
    print("totals", tot, "small fraction", round(tot["small"] / tot["full"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
