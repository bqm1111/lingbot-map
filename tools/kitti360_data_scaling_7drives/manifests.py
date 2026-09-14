#!/usr/bin/env python
"""Freeze the seven-drive manifests and the common step budget (sections 3-4).

Full  = every eligible anchor (``kitti360_splits.anchors``: >= 4 preceding stream frames for
        the frozen five-frame scale, >= 5 future stream frames, GT pose on the anchor and
        every future frame) of the seven training drives that has a built sample.
Small = temporal blocks of 20 consecutive eligible anchors, every 4th block of each drive
        (offset 0) -- ~25 % within every drive, nested in Full. Time metadata only.

Budget = max(established 6 000 steps, ceil(5 passes x |Full| / batch 4)), common to all runs.

    python tools/kitti360_data_scaling_7drives/manifests.py
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

BLOCK = 20; EVERY = 4; PASSES = 5; BATCH = 4; ESTABLISHED = 6000; SEC_PER_STEP = 0.151   # gate8c1 sky runs: 6000 steps / ~905 s
SAMPLES = os.path.join(drives7.DATA_ROOT, "samples_gtsem")
CELL_M = 20.0


def main() -> int:
    out_dir = os.path.join(drives7.ART, "manifests")
    if os.path.exists(os.path.join(out_dir, "full.json")):
        raise SystemExit("manifests already frozen")
    man = {"full": {}, "small": {}}; per = {}
    for d in drives7.TRAIN_DRIVES7:
        elig = [a.stream_index for a in SRC.anchors(d, REPO_ROOT)]
        built = [i for i in elig if os.path.exists(os.path.join(SAMPLES, d, f"{i:05d}.npz"))]
        excluded = sorted(set(elig) - set(built))
        seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
        n_stream = len(nat); not_eligible = n_stream - len(elig)
        blocks = [built[i:i + BLOCK] for i in range(0, len(built), BLOCK)]
        small = sorted(a for k, b in enumerate(blocks) if k % EVERY == 0 for a in b)
        man["full"][d] = built; man["small"][d] = small
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        pos = np.stack([geo.rect_cam_to_world(n)[:3, 3] for n in nat]); step = np.r_[np.linalg.norm(np.diff(pos, axis=0), axis=1), 0.0]
        def cov(sel):
            sel = sorted(sel); fut = {j for t in sel for j in range(t + 1, min(t + 21, n_stream))}
            win = {j for t in sel for j in range(max(0, t - 19), t + 1)}
            cells = {tuple(np.floor(pos[t][[0, 1]] / CELL_M).astype(int)) for t in sel}
            gaps = np.diff(sel) if len(sel) > 1 else np.array([0])
            return {"anchors": len(sel), "unique_input_frames_all_past": (max(sel) + 1) if sel else 0, "unique_input_frames_last20": len(win),
                    "unique_future_target_frames": len(fut), "temporal_span_stream": [int(sel[0]), int(sel[-1])] if sel else None,
                    "temporal_span_native": [int(nat[sel[0]]), int(nat[sel[-1]])] if sel else None,
                    "trajectory_m_at_anchors": float(step[sel].sum()), "spatial_cells_20m": len(cells), "n_contiguous_runs": int((gaps > 1).sum() + 1)}
        per[d] = {"stream_frames": n_stream, "not_eligible_by_rule": not_eligible, "eligible": len(elig), "built_samples": len(built),
                  "excluded_eligible": {"n": len(excluded), "reason": "no sample written (scale not frozen / no target)" if excluded else None, "stream_indices": excluded[:50]},
                  "blocks": len(blocks), "small_blocks": len([b for k, b in enumerate(blocks) if k % EVERY == 0]),
                  "full": cov(built), "small": cov(small), "drive_trajectory_m": float(step.sum()), "drive_cells_20m": len({tuple(np.floor(p[[0, 1]] / CELL_M).astype(int)) for p in pos})}
        print(f"{d[17:21]}: stream {n_stream} eligible {len(elig)} built {len(built)} small {len(small)} ({len(small)/max(len(built),1):.3f}); cells full {per[d]['full']['spatial_cells_20m']} small {per[d]['small']['spatial_cells_20m']}")
    for r in ("full", "small"):
        C0.write_json(os.path.join(out_dir, f"{r}.json"), {"run": r, "definition": ("100 % = all eligible anchors with a built sample across the seven SSCBench-KITTI-360 training drives 0000, 0002, 0003, 0004, 0005, 0007, 0010"
                      if r == "full" else "every 4th block of 20 consecutive eligible anchors within each of the seven drives (offset 0), nested in full"),
                      "drives": man[r], "n_anchors": sum(len(v) for v in man[r].values()), "sha256": C0.sha256_json(man[r]),
                      "nested_in_full": all(set(man["small"][d]) <= set(man["full"][d]) for d in man["full"])})
    nf = sum(len(v) for v in man["full"].values()); ns = sum(len(v) for v in man["small"].values())
    steps = max(ESTABLISHED, math.ceil(PASSES * nf / BATCH / 100) * 100)
    C0.write_json(os.path.join(out_dir, "coverage.json"), {"per_drive": per, "totals": {"full": nf, "small": ns, "small_fraction": ns / nf},
                  "block_anchors": BLOCK, "small_every_block": EVERY, "shared_context_note": "inputs are all-past causal maps, so every block shares input history with earlier stream frames; "
                  "selected small blocks are 3 blocks (60 stream frames) apart, so no future-target frame is shared between selected blocks"})
    budget = {"steps": steps, "rule": f"max(established {ESTABLISHED}, ceil({PASSES} x |full| / batch {BATCH}) rounded up to 100)", "batch_size": BATCH,
              "full_anchors": nf, "small_anchors": ns, "effective_epochs": {"full": steps * BATCH / nf, "small": steps * BATCH / ns},
              "estimated_gpu_minutes_per_run": steps * SEC_PER_STEP / 60, "estimated_gpu_hours_total_4_runs": 4 * steps * SEC_PER_STEP / 3600,
              "checkpoint_every": 1000, "frozen_before_training": True, "env": C0.environment()}
    C0.write_json(os.path.join(drives7.ART, "budget.json"), budget)
    print(json.dumps(budget, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
