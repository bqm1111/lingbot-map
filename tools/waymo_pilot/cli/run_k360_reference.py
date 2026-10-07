#!/usr/bin/env python
"""Measure the KITTI-360 reference arm of the Waymo supervision pilot.

The existing recipe, on the existing caches, over a reproducibly selected sample of the
official SSCBench *train* drives. Drive 0006 is development data and is asserted absent.
Nothing is trained and no target benchmark is opened.

    python tools/waymo_pilot/cli/run_k360_reference.py --device cuda:0
    python tools/waymo_pilot/cli/run_k360_reference.py --smoke          # 2 scenes, 10 anchors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.datasets.audit import FileAudit                                     # noqa: E402
from tools.semantic_completion.evidence import trident as TR                  # noqa: E402
from tools.waymo_pilot import config as C, k360 as K, populations as POP       # noqa: E402
from tools.waymo_pilot import sample as SA, stats as ST                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="the 2-scene / 10-anchor smoke test")
    ap.add_argument("--drives", choices=("frozen", "all"), default="frozen",
                    help="'frozen' = the three drives the current recipe trains on; "
                         "'all' = every KITTI-360 training drive with the needed caches, "
                         "which varies SCENE COUNT with everything else held fixed")
    ap.add_argument("--per-scene", type=int, default=None,
                    help="anchors per scene; the default lets the independence separation "
                         "bind instead of an arbitrary per-scene cap, because KITTI-360 "
                         "offers only three training drives")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    C.ensure_dirs()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    max_scenes = C.SMOKE_SEGMENTS if a.smoke else C.MAX_SEGMENTS
    max_anchors = C.SMOKE_ANCHORS if a.smoke else C.MAX_ANCHORS
    # On Waymo the 5-per-segment cap is what keeps 20 segments diverse. KITTI-360 has three
    # drives in total, so applying the same cap would sample 15 anchors and understate what
    # the existing pool can actually supply. The anchor budget is kept; the min-separation
    # rule decides how many of them a drive can give independently.
    # With one drive pool the separation rule should bind; with several drives the budget
    # must be shared so the sample actually spans the scenes it is meant to compare.
    pool_size = len(C.K360_ALL_TRAIN_DRIVES if a.drives == "all" else C.K360_SAMPLE_DRIVES)
    per_scene = a.per_scene or (max_anchors if pool_size <= len(C.K360_SAMPLE_DRIVES)
                                else -(-max_anchors // pool_size))

    t0 = time.time()
    pool = C.K360_ALL_TRAIN_DRIVES if a.drives == "all" else C.K360_SAMPLE_DRIVES
    variant = "k360_7drives" if a.drives == "all" else K.VARIANT
    # The four unused drives have coverage caches but no causal clip, so the scene-count
    # control admits them and reports their sub-cases as unavailable rather than invented.
    eligible = K.available(pool, require_clip=(a.drives != "all"))
    anchors, selection = SA.select(eligible, K.ARM, max_scenes, max_anchors, per_scene)
    SA.assert_dev_drive_absent(anchors)
    selection["eligible_per_scene"] = {k: len(v) for k, v in eligible.items()}
    print(f"[k360] {selection['n_anchors']} anchors from {selection['n_scenes_used']} drives "
          f"(digest {selection['digest']})", flush=True)

    text, phrases, labels = TR.load_text_embeddings()
    text = text.to(dev)
    centres = POP.grid_centres(dev)

    rows, t1 = [], time.time()
    # The benchmark firewall stays on for the whole measurement: SemanticKITTI, nuScenes and
    # Occ3D are never opened, by this arm or by anything it calls.
    with FileAudit(patterns=C.BENCHMARK_FORBIDDEN, strict=True) as audit:
        for i, an in enumerate(anchors):
            rows.append(K.measure_anchor(an.scene, an.index, dev, text, centres))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(anchors)} {(time.time() - t1) / (i + 1):.2f}s each",
                      flush=True)
    agg = ST.aggregate(rows, variant)
    rec = {
        "version": C.VERSION, "arm": K.ARM, "variant": variant,
        "drive_pool": list(pool),
        "device": str(dev), "smoke": bool(a.smoke),
        "seconds": time.time() - t0, "seconds_per_anchor": (time.time() - t1) / max(len(rows), 1),
        "gpu_hours": (time.time() - t1) / 3600.0,
        "selection": selection,
        "teacher": {"phrases": list(map(str, phrases)), "labels": [int(x) for x in labels],
                    "dim": C.TEACHER_DIM},
        "firewall": {"n_opened": len(audit.opened), "violations": audit.violations,
                     "patterns": list(C.BENCHMARK_FORBIDDEN)},
        "aggregate": agg,
        "per_anchor": rows,
    }
    out = a.out or os.path.join(C.Paths().logs,
                               f"{variant}{'_smoke' if a.smoke else ''}.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    print(ST.table([agg]))
    print(f"[k360] {len(rows)} anchors in {(time.time() - t1) / 60:.1f} min "
          f"({rec['gpu_hours']:.3f} GPU-h of the {C.MAX_PREPROCESS_GPU_HOURS} h cap) -> {out}")
    if audit.violations:
        print(f"FIREWALL VIOLATION: {audit.violations[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
