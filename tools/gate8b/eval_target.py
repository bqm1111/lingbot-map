#!/usr/bin/env python
"""Gate 8B Stage 4 for one fold: the locked target evaluation in both settings.

Takes **no checkpoint and no threshold**. Everything it needs is read from the fold's
frozen manifest (``configs/gate8b/frozen_fold_<fold>.yaml``), which must already exist --
that file is written by ``select_fold.py`` from source-validation data only, and this
script refuses to run without it. The existing KITTI-360 fold reads Gate 8A's frozen
selection the same way.

    python tools/gate8b/eval_target.py --fold semantickitti --setting stream
    python tools/gate8b/eval_target.py --fold semantickitti --setting clips
    python tools/gate8b/eval_target.py --fold kitti360 --setting clips     # 8A model, unmodified
"""
from __future__ import annotations
import argparse, os, sys
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402

FROZEN = {"semantickitti": "configs/gate8b/frozen_fold_semantickitti.yaml",
          "occ3d": "configs/gate8b/frozen_fold_occ3d.yaml",
          "kitti360": "configs/gate8a/frozen_selection.yaml"}
TARGET = {"semantickitti": "semantickitti", "occ3d": "occ3d", "kitti360": "kitti360"}


def frozen(fold: str) -> dict:
    p = os.path.join(REPO_ROOT, FROZEN[fold])
    if not os.path.exists(p):
        raise SystemExit(f"refusing to evaluate {fold}: frozen manifest {p} does not exist")
    f = yaml.safe_load(open(p))
    return {"checkpoint": f["checkpoint"], "threshold": float(f["global_threshold_final_logodds"]),
            "mapper_threshold": float(f["mapper_calibrated_threshold_logodds"]),
            "sha256": f["checkpoint_sha256"], "path": p}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True, choices=list(FROZEN))
    ap.add_argument("--setting", required=True, choices=["stream", "clips"])
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    fz = frozen(a.fold); tgt = TARGET[a.fold]
    import hashlib
    h = hashlib.sha256(open(os.path.join(REPO_ROOT, fz["checkpoint"]), "rb").read()).hexdigest()
    assert h == fz["sha256"], f"checkpoint hash mismatch for fold {a.fold}"
    print(f"[fold {a.fold}] target {tgt} setting {a.setting} ckpt {fz['checkpoint']} "
          f"tau {fz['threshold']:+.4f} tau_map {fz['mapper_threshold']:+.4f} (from {fz['path']})",
          flush=True)
    if a.setting == "stream":
        if a.fold == "kitti360":
            raise SystemExit("the KITTI-360 streaming result is Gate 8A's; reused, not re-run")
        from tools.gate8a.evaluate import run
        run(tgt, a.device, {"selected": fz["checkpoint"]}, tag=f"stream_{a.fold}",
            limit_anchors=a.limit, locked="selected", threshold=fz["threshold"],
            mapper_threshold=fz["mapper_threshold"], art=ART)
    else:
        from tools.gate8b.eval_clips import run
        run(tgt, a.device, fz["checkpoint"], fz["threshold"], fz["mapper_threshold"],
            tag=f"clips_{a.fold}", limit=a.limit, art=ART)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
