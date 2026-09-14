#!/usr/bin/env python
"""One frozen transfer evaluation per checkpoint and dataset. Nothing is tuned here.

Runs the gate's own evaluator (``core.run.eval_target``) unchanged, through the study's
frozen manifest, on the declared ``past5`` protocol and the existing complete anchor
populations. The evaluator's grid, validity masks, scoring conventions, baselines and
semantic pipeline are untouched; only the manifest it reads and the directory it writes to
are redirected, so a study run cannot overwrite a gate artefact.

The expected populations are asserted, not assumed: 163 SemanticKITTI anchors and 1 182
Occ3D anchors. A different count stops the run instead of being quietly reported.

    python tools/feature_study/transfer.py --condition C --seed 0 --dataset semantickitti
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402

#: the verified current manifests, from the adopted ceiling-fixed evaluation
EXPECTED_CLIPS = {"semantickitti": 163, "occ3d": 1182}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(ST.CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--mode", default="past5", choices=["past5"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    man = ST.manifest_path(a.condition)
    if not os.path.exists(man):
        raise SystemExit(f"refusing to open {a.dataset}: {man} is not frozen yet")
    os.environ["G8C1_MANIFEST"] = man
    from core.run import eval_target as EV                                        # noqa: E402
    EV.MANIFEST = man                                     # read inside run(), at call time
    art = os.path.join(ST.ART, "transfer", a.condition)
    os.makedirs(art, exist_ok=True)
    out = EV.run(a.dataset, a.mode, a.seed, a.device, None, art=art)
    n, want = out["n_clips"], EXPECTED_CLIPS[a.dataset]
    if n != want:
        raise SystemExit(f"population mismatch on {a.dataset}: scored {n} anchors, the "
                         f"verified manifest has {want}. Not reporting a different "
                         f"population under the same name.")
    print(f"[{a.condition} seed{a.seed} {a.dataset}] population verified: {n} anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
