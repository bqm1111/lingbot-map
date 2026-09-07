#!/usr/bin/env python
"""What each crop sampler actually shows the network.

Gate 8's sampler centres 70% of crops on a ground-truth-occupied, still-unobserved voxel.
This measures the consequence directly: the occupied prevalence inside the crops each
sampler draws, and the occupied prevalence inside the *editable* part of those crops,
separately for SemanticKITTI and Occ3D training samples.

    python tools/gate8a/sampler_stats.py --n 400 --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate8a.regions import editable_native                                       # noqa: E402
from tools.gate8.train import _default_device                                    # noqa: E402
from tools.gate8a.train import AblationSamples                                   # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")


def measure(source, sampler, crop, n, device, seed, lock):
    s = AblationSamples([source], crop, device, seed, sampler=sampler, centre_on_occ_p=0.7)
    s.verify()
    acc = {k: 0 for k in ("valid", "occ", "edit", "edit_occ", "vol")}
    full = {"valid": 0, "occ": 0, "edit": 0, "edit_occ": 0}
    for _ in range(n):
        d = s.load(s.rng.randrange(len(s)))
        e = editable_native(d["base_logodds"], lock)
        v, o = d["gt_valid"], d["gt_occ"] > 0
        full["valid"] += int(v.sum()); full["occ"] += int((o & v).sum())
        full["edit"] += int((e & v).sum()); full["edit_occ"] += int((o & v & e).sum())
        x0, y0, z0 = s.crop_of(d)
        sl = (slice(x0, x0 + crop[0]), slice(y0, y0 + crop[1]), slice(z0, z0 + crop[2]))
        v, o, e = v[sl], o[sl], e[sl]
        acc["vol"] += int(v.numel()); acc["valid"] += int(v.sum())
        acc["occ"] += int((o & v).sum()); acc["edit"] += int((e & v).sum())
        acc["edit_occ"] += int((o & v & e).sum())
    return {"source": source, "sampler": sampler, "n_crops": n, "n_samples": len(s),
            "crop_valid_voxels": acc["valid"], "crop_volume": acc["vol"],
            "crop_occupied_prevalence": acc["occ"] / max(acc["valid"], 1),
            "crop_editable_fraction": acc["edit"] / max(acc["valid"], 1),
            "crop_editable_occupied_prevalence": acc["edit_occ"] / max(acc["edit"], 1),
            "volume_occupied_prevalence": full["occ"] / max(full["valid"], 1),
            "volume_editable_occupied_prevalence": full["edit_occ"] / max(full["edit"], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--config", default="configs/gate8a/ablation.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, a.config)))
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    out, t0 = [], time.time()
    for src in ("sk_train", "occ3d_train"):
        for smp in ("occ_centred", "uniform"):
            r = measure(src, smp, tuple(cfg["crop"]), a.n, dev, cfg["seed"],
                        cfg["lock_logodds"])
            out.append(r)
            print(f"[{src:12s} {smp:12s}] crop occ prevalence "
                  f"{r['crop_occupied_prevalence']:.4f}  editable-region "
                  f"{r['crop_editable_occupied_prevalence']:.4f}  (whole volume "
                  f"{r['volume_occupied_prevalence']:.4f} / "
                  f"{r['volume_editable_occupied_prevalence']:.4f})", flush=True)
    write_json(os.path.join(ART, "sampler_stats.json"),
               {"n_crops": a.n, "crop": cfg["crop"], "seed": cfg["seed"],
                "lock_logodds": cfg["lock_logodds"], "seconds": time.time() - t0,
                "rows": out})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
