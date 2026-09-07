#!/usr/bin/env python
"""Gate 8C-0: descriptive map-channel distributions across the three KITTI-360 partitions.

Stage 7's training experiment is **gated** on Stages 1-6 showing correct alignment, which
they do not. The channel comparison it asked for is cheap and independent of that gate, so
it is reported here as description only. Nothing is normalised, and no conclusion in this
report rests on it.

    python tools/gate8c0/stage7_channel_stats.py --device cuda:1
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg, default_device                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import targets as TG, vocab as V8                                     # noqa: E402
from gates.gate8a.regions import editable_native                                       # noqa: E402
from gates.gate8b.sources import G8B_ROOT                                              # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402

NAMES = ["logodds/4", "w_free/20", "observed", "unobserved", "n_obs/50", "age/100", "sem_w/20"] \
    + [f"sem[{u}]" for u in range(V8.U)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--per-partition", type=int, default=12)
    a = ap.parse_args()
    c = cfg(); dev = torch.device(a.device); t0 = time.time()
    parts = {"train": c["train_drives"], "source_validation": [c["val_drive"]],
             "heldout": [c["heldout_drive"]]}
    rng = np.random.default_rng(c["seed"])
    out = {"channel_names": NAMES, "partitions": {}, "note": "descriptive only; Gate 8C-0 "
           "does not normalise inputs", "seconds": None}
    for part, drives in parts.items():
        src = "kitti360" if part == "heldout" else "k360_train"
        files = [p for p in sorted(glob.glob(f"{G8B_ROOT}/samples/{src}/*.npz"))
                 if os.path.basename(p).rsplit("_", 1)[0] in drives]
        if not files:
            continue
        files = [files[i] for i in rng.choice(len(files), size=min(a.per_partition, len(files)),
                                              replace=False)]
        acc = {"sum": np.zeros(len(NAMES)), "sq": np.zeros(len(NAMES)), "n": 0,
               "zero": np.zeros(len(NAMES)), "sat": np.zeros(len(NAMES))}
        qs, extra = [], {"n_obs": [], "age": [], "logodds": [], "editable_frac": [],
                         "observed_frac": [], "prevalence": [], "valid_frac": [],
                         "z_profile": [], "range_profile": []}
        for p in files:
            with np.load(p) as z:
                d = TG.unpack_sample(z, dev)
            x = d["input"].reshape(len(NAMES), -1)
            acc["sum"] += x.sum(1).double().cpu().numpy()
            acc["sq"] += (x.double() ** 2).sum(1).cpu().numpy()
            acc["n"] += x.shape[1]
            acc["zero"] += (x == 0).sum(1).double().cpu().numpy()
            acc["sat"] += (x >= 1.0).sum(1).double().cpu().numpy()
            qs.append(np.percentile(x[:7].float().cpu().numpy(), [5, 25, 50, 75, 95], axis=1))
            lo = d["base_logodds"]; obs = d["observed"]; val = d["gt_valid"]; occ = d["gt_occ"] > 0
            extra["logodds"].append(float(lo[obs].mean()) if obs.any() else 0.0)
            extra["n_obs"].append(float((x[4] * 50).mean()))
            extra["age"].append(float((x[5] * 100).mean()))
            extra["editable_frac"].append(float((editable_native(lo) & val).sum() / max(val.sum(), 1)))
            extra["observed_frac"].append(float(obs.float().mean()))
            extra["prevalence"].append(float((occ & val).sum() / max(val.sum(), 1)))
            extra["valid_frac"].append(float(val.float().mean()))
            o = (lo > 0).cpu().numpy().reshape(TF.DIMS)
            extra["z_profile"].append(o.sum(axis=(0, 1)) / max(o.sum(), 1))
            xr = np.arange(TF.DIMS[0]) * TF.VOXEL
            extra["range_profile"].append(np.histogram(
                np.repeat(xr, o.shape[1] * o.shape[2]).reshape(o.shape)[o], bins=16,
                range=(0, 51.2))[0] / max(o.sum(), 1))
        m = acc["sum"] / acc["n"]
        sd = np.sqrt(np.maximum(acc["sq"] / acc["n"] - m ** 2, 0))
        Q = np.mean(np.stack(qs), 0)
        out["partitions"][part] = {
            "n_samples": len(files), "drives": drives,
            "channel_mean": m.tolist(), "channel_sd": sd.tolist(),
            "channel_zero_fraction": (acc["zero"] / acc["n"]).tolist(),
            "channel_saturation_fraction": (acc["sat"] / acc["n"]).tolist(),
            "quantiles_first7": {"p5": Q[0].tolist(), "p25": Q[1].tolist(), "p50": Q[2].tolist(),
                                 "p75": Q[3].tolist(), "p95": Q[4].tolist()},
            "mean_observed_logodds": float(np.mean(extra["logodds"])),
            "mean_n_obs": float(np.mean(extra["n_obs"])),
            "mean_age": float(np.mean(extra["age"])),
            "editable_fraction_of_valid": float(np.mean(extra["editable_frac"])),
            "observed_fraction_of_grid": float(np.mean(extra["observed_frac"])),
            "target_prevalence": float(np.mean(extra["prevalence"])),
            "valid_fraction_of_grid": float(np.mean(extra["valid_frac"])),
            "occupancy_z_profile": np.mean(np.stack(extra["z_profile"]), 0).tolist(),
            "occupancy_range_profile": np.mean(np.stack(extra["range_profile"]), 0).tolist()}
        v = out["partitions"][part]
        print(f"  {part:17s} n={len(files):3d} observed {v['observed_fraction_of_grid']:.4f} "
              f"editable {v['editable_fraction_of_valid']:.4f} prevalence {v['target_prevalence']:.4f} "
              f"valid {v['valid_fraction_of_grid']:.4f} mean n_obs {v['mean_n_obs']:.2f} "
              f"mean age {v['mean_age']:.1f}", flush=True)
    out["seconds"] = time.time() - t0
    write_json(os.path.join(ART, "stage7_channel_stats.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
