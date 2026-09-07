#!/usr/bin/env python
"""Recompute ``pred_pose_c2w`` in every cached clip from the stored ``pose_enc``.

An earlier cache build inverted the output of ``pose_encoding_to_extri_intri`` on the
assumption -- taken from ``demo.py`` -- that it returns world-to-camera. It returns
camera-to-world, so the stored transforms were the inverse of the intended ones. Because
``pose_enc`` is cached, this is repaired without re-running inference.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, load_config
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/scale_gate/semantickitti.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.cache.root, "lingbot")
    files = sorted(f for f in os.listdir(root) if f.endswith(".npz"))
    n_fixed = 0
    for i, fn in enumerate(files):
        p = os.path.join(root, fn)
        z = dict(np.load(p, allow_pickle=False))
        pe = torch.from_numpy(z["pose_enc"]).float().unsqueeze(0)
        hw = tuple(int(x) for x in z["proc_hw"])
        extr, _ = pose_encoding_to_extri_intri(pe, hw)
        c2w = np.tile(np.eye(4), (extr.shape[1], 1, 1))
        c2w[:, :3, :4] = extr[0].double().numpy()
        if np.allclose(z["pred_pose_c2w"], c2w.astype(np.float32), atol=1e-5):
            continue
        if not a.dry_run:
            z["pred_pose_c2w"] = c2w.astype(np.float32)
            np.savez_compressed(p + ".tmp.npz", **z)
            os.replace(p + ".tmp.npz", p)
        n_fixed += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    print(f"{'would repair' if a.dry_run else 'repaired'} {n_fixed}/{len(files)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
