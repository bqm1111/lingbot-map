#!/usr/bin/env python
"""Gate 5.2 — prove the SSCBench -> KITTI-360 frame mapping by pixel equality.

SSCBench renumbered KITTI-360, so ``sscbench index i`` is not KITTI-360 frame ``i``. This
tool reads a sample of the archive's own ``image_00/data_rect`` PNGs by byte-range and
checks them pixel-for-pixel against the local KITTI-360 release under the claimed mapping
``kitti360_frame = pose_frames[i + 1]``. Local imagery is only usable because of this.

    python tools/gate5_2/verify_index_mapping.py --n 40
"""
from __future__ import annotations

import argparse, hashlib, io, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from sscbench_kitti360.adapter import SEQUENCE, pose_frames, sscbench_to_native
from tools.gate5_2.prepare_data import ARCHIVE_ROOT, open_archive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from PIL import Image

    frames = pose_frames(os.path.join(cfg.dataset.root, "data_poses", SEQUENCE, "poses.txt"))
    rf, fs = open_archive(cfg.dataset.root)
    d = f"{ARCHIVE_ROOT}/data_2d_raw/{SEQUENCE}/image_00/data_rect"
    ents = {e.path.rsplit("/", 1)[-1]: e for e in fs.listdir(fs.resolve(d).inode_ref, d)}
    names = sorted(ents)
    picks = [names[int(round(i * (len(names) - 1) / (a.n - 1)))] for i in range(a.n)]
    local = os.path.join(cfg.dataset.kitti360_root, "data_2d_raw", SEQUENCE,
                         "image_00", "data_rect")

    rows, n_match = [], 0
    for nm in picks:
        i = int(nm[:6])
        f = int(sscbench_to_native(i, frames))
        arch = np.asarray(Image.open(io.BytesIO(fs.read_file(ents[nm].inode_ref))))
        loc = np.asarray(Image.open(os.path.join(local, f"{f:010d}.png")))
        same = arch.shape == loc.shape and np.array_equal(arch, loc)
        n_match += same
        rows.append({"sscbench_index": i, "kitti360_frame": f, "identical": bool(same),
                     "shape": list(arch.shape),
                     "sha256_local": hashlib.sha256(loc.tobytes()).hexdigest()[:32]})
    out = {"mapping": "kitti360_frame = pose_frames[sscbench_index + 1]",
           "n_checked": len(rows), "n_identical": int(n_match),
           "all_identical": n_match == len(rows),
           "n_archive_images": len(names), "n_pose_frames": int(len(frames)),
           "bytes_downloaded": rf.bytes_fetched, "samples": rows}
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir,
                            "index_mapping_verification.json"), out)
    print(f"{n_match}/{len(rows)} archive images identical to the local KITTI-360 release "
          f"under the claimed mapping ({rf.bytes_fetched/1e6:.1f} MB fetched)")
    return 0 if out["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
