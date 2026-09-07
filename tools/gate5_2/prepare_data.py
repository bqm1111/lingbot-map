#!/usr/bin/env python
"""Gate 5.2 step 0 — validation-only extraction of SSCBench-KITTI-360 sequence 0006.

SSCBench publishes KITTI-360 as a single 206.7 GB SquashFS image split into ten parts on
the Hugging Face Hub.  SquashFS keeps its inode and directory tables near the end of the
image, so the whole tree is enumerable from ~16 MB of metadata and individual files can be
pulled with HTTP byte-range reads.  This tool therefore downloads **only the official
validation sequence's targets** -- roughly 0.4 GB of the 206.7 GB archive -- instead of the
whole train+val+test package.

Nothing here touches training or test sequences, fisheye or right-camera imagery, or the
nuScenes/Waymo subsets.

    python tools/gate5_2/prepare_data.py --what voxels --limit 20
    python tools/gate5_2/prepare_data.py --what all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sscbench_kitti360.remote import (HF_REPO, HF_REVISION, IMAGE_BYTES, PART_NAMES,
                                      RemoteSplitFile)
from sscbench_kitti360.squashfs import SquashFS

SEQ = "2013_05_28_drive_0006_sync"
ARCHIVE_ROOT = "sscbench-kitti"
GROUPS = {
    # what                            archive directory                       suffixes
    "voxels": (f"{ARCHIVE_ROOT}/data_2d_raw/{SEQ}/voxels", (".bin", ".label", ".invalid")),
    "labels": (f"{ARCHIVE_ROOT}/preprocess/labels/{SEQ}", ("_1_1.npy",)),
}


def open_archive(root: str):
    meta = os.path.join(root, "_meta")
    os.makedirs(meta, exist_ok=True)
    rf = RemoteSplitFile(cache_dir=meta)
    fs = SquashFS(rf.read)
    sb = fs.sb
    rf.pin(sb.inode_table, sb.bytes_used - sb.inode_table, "metadata_tail.bin")
    return rf, fs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/media/SSD1/MINH_DATASETS/sscbench_kitti360")
    ap.add_argument("--what", default="all", choices=["all", "voxels", "labels"])
    ap.add_argument("--limit", type=int, default=None, help="first N anchors only")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    rf, fs = open_archive(a.root)
    groups = list(GROUPS) if a.what == "all" else [a.what]
    index, t0 = [], time.time()
    for g in groups:
        adir, suffixes = GROUPS[g]
        entries = fs.listdir(fs.resolve(adir).inode_ref, adir)
        entries = [e for e in entries if e.path.endswith(suffixes)]
        entries.sort(key=lambda e: e.path)
        if a.limit:
            keep = sorted({e.path.rsplit("/", 1)[-1].split(".")[0].split("_")[0]
                           for e in entries})[: a.limit]
            entries = [e for e in entries
                       if e.path.rsplit("/", 1)[-1].split(".")[0].split("_")[0] in set(keep)]
        out_dir = os.path.join(a.root, adir[len(ARCHIVE_ROOT) + 1:])
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{g}] {len(entries)} files -> {out_dir}")
        for i, e in enumerate(entries):
            name = e.path.rsplit("/", 1)[-1]
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst) and os.path.getsize(dst) == e.size and not a.overwrite:
                continue
            blob = fs.read_file(e.inode_ref)
            if len(blob) != e.size:
                raise IOError(f"{name}: {len(blob)} != {e.size}")
            with open(dst + ".part", "wb") as fh:
                fh.write(blob)
            os.replace(dst + ".part", dst)
            index.append({"group": g, "name": name, "size": e.size,
                          "sha256": hashlib.sha256(blob).hexdigest()})
            if (i + 1) % 250 == 0:
                print(f"   {i+1}/{len(entries)}  {rf.bytes_fetched/1e9:.3f} GB  "
                      f"{time.time()-t0:.0f}s", flush=True)

    prov = {
        "source": f"https://huggingface.co/datasets/{HF_REPO}",
        "revision": HF_REVISION,
        "parts": list(PART_NAMES),
        "archive_format": "squashfs 4.0, zlib, split into 10 parts",
        "archive_bytes": IMAGE_BYTES,
        "license": "CC BY-NC-SA 3.0 (SSCBench-KITTI-360)",
        "sequence": SEQ,
        "official_split_role": "validation",
        "extraction": "HTTP byte-range reads of the squashfs data blocks for this "
                      "sequence only; training and test sequences never transferred",
        "bytes_downloaded": rf.bytes_fetched,
        "http_requests": rf.requests_made,
        "elapsed_s": time.time() - t0,
        "files": index,
    }
    os.makedirs(os.path.join(a.root, "_meta"), exist_ok=True)
    tag = a.what if not a.limit else f"{a.what}_limit{a.limit}"
    with open(os.path.join(a.root, "_meta", f"provenance_{tag}.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"\n{len(index)} new files; {rf.bytes_fetched/1e9:.3f} GB transferred "
          f"in {time.time()-t0:.0f}s ({rf.requests_made} range requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
