#!/usr/bin/env python
"""Gate 8B step 0 — fetch SSCBench-KITTI-360 **training-drive** occupancy targets.

Same mechanism as Gate 5.2's validation-only extraction (byte-range reads of the SquashFS
data blocks on the Hugging Face Hub, ~16 MB of cached metadata, nothing else transferred),
generalised to the official *train* drives that Gate 8B uses as a completion-training
source. Only the ``preprocess/labels/<drive>/<idx>_1_1.npy`` targets are pulled; images
and velodyne for these drives are already on disk.

    python tools/gate8b/fetch_k360_labels.py --drives 0003 0007 0010
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sscbench_kitti360.remote import HF_REPO, HF_REVISION, IMAGE_BYTES, PART_NAMES, RemoteSplitFile
from sscbench_kitti360.squashfs import SquashFS
from sscbench_kitti360.adapter import OFFICIAL_SPLIT

ROOT = "/media/SSD1/MINH_DATASETS/sscbench_kitti360"
ARCHIVE_ROOT = "sscbench-kitti"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--drives", nargs="+", default=["0003", "0007", "0010"])
    a = ap.parse_args()
    drives = [f"2013_05_28_drive_{d}_sync" for d in a.drives]
    for d in drives:
        assert d in OFFICIAL_SPLIT["train"], f"{d} is not an official SSCBench train drive"
    meta = os.path.join(a.root, "_meta"); os.makedirs(meta, exist_ok=True)
    rf = RemoteSplitFile(cache_dir=meta)
    fs = SquashFS(rf.read); sb = fs.sb
    rf.pin(sb.inode_table, sb.bytes_used - sb.inode_table, "metadata_tail.bin")
    index, t0 = [], time.time()
    for d in drives:
        adir = f"{ARCHIVE_ROOT}/preprocess/labels/{d}"
        entries = sorted([e for e in fs.listdir(fs.resolve(adir).inode_ref, adir)
                          if e.path.endswith("_1_1.npy")], key=lambda e: e.path)
        out_dir = os.path.join(a.root, "preprocess", "labels", d)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{d}] {len(entries)} targets -> {out_dir}", flush=True)
        for i, e in enumerate(entries):
            name = e.path.rsplit("/", 1)[-1]
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst) and os.path.getsize(dst) == e.size:
                continue
            blob = fs.read_file(e.inode_ref)
            if len(blob) != e.size:
                raise IOError(f"{name}: {len(blob)} != {e.size}")
            with open(dst + ".part", "wb") as fh:
                fh.write(blob)
            os.replace(dst + ".part", dst)
            index.append({"drive": d, "name": name, "size": e.size,
                          "sha256": hashlib.sha256(blob).hexdigest()})
            if (i + 1) % 100 == 0:
                print(f"   {i+1}/{len(entries)}  {rf.bytes_fetched/1e9:.3f} GB  "
                      f"{time.time()-t0:.0f}s", flush=True)
    prov = {"source": f"https://huggingface.co/datasets/{HF_REPO}", "revision": HF_REVISION,
            "parts": list(PART_NAMES), "archive_bytes": IMAGE_BYTES,
            "license": "CC BY-NC-SA 3.0 (SSCBench-KITTI-360)", "drives": drives,
            "official_split_role": "train (completion-training source for Gate 8B)",
            "extraction": "HTTP byte-range reads of the squashfs data blocks for the "
                          "listed drives' preprocess/labels only",
            "bytes_downloaded": rf.bytes_fetched, "http_requests": rf.requests_made,
            "elapsed_s": time.time() - t0, "files": index}
    with open(os.path.join(meta, "provenance_gate8b_train_labels.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"\n{len(index)} new files; {rf.bytes_fetched/1e9:.3f} GB in {time.time()-t0:.0f}s "
          f"({rf.requests_made} range requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
