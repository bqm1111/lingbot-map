#!/usr/bin/env python
"""Validate (and optionally delete) corrupt frames in the broad corpus.

    python tools/verify_corpus_images.py --corpus <dir> [--delete]

A frame that exists with non-zero size can still be truncated — an interrupted
in-place download leaves exactly that.  Every image is fully decoded here, so the
caching stage cannot fail halfway through on a bad file.
"""

from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from PIL import Image


def check(path: str) -> Optional[str]:
    try:
        with Image.open(path) as im:
            im.load()
        return None
    except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies the frame
        return f"{path}: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--delete", action="store_true", help="remove corrupt frames so they can be refetched")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.corpus, "*", "*.png")))
    print(f"checking {len(paths)} frames ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, paths))
    bad: List[str] = [r for r in results if r]
    for b in bad[:20]:
        print("  CORRUPT", b)
    print(f"{len(bad)} corrupt of {len(paths)}")
    if bad and args.delete:
        for line in bad:
            os.remove(line.split(":")[0])
        print(f"deleted {len(bad)} corrupt frames; re-run the corpus builder to refetch")
    raise SystemExit(1 if bad and not args.delete else 0)


if __name__ == "__main__":
    main()
