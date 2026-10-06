"""Drop the kernel page cache held by this study's own large files (``posix_fadvise(POSIX_FADV_DONTNEED)``; no root, no data change).

Why: preparation and training read and write hundreds of GB. The cached pages are reclaimable, but they drive *free* memory to
near zero, and the session's memory guard stopped background jobs at such a moment (2026-09-15 19:50) although ~109 GB was
still available. Only files under the study's own roots are touched; other users' files and the page cache in general are not.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.cache_trim --once          # one pass over all roots
    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.cache_trim --every 60      # keep trimming files modified recently
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, Tuple

ROOTS: Tuple[str, ...] = ("/media/welf/SSD2/MINH_DATASETS/lingbot_b4_scaling", "/media/SSD1/MINH_DATASETS/lingbot_b4_scaling",
                          "/media/minh/MINH1/lingbot_b4_scaling")
#: existing caches this study READS in bulk (current-pool training files, tokens, 0006 clips); trimming them only drops cached pages
#: of our own read-only files. Raw KITTI-360 on MINH1 is not listed: walking its ~300k files on USB exFAT takes minutes per pass.
READ_ROOTS: Tuple[str, ...] = ("/media/SSD1/MINH_DATASETS/lingbot_gate15_semantic", "/media/SSD1/MINH_DATASETS/lingbot_delayed_semantic",
                               "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf", "/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d",
                               "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit")


def _files(roots: Iterable[str], newer_than: float) -> Iterable[str]:
    for root in roots:
        for dirpath, _, names in os.walk(root, followlinks=False):
            for n in names:
                p = os.path.join(dirpath, n)
                try:
                    if os.path.getmtime(p) >= newer_than:
                        yield p
                except FileNotFoundError:          # renamed by a writer between listing and stat (tmp -> final)
                    continue


def trim(roots: Iterable[str] = ROOTS, newer_than: float = 0.0) -> Tuple[int, int]:
    """Advise DONTNEED on every regular file under ``roots`` modified at or after ``newer_than``. Returns (files, bytes)."""
    n = size = 0
    for p in _files(roots, newer_than):
        try:
            fd = os.open(p, os.O_RDONLY)
        except (FileNotFoundError, PermissionError):
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1; size += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return n, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true"); ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--with-read-roots", action="store_true", help="also trim READ_ROOTS (existing caches and raw data read in bulk)")
    a = ap.parse_args()
    roots = ROOTS + (READ_ROOTS if a.with_read_roots else ())
    if a.once:
        n, size = trim(roots); print(f"trimmed {n} files ({size / 1e9:.1f} GB)", flush=True); return
    while True:                                         # training reads old files too, so every pass trims all listed files
        start = time.time(); n, size = trim(roots)
        print(f"{time.strftime('%H:%M:%S')} trimmed {n} files ({size / 1e9:.1f} GB) in {time.time() - start:.1f} s", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
