"""Byte-range access to the split SSCBench SquashFS image on Hugging Face.

The image is published as ten ``split``-produced parts.  Concatenated they form
one 206.7 GB SquashFS filesystem; part *i* holds bytes
``[i * PART_SIZE, (i+1) * PART_SIZE)`` of that filesystem, so an absolute image
offset maps to a part index and an offset inside it by plain division.  Only the
regions actually requested are transferred.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

HF_REPO = "ai4ce/SSCBench"
HF_REVISION = "badde69bbd01552be186dc0ddd0989885b01c532"
PART_NAMES = tuple(f"sscbench-kitti/sscbench-kitti-part_a{c}" for c in "abcdefghij")
PART_SIZE = 21474836480  # 20 GiB, the size of every part except the last
IMAGE_BYTES = 206742990564

_CHUNK = 1 << 20  # alignment for opportunistic caching of small reads


def part_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{name}"


@dataclass
class _Pinned:
    start: int
    data: bytes

    def covers(self, off: int, length: int) -> bool:
        return self.start <= off and off + length <= self.start + len(self.data)

    def slice(self, off: int, length: int) -> bytes:
        base = off - self.start
        return self.data[base : base + length]


class RemoteSplitFile:
    """Random-access reader over the concatenated Hugging Face parts."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        session: Optional[requests.Session] = None,
        max_retries: int = 5,
    ):
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.bytes_fetched = 0
        self.requests_made = 0
        self._pinned: List[_Pinned] = []
        self._chunks: Dict[int, bytes] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- part mapping ------------------------------------------------------
    @staticmethod
    def locate(offset: int) -> Tuple[int, int]:
        """Map an absolute image offset to (part index, offset within part)."""
        if not 0 <= offset < IMAGE_BYTES:
            raise ValueError(f"offset {offset} outside image of {IMAGE_BYTES} bytes")
        return offset // PART_SIZE, offset % PART_SIZE

    def _http_range(self, part: int, start: int, length: int) -> bytes:
        url = part_url(PART_NAMES[part])
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, headers=headers, timeout=120)
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code} for {url} range {headers['Range']}")
                data = r.content
                if len(data) != length:
                    raise IOError(f"short range read: {len(data)} != {length}")
                self.requests_made += 1
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # transient network failure
                last = exc
                time.sleep(min(2**attempt, 30))
        raise IOError(f"range read failed after {self.max_retries} attempts: {last}")

    def _raw(self, offset: int, length: int) -> bytes:
        """Fetch bytes, splitting the request across part boundaries."""
        out = bytearray()
        while length > 0:
            part, off = self.locate(offset)
            part_bytes = min(PART_SIZE, IMAGE_BYTES - part * PART_SIZE)
            take = min(length, part_bytes - off)
            out += self._http_range(part, off, take)
            offset += take
            length -= take
        return bytes(out)

    # -- public API --------------------------------------------------------
    def read(self, offset: int, length: int) -> bytes:
        if length == 0:
            return b""
        for p in self._pinned:
            if p.covers(offset, length):
                return p.slice(offset, length)
        if length <= _CHUNK // 4:
            base = (offset // _CHUNK) * _CHUNK
            if offset + length <= base + _CHUNK:
                chunk = self._chunks.get(base)
                if chunk is None:
                    size = min(_CHUNK, IMAGE_BYTES - base)
                    chunk = self._raw(base, size)
                    if len(self._chunks) > 512:
                        self._chunks.clear()
                    self._chunks[base] = chunk
                return chunk[offset - base : offset - base + length]
        return self._raw(offset, length)

    def pin(self, offset: int, length: int, name: str) -> None:
        """Fetch a region once and keep it resident (optionally cached on disk)."""
        for p in self._pinned:
            if p.covers(offset, length):
                return
        path = os.path.join(self.cache_dir, name) if self.cache_dir else None
        if path and os.path.exists(path) and os.path.getsize(path) == length:
            with open(path, "rb") as fh:
                data = fh.read()
        else:
            data = self._raw(offset, length)
            if path:
                tmp = path + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, path)
        self._pinned.append(_Pinned(offset, data))
