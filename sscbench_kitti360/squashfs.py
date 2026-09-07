"""Minimal read-only SquashFS 4.0 reader.

SSCBench-KITTI-360 is published as a single 206.7 GB SquashFS image split into
ten 20 GiB parts.  SquashFS stores its inode and directory tables near the *end*
of the image, so the whole tree can be enumerated from ~16 MB of metadata and
individual files can be pulled with byte-range reads.  That is what makes a
validation-only extraction possible without downloading the full archive.

The reader is deliberately restricted to what this image actually uses:
zlib-compressed metadata, data blocks and fragments (superblock flags 0xC0 =
DUPLICATES | EXPORTABLE, so nothing is stored uncompressed by default).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

SQUASHFS_MAGIC = 0x73717368
METADATA_MAX = 8192

INODE_BASIC_DIR = 1
INODE_BASIC_FILE = 2
INODE_BASIC_SYMLINK = 3
INODE_EXT_DIR = 8
INODE_EXT_FILE = 9
INODE_EXT_SYMLINK = 10

_SUPERBLOCK = "<IIIIIHHHHHHQQQQQQQQ"
_SUPERBLOCK_SIZE = struct.calcsize(_SUPERBLOCK)


@dataclass(frozen=True)
class Superblock:
    inodes: int
    mkfs_time: int
    block_size: int
    fragments: int
    compression: int
    block_log: int
    flags: int
    no_ids: int
    s_major: int
    s_minor: int
    root_inode: int
    bytes_used: int
    id_table: int
    xattr_id_table: int
    inode_table: int
    directory_table: int
    fragment_table: int
    lookup_table: int

    @classmethod
    def parse(cls, raw: bytes) -> "Superblock":
        f = struct.unpack(_SUPERBLOCK, raw[:_SUPERBLOCK_SIZE])
        if f[0] != SQUASHFS_MAGIC:
            raise ValueError(f"not a squashfs image (magic 0x{f[0]:08x})")
        if (f[9], f[10]) != (4, 0):
            raise ValueError(f"unsupported squashfs version {f[9]}.{f[10]}")
        if f[5] != 1:
            raise ValueError(f"unsupported compressor id {f[5]} (only zlib=1)")
        return cls(*f[1:])


@dataclass(frozen=True)
class Entry:
    """One directory entry resolved to its inode."""

    path: str
    inode_ref: int
    inode_type: int
    size: int
    mtime: int
    mode: int

    @property
    def is_dir(self) -> bool:
        return self.inode_type in (INODE_BASIC_DIR, INODE_EXT_DIR)

    @property
    def is_file(self) -> bool:
        return self.inode_type in (INODE_BASIC_FILE, INODE_EXT_FILE)


@dataclass(frozen=True)
class FileInode:
    blocks_start: int
    file_size: int
    frag_index: int
    block_offset: int
    block_sizes: Tuple[int, ...]


def _decompress_metadata(raw: bytes) -> bytes:
    out = zlib.decompress(raw)
    if len(out) > METADATA_MAX:
        raise ValueError(f"metadata block too large ({len(out)} > {METADATA_MAX})")
    return out


class SquashFS:
    """Random-access SquashFS reader.

    ``read(offset, length)`` is any callable returning exactly ``length`` bytes
    from absolute image offset ``offset``.
    """

    def __init__(self, read: Callable[[int, int], bytes]):
        self._read = read
        self.sb = Superblock.parse(read(0, _SUPERBLOCK_SIZE))
        self._meta_cache: Dict[int, Tuple[bytes, int]] = {}
        self._frag_entries: Optional[List[Tuple[int, int]]] = None

    # -- metadata blocks ---------------------------------------------------
    def _metadata_block(self, offset: int) -> Tuple[bytes, int]:
        """Return (uncompressed payload, absolute offset of the next block)."""
        cached = self._meta_cache.get(offset)
        if cached is not None:
            return cached
        header = struct.unpack("<H", self._read(offset, 2))[0]
        size = header & 0x7FFF
        raw = self._read(offset + 2, size)
        data = raw if header & 0x8000 else _decompress_metadata(raw)
        result = (data, offset + 2 + size)
        self._meta_cache[offset] = result
        return result

    def _read_metadata(self, start: int, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting ``offset`` into the block at ``start``."""
        out = bytearray()
        pos = start
        skip = offset
        while len(out) < length:
            block, nxt = self._metadata_block(pos)
            if skip:
                block = block[skip:]
                skip = 0
            out += block[: length - len(out)]
            if len(out) < length:
                if nxt == pos:
                    raise ValueError("metadata read made no progress")
                pos = nxt
        return bytes(out)

    def _metadata_stream(self, start: int, offset: int) -> Iterator[bytes]:
        pos, skip = start, offset
        while True:
            block, nxt = self._metadata_block(pos)
            yield block[skip:] if skip else block
            skip = 0
            pos = nxt

    # -- inodes ------------------------------------------------------------
    @staticmethod
    def split_ref(ref: int) -> Tuple[int, int]:
        """Split a squashfs metadata reference into (block offset, byte offset)."""
        return ref >> 16, ref & 0xFFFF

    def _inode_bytes(self, ref: int, length: int) -> bytes:
        block, off = self.split_ref(ref)
        return self._read_metadata(self.sb.inode_table + block, off, length)

    def inode_type(self, ref: int) -> int:
        return struct.unpack("<H", self._inode_bytes(ref, 2))[0]

    def _dir_location(self, ref: int) -> Tuple[int, int, int]:
        """Return (dir table start_block, byte offset, payload size) for a dir."""
        itype = self.inode_type(ref)
        if itype == INODE_BASIC_DIR:
            raw = self._inode_bytes(ref, 16 + 16)
            start_block, _nlink, size, offset, _parent = struct.unpack("<IIHHI", raw[16:32])
            return start_block, offset, size
        if itype == INODE_EXT_DIR:
            raw = self._inode_bytes(ref, 16 + 24)
            _nlink, size, start_block, _parent, _icount, offset, _xattr = struct.unpack(
                "<IIIIHHI", raw[16:40]
            )
            return start_block, offset, size
        raise ValueError(f"inode type {itype} is not a directory")

    def file_inode(self, ref: int) -> FileInode:
        itype = self.inode_type(ref)
        bs = self.sb.block_size
        if itype == INODE_BASIC_FILE:
            raw = self._inode_bytes(ref, 16 + 16)
            blocks_start, frag_index, block_offset, file_size = struct.unpack("<IIII", raw[16:32])
            n = _n_blocks(file_size, bs, frag_index)
            tail = self._inode_bytes(ref, 32 + 4 * n)[32:]
        elif itype == INODE_EXT_FILE:
            raw = self._inode_bytes(ref, 16 + 40)
            blocks_start, file_size, _sparse, _nlink, frag_index, block_offset, _xattr = (
                struct.unpack("<QQQIIII", raw[16:56])
            )
            n = _n_blocks(file_size, bs, frag_index)
            tail = self._inode_bytes(ref, 56 + 4 * n)[56:]
        else:
            raise ValueError(f"inode type {itype} is not a regular file")
        sizes = struct.unpack(f"<{n}I", tail) if n else ()
        return FileInode(blocks_start, file_size, frag_index, block_offset, tuple(sizes))

    # -- directories -------------------------------------------------------
    def listdir(self, ref: int, prefix: str = "") -> List[Entry]:
        start_block, offset, size = self._dir_location(ref)
        payload = size - 3  # squashfs stores size including a 3-byte sentinel
        if payload <= 0:
            return []
        buf = bytearray()
        stream = self._metadata_stream(self.sb.directory_table + start_block, offset)
        while len(buf) < payload:
            buf += next(stream)
        data = bytes(buf[:payload])

        entries: List[Entry] = []
        pos = 0
        while pos + 12 <= len(data):
            count, hdr_start_block, base_inode = struct.unpack("<IIi", data[pos : pos + 12])
            pos += 12
            for _ in range(count + 1):
                e_off, e_delta, e_type, name_size = struct.unpack("<HhHH", data[pos : pos + 8])
                pos += 8
                name = data[pos : pos + name_size + 1].decode("utf-8", "surrogateescape")
                pos += name_size + 1
                child_ref = (hdr_start_block << 16) | e_off
                entries.append(
                    self._entry(f"{prefix}/{name}" if prefix else name, child_ref, e_type)
                )
            _ = base_inode, e_delta
        return entries

    def _entry(self, path: str, ref: int, dir_type: int) -> Entry:
        raw = self._inode_bytes(ref, 16)
        itype, mode, _uid, _gid, mtime, _num = struct.unpack("<HHHHII", raw)
        size = 0
        if itype in (INODE_BASIC_FILE, INODE_EXT_FILE):
            size = self.file_inode(ref).file_size
        _ = dir_type
        return Entry(path, ref, itype, size, mtime, mode)

    def root(self) -> int:
        return self.sb.root_inode

    def resolve(self, path: str) -> Entry:
        ref = self.root()
        parts = [p for p in path.strip("/").split("/") if p]
        entry = Entry("", ref, INODE_BASIC_DIR, 0, 0, 0)
        walked = ""
        for part in parts:
            match = None
            for e in self.listdir(ref, walked):
                if e.path.rsplit("/", 1)[-1] == part:
                    match = e
                    break
            if match is None:
                raise FileNotFoundError(f"{path!r}: no entry {part!r} under {walked or '/'!r}")
            entry, ref, walked = match, match.inode_ref, match.path
        return entry

    def walk(self, ref: int, prefix: str = "", max_depth: int = 64) -> Iterator[Entry]:
        if max_depth < 0:
            return
        for e in self.listdir(ref, prefix):
            yield e
            if e.is_dir:
                yield from self.walk(e.inode_ref, e.path, max_depth - 1)

    # -- fragments ---------------------------------------------------------
    def _fragment_entry(self, index: int) -> Tuple[int, int]:
        if self._frag_entries is None:
            n_index = (self.sb.fragments + 511) // 512
            raw = self._read(self.sb.fragment_table, 8 * n_index)
            starts = struct.unpack(f"<{n_index}Q", raw)
            blob = b""
            for s in starts:
                blob += self._metadata_block(s)[0]
            self._frag_entries = [
                struct.unpack("<QII", blob[i * 16 : i * 16 + 16])[:2]
                for i in range(self.sb.fragments)
            ]
        return self._frag_entries[index]

    # -- file data ---------------------------------------------------------
    def read_file(self, ref: int) -> bytes:
        ino = self.file_inode(ref)
        bs = self.sb.block_size
        out = bytearray()
        pos = ino.blocks_start
        for raw_size in ino.block_sizes:
            size = raw_size & 0xFFFFFF
            if size == 0:  # sparse block
                out += b"\0" * min(bs, ino.file_size - len(out))
                continue
            chunk = self._read(pos, size)
            out += chunk if raw_size & 0x1000000 else zlib.decompress(chunk)
            pos += size
        if ino.frag_index != 0xFFFFFFFF:
            start, fsize = self._fragment_entry(ino.frag_index)
            raw = self._read(start, fsize & 0xFFFFFF)
            block = raw if fsize & 0x1000000 else zlib.decompress(raw)
            remaining = ino.file_size - len(out)
            out += block[ino.block_offset : ino.block_offset + remaining]
        if len(out) != ino.file_size:
            raise ValueError(f"short read: {len(out)} != {ino.file_size}")
        return bytes(out)

    def file_byte_ranges(self, ref: int) -> List[Tuple[int, int]]:
        """(offset, length) of every archive region needed to reconstruct a file."""
        ino = self.file_inode(ref)
        ranges: List[Tuple[int, int]] = []
        pos = ino.blocks_start
        for raw_size in ino.block_sizes:
            size = raw_size & 0xFFFFFF
            if size:
                ranges.append((pos, size))
                pos += size
        if ino.frag_index != 0xFFFFFFFF:
            start, fsize = self._fragment_entry(ino.frag_index)
            ranges.append((start, fsize & 0xFFFFFF))
        return ranges


def _n_blocks(file_size: int, block_size: int, frag_index: int) -> int:
    if frag_index == 0xFFFFFFFF:
        return (file_size + block_size - 1) // block_size
    return file_size // block_size
