"""Disk-backed training anchors with a bounded host cache.

The original trainers load every anchor onto the GPU before step 1 (``trainer.py``: ``anchors = [load_training_anchor(...)]``;
pilot ``load_training_anchors``). Here ``anchors[i]`` loads anchor i when a step asks for it, using the SAME per-anchor loader
on the CPU, then moves every tensor to the training device. Values, dtypes (fp16 teacher features, int64 flat indices, packed
uint8 masks) and dict keys are therefore those of the original loader; only when the bytes are read changes.

Memory bounds:
    host  <= LoaderConfig.host_cache_bytes of decoded tensors (LRU; 0 disables the cache)
    GPU   only the anchors of the current step (the returned dict is a fresh device copy; freed when the step drops it)
Loading consumes no random numbers, so sampling is unaffected by cache hits or misses.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

AnchorKey = Tuple[str, int]                     # (drive, stream index t)
AnchorDict = Dict[str, Any]                     # the original loader's dict: tensors + "drive" / "t"
HostLoader = Callable[[AnchorKey], AnchorDict]  # loads one anchor onto the CPU
FileLister = Callable[[AnchorKey], List[str]]   # the files a loader reads (for error messages)


class AnchorLoadError(RuntimeError):
    """An anchor could not be loaded. Training stops; no anchor is ever skipped or replaced."""


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_now: int = 0
    bytes_peak: int = 0


def tensor_bytes(anchor: AnchorDict) -> int:
    return sum(v.numel() * v.element_size() for v in anchor.values() if isinstance(v, torch.Tensor))


class DiskAnchorSequence(Sequence[AnchorDict]):
    """``Sequence`` of anchors in a fixed, frozen order. ``len`` and indexing match the original in-memory list."""

    def __init__(self, keys: Sequence[AnchorKey], load_to_host: HostLoader, files_of: FileLister, device: torch.device | str,
                 host_cache_bytes: int):
        if host_cache_bytes < 0:
            raise ValueError("host_cache_bytes must be >= 0")
        self.keys: List[AnchorKey] = list(keys)
        self._load_to_host, self._files_of = load_to_host, files_of
        self.device = torch.device(device)
        self.host_cache_bytes = host_cache_bytes
        self._cache: "OrderedDict[AnchorKey, AnchorDict]" = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> AnchorDict:           # type: ignore[override]
        key = self.keys[index]
        host = self._host_anchor(key)
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in host.items()}

    def _host_anchor(self, key: AnchorKey) -> AnchorDict:
        if key in self._cache:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.stats.misses += 1
        try:
            anchor = self._load_to_host(key)
        except Exception as exc:                                   # re-raised with the anchor and its files, never swallowed
            raise AnchorLoadError(f"loading anchor drive={key[0]} t={key[1]} failed while reading {self._files_of(key)}: "
                                  f"{type(exc).__name__}: {exc}") from exc
        if any(isinstance(v, torch.Tensor) and v.device.type != "cpu" for v in anchor.values()):
            raise AnchorLoadError(f"anchor drive={key[0]} t={key[1]}: the host loader must return CPU tensors")
        self._remember(key, anchor)
        return anchor

    def _remember(self, key: AnchorKey, anchor: AnchorDict) -> None:
        size = tensor_bytes(anchor)
        if size > self.host_cache_bytes:
            return
        while self.stats.bytes_now + size > self.host_cache_bytes:
            _, old = self._cache.popitem(last=False)
            self.stats.bytes_now -= tensor_bytes(old); self.stats.evictions += 1
        self._cache[key] = anchor
        self.stats.bytes_now += size; self.stats.bytes_peak = max(self.stats.bytes_peak, self.stats.bytes_now)


__all__ = ["AnchorKey", "AnchorDict", "AnchorLoadError", "CacheStats", "tensor_bytes", "DiskAnchorSequence"]
