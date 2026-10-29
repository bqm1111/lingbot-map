"""Instrumentation for a persistent-state pass: where it reset, and how far its clock ran.

Needed because the obvious readings are both wrong on the frozen code:

* ``gates/gate7b/replay.replay_segment`` records ``rope_slots_used`` by calling
  ``frame_index_trace`` **after** ``inference_streaming`` returns -- but
  ``inference_streaming`` ends with ``clean_kv_cache()`` whenever ``output_device`` is set,
  and that call resets ``aggregator.total_frames_processed`` to 0. Every
  ``rope_slots_used`` the gate ever stored is therefore 0, not the slot count. This module
  records the counter's **peak**, from a forward hook, which is the real figure.
* ``clean_kv_cache`` is invoked three times per pass (``replay_segment``'s own call, then
  ``inference_streaming`` at entry and again at exit), so counting calls says nothing about
  mid-stream resets. What matters is the *frame counter at each call*: a reset with the
  counter strictly between 0 and the final slot count would be a mid-stream reset. This
  module records that value for every call.
"""
from __future__ import annotations

from typing import Dict, List


class StateTrace:
    """``with StateTrace(model) as tr:`` -- then read ``tr.peak_frames`` / ``tr.resets_at``."""

    def __init__(self, model):
        self.model = model
        self.peak_frames = 0
        self.resets_at: List[int] = []
        self.forward_calls = 0
        self._orig_clean = None
        self._handle = None

    def _counter(self) -> int:
        return int(getattr(self.model.aggregator, "total_frames_processed", -1))

    def __enter__(self) -> "StateTrace":
        self._orig_clean = self.model.clean_kv_cache

        def wrapped(*a, **k):
            self.resets_at.append(self._counter())
            return self._orig_clean(*a, **k)
        self.model.clean_kv_cache = wrapped

        def hook(_m, _i, _o):
            self.forward_calls += 1
            self.peak_frames = max(self.peak_frames, self._counter())
        self._handle = self.model.aggregator.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.model.clean_kv_cache = self._orig_clean
        if self._handle is not None:
            self._handle.remove()
        return False

    def mid_stream_resets(self) -> List[int]:
        """Resets that happened with frames already processed and more still to come."""
        return [c for c in self.resets_at if 0 < c < self.peak_frames]

    def summary(self) -> Dict:
        return {"peak_frames_processed": self.peak_frames,
                "aggregator_forward_calls": self.forward_calls,
                "clean_kv_cache_calls": len(self.resets_at),
                "frame_counter_at_each_reset": list(self.resets_at),
                "mid_stream_resets": self.mid_stream_resets(),
                "note": ("resets at counter 0 precede any frame; a reset at the peak follows "
                         "the last frame. Only a reset strictly between the two would break "
                         "the persistent-state requirement.")}


__all__ = ["StateTrace"]
