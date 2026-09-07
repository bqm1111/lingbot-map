"""The streaming metric gauge, kept in one place and never buried in fused voxels.

A voxel map that has already been built in metres cannot be re-gauged: the surfaces are
baked in. So the scale lives here, as an explicit state, and every consumer asks it for
``s(t)``. Observations are stored in LingBot's **canonical** frame; the metric crop is
materialised at evaluation time with the scale in force at that timestamp, which is what
makes a changing gauge transform old and new observations *together*.

Three policies, all causal, all using only RGB, the calibrated FOV and frozen MoGe:

* **G-A, fixed anchor scale (primary)** -- one scalar from the stream's initial anchor
  frames, frozen for the rest of the stream;
* **G-B, causal running median** -- the expanding median of every per-frame candidate up
  to and including ``t``;
* **G-C, per-frame scale (ablation only)** -- each frame gauged by itself. It is expected
  to duplicate surfaces and can never be selected as primary.

The per-frame candidate is the Gate-5.1 G51-B estimator applied to one frame: the median
of ``log D_moge - log D_lingbot`` over pixels valid for both, after a predeclared
3 x MAD clip. It never sees a target, a LiDAR sweep or an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .depth import CONF_THRESHOLD, MAX_DEPTH_M, MIN_DEPTH_M

#: Robust rejection of gross outliers inside one frame, before its median is taken.
MAD_CLIP = 3.0
#: A frame contributes a candidate only with at least this many jointly valid pixels.
MIN_VALID_PIXELS = 1000
#: Anchor frames used by G-A: the stream's own LingBot scale frames.
N_ANCHOR_FRAMES = 5

POLICIES = ("G-A", "G-B", "G-C")


def frame_candidate(moge_depth: np.ndarray, moge_mask: np.ndarray,
                    lingbot_depth: np.ndarray, lingbot_conf: np.ndarray,
                    conf_threshold: float = CONF_THRESHOLD,
                    min_depth_m: float = MIN_DEPTH_M,
                    max_depth_m: float = MAX_DEPTH_M,
                    mad_clip: float = MAD_CLIP) -> Dict[str, float]:
    """One frame's log-scale candidate, or ``nan`` when the overlap is too small.

    Validity is the Gate-5 intersection: MoGe's own mask, finite positive MoGe z-depth
    inside the frozen metric range, finite positive canonical LingBot depth, and the
    frozen LingBot confidence criterion. The range test is applied to the **MoGe** depth
    because it is the one already in metres.
    """
    md = np.asarray(moge_depth, np.float64)
    ld = np.asarray(lingbot_depth, np.float64)
    lc = np.asarray(lingbot_conf, np.float64)
    v = (np.asarray(moge_mask, bool) & np.isfinite(md) & (md > min_depth_m)
         & (md < max_depth_m) & np.isfinite(ld) & (ld > 0) & (lc >= conf_threshold))
    n = int(v.sum())
    if n < MIN_VALID_PIXELS:
        return {"n_valid": n, "log_s": float("nan"), "mad": float("nan")}
    r = np.log(md[v]) - np.log(ld[v])
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    if mad > 0:
        keep = np.abs(r - med) <= mad_clip * mad
        if keep.sum() >= MIN_VALID_PIXELS:
            r = r[keep]
    return {"n_valid": n, "log_s": float(np.median(r)), "mad": mad}


@dataclass
class ScaleState:
    """The gauge in force at each timestamp of one stream, under one policy."""
    policy: str
    n_anchor_frames: int = N_ANCHOR_FRAMES
    log_candidates: List[float] = field(default_factory=list)
    _anchor_logs: List[float] = field(default_factory=list)
    _frozen: Optional[float] = None
    _series: List[float] = field(default_factory=list)

    def observe(self, time_index: int, cand: Dict[str, float]) -> float:
        """Feed frame ``time_index``'s candidate and return ``s`` in force **at** ``t``.

        Strictly causal: only candidates from frames ``<= t`` are ever used, which the
        tests assert by replaying a truncated stream and comparing the series.
        """
        ls = float(cand["log_s"])
        self.log_candidates.append(ls)
        if self.policy == "G-A":
            if time_index < self.n_anchor_frames and np.isfinite(ls):
                self._anchor_logs.append(ls)
            if self._frozen is None and time_index >= self.n_anchor_frames - 1:
                if self._anchor_logs:
                    self._frozen = float(np.exp(np.median(self._anchor_logs)))
            s = self._frozen if self._frozen is not None else self._fallback()
        elif self.policy == "G-B":
            fin = [x for x in self.log_candidates if np.isfinite(x)]
            s = float(np.exp(np.median(fin))) if fin else float("nan")
        elif self.policy == "G-C":
            s = float(np.exp(ls)) if np.isfinite(ls) else self._fallback()
        else:
            raise KeyError(self.policy)
        self._series.append(s)
        return s

    def _fallback(self) -> float:
        """Before any valid candidate exists, the running median of what there is."""
        fin = [x for x in self.log_candidates if np.isfinite(x)]
        return float(np.exp(np.median(fin))) if fin else float("nan")

    @property
    def series(self) -> np.ndarray:
        return np.asarray(self._series, np.float64)

    def diagnostics(self) -> Dict[str, float]:
        s = self.series
        fin = s[np.isfinite(s)]
        if fin.size == 0:
            return {"n": 0}
        ls = np.log(fin)
        adj = np.abs(np.diff(ls))
        return {"n": int(fin.size),
                "median_scale": float(np.median(fin)),
                "log_mad": float(np.median(np.abs(ls - np.median(ls)))),
                "adjacent_log_mad": float(np.median(adj)) if adj.size else 0.0,
                "adjacent_log_p95": float(np.percentile(adj, 95)) if adj.size else 0.0,
                "first_to_final_log_drift": float(ls[-1] - ls[0]),
                "first_to_final_ratio": float(fin[-1] / fin[0]),
                "min_scale": float(fin.min()), "max_scale": float(fin.max())}


__all__ = ["ScaleState", "frame_candidate", "POLICIES", "MAD_CLIP", "MIN_VALID_PIXELS",
           "N_ANCHOR_FRAMES"]
