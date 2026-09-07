"""Feed the mapper one frame at a time from the cached frozen-model outputs.

The cached stream is exactly what the per-frame direct-mode calls produce (Gate 7B ran
``inference_streaming`` once per segment; its per-frame outputs are stored in order), so
feeding the mapper from the cache is causally identical to feeding it live: frame ``t``'s
``FrameInput`` is built from row ``t`` only. The live feed (``LiveFeed``) exists for the
runtime audit and for the test that the two agree.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import torch

from gate7b import depth as D7
from gate8 import sources as S
from gate8.mapper import FrameInput


class CachedFeed:
    def __init__(self, seg: S.Seg, device, with_semantics: bool = True,
                 with_moge: bool = False):
        self.seg, self.device = seg, device
        with np.load(S.stream_path(seg.source, seg.name)) as z:
            self.dep = z["pred_depth"]; self.conf = z["pred_depth_conf"]
            self.pose = z["pred_pose_c2w"].astype(np.float64)
            self.K = z["pred_K"].astype(np.float64)
            keys = [str(k) for k in z["keys"]]
        assert keys == [f.key for f in seg.frames], "stream cache does not match the segment"
        with np.load(S.scale_path(seg.source, seg.name)) as z:
            self.log_s = z["log_s"].astype(np.float64)
        self.with_semantics, self.with_moge = with_semantics, with_moge
        self.n_missing_sem = 0

    def __len__(self) -> int:
        return len(self.seg)

    def frame(self, i: int) -> FrameInput:
        f = self.seg.frames[i]
        sem = None
        if self.with_semantics:
            p = S.trident_path(self.seg.source, f.key)
            if os.path.exists(p):
                with np.load(p) as z:
                    sem = torch.from_numpy(z["probs"].astype(np.float32)).permute(1, 2, 0)
            else:
                self.n_missing_sem += 1
        mg = mm = None
        if self.with_moge:
            mp = S.moge_path(self.seg.source, f.key)
            if os.path.exists(mp):
                with np.load(mp) as z:
                    mg = torch.from_numpy(z["moge_depth"].astype(np.float32))
                    mm = torch.from_numpy(D7.unpack_mask(z["moge_mask"], z["mask_shape"]))
        return FrameInput(index=i,
                          depth_canonical=torch.from_numpy(self.dep[i].astype(np.float32)).to(self.device),
                          conf=torch.from_numpy(self.conf[i].astype(np.float32)).to(self.device),
                          K=self.K[i], pose_c2w_canonical=self.pose[i],
                          log_scale_candidate=float(self.log_s[i]),
                          sem_probs=sem, moge_depth=mg, moge_mask=mm)

    def __iter__(self) -> Iterator[FrameInput]:
        for i in range(len(self)):
            yield self.frame(i)


__all__ = ["CachedFeed"]
