"""Scene discovery, chunking and the training sampler.

A "scene" here is always a *contiguous* run of frames that the streaming model can
process in one KV-cache session.  Long sequences are split into fixed-size chunks;
each chunk gets its own world frame, its own tracks and its own map.  That keeps
every sequence inside the RoPE training range and makes the geometry of a chunk
independent of how many chunks precede it.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from semantic_sidecar.config import SceneSpec
from semantic_sidecar.lingbot_features import FeatureCacheReader, list_scene_images
from semantic_sidecar.tracks import TrackSet

logger = logging.getLogger(__name__)


def expand_scene(spec: SceneSpec) -> List[SceneSpec]:
    """Split a scene into contiguous chunks when ``chunk_size`` is set."""
    if not spec.chunk_size:
        return [spec]
    paths = list_scene_images(spec)
    chunks: List[SceneSpec] = []
    for c, start in enumerate(range(0, len(paths), spec.chunk_size)):
        n = len(paths[start : start + spec.chunk_size])
        if n < 8:  # too short for the scale-frame phase to mean anything
            continue
        chunks.append(
            replace(
                spec,
                name=f"{spec.name}_c{c:03d}",
                start=spec.start + start * spec.stride,
                stride=spec.stride,
                max_frames=n,
                chunk_size=None,
            )
        )
    return chunks


def expand_scenes(specs: Sequence[SceneSpec]) -> List[SceneSpec]:
    out: List[SceneSpec] = []
    for spec in specs:
        out.extend(expand_scene(spec))
    return out


def discover_tartanair(root: str, cameras: Sequence[str] = ("image_lcam_front",)) -> List[Tuple[str, str]]:
    """Every TartanAir trajectory folder under ``root`` as ``(name, path)``."""
    found: List[Tuple[str, str]] = []
    for cam in cameras:
        for path in sorted(glob.glob(os.path.join(root, "*", "Data_*", "P*", cam))):
            parts = path.split(os.sep)
            found.append((f"tartanair_{parts[-4]}_{parts[-3]}_{parts[-2]}", path))
    return found


# --------------------------------------------------------------------------- #
# Training sample assembly
# --------------------------------------------------------------------------- #
class SceneBundle:
    """Everything one cached scene contributes to training.

    Holds the frozen feature reader, the tracks, the per-observation teacher features
    (already PCA-projected), the per-track consensus target, and the frame-level
    teacher availability mask.  Also builds the ``(frame, token) -> observation``
    lookup that lets a frame batch be turned into per-token targets in O(1).
    """

    def __init__(
        self,
        name: str,
        reader: FeatureCacheReader,
        tracks: TrackSet,
        obs_teacher: torch.Tensor,
        obs_has_teacher: torch.Tensor,
        consensus: torch.Tensor,
        consensus_valid: torch.Tensor,
        frame_teacher_mask: np.ndarray,
    ) -> None:
        self.name = name
        self.reader = reader
        self.tracks = tracks
        self.obs_teacher = obs_teacher
        self.obs_has_teacher = obs_has_teacher
        self.consensus = consensus
        self.consensus_valid = consensus_valid
        self.frame_teacher_mask = frame_teacher_mask

        h, w = reader.patch_hw
        self.num_tokens = h * w
        self.num_frames = reader.num_frames

        from semantic_sidecar.consensus import segment_ids

        track_of_obs = segment_ids(tracks.obs_ptr)
        flat = tracks.obs_frame.long() * self.num_tokens + tracks.obs_token.long()
        self._obs_lookup = torch.full((self.num_frames * self.num_tokens,), -1, dtype=torch.int64)
        self._obs_lookup[flat] = torch.arange(flat.numel(), dtype=torch.int64)
        self._track_of_obs = track_of_obs

    def frame_targets(self, frame: int) -> Dict[str, torch.Tensor]:
        """Per-token targets for one frame.

        Returns tensors of length ``num_tokens``:
            ``obs_index`` (−1 where the token belongs to no track), ``track_id``,
            ``consensus``, ``consensus_mask``, ``pixel``, ``pixel_mask``.
        """
        lo = frame * self.num_tokens
        obs_index = self._obs_lookup[lo : lo + self.num_tokens]
        has_obs = obs_index >= 0
        safe = obs_index.clamp_min(0)

        track_id = torch.where(has_obs, self._track_of_obs[safe], torch.full_like(safe, -1))
        cons = torch.zeros(self.num_tokens, self.consensus.shape[1])
        cons_mask = torch.zeros(self.num_tokens, dtype=torch.bool)
        if bool(has_obs.any()):
            tid = track_id.clamp_min(0)
            cons[has_obs] = self.consensus[tid[has_obs]]
            cons_mask[has_obs] = self.consensus_valid[tid[has_obs]]

        pixel = torch.zeros(self.num_tokens, self.obs_teacher.shape[1])
        pixel_mask = torch.zeros(self.num_tokens, dtype=torch.bool)
        if bool(has_obs.any()):
            pixel[has_obs] = self.obs_teacher[safe[has_obs]]
            pixel_mask[has_obs] = self.obs_has_teacher[safe[has_obs]]

        return {
            "obs_index": obs_index,
            "track_id": track_id,
            "consensus": cons,
            "consensus_mask": cons_mask & has_obs,
            "pixel": pixel,
            "pixel_mask": pixel_mask & has_obs,
        }


class FrameBatchSampler:
    """Draws groups of temporally-nearby frames from a single scene.

    Cross-view terms only have signal when the frames in a batch actually see the
    same surfaces, so a batch is a window of frames from one scene rather than a
    uniform random draw over the whole corpus.
    """

    def __init__(
        self,
        bundles: Sequence[SceneBundle],
        frames_per_batch: int = 4,
        window: int = 24,
        seed: int = 0,
        resample_every: int = 8,
    ) -> None:
        """
        Args:
            resample_every: How many consecutive batches reuse the same scene and
                window.  Feature shards are hundreds of megabytes, so jumping scenes
                every step would thrash the shard cache; staying put for a few steps
                makes almost every batch a cache hit.
        """
        if not bundles:
            raise ValueError("no scene bundles to sample from")
        self.bundles = list(bundles)
        self.frames_per_batch = frames_per_batch
        self.window = window
        self.resample_every = max(1, resample_every)
        self.rng = np.random.default_rng(seed)
        weights = np.array([b.num_frames for b in self.bundles], dtype=np.float64)
        self.weights = weights / weights.sum()
        self._calls = 0
        self._current: Optional[Tuple[SceneBundle, int, int]] = None

    def _pick_window(self) -> Tuple[SceneBundle, int, int]:
        bundle = self.bundles[int(self.rng.choice(len(self.bundles), p=self.weights))]
        n = bundle.num_frames
        span = min(self.window, n)
        start = int(self.rng.integers(0, n - span + 1))
        return bundle, start, span

    def sample(self) -> Tuple[SceneBundle, List[int]]:
        if self._current is None or self._calls % self.resample_every == 0:
            self._current = self._pick_window()
        self._calls += 1
        bundle, start, span = self._current
        k = min(self.frames_per_batch, span)
        frames = sorted(self.rng.choice(np.arange(start, start + span), size=k, replace=False).tolist())
        return bundle, frames

    def __iter__(self) -> Iterator[Tuple[SceneBundle, List[int]]]:
        while True:
            yield self.sample()


def collate_frame_batch(
    bundle: SceneBundle,
    frames: Sequence[int],
    tokens_per_frame: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Gather frozen tokens and targets for a frame group into flat tensors.

    Tokens are sub-sampled per frame to bound memory; the sample is biased towards
    tokens that carry a target, because tokens outside every track contribute nothing
    to any loss term.
    """
    feats, cons, cons_m, pix, pix_m, tid = [], [], [], [], [], []
    for f in frames:
        tgt = bundle.frame_targets(f)
        usable = torch.nonzero(tgt["consensus_mask"] | tgt["pixel_mask"], as_tuple=False).squeeze(-1)
        if usable.numel() == 0:
            continue
        if usable.numel() > tokens_per_frame:
            sel = torch.from_numpy(
                rng.choice(usable.numel(), size=tokens_per_frame, replace=False)
            )
            usable = usable[sel]
        tok = bundle.reader.tokens(f)[:, usable, :]  # [L, n, C]
        feats.append(tok.permute(1, 0, 2).contiguous())  # [n, L, C]
        cons.append(tgt["consensus"][usable])
        cons_m.append(tgt["consensus_mask"][usable])
        pix.append(tgt["pixel"][usable])
        pix_m.append(tgt["pixel_mask"][usable])
        # Track ids are scene-local, which is safe because a batch never spans scenes.
        tid.append(tgt["track_id"][usable])

    if not feats:
        raise RuntimeError(f"scene {bundle.name}: no usable tokens in frames {list(frames)}")

    return {
        "tokens": torch.cat(feats).to(device=device, dtype=torch.float32),
        "consensus": torch.cat(cons).to(device),
        "consensus_mask": torch.cat(cons_m).to(device),
        "pixel": torch.cat(pix).to(device),
        "pixel_mask": torch.cat(pix_m).to(device),
        "track_id": torch.cat(tid).to(device),
    }
