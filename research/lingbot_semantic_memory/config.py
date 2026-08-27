"""Configuration objects for the Phase-1 geometry-conditioned semantic probe study.

Mirrors the conventions of :mod:`semantic_sidecar.config`: nested dataclasses that
round-trip through YAML so no script hardcodes a path, plus a provenance block that
is stamped into every artefact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

#: Tokens preceding the patch tokens in an aggregator block output
#: (1 camera + 4 register + 1 scale), measured in Phase 0.
PATCH_START_IDX = 6
#: LingBot aggregator width; a block output is ``frame ‖ global`` = 2 x this.
EMBED_DIM = 1024


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
@dataclass
class RepresentationSpec:
    """One LingBot representation to attach a probe to.

    Attributes:
        name: Identifier used for cache keys, metrics and report rows.
        kind: ``"patch_embed"`` (frame-independent DINOv2 tokens, pre-GCT) or
            ``"block"`` (``frame_blocks[i] ‖ global_blocks[i]``, post-GCT).
        block: Aggregator block group index for ``kind == "block"``.
        dim: Feature width.
    """

    name: str
    kind: str
    block: Optional[int] = None
    dim: int = 1024


def default_representations() -> List[RepresentationSpec]:
    """The representation ladder: pre-GCT encoder, then four GCT depths."""
    return [
        RepresentationSpec("lingbot_encoder", "patch_embed", None, EMBED_DIM),
        RepresentationSpec("lingbot_gct_b04", "block", 4, 2 * EMBED_DIM),
        RepresentationSpec("lingbot_gct_mid", "block", 11, 2 * EMBED_DIM),
        RepresentationSpec("lingbot_gct_b17", "block", 17, 2 * EMBED_DIM),
        RepresentationSpec("lingbot_gct_final", "block", 23, 2 * EMBED_DIM),
    ]


# --------------------------------------------------------------------------- #
# Config blocks
# --------------------------------------------------------------------------- #
@dataclass
class ChunkSpec:
    """A run of frames fed to one streaming inference call.

    ``stride`` samples every n-th raw frame; the chunk still spans a contiguous piece of
    the sequence, so cross-frame attention sees a coherent trajectory. Frame gaps in the
    evaluation are expressed in *sampled* units.
    """

    sequence: str
    start: int
    length: int
    role: str  # "train" | "val"
    stride: int = 1

    @property
    def frame_indices(self) -> range:
        return range(self.start, self.start + self.length * self.stride, self.stride)

    @property
    def name(self) -> str:
        tag = f"_s{self.stride}" if self.stride != 1 else ""
        return f"seq{self.sequence}_{self.start:06d}_{self.length}{tag}"


@dataclass
class DataConfig:
    """Dataset paths and deterministic split construction.

    ``dataset`` selects the backend in :mod:`dataset_adapter` (``"kitti"`` or
    ``"replica"``); everything else has the same meaning for both.
    """

    dataset: str = "kitti"
    root: str = "data/kitti/dataset"
    #: Sequences 04-10 share a 1226x370 resolution, hence one identical patch grid.
    train_sequences: List[str] = field(default_factory=lambda: ["04", "05", "06", "07", "09", "10"])
    val_sequences: List[str] = field(default_factory=lambda: ["08"])
    chunk_length: int = 64
    #: Sample every n-th raw frame within a chunk. KITTI at 10 Hz needs no stride;
    #: Replica's 9 mm / 0.6 deg per-frame motion does.
    stride: int = 1
    train_chunks_per_sequence: int = 1
    val_chunks: int = 3
    image_size: int = 518
    patch_size: int = 14
    image_dirname: str = "image_2"


@dataclass
class LingBotConfig:
    """How to build and run the frozen geometry model (Phase-0 verified values)."""

    model_path: str = "checkpoints/lingbot-map/204754b/lingbot-map.pt"
    image_size: int = 518
    patch_size: int = 14
    num_scale_frames: int = 8
    keyframe_interval: int = 1
    enable_3d_rope: bool = True
    max_frame_num: int = 1024
    kv_cache_sliding_window: int = 64
    use_sdpa: bool = False
    camera_num_iterations: int = 4
    autocast_dtype: str = "bfloat16"


@dataclass
class TeacherConfig:
    """The frozen external DINO teacher (distillation target, never trained)."""

    hub_dir: str = "/home/minh/.cache/torch/hub"
    repo_subdir: str = "facebookresearch_dinov2_main"
    entrypoint: str = "dinov2_vitb14_reg"
    weights: str = "checkpoints/dinov2_vitb14_reg4_pretrain.pth"
    dim: int = 768
    #: ImageNet statistics DINOv2 was trained with.
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ProbeConfig:
    """Probe architecture and the shared optimisation schedule."""

    #: Total trainable parameters every probe is matched to (hidden width is solved
    #: per input dimension so capacity is equal, not merely similar).
    target_params: int = 2_000_000
    max_params: int = 10_000_000
    steps: int = 4000
    batch_frames: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"


@dataclass
class EvalConfig:
    """Cross-view correspondence and metric settings."""

    #: Frame gaps (in sampled-frame units) grouped into short and long bands.
    short_gaps: List[int] = field(default_factory=lambda: [1, 2, 3])
    long_gaps: List[int] = field(default_factory=lambda: [8, 16, 24])
    #: Relative depth agreement for the occlusion test.
    occlusion_rel_tol: float = 0.05
    #: Forward-backward reprojection tolerance, in patch units.
    fb_tol_patches: float = 1.0
    #: Depth-confidence percentile below which patches are dropped when
    #: confidence filtering is enabled.
    conf_percentile: float = 25.0
    #: Random non-corresponding pairs sampled per frame pair for the
    #: dispersion control (guards against a collapsed probe scoring 1.0).
    num_negatives: int = 4096
    max_pairs_per_chunk: int = 96


@dataclass
class Phase1Config:
    """Top-level Phase-1 experiment configuration."""

    seed: int = 0
    device: str = "cuda:0"
    output_dir: str = "research/lingbot_semantic_memory/outputs/phase1"
    cache_dir: str = "research/lingbot_semantic_memory/outputs/cache"
    teacher_budgets: List[float] = field(default_factory=lambda: [0.10, 0.25, 1.0])
    data: DataConfig = field(default_factory=DataConfig)
    lingbot: LingBotConfig = field(default_factory=LingBotConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    representations: List[RepresentationSpec] = field(default_factory=default_representations)

    # -- YAML round-trip ---------------------------------------------------- #
    @classmethod
    def load(cls, path: str) -> "Phase1Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Phase1Config":
        kw: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in raw:
                continue
            v = raw[f.name]
            if f.name == "representations":
                kw[f.name] = [RepresentationSpec(**r) for r in v]
            elif f.name in ("data", "lingbot", "teacher", "probe", "eval"):
                sub = {"data": DataConfig, "lingbot": LingBotConfig, "teacher": TeacherConfig,
                       "probe": ProbeConfig, "eval": EvalConfig}[f.name]
                kw[f.name] = sub(**v)
            else:
                kw[f.name] = v
        return cls(**kw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


# --------------------------------------------------------------------------- #
# Determinism + provenance
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG the experiment touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def collect_provenance(cfg: Phase1Config, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record git state, checkpoint hashes, environment and seeds."""
    ckpt = os.path.join(REPO_ROOT, cfg.lingbot.model_path)
    prov: Dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": cfg.seed,
        "lingbot_checkpoint": cfg.lingbot.model_path,
        "lingbot_checkpoint_sha256": file_sha256(ckpt) if os.path.exists(ckpt) else None,
        "teacher_entrypoint": cfg.teacher.entrypoint,
        "teacher_weights": cfg.teacher.weights,
    }
    if extra:
        prov.update(extra)
    return prov


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
