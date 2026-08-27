"""Configuration objects and run provenance for the semantic sidecar.

Everything the experiments need is expressed as nested dataclasses that round-trip
through YAML, so no script hardcodes a path.  :func:`collect_provenance` records the
git commit, checkpoint hash, environment and seeds into every artefact written.
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Aggregator block indices returned by ``GCTStream._aggregate_features``.
AGGREGATOR_SELECTED_BLOCKS: Sequence[int] = (4, 11, 17, 23)
#: Number of tokens preceding the patch tokens (1 camera + 4 register + 1 scale).
PATCH_START_IDX = 6
#: Channel width of one aggregator output (frame-attn ‖ global-attn = 2 x embed_dim).
TOKEN_CHANNELS = 2048


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class SceneSpec:
    """One RGB sequence to run the frozen model over.

    Attributes:
        name: Unique scene identifier; used as the cache subdirectory name.
        image_folder: Directory of frames (sorted lexicographically).
        role: ``"train"``, ``"eval"`` or ``"qualitative"``.
        stride: Take every ``stride``-th frame.
        start: Index of the first frame to take (after sorting).
        max_frames: Cap on the number of frames after start/stride.
        image_ext: Glob extension filter.
        dataset: Free-form dataset tag recorded in the manifest.
    """

    name: str
    image_folder: str
    role: str = "train"
    stride: int = 1
    start: int = 0
    max_frames: Optional[int] = None
    image_ext: str = ".png"
    dataset: str = "unknown"
    #: Split the frame list into contiguous chunks of this many frames.  Each chunk
    #: is cached and mapped independently (its own world frame), which keeps every
    #: sequence inside the RoPE training range and inside one streaming KV cache.
    chunk_size: Optional[int] = None


@dataclass
class LingBotConfig:
    """How to build and run the frozen geometry model."""

    model_path: str = "checkpoints/lingbot-map/204754b/lingbot-map.pt"
    image_size: int = 518
    patch_size: int = 14
    mode: str = "streaming"  # "streaming" | "windowed"
    num_scale_frames: int = 8
    keyframe_interval: int = 1
    window_size: int = 128
    overlap_keyframes: int = 8
    max_non_keyframe_gap: int = 100
    enable_3d_rope: bool = True
    max_frame_num: int = 1024
    kv_cache_sliding_window: int = 64
    use_sdpa: bool = False
    camera_num_iterations: int = 4
    #: Aggregator block indices to keep (subset of AGGREGATOR_SELECTED_BLOCKS).
    layers: List[int] = field(default_factory=lambda: [11, 23])
    device: str = "cuda"
    autocast_dtype: str = "bfloat16"


@dataclass
class TeacherConfig:
    """Frozen 2D semantic teacher and its compression basis."""

    backend: str = "maskclip"  # "maskclip" | "lseg"
    model_name: str = "ViT-B-16-quickgelu"
    pretrained: str = "openai"
    variant: str = "maskclip"
    lseg_checkpoint: Optional[str] = None
    #: Upsample factor applied before the teacher ViT (finer dense grid).
    input_scale: float = 2.0
    embed_dim: int = 512
    #: Compressed dimensionality of the sidecar's output space.
    pca_dim: int = 64
    pca_frames: int = 96
    pca_samples_per_frame: int = 2048
    device: str = "cuda"


@dataclass
class TrackConfig:
    """3D observation-track construction."""

    #: Voxel edge measured in patch-token footprints at the anchor median depth.
    #: A voxel smaller than one token footprint cannot merge two views' tokens, so
    #: this is the knob that actually controls how many multi-view tracks exist.
    #: Set to ``None`` to fall back to ``rel_voxel_size``.
    voxel_token_scale: Optional[float] = 1.5
    #: Voxel size as a fraction of the scene's anchor median depth (fallback).
    rel_voxel_size: float = 0.02
    #: Explicit metric voxel size; overrides both of the above when set.
    metric_voxel_size: Optional[float] = None
    anchor_frames: int = 32
    min_depth: float = 1e-3
    max_depth_rel: float = 20.0
    conf_threshold: float = 1.3
    reproj_tol: float = 0.10
    min_observations: int = 3
    max_observations: int = 32
    outlier_mad_scale: float = 3.0
    min_teacher_obs: int = 1
    seed: int = 0


@dataclass
class ConsensusConfig:
    """Robust semantic consensus over a track's observations."""

    estimator: str = "robust"  # "mean" | "geometric" | "robust"
    conf_power: float = 1.0
    residual_tau: float = 0.05
    angle_power: float = 1.0
    agreement_floor: float = 0.0
    #: Tracks with fewer teacher observations than TrackConfig.min_teacher_obs are
    #: marked invalid unless this is "pixelwise".
    fallback: str = "invalid"  # "invalid" | "pixelwise"


@dataclass
class ModelConfig:
    """Sidecar architecture."""

    arch: str = "mlp"  # "linear" | "mlp"
    hidden_dim: int = 768
    num_blocks: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    out_dim: int = 64
    fuse: str = "concat"  # "concat" | "sum"


@dataclass
class LossConfig:
    """Loss weights; every term is logged independently."""

    lambda_pixel: float = 0.0
    lambda_consensus: float = 1.0
    lambda_mv: float = 0.0
    lambda_rel: float = 0.0
    #: Cosine on the mean-removed residual.  Optional; see losses.centered_cosine_loss
    #: for why the plain cosine is a weak signal in CLIP space.
    lambda_center: float = 0.0
    rel_group_size: int = 256
    rel_groups_per_batch: int = 4


@dataclass
class TrainConfig:
    """Single-GPU training loop."""

    steps: int = 2000
    frames_per_batch: int = 4
    tokens_per_frame: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 250
    seed: int = 0
    device: str = "cuda"
    amp: bool = True
    #: Fraction of frames for which the teacher is available (frame-level sampling).
    teacher_density: float = 1.0
    teacher_density_seed: int = 1234
    num_workers: int = 4
    #: Consecutive batches drawn from the same scene/window (shard-cache locality).
    resample_every: int = 8
    #: Frames the batch window spans.
    window: int = 24
    #: Feature shards kept resident in RAM across all open scenes.
    shard_cache_limit: int = 4


@dataclass
class EvalConfig:
    """Evaluation protocol knobs."""

    kitti_root: str = "data/kitti/dataset"
    sequence: str = "08"
    frame_stride: int = 5
    orig_width: int = 1226
    orig_height: int = 370
    depth_tolerance: float = 0.10
    grid_resolution: int = 4096
    min_count: int = 2
    pixel_stride: int = 2
    conf_threshold: float = 1.3
    sky_mask_dir: Optional[str] = None
    #: Optional stanzas for benchmarks not present on this machine.
    extra_benchmarks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathsConfig:
    """Where artefacts live."""

    cache_root: str = "output/semantic_sidecar/cache"
    tracks_root: str = "output/semantic_sidecar/tracks"
    runs_root: str = "output/semantic_sidecar/runs"
    maps_root: str = "output/semantic_sidecar/maps"
    reports_root: str = "output/semantic_sidecar/reports"


@dataclass
class SidecarConfig:
    """Top-level experiment configuration."""

    name: str = "feasibility"
    seed: int = 0
    scenes: List[SceneSpec] = field(default_factory=list)
    paths: PathsConfig = field(default_factory=PathsConfig)
    lingbot: LingBotConfig = field(default_factory=LingBotConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    tracks: TrackConfig = field(default_factory=TrackConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # -- convenience views ------------------------------------------------- #
    def scenes_with_role(self, role: str) -> List[SceneSpec]:
        return [s for s in self.scenes if s.role == role]

    @property
    def train_scenes(self) -> List[SceneSpec]:
        return self.scenes_with_role("train")


_DATACLASS_FIELDS = {
    "paths": PathsConfig,
    "lingbot": LingBotConfig,
    "teacher": TeacherConfig,
    "tracks": TrackConfig,
    "consensus": ConsensusConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "evaluation": EvalConfig,
}


def _build_dataclass(cls, data: Dict[str, Any]):
    """Instantiate ``cls`` from ``data``, rejecting unknown keys loudly."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**data)


def _parse_override_value(text: str) -> Any:
    """Parse a CLI override value as YAML, with a numeric fallback.

    YAML 1.1 does not recognise ``5e-4`` as a float (it needs ``5.0e-4``), which is
    exactly the form people type on a command line, so try ``float``/``int`` before
    giving up and keeping the string.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
    return value


def load_config(path: str, overrides: Optional[Sequence[str]] = None) -> SidecarConfig:
    """Load a YAML config, applying ``key.sub=value`` dotted overrides.

    Args:
        path: YAML file path.
        overrides: Sequence of ``dotted.key=value`` strings; values are parsed as YAML
            so ``lr=1e-3``, ``model.arch=linear`` and ``lingbot.layers=[23]`` all work.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, value = item.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_override_value(value)

    scenes = [_build_dataclass(SceneSpec, s) for s in raw.pop("scenes", [])]
    sub = {name: _build_dataclass(cls, raw.pop(name, {}) or {}) for name, cls in _DATACLASS_FIELDS.items()}
    cfg = SidecarConfig(scenes=scenes, **sub, **raw)
    return cfg


def config_to_dict(cfg: SidecarConfig) -> Dict[str, Any]:
    """Plain-dict view of a config, suitable for JSON/YAML serialisation."""
    return dataclasses.asdict(cfg)


def save_config(cfg: SidecarConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_commit(repo_root: str = REPO_ROOT) -> str:
    """Current git commit (with ``-dirty`` suffix), or ``"unknown"``."""
    try:
        head = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", repo_root, "diff", "--quiet", "--ignore-submodules", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return head + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - provenance must never be fatal
        return "unknown"


def file_sha256(path: str, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 of a (possibly multi-GB) file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect_provenance(cfg: Optional[SidecarConfig] = None, **extra: Any) -> Dict[str, Any]:
    """Environment + code identity recorded into every artefact."""
    info: Dict[str, Any] = {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "argv": sys.argv,
    }
    if cfg is not None:
        info["seed"] = cfg.seed
        info["config_name"] = cfg.name
    info.update(extra)
    return info


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (all devices) and log it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
