"""Gate 15 -- sparse causal Trident-H evidence propagated into completed 3D space. Single source of truth.

Frozen upstream (never modified here): the Current Geometry-Only Baseline of
``tools/baseline_reconstruction`` (local-reset L + LB-M + hard mapper H -> CompletionUNet [S, 0, F],
seeds 0/1/2, step 6000, occupied := final log-odds > 0) and frozen Trident-H.

Grid convention used by every tensor in this package (identical to the baseline):
    SSCBench-KITTI-360 anchor velodyne grid, dims (X, Y, Z) = (256, 256, 32), 0.2 m voxels,
    origin (0, -25.6, -2) m, flat index = ravel_multi_index((x, y, z)) (C order, "ij").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "gate15_semantic_completion")
DATA_ROOT = os.environ.get("GATE15_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate15_semantic")
PROTOCOL_VERSION = "gate15_semantic_completion/2026-09-14"

TRAIN_DRIVES: Tuple[str, ...] = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync", "2013_05_28_drive_0010_sync")
VAL_DRIVE = "2013_05_28_drive_0006_sync"
SEEDS: Tuple[int, ...] = (0, 1, 2)
GRID_DIMS: Tuple[int, int, int] = (256, 256, 32)
N_VOXELS = GRID_DIMS[0] * GRID_DIMS[1] * GRID_DIMS[2]
VOXEL_SIZE_M = 0.2
GRID_ORIGIN_M: Tuple[float, float, float] = (0.0, -25.6, -2.0)

# ---- frozen teacher -------------------------------------------------------------------
TEACHER_DIM = 1024                              # Trident-H CLIP ViT-H-14 token feature, unit norm
TOKEN_GRID_HW: Tuple[int, int] = (34, 128)      # spans the full 376x1408 native frame
N_CLASSES = 18                                  # KITTI-360 / SSCBench labels 1..18; 0 = empty, 255 = ignore


@dataclass(frozen=True)
class Paths:
    token_root: str = "/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d/trident_tokens"
    text_embeddings: str = "/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d/text_embeddings.npz"
    #: local-reset clips and hard-map inputs of the frozen geometry baseline
    clips_train_root: str = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/local_lingbot"
    clips_val_root: str = "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot"
    inputs_root: str = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs"
    geometry_checkpoints: str = os.path.join(REPO_ROOT, "artifacts", "baseline_reconstruction", "checkpoints")
    geometry_freeze: str = os.path.join(REPO_ROOT, "artifacts", "baseline_reconstruction", "code_freeze.json")
    geometry_source_eval: str = os.path.join(REPO_ROOT, "artifacts", "baseline_reconstruction", "source_eval")
    #: evaluation-only human labels (SSCBench voxels resampled onto the grid) -- never opened by training
    eval_masks_root: str = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/eval_masks"
    evidence_root: str = os.path.join(DATA_ROOT, "evidence")
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    runs: str = os.path.join(ARTIFACT_DIR, "runs")
    source_eval: str = os.path.join(ARTIFACT_DIR, "source_eval")

    def clip_path(self, drive: str, t: int) -> str:
        root = self.clips_val_root if drive == VAL_DRIVE else self.clips_train_root
        return os.path.join(root, drive, f"{t:05d}.npz")

    def input_path(self, drive: str, t: int) -> str:
        return os.path.join(self.inputs_root, drive, f"{t:05d}.npz")

    def token_path(self, drive: str, key: str) -> str:
        source = "kitti360" if drive == VAL_DRIVE else "k360_train"
        return os.path.join(self.token_root, source, f"{key}.npz")

    def evidence_path(self, drive: str, t: int) -> str:
        return os.path.join(self.evidence_root, drive, f"{t:05d}.npz")

    def eval_mask_path(self, t: int) -> str:
        return os.path.join(self.eval_masks_root, VAL_DRIVE, f"{t:05d}.npz")

    def geometry_checkpoint(self, seed: int) -> str:
        return os.path.join(self.geometry_checkpoints, f"geo_seed{seed}_step06000.pt")

    def head_checkpoint(self, seed: int, step: int) -> str:
        return os.path.join(self.checkpoints, f"sem_seed{seed}_step{step:05d}.pt")


# ---- the learned propagation head (B1) ------------------------------------------------
@dataclass(frozen=True)
class HeadConfig:
    evidence_dim: int = 32                      # learned 1024 -> 32 projection of the sparse evidence
    width: int = 32
    dilations: Tuple[int, ...] = (1, 2, 4, 8)   # residual blocks; receptive-field radius 2*(1+2+4+8) = 30 voxels = 6 m
    trunk_channels: int = 24                    # frozen CompletionUNet decoder feature d1


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 6000
    batch_size: int = 4
    crop: Tuple[int, int, int] = (128, 128, 32)
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    #: evidence hiding. Loss is applied on every evidence voxel of the crop; the evidence of voxels inside
    #: randomly hidden blocks is removed from the INPUT, so those losses can only be met by propagation.
    hide_block_vox: Tuple[int, int, int] = (8, 8, 4)     # 1.6 x 1.6 x 0.8 m blocks
    hide_block_prob: float = 0.5
    log_every: int = 50
    ckpt_every: int = 1000


#: forbidden while building evidence and training (human labels, LiDAR, GT poses, target datasets)
TRAIN_FORBIDDEN_TOKENS: Tuple[str, ...] = ("data/kitti/dataset", "occ3d_gt", "nuscenes", "lingbot_occ3d_zeroshot",
                                           "scale_gate/cache", "preprocess/labels", "eval_masks", "samples_gtsem",
                                           "data_3d_semantics", "velodyne", "data_3d_raw", "data_poses", "targets")
#: forbidden during source evaluation (target datasets only)
EVAL_FORBIDDEN_TOKENS: Tuple[str, ...] = ("data/kitti/dataset", "occ3d_gt", "nuscenes", "lingbot_occ3d_zeroshot", "scale_gate/cache")

#: propagation-distance bins (metres, distance from a voxel centre to the nearest evidence voxel centre)
DISTANCE_EDGES_M: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
DISTANCE_NAMES: Tuple[str, ...] = ("0-1 m", "1-2 m", "2-4 m", "4-8 m", ">8 m")


def ensure_dirs() -> None:
    p = Paths()
    for d in (ARTIFACT_DIR, os.path.join(ARTIFACT_DIR, "tables"), os.path.join(ARTIFACT_DIR, "figures"), os.path.join(ARTIFACT_DIR, "logs"),
              p.checkpoints, p.runs, p.source_eval, p.evidence_root):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
