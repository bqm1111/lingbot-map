"""Frozen 2D semantic teacher and the fixed compression basis shared with text.

The teacher is only ever used to *build training targets*.  The final sidecar
inference path never instantiates it.

``TeacherBackend`` is the abstraction; ``MaskCLIPTeacher`` is the concrete backend
available in this environment (``semantic.dense_clip.DenseCLIP``, a MaskCLIP-style
rewrite of the last CLIP block that yields per-patch, text-aligned features).
``LSegTeacher`` is a thin adapter kept for the machine where the lang-seg stack is
importable; it raises a clear error otherwise instead of silently changing spaces.

Image features and text embeddings go through the *identical* transform:
``normalize(feature) @ componentsᵀ`` followed by another normalise.  The basis is
orthonormal and uncentered (``semantic.feature_field.fit_projection``), so cosine
similarity is preserved inside the retained subspace.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from semantic.feature_field import fit_projection
from semantic_sidecar.config import TeacherConfig
from semantic_sidecar.lingbot_features import pool_to_patch_grid

logger = logging.getLogger(__name__)

#: Background prompts for LERF-style relevancy; deliberately generic and fixed.
CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture", "surface")


class TeacherBackend(ABC):
    """A frozen, language-aligned dense feature extractor."""

    embed_dim: int
    name: str

    @abstractmethod
    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        """Dense per-patch features for ``[B, 3, H, W]`` images in ``[0, 1]``.

        Returns ``[B, h, w, D]``, L2-normalised.
        """

    @abstractmethod
    def encode_text(self, prompts: Sequence[str], templates: Optional[Sequence[str]] = None) -> torch.Tensor:
        """``[T, D]`` L2-normalised text embeddings in the same space."""

    def encode_patch_grid(self, images: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Dense features area-pooled onto the LingBot token grid → ``[B, h*w, D]``."""
        dense = self.encode_dense(images)  # [B, hd, wd, D]
        dense = dense.permute(0, 3, 1, 2).float()  # [B, D, hd, wd]
        pooled = torch.stack([pool_to_patch_grid(d, h, w) for d in dense])
        return F.normalize(pooled, dim=-1)


class MaskCLIPTeacher(TeacherBackend):
    """MaskCLIP dense CLIP teacher (``semantic.dense_clip.DenseCLIP``), frozen."""

    def __init__(self, cfg: TeacherConfig) -> None:
        from semantic.dense_clip import DenseCLIP

        self.cfg = cfg
        self.clip = DenseCLIP(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            device=cfg.device,
            variant=cfg.variant,
        )
        for p in self.clip.model.parameters():
            p.requires_grad_(False)
        self.clip.model.eval()
        self.embed_dim = int(self.clip.embed_dim)
        self.name = f"maskclip:{cfg.model_name}/{cfg.pretrained}/{cfg.variant}"

    @torch.no_grad()
    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:
        if self.cfg.input_scale != 1.0:
            h, w = images.shape[-2:]
            images = F.interpolate(
                images,
                size=(int(round(h * self.cfg.input_scale)), int(round(w * self.cfg.input_scale))),
                mode="bilinear",
                align_corners=False,
            )
        return self.clip.encode_dense(images.to(self.clip.device)).float()

    @torch.no_grad()
    def encode_text(self, prompts, templates=None) -> torch.Tensor:
        from semantic.dense_clip import DEFAULT_TEMPLATES

        return self.clip.encode_text(list(prompts), templates or DEFAULT_TEMPLATES).float()


class LSegTeacher(TeacherBackend):
    """LSeg adapter.

    LSeg is the preferred teacher (its dense features are trained to be language
    aligned), but it needs the ``lang-seg`` package on ``sys.path`` together with its
    legacy pytorch-lightning stack.  When that import fails we raise rather than
    quietly substituting a different embedding space.
    """

    def __init__(self, cfg: TeacherConfig) -> None:
        if not cfg.lseg_checkpoint or not os.path.exists(cfg.lseg_checkpoint):
            raise FileNotFoundError(
                f"LSeg checkpoint {cfg.lseg_checkpoint!r} not found; set "
                "teacher.lseg_checkpoint or use teacher.backend=maskclip"
            )
        try:  # pragma: no cover - environment dependent
            from modules.lseg_module import LSegModule  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "lang-seg is not importable in this environment; add its repo root to "
                "PYTHONPATH or use teacher.backend=maskclip"
            ) from exc

        module = LSegModule.load_from_checkpoint(  # pragma: no cover
            checkpoint_path=cfg.lseg_checkpoint, backbone="clip_vitl16_384", num_features=256
        )
        self.net = module.net.eval().to(cfg.device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        self.embed_dim = 512
        self.name = f"lseg:{os.path.basename(cfg.lseg_checkpoint)}"

    @torch.no_grad()
    def encode_dense(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        feats = self.net.forward_features(images.to(self.cfg.device))
        return F.normalize(feats.permute(0, 2, 3, 1).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts, templates=None) -> torch.Tensor:  # pragma: no cover
        import clip as openai_clip

        tokens = openai_clip.tokenize(list(prompts)).to(self.cfg.device)
        emb = self.net.clip_pretrained.encode_text(tokens).float()
        return F.normalize(emb, dim=-1)


def build_teacher(cfg: TeacherConfig) -> TeacherBackend:
    """Instantiate the configured teacher, falling back with a loud log if asked."""
    if cfg.backend == "maskclip":
        return MaskCLIPTeacher(cfg)
    if cfg.backend == "lseg":
        return LSegTeacher(cfg)
    if cfg.backend == "lseg_or_maskclip":
        try:
            return LSegTeacher(cfg)
        except Exception as exc:
            logger.warning("LSeg unavailable (%s); falling back to MaskCLIP", exc)
            return MaskCLIPTeacher(cfg)
    raise ValueError(f"unknown teacher backend {cfg.backend!r}")


# --------------------------------------------------------------------------- #
# Fixed PCA basis
# --------------------------------------------------------------------------- #
@dataclass
class SemanticProjection:
    """Fixed orthonormal, uncentered basis shared by image features and text.

    Attributes:
        components: ``[d, D]`` orthonormal rows.
        mean: ``[D]`` feature mean, stored for completeness/diagnostics only — the
            projection deliberately does **not** subtract it, which is what keeps
            cosine similarity valid in the compressed space.
        explained_ratio: Fraction of (uncentered) feature energy retained.
        meta: Provenance, including the exact scenes the basis was fitted on.
    """

    components: torch.Tensor
    mean: torch.Tensor
    explained_ratio: float
    meta: Dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def source_dim(self) -> int:
        return int(self.components.shape[1])

    def project(self, features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Project ``[..., D]`` features to ``[..., d]``; normalises by default."""
        if features.shape[-1] != self.source_dim:
            raise ValueError(
                f"expected last dim {self.source_dim}, got {features.shape[-1]}"
            )
        comps = self.components.to(features.device, features.dtype)
        out = features @ comps.T
        return F.normalize(out, dim=-1) if normalize else out

    def to(self, device) -> "SemanticProjection":
        return SemanticProjection(
            self.components.to(device), self.mean.to(device), self.explained_ratio, self.meta
        )

    def save(self, path: str) -> None:
        from safetensors.torch import save_file

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_file(
            {"components": self.components.cpu().contiguous(), "mean": self.mean.cpu().contiguous()},
            path,
            metadata={"explained_ratio": str(self.explained_ratio), "meta": json.dumps(self.meta, default=str)},
        )

    @classmethod
    def load(cls, path: str) -> "SemanticProjection":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as fh:
            comps = fh.get_tensor("components")
            mean = fh.get_tensor("mean")
            md = fh.metadata() or {}
        return cls(
            components=comps,
            mean=mean,
            explained_ratio=float(md.get("explained_ratio", "nan")),
            meta=json.loads(md.get("meta", "{}")),
        )


def fit_semantic_projection(
    features: torch.Tensor, dim: int, meta: Optional[Dict[str, Any]] = None
) -> SemanticProjection:
    """Fit the shared basis on a sample of **training-source** teacher features.

    Args:
        features: ``[N, D]`` L2-normalised teacher features.
        dim: Target dimensionality.
        meta: Provenance dict; the caller must record the scenes used.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if dim > features.shape[1]:
        raise ValueError(f"pca_dim {dim} exceeds feature dim {features.shape[1]}")
    comps, explained = fit_projection(features.float(), dim)
    return SemanticProjection(
        components=comps.cpu(),
        mean=features.float().mean(0).cpu(),
        explained_ratio=float(explained),
        meta=dict(meta or {}),
    )


def encode_text_queries(
    teacher: TeacherBackend,
    prompts: Sequence[str],
    projection: SemanticProjection,
    templates: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Text prompts → normalised embeddings in the sidecar's output space.

    Uses exactly the transform applied to the training targets: encode, project onto
    the fixed basis, renormalise.
    """
    emb = teacher.encode_text(prompts, templates)
    return projection.project(emb.cpu(), normalize=True)


# --------------------------------------------------------------------------- #
# Teacher-frame density
# --------------------------------------------------------------------------- #
def sample_teacher_frames(num_frames: int, density: float, seed: int) -> np.ndarray:
    """Frame-level teacher availability mask, reproducible from ``seed``.

    Always keeps at least one frame so a scene never silently loses all supervision.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if density >= 1.0:
        return np.ones(num_frames, dtype=bool)
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(num_frames * density)))
    mask = np.zeros(num_frames, dtype=bool)
    mask[rng.choice(num_frames, size=keep, replace=False)] = True
    return mask
