"""``LingBotSemBypass`` -- frozen geometry + a lightweight MaskCLIP-space sidecar.

    geometry, encoder_tokens = frozen_lingbot(images)
    semantic_features        = sidecar(encoder_tokens)

The sidecar is the **unmodified**
:class:`~semantic_sidecar.models.semantic_sidecar.SemanticSidecar`, instantiated with
``num_layers=1, in_channels=1024`` because SemBypass reads pre-GCT encoder tokens rather
than the previous study's two 2048-channel aggregator blocks.  Nothing about the
architecture changes; only the input width does, which lowers the count from 9 112 896
to **6 156 864** trainable parameters (0.532 % of LingBot).

Output space is the frozen 64-d PCA subspace of MaskCLIP's 512-d embedding
(``pca.safetensors``: orthonormal, applied uncentered), so cosine similarity against
text prompts projected through the same basis is preserved exactly.  The MaskCLIP
**image** tower is never loaded at inference; text embeddings are precomputed once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from semantic_sidecar.config import LingBotConfig
from semantic_sidecar.models.semantic_sidecar import SemanticSidecar
from semantic_sidecar.teacher_features import SemanticProjection

from research.sem_bypass.encoder_features import ENCODER_CHANNELS, FrozenLingBotEncoder


@dataclass
class SemBypassOutput:
    """One streaming step: unchanged LingBot geometry plus semantic features."""

    depth: torch.Tensor            # [S, H, W] LingBot depth, arbitrary monocular scale
    depth_conf: torch.Tensor       # [S, H, W] confidence/precision in [1, inf)
    pose_enc: torch.Tensor         # [S, 9]
    extrinsic: torch.Tensor        # [S, 3, 4] world-to-camera
    intrinsic: torch.Tensor        # [S, 3, 3]
    semantic: torch.Tensor         # [S, P, out_dim] L2-normalised, PCA-MaskCLIP space
    patch_grid: Sequence[int]      # (h, w)

    def query(self, text_embed: torch.Tensor) -> torch.Tensor:
        """Cosine logits of every patch against ``[C, out_dim]`` text embeddings."""
        return self.semantic.to(text_embed.dtype) @ text_embed.T


def build_sidecar(out_dim: int = 64, hidden_dim: int = 768, num_blocks: int = 2,
                  mlp_ratio: float = 2.0, dropout: float = 0.0) -> SemanticSidecar:
    """The audited architecture, sized for a single 1024-channel pre-GCT input."""
    return SemanticSidecar(
        num_layers=1, in_channels=ENCODER_CHANNELS, hidden_dim=hidden_dim,
        num_blocks=num_blocks, mlp_ratio=mlp_ratio, out_dim=out_dim,
        fuse="concat", dropout=dropout,
    )


class LingBotSemBypass:
    """Frozen LingBot geometry with a trained semantic bypass.

    Args:
        lingbot_cfg: frozen-model configuration.
        sidecar_ckpt: path to a trained sidecar ``best.pt``/``last.pt``.
        projection: fitted PCA basis (for text-side projection); optional.
        device: compute device.
    """

    def __init__(self, lingbot_cfg: LingBotConfig, sidecar_ckpt: Optional[str] = None,
                 projection: Optional[SemanticProjection] = None,
                 device: Optional[str] = None, out_dim: int = 64):
        self.device = torch.device(device or lingbot_cfg.device)
        self.lingbot = FrozenLingBotEncoder(lingbot_cfg, capture_tokens=True)
        self.lingbot.assert_frozen()
        self.sidecar = build_sidecar(out_dim=out_dim).to(self.device).eval()
        if sidecar_ckpt:
            self.load_sidecar(sidecar_ckpt)
        self.projection = projection

    # -- checkpoint --------------------------------------------------------- #
    def load_sidecar(self, path: str) -> Dict[str, int]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck.get("model", ck.get("state_dict", ck))
        missing, unexpected = self.sidecar.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"sidecar checkpoint mismatch: {len(missing)}/{len(unexpected)}")
        return self.parameter_report()

    def parameter_report(self) -> Dict[str, int]:
        rep = self.lingbot.parameter_report()
        total = sum(p.numel() for p in self.sidecar.parameters())
        rep.update({
            "sidecar_total": total,
            "sidecar_trainable": sum(p.numel() for p in self.sidecar.parameters() if p.requires_grad),
            "sidecar_fraction_of_lingbot": total / rep["lingbot_total"],
        })
        return rep

    def assert_frozen_geometry(self) -> None:
        self.lingbot.assert_frozen()

    # -- text side ---------------------------------------------------------- #
    @staticmethod
    def encode_text(prompts: Sequence[str], projection: SemanticProjection,
                    model_name: str = "ViT-B-16-quickgelu", pretrained: str = "openai",
                    device: str = "cuda") -> torch.Tensor:
        """Precompute ``[C, out_dim]`` text embeddings in the sidecar's output space.

        Loads only CLIP's **text** tower, once, offline.  The result is meant to be
        cached and shipped, so inference never touches MaskCLIP.
        """
        from semantic.dense_clip import DenseCLIP

        clip = DenseCLIP(model_name=model_name, pretrained=pretrained, device=device)
        text = clip.encode_text(list(prompts)).float().cpu()
        return F.normalize(projection.project(text, normalize=False), dim=-1)

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def run(self, images: torch.Tensor) -> SemBypassOutput:
        """Run one causal RGB chunk ``[S, 3, H, W]`` in ``[0, 1]``.

        LingBot runs exactly as it would without the sidecar -- the token hook returns
        ``None`` and cannot change its outputs.
        """
        preds, tokens = self.lingbot.run_sequence(images)
        extr, intr = FrozenLingBotEncoder.to_extrinsics(preds["pose_enc"], images.shape[-2:])
        H, W = images.shape[-2:]
        ps = self.lingbot.cfg.patch_size
        sem = self.sidecar(tokens.to(self.device).float(), normalize=True).cpu()
        return SemBypassOutput(
            depth=preds["depth"][..., 0], depth_conf=preds["depth_conf"],
            pose_enc=preds["pose_enc"], extrinsic=extr.float(), intrinsic=intr.float(),
            semantic=sem, patch_grid=(H // ps, W // ps),
        )

    # 