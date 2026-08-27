"""Dense, text-aligned CLIP features (MaskCLIP-style) for open-vocabulary fusion.

A standard CLIP ViT only exposes one text-aligned vector per image (the CLS
token after attention pooling); its patch tokens are *not* in the text
embedding space.  MaskCLIP recovers a per-patch text-aligned feature by
rewriting the final residual attention block: instead of letting each patch
attend over the image (which mixes in global context and destroys locality),
the value projection is applied per token and pushed straight through the
output projection, ``ln_post`` and the visual projection.

The result is a [h, w, D] grid living in the same space as ``encode_text``,
so cosine similarity against a text embedding is meaningful per patch.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

# A small subset of the OpenAI CLIP prompt ensemble.  Averaging over templates
# is worth ~1-2 points of open-vocabulary accuracy versus a bare class name.
DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "there is a {} in the scene.",
    "a street photo of a {}.",
)


class DenseCLIP:
    """Wraps an open_clip ViT to emit per-patch features aligned with text.

    Args:
        model_name: open_clip architecture. Must be a ViT (patch-based).
        pretrained: open_clip pretrained tag.
        variant: ``"maskclip"`` replaces the last block entirely with the
            value path (canonical MaskCLIP; sharpest localization).
            ``"vv_residual"`` keeps the residual + MLP of the last block and
            only swaps attention for the value path (closer to the original
            activation statistics, smoother maps).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16-quickgelu",
        pretrained: str = "openai",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        variant: str = "maskclip",
    ):
        import open_clip

        if variant not in ("maskclip", "vv_residual"):
            raise ValueError(f"unknown variant {variant!r}")

        self.device = torch.device(device)
        self.dtype = dtype
        self.variant = variant
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(self.device, dtype=dtype)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        visual = self.model.visual
        if not hasattr(visual, "conv1"):
            raise ValueError(f"{model_name} is not a patch-based ViT; dense features unsupported")
        self.visual = visual
        self.patch_size = visual.conv1.kernel_size[0]
        self.embed_dim = visual.proj.shape[1]

        # Grid the pretrained positional embedding was trained at.
        num_pos = visual.positional_embedding.shape[0] - 1
        self.pretrain_grid = int(round(num_pos**0.5))

        mean = getattr(visual, "image_mean", None) or (0.48145466, 0.4578275, 0.40821073)
        std = getattr(visual, "image_std", None) or (0.26862954, 0.26130258, 0.27577711)
        self._mean = torch.tensor(mean, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device, dtype=dtype).view(1, 3, 1, 1)

    # ── positional embedding ────────────────────────────────────────────────

    def _interpolate_pos(self, h: int, w: int) -> torch.Tensor:
        """Resample the square pretrain grid onto an (h, w) grid."""
        pos = self.visual.positional_embedding  # [1 + g*g, C]
        cls_pos, grid_pos = pos[:1], pos[1:]
        g = self.pretrain_grid
        if (h, w) == (g, g):
            return pos
        grid = grid_pos.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(h * w, -1).to(pos.dtype)
        return torch.cat([cls_pos, grid], dim=0)

    # ── dense image features ────────────────────────────────────────────────

    @torch.no_grad()
    def encode_dense(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Per-patch text-aligned features.

        Args:
            images: [B, 3, H, W] float tensor in [0, 1], RGB.
            normalize: L2-normalize the output features.

        Returns:
            [B, h, w, D] where h = H // patch_size, w = W // patch_size.
        """
        v = self.visual
        x = images.to(self.device, dtype=self.dtype)
        x = (x - self._mean) / self._std

        x = v.conv1(x)  # [B, C, h, w]
        B, C, h, w = x.shape
        x = x.reshape(B, C, h * w).permute(0, 2, 1)  # [B, N, C]

        cls = v.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interpolate_pos(h, w).to(x.dtype).unsqueeze(0)
        x = v.ln_pre(x)

        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)

        x = self._dense_last_block(blocks[-1], x)

        x = x[:, 1:]  # drop CLS, keep patches
        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj
        if normalize:
            x = F.normalize(x.float(), dim=-1).to(self.dtype)
        return x.reshape(B, h, w, -1)

    def _dense_last_block(self, blk, x: torch.Tensor) -> torch.Tensor:
        """Run the final block with attention replaced by the value path."""
        y = blk.ln_1(x)
        attn = blk.attn
        dim = y.shape[-1]
        w_v = attn.in_proj_weight[2 * dim :]
        b_v = attn.in_proj_bias[2 * dim :] if attn.in_proj_bias is not None else None
        val = F.linear(y, w_v, b_v)
        val = attn.out_proj(val)

        if self.variant == "maskclip":
            # Canonical MaskCLIP: the dense branch *is* the value path.  Dropping
            # the residual keeps each token's feature purely local.
            return val

        # vv_residual: keep the block's residual structure, swap only attention.
        ls1 = getattr(blk, "ls_1", None)
        x = x + (ls1(val) if ls1 is not None else val)
        mlp_out = blk.mlp(blk.ln_2(x))
        ls2 = getattr(blk, "ls_2", None)
        return x + (ls2(mlp_out) if ls2 is not None else mlp_out)

    # ── text features ───────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(
        self,
        prompts: Sequence[str],
        templates: Optional[Sequence[str]] = DEFAULT_TEMPLATES,
    ) -> torch.Tensor:
        """Embed class names, averaging over a prompt ensemble.

        Returns [len(prompts), D], L2-normalized, matching ``encode_dense``.
        """
        outs: List[torch.Tensor] = []
        for prompt in prompts:
            texts = [t.format(prompt) for t in templates] if templates else [prompt]
            tokens = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tokens).float()
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            outs.append(F.normalize(emb, dim=-1))
        return torch.stack(outs)
