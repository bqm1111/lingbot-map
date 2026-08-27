"""Frozen model wrappers: LingBot representation capture and the external DINO teacher.

Both models are loaded with ``requires_grad_(False)`` on every parameter and run under
``torch.no_grad``.  LingBot representations are read with **forward hooks**, which
cannot alter a module's return value; ``tests/test_hooks.py`` asserts bit-exactness of
``depth`` and ``pose_enc`` with the hooks registered and removed.

Two capture points are used:

``patch_embed``
    ``aggregator.patch_embed`` returns ``x_norm_patchtokens`` ``[B*S, P, 1024]`` -- the
    frame-independent DINOv2 tokens, computed before any cross-frame attention.

``aggregator``
    the module's own return value ``(output_list, patch_start_idx)`` where
    ``output_list[k]`` is ``[B, S, 6+P, 2048]`` = ``frame_blocks[b] ‖ global_blocks[b]``
    for the ``b`` in ``selected_idx=[4, 11, 17, 23]`` chosen by ``GCTStream``.
    Reading the module output rather than re-running blocks guarantees the captured
    tokens are exactly the ones the depth and camera heads consume.
"""

from __future__ import annotations

import contextlib
import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from research.lingbot_semantic_memory.config import (
    PATCH_START_IDX, LingBotConfig, REPO_ROOT, RepresentationSpec, TeacherConfig,
)

#: Aggregator block indices ``GCTStream._aggregate_features`` emits, in order.
SELECTED_BLOCKS: Tuple[int, ...] = (4, 11, 17, 23)


def assert_frozen(module: torch.nn.Module, name: str) -> None:
    """Raise if any parameter is trainable or carries a gradient."""
    bad = [n for n, p in module.named_parameters() if p.requires_grad]
    if bad:
        raise RuntimeError(f"{name}: {len(bad)} trainable parameters, e.g. {bad[:3]}")
    grads = [n for n, p in module.named_parameters() if p.grad is not None]
    if grads:
        raise RuntimeError(f"{name}: {len(grads)} parameters carry a gradient, e.g. {grads[:3]}")


# --------------------------------------------------------------------------- #
# LingBot
# --------------------------------------------------------------------------- #
class FrozenLingBot:
    """Frozen ``GCTStream`` that additionally yields detached intermediate tokens."""

    def __init__(self, cfg: LingBotConfig, reps: List[RepresentationSpec], device: torch.device):
        from lingbot_map.models.gct_stream import GCTStream

        self.cfg = cfg
        self.device = device
        self.reps = list(reps)
        for r in self.reps:
            if r.kind == "block" and r.block not in SELECTED_BLOCKS:
                raise ValueError(
                    f"block {r.block} is not emitted by GCTStream (selected={SELECTED_BLOCKS}); "
                    "capturing it would require changing what the heads consume"
                )

        model = GCTStream(
            img_size=cfg.image_size, patch_size=cfg.patch_size,
            enable_3d_rope=cfg.enable_3d_rope, max_frame_num=cfg.max_frame_num,
            kv_cache_sliding_window=cfg.kv_cache_sliding_window,
            kv_cache_scale_frames=cfg.num_scale_frames,
            kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
            use_sdpa=cfg.use_sdpa, camera_num_iterations=cfg.camera_num_iterations,
        )
        ck = torch.load(os.path.join(REPO_ROOT, cfg.model_path), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert_frozen(self.model, "lingbot")
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self._patch_buf: List[torch.Tensor] = []
        self._rep_buf: Dict[str, List[torch.Tensor]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._capture = False

    # -- hooks -------------------------------------------------------------- #
    def _patch_hook(self, _m, _i, out):
        if not self._capture:
            return None
        t = out["x_norm_patchtokens"] if isinstance(out, dict) else out
        self._patch_buf.append(t.detach())
        return None  # never modify the module output

    def _agg_hook(self, _m, _i, out):
        if not self._capture:
            return None
        tokens_list, patch_start_idx = out
        if patch_start_idx != PATCH_START_IDX:
            raise RuntimeError(f"unexpected patch_start_idx {patch_start_idx}")
        if len(tokens_list) != len(SELECTED_BLOCKS):
            raise RuntimeError(f"expected {len(SELECTED_BLOCKS)} block outputs, got {len(tokens_list)}")
        patch = self._patch_buf.pop(0)                      # [B*S, P, 1024]
        for rep in self.reps:
            if rep.kind == "patch_embed":
                t = patch                                          # [B*S, P, 1024], B == 1
            else:
                k = SELECTED_BLOCKS.index(rep.block)
                t = tokens_list[k][:, :, patch_start_idx:, :][0]   # [S, P, 2048]
            self._rep_buf.setdefault(rep.name, []).append(t.detach().to(torch.float16).cpu())
        return None

    @contextlib.contextmanager
    def capturing(self) -> Iterator[None]:
        self._patch_buf, self._rep_buf, self._capture = [], {}, True
        self._handles = [
            self.model.aggregator.patch_embed.register_forward_hook(self._patch_hook),
            self.model.aggregator.register_forward_hook(self._agg_hook),
        ]
        try:
            yield
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
            self._capture = False

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def run_chunk(self, images: torch.Tensor, capture: bool = True):
        """Run streaming inference over ``[S, 3, H, W]`` images in ``[0, 1]``.

        Returns:
            ``(predictions, reps)`` -- predictions on CPU with the batch dim squeezed,
            and ``reps[name]`` of shape ``[S, P, C]`` in fp16, or ``{}`` when
            ``capture`` is False.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [S, 3, H, W], got {tuple(images.shape)}")
        images = images.to(self.device)
        autocast = torch.amp.autocast("cuda", dtype=getattr(torch, self.cfg.autocast_dtype))
        ctx = self.capturing() if capture else contextlib.nullcontext()
        with ctx, autocast:
            self.model.clean_kv_cache()
            preds = self.model.inference_streaming(
                images, num_scale_frames=self.cfg.num_scale_frames,
                keyframe_interval=self.cfg.keyframe_interval,
                output_device=torch.device("cpu"),
            )
        preds = {k: (v[0] if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in preds.items()}
        reps: Dict[str, torch.Tensor] = {}
        if capture:
            S = images.shape[0]
            for name, parts in self._rep_buf.items():
                t = torch.cat(parts, dim=0)
                if t.shape[0] != S:
                    raise RuntimeError(f"{name}: captured {t.shape[0]} frames, expected {S}")
                reps[name] = t
            self._rep_buf = {}
        return preds, reps


# --------------------------------------------------------------------------- #
# External DINO teacher
# --------------------------------------------------------------------------- #
class FrozenDinoTeacher:
    """Frozen DINOv2 ViT-B/14-reg producing dense patch features.

    Both LingBot's encoder and this teacher use a patch size of 14, so on the shared
    518-wide input grid the teacher's patch tokens land on **exactly** the same
    ``(H/14, W/14)`` lattice as LingBot's.  Distillation therefore needs no resize and
    no interpolation -- the token-to-token mapping is the identity.
    """

    def __init__(self, cfg: TeacherConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        torch.hub.set_dir(cfg.hub_dir)
        repo = os.path.join(cfg.hub_dir, cfg.repo_subdir)
        model = torch.hub.load(repo, cfg.entrypoint, source="local", pretrained=False)
        sd = torch.load(os.path.join(cfg.hub_dir, cfg.weights), map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"teacher weights mismatch: {len(missing)}/{len(unexpected)}")
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert_frozen(self.model, "teacher")
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self._mean = torch.tensor(cfg.mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(cfg.std, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, batch: int = 16) -> torch.Tensor:
        """Encode ``[S, 3, H, W]`` in ``[0, 1]`` into ``[S, P, 768]`` fp16 patch tokens."""
        out: List[torch.Tensor] = []
        for i in range(0, images.shape[0], batch):
            x = images[i: i + batch].to(self.device)
            x = (x - self._mean) / self._std
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = self.model.forward_features(x)["x_norm_patchtokens"]
            out.append(f.float().to(torch.float16).cpu())
        return torch.cat(out, dim=0)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
