"""Frozen Trident-H teacher: official hard prediction plus a dense per-class score tensor.

The upstream repository is used **byte-identically** -- nothing in
``third_party/Trident`` is edited. Everything this module does is either (a) calling the
official single-image entry point that ``trident_demo.py`` itself calls, or (b) observing
the official SAM-refinement call through a recording wrapper.

Why a dense tensor needs recovering at all
------------------------------------------
With SAM refinement on, the official ``Trident.postprocess_result`` ends at::

    refined_masks, scores, refined_logits, boxes = sam_refinement(...)
    seg_pred, seg_logits = refined_masks, refined_logits

Inside ``sam_refinement``, ``map_refinement_coarse`` builds a dense ``[C, H, W]`` score
volume and takes ``argmax(0)`` -- so on most pixels the hard label already *is* the argmax
of a dense tensor. Exactly one step then breaks that identity::

    refined_masks = torch.where(refined_logits.sum(0, keepdim=True) == 0,
                                segmentations, refined_masks)

i.e. on pixels no SAM region claimed, the official code falls back to the *coarse* CLIP
prediction ``segmentations``, which is itself ``argmax`` of the coarse probability volume.
So the dense tensor whose argmax reproduces the official prediction everywhere is::

    dense = where(refined_logits.sum(0) == 0, coarse_probs, refined_logits)

``map_failed_regions`` runs afterwards. It provably cannot move the hard label -- it first
does ``refined_logits = max(refined_logits, failed_logit)`` and only then tests
``failed_logit.max(0) > refined_logits.max(0)``, which is False by construction -- but it
*does* rewrite the returned score tensor, so the tensor Trident returns is no longer the
one its own label was computed from. We therefore read the scores at the point the label
was taken (the output of ``map_refinement_coarse``) and reproduce the official behaviour
rather than repair it. Both facts are asserted in ``tests/gate6``.

Probabilities
-------------
The coarse volume is already a distribution: the official code applies the official
temperature (``logit_scale = 40``) and ``softmax`` over classes. The SAM-refined volume is
``sigmoid(sam_logit) * coarse_prob`` masked to the region, which is non-negative but
unnormalised. We divide by the per-pixel sum. That is strictly monotone per pixel, so the
official ``argmax`` is preserved exactly, and it yields the proper distribution that
multi-view voxel fusion needs. No second softmax is applied: the scores live in [0, 1] and
re-softmaxing them would flatten the signal to near-uniform.

The one upstream defect
-----------------------
``trident.py:85`` reads::

    if sam_model_type != 'vit_h' or not sam_refinement:
        self.sam = sam_model_registry[sam_model_type](...).eval().half()
    self.sam.prompt_encoder = ...

For the predeclared large configuration (SAM-H **and** refinement) the guard skips the
assignment and the next line raises ``AttributeError``; the ``else`` branch is missing.
The comment on that line points at segment-anything issue #540 (ViT-H overflows in fp16),
so the intent is unambiguous: build ViT-H in float32. We do not edit the file. We build
the model with ``sam_refinement=False`` (which constructs SAM by the official code path)
and then install the float32 SAM-H and re-enable refinement, which is the state the
missing branch would have produced. Recorded as deviation D1 in the report.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

TRIDENT_ROOT = os.environ.get("TRIDENT_ROOT", "/home/minh/workspace/third_party/Trident")

# The official Cityscapes configuration: the only official Trident config for automotive
# imagery, adopted unchanged for all three automotive benchmarks and pinned in the
# precommit before any Gate-6 target was opened.
OFFICIAL_CITYSCAPES = dict(
    clip_type="laion2b_s32b_b79k",
    model_type="ViT-H-14",
    vfm_model="dino",
    sam_model_type="vit_h",
    sam_refinement=True,
    beta=1.2,
    gamma=3.0,
    cos_fac=3.0,               # configs/cfg_city_scapes.py
    refine_neg_cos=False,      # configs/cfg_city_scapes.py
    coarse_thresh=0.10,        # configs/cfg_city_scapes.py
    slide_stride=224,          # configs/base_config.py
    slide_crop=336,            # configs/base_config.py
    sam_iou_thresh=0.80,       # configs/base_config.py
    minimal_area=225,          # configs/base_config.py
    sam_mask_coff=0.005,       # trident.Trident default
    logit_scale=40,            # trident.Trident default (the official temperature)
    prob_thd=0.0,              # trident.Trident default
    dataset_type="CityscapesDataset",
)
# configs/cfg_city_scapes.py test_pipeline: Resize(scale=(2048, 688), keep_ratio=True)
OFFICIAL_RESIZE_SCALE = (2048, 688)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _ensure_path() -> None:
    for p in (TRIDENT_ROOT, os.path.dirname(TRIDENT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def keep_ratio_size(hw: Tuple[int, int], scale: Tuple[int, int] = OFFICIAL_RESIZE_SCALE):
    """mmcv ``Resize(scale, keep_ratio=True)`` output size for an ``(H, W)`` image.

    ``scale`` is mmcv's ``(long_edge, short_edge)`` pair; the factor is the smaller of the
    two ratios, so the whole image is rescaled uniformly and never cropped or padded.
    """
    h, w = hw
    long_edge, short_edge = max(scale), min(scale)
    f = min(long_edge / max(h, w), short_edge / min(h, w))
    return int(h * f + 0.5), int(w * f + 0.5)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TeacherOutput:
    """One frame of teacher output, all tensors on CPU."""
    probs: torch.Tensor          # [C, H, W] float32, sums to 1 over C
    hard: torch.Tensor           # [H, W] int64, the OFFICIAL Trident prediction
    coarse_probs: torch.Tensor   # [C, H, W] float32, pre-refinement official softmax
    ties: int                    # pixels where argmax(probs) != hard (documented ties)
    lattice_hw: Tuple[int, int]


class TridentTeacher:
    """The frozen teacher. Construct once per process; call :meth:`predict_frame` per image."""

    def __init__(self, phrases: Sequence[str], sam_ckpt: str, device: str = "cuda:0",
                 name_file: Optional[str] = None, cfg: Optional[Dict] = None):
        _ensure_path()
        import trident as trident_mod                                   # noqa: E402
        from segment_anything import sam_model_registry, SamPredictor   # noqa: E402

        self.cfg = dict(OFFICIAL_CITYSCAPES)
        if cfg:
            self.cfg.update(cfg)
        self.phrases = list(phrases)
        self.device = torch.device(device)
        self._mod = trident_mod

        # Trident reads the vocabulary from a file, one entry per line, in channel order.
        self.name_file = name_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_gate6_names.txt")
        with open(self.name_file, "w") as fh:
            fh.write("\n".join(self.phrases))

        kw = dict(self.cfg)
        kw.pop("sam_refinement")
        self.model = trident_mod.Trident(
            name_path=self.name_file, device=self.device, debug=True,
            sam_ckpt=sam_ckpt, sam_refinement=False, **kw)

        # --- the missing ``else`` branch of trident.py:85, applied post-construction ---
        sam = sam_model_registry[self.cfg["sam_model_type"]](checkpoint=sam_ckpt)
        sam = sam.to(device=self.device).eval()                 # float32: SAM issue #540
        sam.prompt_encoder = sam.prompt_encoder.float()
        sam.mask_decoder = sam.mask_decoder.float()
        self.model.sam = sam
        self.model.sam_predictor = SamPredictor(sam)
        self.model.sam_refine = True
        self.model.coarse_thresh = float(self.cfg["coarse_thresh"])
        self.model.minimal_area = int(self.cfg["minimal_area"])
        self.model.sam_mask_coff = float(self.cfg["sam_mask_coff"])

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_classes = self.model.num_classes
        if self.num_classes != len(self.phrases):
            raise AssertionError(
                f"Trident built {self.num_classes} classes for {len(self.phrases)} phrases; "
                "one phrase per class is required so the channel mapping stays injective")

        self._record: Dict[str, torch.Tensor] = {}
        self._install_recorder()

    # ------------------------------------------------------------------ #
    def _install_recorder(self) -> None:
        """Wrap the official ``sam_refinement`` so its inputs/outputs can be read.

        The wrapper calls the original and changes nothing, so the official prediction is
        bit-identical to an unwrapped run (asserted in ``tests/gate6``).
        """
        import seg_utils.utils as su                                    # noqa: E402
        original = self._mod.sam_refinement
        original_map = su.map_refinement_coarse
        rec = self._record

        def recording(img, segmentations, seg_logits, num_classes, predictor, *a, **k):
            rec["coarse_probs"] = seg_logits
            rec["coarse_hard"] = segmentations
            out = original(img, segmentations, seg_logits, num_classes, predictor, *a, **k)
            rec["refined_masks"], rec["returned_logits"] = out[0], out[2]
            return out

        def recording_map(*a, **k):
            masks, logits = original_map(*a, **k)
            # the scores the official label is taken from, before map_failed_regions
            rec["scored_logits"] = logits
            rec["scored_masks"] = masks
            return masks, logits

        recording._gate6_original = original          # for the no-op tests
        recording_map._gate6_original = original_map
        self._mod.sam_refinement = recording
        su.map_refinement_coarse = recording_map

    # ------------------------------------------------------------------ #
    def load_input(self, img_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Native image file -> the CLIP-normalised input tensor at official resolution."""
        import cv2
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        native_hw = (bgr.shape[0], bgr.shape[1])
        th, tw = keep_ratio_size(native_hw)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (th, tw) != native_hw:
            interp = cv2.INTER_LINEAR if th * tw >= native_hw[0] * native_hw[1] else cv2.INTER_AREA
            rgb = cv2.resize(rgb, (tw, th), interpolation=interp)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0).to(self.device), native_hw

    @torch.no_grad()
    def predict_frame(self, img_path: str) -> TeacherOutput:
        """Run the official pipeline once and return the official label + dense scores."""
        inp, native_hw = self.load_input(img_path)
        # mmseg's LoadImageFromFile sets ``ori_shape`` from the file, *before* Resize, so
        # the official code emits scores at native resolution -- which is also the frame
        # SAM works in (it reads ``img_path`` itself). Passing the resized tensor's shape
        # instead desynchronises the two and ``map_refinement_coarse`` fails outright.
        metas = [dict(ori_shape=native_hw, img_shape=tuple(inp.shape[-2:]),
                      pad_shape=tuple(inp.shape[-2:]), padding_size=[0, 0, 0, 0])]
        self._record.clear()
        hard, refined = self.model.predict(inp, data_samples=metas, debug_img_path=img_path)
        hard = hard.squeeze(0).long()                                    # [H, W]

        coarse = self._record.get("coarse_probs")
        scored = self._record.get("scored_logits")
        if coarse is None:                  # refinement disabled: the scores are official
            dense = refined.float()
        elif scored is None:                # no SAM region survived: official early return
            dense = coarse.float()
        else:
            coarse, scored = coarse.float(), scored.float()
            zero = (scored.sum(0, keepdim=True) == 0)
            dense = torch.where(zero, coarse, scored)

        total = dense.sum(0, keepdim=True)
        probs = torch.where(total > 0, dense / total.clamp_min(torch.finfo(torch.float32).tiny),
                            dense)
        ties = int((probs.argmax(0) != hard).sum().item())
        return TeacherOutput(probs=probs.cpu(), hard=hard.cpu(),
                             coarse_probs=(coarse.cpu() if coarse is not None
                                           else probs.cpu()),
                             ties=ties, lattice_hw=tuple(probs.shape[-2:]))


def _uninstall(teacher):
    """Restore the pristine official functions (used to prove the recorder is a no-op)."""
    import seg_utils.utils as su
    mod, cur_ref, cur_map = teacher._mod, teacher._mod.sam_refinement, su.map_refinement_coarse
    mod.sam_refinement = getattr(cur_ref, "_gate6_original", cur_ref)
    su.map_refinement_coarse = getattr(cur_map, "_gate6_original", cur_map)
    return cur_ref, cur_map


def _reinstall(teacher, saved):
    import seg_utils.utils as su
    teacher._mod.sam_refinement, su.map_refinement_coarse = saved


def official_only(teacher, img_path):
    """Run the OFFICIAL pipeline with the recorder removed and return its hard label."""
    saved = _uninstall(teacher)
    try:
        inp, native_hw = teacher.load_input(img_path)
        metas = [dict(ori_shape=native_hw, img_shape=tuple(inp.shape[-2:]),
                      pad_shape=tuple(inp.shape[-2:]), padding_size=[0, 0, 0, 0])]
        with torch.no_grad():
            hard, _ = teacher.model.predict(inp, data_samples=metas, debug_img_path=img_path)
        return hard.squeeze(0).long().cpu()
    finally:
        _reinstall(teacher, saved)


def resample_to_lattice(probs: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Dense ``[C, H, W]`` teacher probabilities -> the LingBot processed lattice.

    Both lattices are *full-extent* resamplings of the same rectified image (Trident's is
    uniform ``keep_ratio``, LingBot's is anisotropic), so a single ``align_corners=False``
    bilinear resize is the exact composition of the two pixel maps. Renormalised afterwards
    because bilinear interpolation of a simplex-valued field is only approximately
    normalised at sub-pixel level.
    """
    out = F.interpolate(probs.unsqueeze(0).float(), size=tuple(hw), mode="bilinear",
                        align_corners=False).squeeze(0)
    return out / out.sum(0, keepdim=True).clamp_min(1e-12)


__all__ = ["TridentTeacher", "TeacherOutput", "OFFICIAL_CITYSCAPES", "OFFICIAL_RESIZE_SCALE",
           "keep_ratio_size", "resample_to_lattice", "sha256_file", "TRIDENT_ROOT",
           "official_only"]
