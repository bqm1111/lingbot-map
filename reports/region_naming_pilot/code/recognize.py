"""Two ways to name exactly the same regions: P = pooled Trident evidence, Q = Qwen-VL.

Both consume the frozen proposals and nothing else. P reads the region's own lifted teacher features; Q
reads the region's causal RGB. Neither sees a human label, and Q never sees a depth map, an occupancy
volume or a frame outside the causal clip.

The two recognizers are compared on an identical region set: every region with at least
``query_min_support`` owned evidence voxels, capped at ``query_max_regions`` per anchor by descending
support. Regions outside that set keep the **pooled Trident** label under both recognizers, so the only
difference between P and Q is who named the queried regions. Q's abstentions, malformed replies and
out-of-vocabulary answers fall back to that region's pooled Trident label -- an abstention therefore
never removes a seed and never changes the support.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import legacy as L
import paths as P
import proposals as PR

SETTINGS = {
    "pooled_trident": {
        "aggregation": "mean of the region's owned evidence voxel features, then L2 normalise",
        "readout": "frozen Trident text embeddings, label = 1 + argmax cosine (evidence.trident.classify)",
    },
    "qwen": {
        "model": P.QWEN_REPO,
        "dtype": "bfloat16",
        "generation": {"do_sample": False, "max_new_tokens": 192},
        "views": "tight crop of the largest view + whole-frame context with the region outlined + "
                 "one tight crop from a second frame when the region spans several frames",
        "crop_margin": 0.15, "crop_min_px": 64, "context_dim_outside": 0.45,
        "vocabulary": "the 18 frozen KITTI-360 phrases, verbatim",
        "fallback": "pooled Trident label of the same region (abstain, malformed, out-of-vocabulary)",
        "query_min_support": 10, "query_max_regions": 80,
        "diagnostic_single_view_top_k": 20,
    },
}

PROMPT = """You are looking at a region of a street scene photographed from a car.

The region is the SAME object or surface in every image you are given:
1. a tight crop of the region,
2. the whole camera frame with the region outlined in magenta and everything else darkened,
3. sometimes a second tight crop of the same region from another frame of the same 5-frame clip.

The outline only tells you WHICH region to name. It says nothing about what the region is.

Name the region with exactly one category from this list, and nothing else:
{vocab}

Reply with one JSON object and no other text:
{{"category": "<one category from the list, or \\"insufficient evidence\\">",
  "alternatives": ["<zero to two other categories from the list>"],
  "justification": "<one short sentence about what you can actually see>"}}

If the images do not show enough to choose, answer "insufficient evidence". Do not guess from context
about what a street usually contains."""


# ---- P: pooled Trident -----------------------------------------------------------------
def pooled_trident_labels(voxel_region: torch.Tensor, ev: "L.LF.VoxelEvidence", text: torch.Tensor,
                          n_regions: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(region label [n_regions] long, 0 where the region owns no evidence; support count [n_regions])``."""
    dev = ev.feature.device
    r_of_ev = voxel_region[ev.flat_index]                                  # [M] region id or -1
    lab = torch.zeros(n_regions, dtype=torch.long, device=dev)
    cnt = torch.zeros(n_regions, dtype=torch.long, device=dev)
    keep = r_of_ev >= 0
    if not bool(keep.any()):
        return lab, cnt
    r, f = r_of_ev[keep], ev.feature[keep]
    s = torch.zeros(n_regions, L.TEACHER_DIM, device=dev).index_add_(0, r, f)
    cnt.index_add_(0, r, torch.ones_like(r))
    has = cnt > 0
    lab[has] = L.TR.classify(F.normalize(s[has], dim=-1), text)
    return lab, cnt


# ---- Q: Qwen-VL ------------------------------------------------------------------------
def load_qwen(device: str = "cuda:0"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(P.QWEN_REPO)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        P.QWEN_REPO, dtype=torch.bfloat16, device_map=device).eval()
    return model, proc


def _crop(img: np.ndarray, bbox, margin: float, min_px: int) -> "Image.Image":
    from PIL import Image
    H, W = img.shape[:2]
    x0, y0, w, h = bbox
    cx, cy = x0 + w / 2, y0 + h / 2
    w = max(w * (1 + 2 * margin), min_px); h = max(h * (1 + 2 * margin), min_px)
    a = int(max(0, round(cx - w / 2))); b = int(min(W, round(cx + w / 2)))
    c = int(max(0, round(cy - h / 2))); d = int(min(H, round(cy + h / 2)))
    return Image.fromarray(img[c:d, a:b])


def _context(img: np.ndarray, mask: np.ndarray, dim: float) -> "Image.Image":
    from PIL import Image
    from scipy import ndimage
    out = (img.astype(np.float32) * dim).astype(np.uint8)
    out[mask] = img[mask]
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)
    out[edge] = (255, 0, 255)
    return Image.fromarray(out)


def region_views(region: Dict, label_native: np.ndarray, images: List[np.ndarray],
                 single_view: bool = False) -> List["Image.Image"]:
    """The images Qwen is shown. ``label_native[q]`` is the painted mask-id image of clip slot ``q``."""
    s = SETTINGS["qwen"]
    q = region["best_frame"]
    views = [_crop(images[q], region["best_bbox"], s["crop_margin"], s["crop_min_px"])]
    if single_view:
        return views
    views.append(_context(images[q], label_native[q] == region["best_mask_id"], s["context_dim_outside"]))
    alt = region.get("alt")
    if alt is not None:
        views.append(_crop(images[alt["frame"]], alt["bbox"], s["crop_margin"], s["crop_min_px"]))
    return views


def queried_regions(regions: List[Dict]) -> List[int]:
    """The region ids both recognizers are compared on -- fixed by support, never by a score."""
    s = SETTINGS["qwen"]
    ok = [r for r in regions if r["n_evidence_voxels"] >= s["query_min_support"]]
    ok.sort(key=lambda r: (-r["n_evidence_voxels"], r["region_id"]))
    return sorted(r["region_id"] for r in ok[:s["query_max_regions"]])


@torch.no_grad()
def ask(model, proc, views: List["Image.Image"], vocab: List[str]) -> str:
    content = [{"type": "image", "image": v} for v in views]
    content.append({"type": "text", "text": PROMPT.format(vocab="\n".join(f"- {v}" for v in vocab))})
    msgs = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **SETTINGS["qwen"]["generation"])
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---- deterministic parsing -------------------------------------------------------------
ABSTAIN = "insufficient evidence"


def normalise(s: str) -> str:
    return re.sub(r"[^a-z ]", " ", str(s).lower()).split("(")[0].strip()


def parse(reply: str, vocab: List[str]) -> Dict:
    """``{outcome, category, label, alternatives}``; ``label`` is None unless the category is in vocabulary.

    outcome is one of ``ok`` / ``abstain`` / ``out_of_vocabulary`` / ``malformed``.
    """
    index = {normalise(v): i + 1 for i, v in enumerate(vocab)}
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return {"outcome": "malformed", "category": None, "label": None, "alternatives": []}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"outcome": "malformed", "category": None, "label": None, "alternatives": []}
    cat = normalise(obj.get("category", ""))
    alts = [normalise(a) for a in (obj.get("alternatives") or []) if isinstance(a, str)]
    if cat == normalise(ABSTAIN) or cat in ("", "none", "unknown"):
        return {"outcome": "abstain", "category": cat, "label": None, "alternatives": alts}
    if cat not in index:
        return {"outcome": "out_of_vocabulary", "category": cat, "label": None, "alternatives": alts}
    return {"outcome": "ok", "category": cat, "label": index[cat], "alternatives": alts}
