#!/usr/bin/env python
"""Optional comparator: the OccAny-aligned Grounded-SAM-2 semantic readout.

Run **after** the primary Trident predictions are SHA-256-pinned. It is a comparator, not
a candidate teacher, and nothing about the primary method depends on it.

What is reproduced from the public OccAny code, unchanged:

* the class prompts ``occany.datasets.kitti.KITTI_CLASS_PROMPTS`` -- per-class synonym
  lists over the full official SemanticKITTI vocabulary, with ``sky`` mapped to empty;
* ``TEXT_PROMPT = '. '.join(FINE_CLASSES)`` and the fine-to-coarse index mapping;
* Grounding DINO SwinB with OccAny's defaults ``box_threshold=0.1``, ``text_threshold=0.0``;
* ``occany.semantic_inference.infer_semantic`` itself, which runs SAM 2.1 on the detected
  boxes and resolves overlaps by painting highest-confidence masks first and never
  overwriting an already-painted pixel;
* OccAny's inference resolution (long side 1216).

The two documented departures, both forced and both recorded in the report:

* **Preprocessing.** OccAny crops around the principal point before rescaling. That crop is
  tied to its own reconstruction, and using it would misalign the readout with the frozen
  LingBot lattice this gate is required to keep. The native image is resized full-extent to
  OccAny's inference resolution instead, which is also the fairer comparator setup: both
  teachers then see the same pixels.
* **SAM 3.** OccAny's newer branch uses ``facebook/sam3``, a gated repository. No access is
  configured on this machine and the brief forbids bypassing licensing, so the
  Grounded-SAM-2 branch is used -- which the brief names as an accepted alternative.

Because the readout emits hard labels rather than scores, a voxel's label is the mode of
its contributing pixels. That is exactly what the frozen fusion already computes if each
pixel is written as a one-hot vector, so the identical lifting code is reused unchanged.
A voxel no detection covers is left **empty**: the teacher declined to name it, and the
report gives its rate separately rather than hiding it in an arbitrary fallback class.

    python tools/gate6/occany_comparator.py --stage cache
    python tools/gate6/occany_comparator.py --stage predict
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

OCCANY = os.environ.get("OCCANY_ROOT", "/home/minh/workspace/third_party/OccAny")
# OccAny's own import list (extract_gdino_boxes_kitti.py), pointed at the checkout that
# has the submodules populated.
_TP = "/home/minh/workspace/OccAny/third_party"
GSAM = os.path.join(_TP, "Grounded-SAM-2")
for p in (OCCANY, _TP, os.path.join(_TP, "dust3r"),
          os.path.join(_TP, "croco", "models", "curope"), GSAM,
          os.path.join(GSAM, "grounding_dino"), os.path.join(_TP, "sam3"),
          os.path.join(_TP, "Depth-Anything-3", "src")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, grids, pipelines, vocab                    # noqa: E402

DATASET = "semantickitti"
CACHE = "/media/SSD1/MINH_DATASETS/lingbot_gate6/occany_semantics"
PRED = "/media/SSD1/MINH_DATASETS/lingbot_gate6/occany_predictions"
IMAGE_SIZE = 1216                       # OccAny extract_gdino_boxes_kitti.py default
BOX_THRESHOLD = 0.1                     # OccAny default
TEXT_THRESHOLD = 0.0                    # OccAny default
SEMANTIC_TXT = "pretrained@SAM2_large"  # feat_src @ SAM2_CONFIGS key, per infer_semantic


def _import_occany():
    """Import OccAny's semantic stack, repairing one upstream import defect.

    ``occany/model/model_sam2.py`` line 26 does ``from PIL import Image`` (the *module*)
    but then writes ``Union[np.ndarray, Image]`` (line 521) and
    ``isinstance(image, Image)`` (line 537), both of which require the *class* -- which is
    what upstream sam2 binds (``from PIL.Image import Image``). As released the module
    therefore cannot be imported at all under Python 3.10. Binding the class is the only
    reading under which the file runs, so it is a repair, not a choice, and it carries no
    methodological content.

    The upstream file is NOT edited. Everything ``model_sam2`` imports before line 26 is
    pre-warmed, then a shim of the ``PIL`` package whose ``Image`` is the class is
    installed for the duration of that single import and removed immediately after, so no
    other module ever sees it. Recorded as comparator deviation C2 in the report.
    """
    import types, importlib
    import PIL, PIL.Image
    for m in ("copy", "collections", "torch", "packaging", "huggingface_hub", "torch.nn",
              "dust3r.utils.misc", "sam2.build_sam", "sam2.sam2_image_predictor",
              "sam2.sam2_video_predictor", "croco.models.blocks", "croco.models.pos_embed",
              "hydra", "hydra.utils", "omegaconf", "typing", "numpy", "logging"):
        importlib.import_module(m)
    shim = types.ModuleType("PIL")
    shim.__dict__.update(PIL.__dict__)
    shim.Image = PIL.Image.Image
    real = sys.modules["PIL"]
    sys.modules["PIL"] = shim
    try:
        importlib.import_module("occany.model.model_sam2")
    finally:
        sys.modules["PIL"] = real
    from occany.semantic_inference import infer_semantic
    return infer_semantic


def occany_prompts():
    """OccAny's ``KITTI_CLASS_PROMPTS``, read as data rather than imported.

    ``occany/datasets/__init__.py`` unconditionally re-exports the vendored dust3r, whose
    ``datasets/base/batched_sampler.py:290`` uses ``np.c_[sample_idxs, *idxs]`` -- a
    starred expression inside a subscript, which is a **SyntaxError** before Python 3.11.
    The whole frozen Gate-1..6 stack runs on Python 3.10, so that package cannot be
    imported here at all. The prompts are a literal constant, so they are parsed straight
    out of the source with ``ast.literal_eval`` -- byte-identical data, no import.
    Recorded as comparator deviation C3.
    """
    import ast
    src = open(os.path.join(OCCANY, "occany", "datasets", "kitti.py")).read()
    tree = ast.parse(src)
    KITTI_CLASS_PROMPTS = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "KITTI_CLASS_PROMPTS" for t in node.targets):
            KITTI_CLASS_PROMPTS = ast.literal_eval(node.value)
    if KITTI_CLASS_PROMPTS is None:
        raise RuntimeError("KITTI_CLASS_PROMPTS not found in OccAny's kitti.py")
    class_names = [n for n, _ in KITTI_CLASS_PROMPTS]
    prompt_list = [p for _, p in KITTI_CLASS_PROMPTS]
    fine = sum(prompt_list, [])
    coarse_of_fine = [i for i, inner in enumerate(prompt_list) for _ in inner]
    return class_names, prompt_list, fine, coarse_of_fine


def load_frame(path, size=IMAGE_SIZE):
    """Native image -> the two normalised tensors OccAny's ``infer_semantic`` expects."""
    import cv2
    from PIL import Image
    from occany.utils.image_util import GroundingDinoImgNorm, get_SAM2_transforms

    bgr = cv2.imread(path)
    native_hw = (bgr.shape[0], bgr.shape[1])
    h, w = native_hw
    f = size / max(h, w)
    th, tw = int(round(h * f)), int(round(w * f))
    rgb = cv2.cvtColor(cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA
                                  if f < 1 else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    gdino_img, _ = GroundingDinoImgNorm(pil, None)          # OccAny's own transform
    sam2_tf = get_SAM2_transforms(resolution=min(1024, size))   # sam2_output_resolution
    sam2_img = sam2_tf(np.array(pil))
    return gdino_img[None], sam2_img[None], native_hw, (th, tw)


def stage_cache(args):
    infer_semantic = _import_occany()
    import torch.nn.functional as Fnn

    class_names, _pl, fine, coarse_of_fine = occany_prompts()
    v = vocab.load(DATASET)
    # OccAny class ids are 0=empty then the 19 official classes in order, which is exactly
    # the official label id. Assert rather than assume.
    assert class_names[0] == "empty" and tuple(class_names[1:]) == tuple(v.names), \
        f"OccAny KITTI class order differs from the official one: {class_names}"

    os.makedirs(os.path.join(CACHE, DATASET), exist_ok=True)
    hp, wp = None, None
    for rec in F.read_manifest(DATASET, REPO_ROOT):
        p = F.lingbot_cache_path(DATASET, rec.clip_id, REPO_ROOT)
        if os.path.exists(p):
            with np.load(p) as z:
                hp, wp = [int(x) for x in z["proc_hw"]]
            break

    allf = F.unique_frames(DATASET, REPO_ROOT)
    mine = F.shard(allf, args.shard, args.num_shards)
    t0, done, nodet = time.time(), 0, 0
    for key, path in mine:
        dst = os.path.join(CACHE, DATASET, f"{key}.npz")
        if os.path.exists(dst) and not args.overwrite:
            continue
        gd, s2, native_hw, proc = load_frame(path)
        with torch.no_grad():
            out = infer_semantic(gd.to(args.device), s2.to(args.device), SEMANTIC_TXT,
                                 fine, args.device, box_threshold=BOX_THRESHOLD,
                                 text_threshold=TEXT_THRESHOLD,
                                 image_size=s2.shape[-1])   # as OccAny's own driver does
        # (sem2d, boxes, confidences, labels); sem2d carries FINE class indices, painted
        # highest-confidence-first with no overwriting -- OccAny's own overlap rule
        fine_sem = np.asarray(out[0]).squeeze()
        n_boxes = len(out[1])
        # OccAny's fine -> coarse remap, verbatim
        sem = np.zeros_like(fine_sem)
        for fid in np.unique(fine_sem):
            if fid == 0:
                continue
            sem[fine_sem == fid] = coarse_of_fine[int(fid)]
        lab = torch.from_numpy(sem.astype(np.int64))[None, None].float()
        lab = Fnn.interpolate(lab, size=(hp, wp), mode="nearest")[0, 0].numpy().astype(np.uint8)
        nodet += int((lab == 0).mean() > 0.999)
        np.savez(dst + ".tmp.npz", label=lab, proc_hw=np.asarray([hp, wp], np.int32),
                 native_hw=np.asarray(native_hw, np.int32),
                 n_boxes=np.int64(n_boxes), n_detected=np.int64((lab > 0).sum()))
        os.replace(dst + ".tmp.npz", dst)
        done += 1
        if done % 100 == 0:
            print(f"[occany {args.shard}] {done} {(time.time()-t0)/done:.2f}s/frame", flush=True)
    print(json.dumps({"stage": "cache", "written": done, "frames_with_no_detection": nodet,
                      "seconds": time.time() - t0}))
    return 0


def stage_predict(args):
    """Lift the comparator's hard labels with the identical frozen geometry."""
    from gates.gate6 import lifting
    v = vocab.load(DATASET)
    C = len(v)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev); torch.cuda.init()
    G = grids.EVAL_GRID[DATASET]
    os.makedirs(os.path.join(PRED, DATASET), exist_ok=True)
    rows, t0 = [], time.time()
    for clip in pipelines.iter_clips(DATASET, REPO_ROOT):
        if clip.scale is None:
            continue
        onehots = []
        ok = True
        for k in clip.frame_keys:
            p = os.path.join(CACHE, DATASET, f"{k}.npz")
            if not os.path.exists(p):
                ok = False
                break
            with np.load(p) as z:
                lab = z["label"].astype(np.int64)
            # C+1 channels: the last one is "the teacher detected nothing here". The
            # frozen fusion (mean of one-hot vectors, then argmax) is then exactly a
            # majority vote among the contributing pixels, with "undetected" a legitimate
            # outcome rather than an arbitrary fallback class.
            oh = np.zeros((C + 1,) + lab.shape, np.float32)
            for c, l in enumerate(v.labels):
                oh[c][lab == l] = 1.0
            oh[C][lab == 0] = 1.0
            onehots.append(torch.from_numpy(oh))
        if not ok:
            continue
        sem = torch.stack(onehots, 0).to(dev)
        pr = pipelines.predict_clip(DATASET, clip, sem, dev)
        labels = np.asarray(list(v.labels) + [v.empty_label], np.int32)
        np.savez(os.path.join(PRED, DATASET, f"{clip.clip_id}.npz"),
                 raw_flat=pr.raw_flat, raw_channel=pr.raw_channel,
                 dil_flat=pr.dil_flat, dil_channel=pr.dil_channel,
                 dil_support=pr.dil_support, labels=labels)
        rows.append(clip.clip_id)
        del sem
    print(json.dumps({"stage": "predict", "clips": len(rows),
                      "seconds": time.time() - t0}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["cache", "predict"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    return stage_cache(a) if a.stage == "cache" else stage_predict(a)


if __name__ == "__main__":
    raise SystemExit(main())
