#!/usr/bin/env python
"""Record the frozen teacher's provenance: revisions, weights, hashes, parameter counts.

Runs inside the Trident environment (it imports the official code). Writes
``artifacts/gate6/teacher_provenance.json``, which the precommit config embeds.
"""
from __future__ import annotations

import hashlib, json, os, subprocess, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.gate6.trident_adapter import TRIDENT_ROOT, sha256_file      # noqa: E402

SAM_CKPT = os.environ.get(
    "GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    os.chdir(TRIDENT_ROOT)
    sys.path.insert(0, TRIDENT_ROOT)
    import torch
    from huggingface_hub import hf_hub_download
    from open_clip import create_model
    from segment_anything import sam_model_registry

    prov = {"teacher": "Trident-H", "role": "primary, predeclared", "trained_by_us": False}
    prov["trident"] = {
        "repo": "https://github.com/YuHengsss/Trident",
        "paper": "https://arxiv.org/abs/2411.09219",
        "commit": sh("git rev-parse HEAD", TRIDENT_ROOT),
        "commit_date": sh("git log -1 --format=%ad --date=iso", TRIDENT_ROOT),
        "license": "Apache-2.0 (LICENSE in repository root)",
        "local_path": TRIDENT_ROOT,
        "tracked_files_unmodified": sh("git diff --stat", TRIDENT_ROOT) == ""
                                   and sh("git diff --cached --stat", TRIDENT_ROOT) == "",
        "untracked_paths": [l for l in sh("git status --porcelain", TRIDENT_ROOT).splitlines()],
        "official_config_adopted": "configs/cfg_city_scapes.py (+ configs/base_config.py)",
    }

    # --- OpenCLIP ViT-H/14, laion2b_s32b_b79k -------------------------------------- #
    w = hf_hub_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        "open_clip_pytorch_model.bin")
    clip = create_model("ViT-H-14", pretrained="laion2b_s32b_b79k", precision="fp32")
    prov["clip"] = {
        "model_type": "ViT-H-14", "pretrained_tag": "laion2b_s32b_b79k",
        "hf_repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "weight_file": os.path.realpath(w), "weight_sha256": sha256_file(w),
        "weight_bytes": os.path.getsize(w),
        "n_params": sum(p.numel() for p in clip.parameters()),
        "license": "MIT (LAION OpenCLIP release)",
        "supervision": "LAION-2B image-text pairs; contrastive, no dense segmentation labels",
        "patch_size": 14,
    }
    del clip

    # --- DINO ViT-B/16 (the VFM the official code hard-codes) ----------------------- #
    vfm = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    prov["vfm"] = {
        "model": "dino_vitb16", "hub": "facebookresearch/dino:main",
        "n_params": sum(p.numel() for p in vfm.parameters()),
        "patch_size": 16, "license": "Apache-2.0",
        "supervision": "ImageNet-1k, self-supervised (no labels)",
        "note": ("trident.py:47 loads dino_vitb16 unconditionally; the ``vfm_model`` "
                 "argument only selects the attention hook. DINOv2 is not reachable "
                 "through the official implementation."),
    }
    del vfm

    # --- SAM ViT-H ------------------------------------------------------------------ #
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    prov["sam"] = {
        "model_type": "vit_h", "checkpoint": SAM_CKPT, "source_url": SAM_URL,
        "weight_sha256": sha256_file(SAM_CKPT),
        "weight_bytes": os.path.getsize(SAM_CKPT),
        "n_params": sum(p.numel() for p in sam.parameters()),
        "license": "Apache-2.0",
        "supervision": "SA-1B, class-agnostic masks (no semantic class labels)",
        "precision_used": "float32 (segment-anything issue #540: ViT-H overflows in fp16)",
    }
    del sam

    prov["total_params"] = (prov["clip"]["n_params"] + prov["vfm"]["n_params"]
                            + prov["sam"]["n_params"])
    prov["source_file_hashes"] = {
        f: sha256_file(os.path.join(TRIDENT_ROOT, f))
        for f in ("trident.py", "seg_utils/utils.py", "open_clip/transformer.py",
                  "configs/base_config.py", "configs/cfg_city_scapes.py",
                  "prompts/imagenet_template.py")}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "artifacts", "gate6", "teacher_provenance.json")
    out = os.path.abspath(out)
    with open(out, "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    print(json.dumps(prov, indent=1, sort_keys=True))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
