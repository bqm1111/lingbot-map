"""Feature-first 3D semantics: are Trident-H's pre-classification dense features more
view-consistent than its 25-class probability vectors? Frozen everything; no training.

Representation (Step 1): inside ``Trident.get_trident_seg`` the frozen stack computes
``clip_features = self.clip.encode_image(...)`` -- CLIP ViT-H-14 (laion2b_s32b_b79k) image
tokens after Trident's DINO/SAM correlation refinement -- L2-normalises them, and takes
``logits = clip_features @ query_features.T``, where ``query_features`` are the CLIP text
embeddings of the vocabulary phrases averaged over the OpenAI ImageNet templates. Those
logits, bilinearly interpolated to native resolution, scaled by 40 and soft-maxed, are the
official coarse probabilities; SAM refinement then rewrites region scores. The cached
25-class vectors are the SAM-refined probabilities mapped into the union vocabulary.

So the candidate is the unit-normalised token feature: 1024-d, per token on a 34 x 128
grid for a 376 x 1408 KITTI-360 frame, language-aligned by construction (any CLIP text
embedding can be compared to it by cosine), one linear map away from the official logits.
"""
from __future__ import annotations
import os
from tools.future_semantic_completion import study as FSC
from tools.future_semantic_completion.target_consistency import common as TC

ART = os.path.join(FSC.REPO_ROOT, "artifacts", "feature_first_3d_semantics")
DATA_ROOT = os.environ.get("FF3D_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d")
TOKEN_DIR = os.path.join(DATA_ROOT, "trident_tokens")        # per-frame dense features
LIFT_DIR = os.path.join(DATA_ROOT, "lifted")                  # per-anchor per-frame lifted evidence
TEXT_PATH = os.path.join(DATA_ROOT, "text_embeddings.npz")
FEAT_DIM = 1024
LOGIT_SCALE = 40.0
SAM_CKPT = os.environ.get("GATE6_SAM_CKPT", "/home/minh/workspace/third_party/checkpoints/sam_vit_h_4b8939.pth")
TRIDENT_ROOT = "/home/minh/workspace/third_party/Trident"
GROUPING = "interleaved"
EVIDENCE_DIRS = {"eligible": TC.EVIDENCE_DIR, "all_future_observed": TC.EVIDENCE_DIR + "_allocc"}

require_data_root = FSC.require_data_root
sha256_file, sha256_json, write_json, environment = (FSC.sha256_file, FSC.sha256_json, FSC.write_json, FSC.environment)


def token_path(source: str, key: str) -> str:
    return os.path.join(TOKEN_DIR, source, f"{key}.npz")


def lift_path(population: str, drive: str, idx: int) -> str:
    return os.path.join(LIFT_DIR, population, drive, f"{idx:05d}.npz")


__all__ = ["ART", "DATA_ROOT", "TOKEN_DIR", "LIFT_DIR", "TEXT_PATH", "FEAT_DIM", "LOGIT_SCALE", "SAM_CKPT",
           "TRIDENT_ROOT", "GROUPING", "EVIDENCE_DIRS", "token_path", "lift_path", "require_data_root",
           "sha256_file", "sha256_json", "write_json", "environment"]
