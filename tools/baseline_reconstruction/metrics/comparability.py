"""Historical-vs-current comparability: the property table and the classification rule.

The classification is mechanical so it cannot bend to the numbers: the *evaluation* side
(anchors, mask, grid, camera, metric) decides between "not comparable" and the rest; the
*pipeline* side (geometry, scale, input, semantics, lock, threshold, targets) decides
between "fully" and "partially" comparable. Magnitudes never enter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from tools.baseline_reconstruction import config as C


@dataclass(frozen=True)
class Property:
    name: str
    historical: str
    current: str
    kind: str          # "evaluation" or "pipeline"

    @property
    def differs(self) -> bool:
        return self.historical.strip().lower() != self.current.strip().lower()


def property_table() -> List[Property]:
    return [
        Property("LingBot mode", "long-stream causal replay (persistent KV cache, stream depth/pose cache)", "local-reset L, five frames t-4..t, KV cache cleared per clip", "pipeline"),
        Property("depth", "LB-M (LingBot canonical depth x metric scale)", "LB-M (LingBot canonical depth x metric scale)", "pipeline"),
        Property("scale", "stream scale: MoGe-2 vs LingBot, median of the FIRST five stream frames, frozen for the stream", "clip scale: MoGe-2 vs LingBot, median over the clip's own five frames", "pipeline"),
        Property("pose", "long-stream LingBot pose", "local LingBot pose (anchor = identity)", "pipeline"),
        Property("geometry representation", "log-odds map with w_free, n_obs, age, semantic evidence (H mapper + accumulators)", "H: hard S / F from the same mapper, semantics off", "pipeline"),
        Property("input channels", "32", "3", "pipeline"),
        Property("semantics", "Trident-H probabilities as input; semantic KL loss 0.5; semantic head", "none", "pipeline"),
        Property("residual lock", "final = base + residual * [|base| < 2]", "none (base log-odds 0: the network output is the map)", "pipeline"),
        Property("completion architecture", "CompletionUNet width 24, cin 32, 986,114 params, replicate padding", "CompletionUNet width 24, cin 3, 967,322 params, replicate padding", "pipeline"),
        Property("training data", "KITTI-360 0003/0007/0010; targets: raw-LiDAR + plain-column open-sky rule (pre-647bcf3)", "KITTI-360 0003/0007/0010; targets: raw-LiDAR + neighbourhood-7 open-sky rule + 647bcf3 fix", "pipeline"),
        Property("validation drive / checkpoint rule", "0006; highest AP -> step 6000 for all seeds", "0006; final step 6000, no selection", "pipeline"),
        Property("decision threshold", "per seed, IoU-maximising on 0006: -0.125 / -0.4375 / -0.3125", "0 for every seed", "pipeline"),
        Property("training recipe", "focal+Dice (gamma 2, cap 8, w_unknown 2) + 0.5 semantic KL; AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; 128x128x32; 6000 steps", "focal+Dice (gamma 2, cap 8, w_unknown 2); AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; 128x128x32; 6000 steps", "pipeline"),
        Property("seed statistic", "median of 3", "mean of 3 (median reported)", "pipeline"),
        Property("SemanticKITTI protocol", C.TARGET_PROTOCOLS["semantickitti"].stream_rule + "; " + C.TARGET_PROTOCOLS["semantickitti"].evaluation_mask,
                 C.TARGET_PROTOCOLS["semantickitti"].stream_rule + "; " + C.TARGET_PROTOCOLS["semantickitti"].evaluation_mask, "evaluation"),
        Property("Occ3D protocol", C.TARGET_PROTOCOLS["occ3d"].stream_rule + "; " + C.TARGET_PROTOCOLS["occ3d"].evaluation_mask + "; " + C.TARGET_PROTOCOLS["occ3d"].reduction,
                 C.TARGET_PROTOCOLS["occ3d"].stream_rule + "; " + C.TARGET_PROTOCOLS["occ3d"].evaluation_mask + "; " + C.TARGET_PROTOCOLS["occ3d"].reduction, "evaluation"),
        Property("metric", "SC IoU = TP/(TP+FP+FN) inside the official mask", "SC IoU = TP/(TP+FP+FN) inside the official mask", "evaluation"),
        Property("anchor set", "163 SemanticKITTI / 1182 Occ3D official anchors", "163 SemanticKITTI / 1182 Occ3D official anchors", "evaluation"),
    ]


def classify_comparability(properties: List[Property], anchors_match: Dict[str, bool]) -> Dict[str, Dict]:
    """Per target dataset: one of config.COMPARABILITY_CLASSES plus the reasons."""
    eval_diffs = [p.name for p in properties if p.kind == "evaluation" and p.differs]
    pipe_diffs = [p.name for p in properties if p.kind == "pipeline" and p.differs]
    out = {}
    for ds in C.HISTORICAL_TARGETS:
        if eval_diffs or not anchors_match.get(ds, False):
            cls = "not comparable"
        elif pipe_diffs:
            cls = "partially comparable"
        else:
            cls = "fully comparable"
        out[ds] = {"class": cls, "evaluation_differences": eval_diffs, "pipeline_differences": pipe_diffs,
                   "anchor_set_identical": anchors_match.get(ds, False)}
    return out


__all__ = ["Property", "property_table", "classify_comparability"]
