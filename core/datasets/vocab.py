# extracted from gate6/vocab.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Official non-empty class vocabularies and the one deterministic phrase per class.

Three rules, tested in ``tests/gate6``:

* every official **non-empty** semantic class of a benchmark appears exactly once, in the
  benchmark's own index order -- no class is dropped for being rare or hard, and no
  nuisance class (sky, "background", "object") is invented;
* the phrase for a class is its canonical benchmark name with **mechanical** normalisation
  only (``-``/``_`` to space, plus the eight rewrites the Gate-6 brief enumerates). No
  phrase was chosen, scored or swapped after looking at a target;
* the empty/free class is **never** in the vocabulary. Occupancy comes from the frozen
  geometry; the teacher only names voxels that geometry has already produced.

Provenance of the class lists (read from the official code, not from memory):

    SemanticKITTI        ``learning_map_inv`` of the official ``semantic-kitti.yaml``
                         (CGFormer ``tools/semantic-kitti.yaml``), classes 1..19.
    SSCBench-KITTI-360   ``class_names`` of the official CGFormer KITTI-360 config,
                         classes 1..18.
    Occ3D-nuScenes       the official Occ3D-nuScenes ontology, classes 0..16
                         (17 = free, excluded).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Official benchmark label names, keyed by the benchmark's own integer label.
# --------------------------------------------------------------------------- #
SEMANTICKITTI_LABELS: Dict[int, str] = {
    1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle",
    6: "person", 7: "bicyclist", 8: "motorcyclist", 9: "road", 10: "parking",
    11: "sidewalk", 12: "other-ground", 13: "building", 14: "fence",
    15: "vegetation", 16: "trunk", 17: "terrain", 18: "pole", 19: "traffic-sign",
}

KITTI360_LABELS: Dict[int, str] = {
    1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle",
    6: "person", 7: "road", 8: "parking", 9: "sidewalk", 10: "other-ground",
    11: "building", 12: "fence", 13: "vegetation", 14: "terrain", 15: "pole",
    16: "traffic-sign", 17: "other-structure", 18: "other-object",
}

OCC3D_LABELS: Dict[int, str] = {
    0: "others", 1: "barrier", 2: "bicycle", 3: "bus", 4: "car",
    5: "construction_vehicle", 6: "motorcycle", 7: "pedestrian", 8: "traffic_cone",
    9: "trailer", 10: "truck", 11: "driveable_surface", 12: "other_flat",
    13: "sidewalk", 14: "terrain", 15: "manmade", 16: "vegetation",
}

# The empty/free label of each benchmark. Never a vocabulary entry.
EMPTY_LABEL = {"semantickitti": 0, "kitti360": 0, "occ3d": 17}
IGNORE_LABEL = {"semantickitti": 255, "kitti360": 255, "occ3d": 255}

# --------------------------------------------------------------------------- #
# Phrases. The eight explicit rewrites the brief lists, then the mechanical rule.
# --------------------------------------------------------------------------- #
EXPLICIT_REWRITES: Dict[str, str] = {
    "other-vehicle": "other vehicle",
    "bicyclist": "person riding a bicycle",
    "motorcyclist": "person riding a motorcycle",
    "traffic-sign": "traffic sign",
    "construction_vehicle": "construction vehicle",
    "driveable_surface": "drivable road surface",
    "other_flat": "other flat ground",
    "manmade": "man-made structure",
}


def phrase_for(name: str) -> str:
    """Canonical benchmark name -> the single deterministic text phrase.

    Explicit rewrites first, then the purely mechanical ``-``/``_`` to space. No other
    transformation exists, so the mapping is reproducible from the class list alone.
    """
    if name in EXPLICIT_REWRITES:
        return EXPLICIT_REWRITES[name]
    return name.replace("-", " ").replace("_", " ")


DATASET_LABELS: Dict[str, Dict[int, str]] = {
    "semantickitti": SEMANTICKITTI_LABELS,
    "kitti360": KITTI360_LABELS,
    "occ3d": OCC3D_LABELS,
}


class Vocabulary:
    """The frozen ``[C]`` vocabulary of one benchmark.

    ``labels[c]`` is the benchmark's integer label of teacher channel ``c``; ``names[c]``
    the official name; ``phrases[c]`` the text handed to the teacher. Channel order is
    ascending benchmark label, which makes the mapping total, injective and inspectable.
    """

    def __init__(self, dataset: str):
        if dataset not in DATASET_LABELS:
            raise KeyError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        table = DATASET_LABELS[dataset]
        self.labels: Tuple[int, ...] = tuple(sorted(table))
        self.names: Tuple[str, ...] = tuple(table[k] for k in self.labels)
        self.phrases: Tuple[str, ...] = tuple(phrase_for(n) for n in self.names)
        self.empty_label = EMPTY_LABEL[dataset]
        self.ignore_label = IGNORE_LABEL[dataset]
        if self.empty_label in self.labels:
            raise AssertionError("the empty class must never enter the vocabulary")
        if len(set(self.phrases)) != len(self.phrases):
            raise AssertionError("phrases must be unique so argmax is invertible")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def channel_of_label(self) -> Dict[int, int]:
        return {lab: c for c, lab in enumerate(self.labels)}

    def name_file_lines(self) -> List[str]:
        """The lines of a Trident ``name_path`` file: one phrase per class, in order."""
        return list(self.phrases)

    def to_dict(self) -> Dict[str, object]:
        return {"dataset": self.dataset, "num_classes": len(self),
                "empty_label": self.empty_label, "ignore_label": self.ignore_label,
                "labels": list(self.labels), "names": list(self.names),
                "phrases": list(self.phrases),
                "channel_of_label": {str(k): v for k, v in self.channel_of_label.items()}}


def load(dataset: str) -> Vocabulary:
    return Vocabulary(dataset)


DATASETS: Sequence[str] = ("semantickitti", "occ3d", "kitti360")

__all__ = ["Vocabulary", "load", "phrase_for", "DATASETS", "DATASET_LABELS",
           "EXPLICIT_REWRITES", "EMPTY_LABEL", "IGNORE_LABEL",
           "SEMANTICKITTI_LABELS", "KITTI360_LABELS", "OCC3D_LABELS"]
