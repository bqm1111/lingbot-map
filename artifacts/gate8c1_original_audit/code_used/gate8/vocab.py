"""The declared union vocabulary, and the fixed maps between it and each benchmark.

Trident-H is cached as *vocabulary-conditioned probability maps*, one vocabulary per
benchmark. Training one semantic head across two source benchmarks therefore needs one
shared label space. This module declares it **once, before any result**, together with the
many-to-one maps *into* it (used when fusing teacher evidence) and the one-to-one maps
*out of* it (used only at evaluation, to score in a benchmark's own classes).

This is stated plainly wherever it matters: the first version is **not query-time
open-vocabulary**. The head predicts a fixed 25-way distribution over the union. A later
version must replace it with fixed-dimensional language-aligned features.

The held-out mapping ``UNION -> kitti360`` was written from class *names* only, at the
same time as the others, and no KITTI-360 result was observed before it was fixed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from gate6 import vocab as G6V

UNION: List[str] = [
    "car", "bicycle", "motorcycle", "truck", "other vehicle", "bus",
    "construction vehicle", "trailer", "person", "rider", "road", "parking", "sidewalk",
    "other ground", "building", "man-made structure", "fence", "vegetation", "trunk",
    "terrain", "pole", "traffic sign", "barrier", "traffic cone", "other object",
]
U = len(UNION)
_UI = {n: i for i, n in enumerate(UNION)}

#: benchmark phrase -> union name (many-to-one)
INTO: Dict[str, Dict[str, str]] = {
    "semantickitti": {
        "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle", "truck": "truck",
        "other vehicle": "other vehicle", "person": "person",
        "person riding a bicycle": "rider", "person riding a motorcycle": "rider",
        "road": "road", "parking": "parking", "sidewalk": "sidewalk",
        "other ground": "other ground", "building": "building", "fence": "fence",
        "vegetation": "vegetation", "trunk": "trunk", "terrain": "terrain", "pole": "pole",
        "traffic sign": "traffic sign"},
    "occ3d": {
        "others": "other object", "barrier": "barrier", "bicycle": "bicycle", "bus": "bus",
        "car": "car", "construction vehicle": "construction vehicle",
        "motorcycle": "motorcycle", "pedestrian": "person", "traffic cone": "traffic cone",
        "trailer": "trailer", "truck": "truck", "drivable road surface": "road",
        "other flat ground": "other ground", "sidewalk": "sidewalk", "terrain": "terrain",
        "man-made structure": "man-made structure", "vegetation": "vegetation"},
    "kitti360": {
        "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle", "truck": "truck",
        "other vehicle": "other vehicle", "person": "person", "road": "road",
        "parking": "parking", "sidewalk": "sidewalk", "other ground": "other ground",
        "building": "building", "fence": "fence", "vegetation": "vegetation",
        "terrain": "terrain", "pole": "pole", "traffic sign": "traffic sign",
        "other structure": "man-made structure", "other object": "other object"},
}

#: union name -> benchmark phrase (one-to-one, evaluation only)
OUT: Dict[str, Dict[str, str]] = {
    "semantickitti": {
        "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle", "truck": "truck",
        "other vehicle": "other vehicle", "bus": "other vehicle",
        "construction vehicle": "other vehicle", "trailer": "other vehicle",
        "person": "person", "rider": "person riding a bicycle", "road": "road",
        "parking": "parking", "sidewalk": "sidewalk", "other ground": "other ground",
        "building": "building", "man-made structure": "building", "fence": "fence",
        "vegetation": "vegetation", "trunk": "trunk", "terrain": "terrain", "pole": "pole",
        "traffic sign": "traffic sign", "barrier": "fence", "traffic cone": "pole",
        "other object": "other ground"},
    "occ3d": {
        "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle", "truck": "truck",
        "other vehicle": "others", "bus": "bus",
        "construction vehicle": "construction vehicle", "trailer": "trailer",
        "person": "pedestrian", "rider": "pedestrian", "road": "drivable road surface",
        "parking": "other flat ground", "sidewalk": "sidewalk",
        "other ground": "other flat ground", "building": "man-made structure",
        "man-made structure": "man-made structure", "fence": "man-made structure",
        "vegetation": "vegetation", "trunk": "vegetation", "terrain": "terrain",
        "pole": "man-made structure", "traffic sign": "man-made structure",
        "barrier": "barrier", "traffic cone": "traffic cone", "other object": "others"},
    "kitti360": {
        "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle", "truck": "truck",
        "other vehicle": "other vehicle", "bus": "other vehicle",
        "construction vehicle": "other vehicle", "trailer": "other vehicle",
        "person": "person", "rider": "person", "road": "road", "parking": "parking",
        "sidewalk": "sidewalk", "other ground": "other ground", "building": "building",
        "man-made structure": "other structure", "fence": "fence",
        "vegetation": "vegetation", "trunk": "vegetation", "terrain": "terrain",
        "pole": "pole", "traffic sign": "traffic sign", "barrier": "other structure",
        "traffic cone": "other object", "other object": "other object"},
}


def into_matrix(dataset: str) -> np.ndarray:
    """``[C_dataset, U]`` 0/1 matrix: teacher channel -> union class."""
    v = G6V.load(dataset)
    M = np.zeros((len(v), U), np.float32)
    for c, ph in enumerate(v.phrases):
        M[c, _UI[INTO[dataset][ph]]] = 1.0
    assert (M.sum(1) == 1).all(), dataset
    return M


def out_channel(dataset: str) -> np.ndarray:
    """``[U]`` int: union class -> teacher channel of the benchmark (evaluation only)."""
    v = G6V.load(dataset)
    ch = {ph: c for c, ph in enumerate(v.phrases)}
    return np.asarray([ch[OUT[dataset][u]] for u in UNION], np.int64)


def check() -> None:
    for ds in INTO:
        v = G6V.load(ds)
        assert set(INTO[ds]) == set(v.phrases), ds
        assert set(OUT[ds]) == set(UNION), ds
        assert all(p in v.phrases for p in OUT[ds].values()), ds
        into_matrix(ds); out_channel(ds)


__all__ = ["UNION", "U", "INTO", "OUT", "into_matrix", "out_channel", "check"]
