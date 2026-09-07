# extracted from gate8a/sampler.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Crop-origin samplers for the 2x2 ablation.

``occ_centred`` is Gate 8's sampler verbatim: with probability 0.7 it centres the crop on
a ground-truth-occupied, still-unobserved voxel. That is a label-conditioned sampler, and
it is the leading suspect for Gate 8's over-prediction -- it presents the network with a
crop prior far denser than any benchmark.

``uniform`` draws the origin uniformly over the valid benchmark lattice. It is written as
a function of the *volume shape and the RNG only*: it receives no labels, no occupancy and
no map state, which is what the test asserts by inspecting its signature and its source.
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

SAMPLERS = ("occ_centred", "uniform")


def uniform_origin(rng: random.Random, dims: Sequence[int], crop: Sequence[int]
                   ) -> Tuple[int, int, int]:
    """Uniform crop origin over the lattice. Sees only ``dims`` and ``crop``."""
    return tuple(rng.randrange(int(n) - int(w) + 1) for n, w in zip(dims, crop))


__all__ = ["uniform_origin", "SAMPLERS"]
