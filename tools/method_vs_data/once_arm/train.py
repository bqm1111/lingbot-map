"""Train the frozen recipe on the ONCE arm.

Nothing about the recipe is re-implemented here. ``tools.waymo_arm.train.train_one_seed`` is
already a transcription of the hash-frozen ``baseline_reconstruction.current.train`` loop,
gated against it by ``tools/waymo_arm/cli/check_trainer_equivalence.py`` -- run the two on the
same KITTI-360 drives and every logged loss, metric and parameter digest must agree exactly.
This module supplies that loop with ONCE's caches and scene lists and changes nothing else:
same recipe yaml, same ``CompletionUNet``, same ``step_loss``, same optimiser and schedule
construction order, same ``HInputDataset``, same RNG discipline, same checkpoint rule.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from tools.baseline_reconstruction import config as BR
from tools.method_vs_data.once_arm import config as A


@dataclass(frozen=True)
class OnceArmPaths:
    """The paths the frozen dataset code and the training loop touch, pointed at this arm.

    ``HInputDataset`` and ``anchor_ids`` touch only ``input_path`` and ``target_path``; the
    loop additionally reads ``training_config`` -- which stays the KITTI-360 arm's recipe
    file, byte for byte -- plus ``checkpoints`` and ``runs``.
    """
    inputs: str = field(default_factory=lambda: A.Paths().inputs)
    targets: str = field(default_factory=lambda: A.Paths().targets)
    checkpoints: str = field(default_factory=lambda: A.Paths().checkpoints)
    runs: str = field(default_factory=lambda: A.Paths().runs)
    #: the recipe is the frozen one; this arm changes the data, not the recipe
    training_config: str = field(default_factory=lambda: BR.CurrentPaths().training_config)
    #: deliberately the KITTI-360 arm's checkpoint NAME, in this arm's own directory, so the
    #: frozen evaluation harness can be pointed at it with
    #: ``dataclasses.replace(CurrentPaths(), checkpoints=...)`` and nothing else.
    checkpoint_prefix: str = "geo"

    def input_path(self, scene: str, t: int) -> str:
        return os.path.join(self.inputs, scene, f"{t:05d}.npz")

    def target_path(self, scene: str, t: int) -> str:
        return os.path.join(self.targets, scene, f"{t:05d}.npz")

    def checkpoint_path(self, seed: int, step: int) -> str:
        return os.path.join(self.checkpoints,
                            f"{self.checkpoint_prefix}_seed{seed}_step{step:05d}.pt")


#: the shared trainer writes to ``paths.checkpoint_path`` only when ``tag_prefix ==
#: "waymo_geo"``; any other tag goes to a ``smoke``/ subdirectory keyed by tag. That guard
#: exists so the Waymo arm's trainer-equivalence runs cannot clobber its real checkpoints, and
#: it is deliberately left alone -- modifying that file would invalidate the gate that proves
#: this loop reproduces the frozen KITTI-360 trainer. :func:`publish_final` instead copies each
#: seed's final checkpoint to the canonical name the FROZEN evaluation harness looks for, so
#: ``dataclasses.replace(CurrentPaths(), checkpoints=...)`` works with nothing else changed.
BY_STEP_SUBDIR = "by_step"


def publish_final(seed: int, steps: int = 6000, paths: "OnceArmPaths" = None) -> dict:
    """Copy ``once_geo_seed{s}_step{N}.pt`` to the canonical ``geo_seed{s}_step{N}.pt``."""
    import hashlib
    import shutil
    paths = paths or OnceArmPaths()
    src = os.path.join(paths.checkpoints, BY_STEP_SUBDIR,
                       f"once_geo_seed{seed}_step{steps:05d}.pt")
    if not os.path.isfile(src):          # older runs land in the trainer's "smoke" subdir
        alt = os.path.join(paths.checkpoints, "smoke",
                           f"once_geo_seed{seed}_step{steps:05d}.pt")
        src = alt if os.path.isfile(alt) else src
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    dst = paths.checkpoint_path(seed, steps)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()           # noqa: E731
    assert h(src) == h(dst), "copy did not reproduce the checkpoint"
    return {"seed": seed, "steps": steps, "source": os.path.relpath(src, A.REPO_ROOT),
            "published": os.path.relpath(dst, A.REPO_ROOT), "sha256": h(dst)}


def train_one_seed(seed: int, device: str, steps: int = None, record_dir: str = None):
    """Delegate to the gated Waymo-arm loop with ONCE's scenes and caches."""
    from tools.waymo_arm.train import train_one_seed as _train
    sp = A.split(A.sequences())
    return _train(seed, device, sp["train"], sp["val"], paths=OnceArmPaths(), steps=steps,
                  tag_prefix="once_geo",
                  record_dir=record_dir or os.path.join(A.ARTIFACT_DIR, "runs"))


__all__ = ["OnceArmPaths", "train_one_seed", "publish_final", "BY_STEP_SUBDIR"]
