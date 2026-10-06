"""The surround setting's own logic: window handling, the 360-degree label rule, and the setting table.

Nothing here touches a GPU or a model checkpoint. What these tests pin down is the part of the surround path that is not
already covered by ``tools/like_for_like/verify.py`` (which checks the shared prediction body against the frozen
predictions array for array).
"""
from __future__ import annotations

import numpy as np
import torch

from tools.like_for_like import evaluate as LE
from tools.like_for_like.evidence import WindowEvidence
from tools.occany_comparison import config as OC


def _evidence(bits, n_slots, anchor_slot):
    n = len(bits)
    return WindowEvidence(torch.arange(n), torch.zeros(n, 1), torch.ones(n, dtype=torch.int32),
                          torch.tensor(bits, dtype=torch.uint8), anchor_slot=anchor_slot, n_slots=n_slots)


def test_n_frames_counts_every_slot_of_the_window():
    """A voxel seen in all six surround views must count six frames, not five."""
    ev = _evidence([0b111111, 0b100001, 0b000001], 6, 0)
    out = ev.scorer_inputs(torch.zeros(1, 3), torch.zeros(3, 3))
    assert out["n_frames"].tolist() == [6, 2, 1]
    assert out["current"].tolist() == [True, True, True]          # anchor slot 0 = CAM_FRONT


def test_current_flag_reads_the_anchor_slot():
    """``current`` means "seen by the anchor camera", which is slot 0 here and slot 4 in the causal window."""
    assert _evidence([0b000010], 6, 0).scorer_inputs(torch.zeros(1, 1), torch.zeros(1, 3))["current"].tolist() == [False]
    assert _evidence([0b010000], 5, 4).scorer_inputs(torch.zeros(1, 1), torch.zeros(1, 3))["current"].tolist() == [True]


def test_five_slot_window_is_unchanged():
    """The default stays exactly the frozen five-slot reading, so neither temporal protocol moves."""
    ev = _evidence([0b11111], 5, 4)
    assert ev.n_slots == 5 and ev.scorer_inputs(torch.zeros(1, 1), torch.zeros(1, 3))["n_frames"].tolist() == [5]


def test_settings_table_pairs_predictions_with_ground_truth():
    """Each evaluated setting names its prediction source and its ground truth; only the free row mixes them."""
    assert LE.SETTINGS["forward"] == ("forward", "occany")
    assert LE.SETTINGS["surround"] == ("surround", "surround")
    assert LE.SETTINGS["forward_surround_label"] == ("forward", "surround"), "the coverage-only row reuses forward predictions"


def test_occany_published_occ3d_row_is_the_surround_setting():
    """Guard against comparing our single-camera number to OccAny's only nuScenes semantic number again."""
    assert "surround" in LE.OCCANY_PUBLISHED["occ3d"]["setting"]
    assert "1 camera" in LE.OCCANY_PUBLISHED["semantickitti"]["setting"]


def test_ring_order_starts_at_the_anchor_camera_and_visits_each_once():
    from tools.like_for_like import surround as SR, surround_clips as SCL
    assert SR.RING[0] == "CAM_FRONT" and SCL.ANCHOR_SLOT == 0
    assert len(set(SR.RING)) == 6
    # neighbouring entries must be adjacent views, so the sequence model always has overlap with the previous slot
    opposite = {"CAM_FRONT": "CAM_BACK", "CAM_BACK": "CAM_FRONT",
                "CAM_FRONT_LEFT": "CAM_BACK_RIGHT", "CAM_BACK_RIGHT": "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT": "CAM_BACK_LEFT", "CAM_BACK_LEFT": "CAM_FRONT_RIGHT"}
    for a, b in zip(SR.RING, SR.RING[1:]):
        assert opposite[a] != b, f"{a} and {b} face opposite ways and share no overlap"


def test_surround_label_restores_exactly_the_rear_half():
    """On a real sample: identical to OccAny's stored label on the front half, valid where that label ignores the rear.

    This is the claim the Occ3D surround comparison rests on -- that our 360-degree ground truth is OccAny's
    ``use_surround_label=True`` label and differs from their single-camera one only by the rear cut."""
    from tools.like_for_like import frames as LF, surround as SR
    from tools.occany_comparison import scoring as SC
    samples = LF.samples("occ3d")
    s = samples[0]
    full = SR.surround_label(s)
    single = SC.load_occany("occ3d", s.occany_pkl, full.shape).label
    assert np.array_equal(full[100:], single[100:]), "front half must be OccAny's own label, voxel for voxel"
    assert (single[:100] == OC.IGNORE).all(), "OccAny's 5frames label ignores the rear half"
    assert (full[:100] != OC.IGNORE).any(), "the surround label must score rear voxels"
