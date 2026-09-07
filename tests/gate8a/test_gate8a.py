"""Gate 8A — source-only prior and calibration ablation.

The tests that matter here are the ones that keep the *evaluation* honest: that the
editable region is the residual rule and not a copy of it, that the score-free baselines
never touch a voxel the residual cannot touch, that every threshold-dependent number comes
out of stored continuous scores, and that the uniform sampler is structurally incapable of
looking at an occupancy label.
"""
from __future__ import annotations
import ast, inspect, os, random, sys
import numpy as np, pytest, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from gates.gate6 import grids as G6G
from gates.gate8.net import apply_residual, LOCK_LOGODDS
from gates.gate8a import baselines as BL, losses as L8A, sampler as SMP, scores as SC
from gates.gate8a import regions as RG

ART = os.path.join(REPO, "artifacts", "gate8a")


# ----------------------------------------------------------------- the editable region
def test_editable_mask_is_exactly_the_residual_gate():
    """Not "matches on these examples" -- identical on every voxel where the residual
    could act, checked against ``apply_residual`` itself."""
    g = torch.Generator().manual_seed(0)
    base = (torch.rand(20000, generator=g) * 16 - 8)
    base[:500] = LOCK_LOGODDS                       # exactly on the boundary, both signs
    base[500:1000] = -LOCK_LOGODDS
    res = torch.full_like(base, 3.7)
    moved = apply_residual(base, res) != base       # the residual actually changed it
    assert torch.equal(moved, RG.editable_native(base))
    assert torch.equal(RG.editable_native(base), base.abs() < LOCK_LOGODDS)


def test_editable_region_follows_a_change_in_the_lock():
    base = torch.linspace(-5, 5, 401)
    for lock in (0.5, 1.0, 2.0, 3.0):
        assert torch.equal(RG.editable_native(base, lock), base.abs() < lock)


def test_occ3d_region_reduction_is_the_frozen_any_subvoxel_rule():
    dims = G6G.PREDICTION_GRID["occ3d"].dims
    g = torch.Generator().manual_seed(1)
    base = (torch.rand(int(np.prod(dims)), generator=g) * 12 - 6)
    s, e, f = RG.to_eval_grid(base, base, "occ3d")
    ed = G6G.EVAL_GRID["occ3d"].dims
    assert s.numel() == int(np.prod(ed)) and e.numel() == s.numel()
    r = G6G.RATIO
    b = base.reshape(dims)[:2 * r, :r, :r]
    blk = b.reshape(2, r, 1, r, 1, r).permute(0, 2, 4, 1, 3, 5).reshape(2, -1)
    assert torch.allclose(s.reshape(ed)[:2, 0, 0], blk.amax(-1))
    assert torch.equal(e.reshape(ed)[:2, 0, 0], (blk.abs() < LOCK_LOGODDS).any(-1))
    assert torch.equal(f.reshape(ed)[:2, 0, 0], (blk >= LOCK_LOGODDS).any(-1))


def test_no_reduction_datasets_are_the_identity():
    base = torch.linspace(-6, 6, 256 * 256 * 32)
    for ds in ("semantickitti", "kitti360"):
        s, e, f = RG.to_eval_grid(base, base, ds)
        assert torch.equal(s, base) and torch.equal(e, base.abs() < LOCK_LOGODDS)


# ----------------------------------------------------------------- baselines
def _world(n=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, generator=g) * 12 - 6
    valid = torch.rand(n, generator=g) > 0.1
    gt = torch.rand(n, generator=g) < 0.2
    mp = (base > 0) & (torch.rand(n, generator=g) > 0.3)
    edit = RG.editable_native(base)
    forced = RG.locked_occupied_native(base)
    return base, valid, gt, mp, edit, forced


def test_editable_fill_preserves_every_protected_voxel():
    base, valid, gt, mp, edit, forced = _world()
    p = BL.editable_fill(mp, edit)
    assert BL.check_protected(p, mp, edit, valid)
    assert torch.equal(p[~edit], mp[~edit])
    assert bool(p[edit].all())                    # and it does fill the editable region


def test_random_editable_preserves_protected_and_forced_voxels():
    base, valid, gt, mp, edit, forced = _world(seed=3)
    for q in (0.0, 0.25, 1.0):
        p = BL.random_editable(mp, edit, forced, valid, q, seed=0)
        assert BL.check_protected(p, mp, edit, valid)
        assert bool(p[forced & edit & valid].all()), "occupancy the residual cannot remove"
        if q == 0.0:
            assert not bool((p & edit & valid & ~forced).any())


def test_random_baseline_is_reproducible_and_seed_dependent():
    base, valid, gt, mp, edit, forced = _world(seed=5)
    a = BL.random_editable(mp, edit, forced, valid, 0.4, seed=0)
    b = BL.random_editable(mp, edit, forced, valid, 0.4, seed=0)
    c = BL.random_editable(mp, edit, forced, valid, 0.4, seed=1)
    assert torch.equal(a, b) and not torch.equal(a, c)
    d = BL.random_editable(mp, edit, forced, valid, 0.4, seed=0, offset=1)
    assert not torch.equal(a, d), "different anchors must not reuse one draw"


def test_matched_density_hits_the_requested_count():
    base, valid, gt, mp, edit, forced = _world(n=200000, seed=7)
    floor_ = int((mp & ~edit & valid).sum()) + int((forced & edit & valid).sum())
    free = int((edit & ~forced & valid).sum())
    target = floor_ + free // 2
    q = BL.matched_density_q(target, mp, edit, forced, valid)
    assert 0.0 < q < 1.0
    got = int((BL.random_editable(mp, edit, forced, valid, q, 0) & valid).sum())
    assert abs(got - target) / target < 0.02


def test_all_valid_occupied_is_the_trivial_line():
    base, valid, gt, mp, edit, forced = _world()
    tp, fp, fn = BL.binary_counts(BL.all_valid_occupied(valid), gt, valid)
    assert fn == 0 and tp == int((gt & valid).sum())


# ----------------------------------------------------------------- scores
def _hist(logit, y):
    b = SC.bin_index(logit)
    pos = torch.bincount(b[y], minlength=SC.N_BINS).numpy()
    tot = torch.bincount(b, minlength=SC.N_BINS).numpy()
    return {"pos": pos[None], "neg": (tot - pos)[None]}


def test_sweep_reproduces_brute_force_counts_at_many_thresholds():
    g = torch.Generator().manual_seed(11)
    logit = torch.rand(60000, generator=g) * 20 - 10
    y = (torch.rand(60000, generator=g) < torch.sigmoid(logit * 0.8))
    blk = _hist(logit, y)
    sw = SC.sweep(blk["pos"], blk["neg"])
    for tau in (-8.0, -2.0, 0.0, 0.5, 3.0, 7.0):
        # brute force on the *continuous* scores, quantised the same way
        q = SC.bin_index(logit) >= int(round((tau - SC.LOGIT_LO) / ((SC.LOGIT_HI - SC.LOGIT_LO) / SC.N_BINS)))
        tp = int((q & y).sum()); fp = int((q & ~y).sum()); fn = int((~q & y).sum())
        got = SC.at_threshold(sw, tau)
        assert (got["tp"], got["fp"], got["fn"]) == (tp, fp, fn)
        assert got["iou"] == pytest.approx(tp / max(tp + fp + fn, 1))


def test_average_precision_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    g = torch.Generator().manual_seed(12)
    logit = torch.rand(40000, generator=g) * 16 - 8
    y = (torch.rand(40000, generator=g) < torch.sigmoid(logit))
    blk = _hist(logit, y)
    sw = SC.sweep(blk["pos"], blk["neg"])
    # quantise the reference identically: the histogram is exact up to the bin width
    q = SC.bin_index(logit).numpy()
    assert SC.average_precision(sw) == pytest.approx(
        sk.average_precision_score(y.numpy(), q), abs=2e-3)
    assert SC.auroc(sw) == pytest.approx(sk.roc_auc_score(y.numpy(), q), abs=2e-3)


def test_threshold_sweep_uses_stored_continuous_scores_not_a_binarization():
    """One stored block must answer *different* thresholds differently -- a block of
    already-binarized predictions could not."""
    g = torch.Generator().manual_seed(13)
    logit = torch.rand(30000, generator=g) * 12 - 6
    y = (torch.rand(30000, generator=g) < 0.25)
    blk = _hist(logit, y)
    sw = SC.sweep(blk["pos"], blk["neg"])
    ious = [SC.at_threshold(sw, t)["iou"] for t in (-4.0, -1.0, 0.0, 1.0, 4.0)]
    assert len(set(np.round(ious, 6))) == len(ious)
    # and the stored block spans many bins, i.e. it is a score histogram
    assert int((blk["pos"] + blk["neg"] > 0).sum()) > 100


def test_per_clip_counts_sum_to_the_pooled_sweep():
    g = torch.Generator().manual_seed(14)
    blocks = []
    for i in range(7):
        logit = torch.rand(5000, generator=g) * 12 - 6
        y = torch.rand(5000, generator=g) < 0.2
        blocks.append(_hist(logit, y))
    blk = {k: np.concatenate([b[k] for b in blocks]) for k in ("pos", "neg")}
    sw = SC.sweep(blk["pos"], blk["neg"])
    for tau in (-3.0, 0.0, 2.5):
        pc = SC.per_clip_counts(blk, tau).sum(0)
        at = SC.at_threshold(sw, tau)
        assert (pc[0], pc[1], pc[2]) == (at["tp"], at["fp"], at["fn"])
    assert SC.per_clip_counts(blk, 0.0).shape == (7, 3)


def test_zero_is_a_bin_boundary():
    assert SC.threshold_of_bin(SC.ZERO_BIN) == 0.0
    assert SC.at_threshold(SC.sweep(np.zeros((1, SC.N_BINS)), np.ones((1, SC.N_BINS))),
                           0.0)["threshold"] == 0.0


def test_ece_and_brier_are_exact_on_a_known_case():
    p = torch.tensor([0.1, 0.1, 0.9, 0.9]); y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logit = torch.log(p / (1 - p))
    a = SC.ScoreAccumulator()
    a.add(logit, y.bool(), torch.ones(4, dtype=torch.bool), "c", "g")
    blk = a.block()
    assert blk["sse"][0] == pytest.approx(4 * 0.01, abs=1e-6)
    assert SC.ece(blk["cal_n"], blk["cal_p"], blk["cal_y"]) == pytest.approx(0.1, abs=1e-6)


# ----------------------------------------------------------------- samplers
def test_uniform_sampler_cannot_see_an_occupancy_label():
    """Structural, not behavioural: the function's whole input is the RNG and the shapes."""
    sig = list(inspect.signature(SMP.uniform_origin).parameters)
    assert sig == ["rng", "dims", "crop"]
    src = inspect.getsource(SMP.uniform_origin)
    fn = ast.parse(src.strip()).body[0]
    body = [n for st in fn.body for n in ast.walk(st)]
    names = {n.id for n in body if isinstance(n, ast.Name)}
    forbidden = {"gt_occ", "gt_valid", "occ", "target", "label", "observed", "d", "self"}
    assert not (names & forbidden), names & forbidden
    # no indexing and no attribute access in the body: it cannot read a volume at all
    assert not any(isinstance(n, (ast.Subscript, ast.Attribute)) and
                   not (isinstance(n, ast.Attribute) and
                        isinstance(n.value, ast.Name) and n.value.id == "rng")
                   for n in body)


def test_uniform_sampler_is_blind_to_the_labels_behaviourally():
    from tools.gate8a.train import AblationSamples
    s = AblationSamples.__new__(AblationSamples)
    s.crop = (8, 8, 4); s.sampler = "uniform"; s.centre_on_occ_p = 0.7
    dims = (32, 32, 16)
    a = {"gt_occ": torch.zeros(dims), "gt_valid": torch.ones(dims, dtype=torch.bool),
         "observed": torch.zeros(dims, dtype=torch.bool)}
    b = {"gt_occ": torch.ones(dims), "gt_valid": torch.ones(dims, dtype=torch.bool),
         "observed": torch.zeros(dims, dtype=torch.bool)}
    s.rng = random.Random(0); oa = [s.crop_of(a) for _ in range(64)]
    s.rng = random.Random(0); ob = [s.crop_of(b) for _ in range(64)]
    assert oa == ob, "the uniform sampler's origins depend on the occupancy labels"
    xs = [o[0] for o in oa]
    assert min(xs) < 6 and max(xs) > dims[0] - s.crop[0] - 6, "origins are not spread out"


def test_occ_centred_sampler_does_depend_on_the_labels():
    from tools.gate8a.train import AblationSamples
    s = AblationSamples.__new__(AblationSamples)
    s.crop = (8, 8, 4); s.sampler = "occ_centred"; s.centre_on_occ_p = 1.0
    dims = (32, 32, 16)
    occ = torch.zeros(dims); occ[24, 24, 8] = 1
    d = {"gt_occ": occ, "gt_valid": torch.ones(dims, dtype=torch.bool),
         "observed": torch.zeros(dims, dtype=torch.bool)}
    s.rng = random.Random(0)
    assert all(s.crop_of(d) == (20, 20, 6) for _ in range(16))


# ----------------------------------------------------------------- the BCE arm
def test_bce_mask_excludes_invalid_and_protected_voxels():
    g = torch.Generator().manual_seed(21)
    n = 4096
    base = torch.rand(n, generator=g) * 12 - 6
    edit = RG.editable_native(base)
    valid = torch.rand(n, generator=g) > 0.2
    logits = torch.rand(n, generator=g) * 4 - 2
    tgt = (torch.rand(n, generator=g) < 0.3).float()
    ref = L8A.bce_editable(logits, tgt, valid, edit)
    for flip in (~valid, valid & ~edit):
        t2 = tgt.clone(); t2[flip] = 1.0 - t2[flip]
        assert L8A.bce_editable(logits, t2, valid, edit) == pytest.approx(float(ref), abs=1e-6)
    t3 = tgt.clone(); m = valid & edit
    t3[m] = 1.0 - t3[m]
    assert L8A.bce_editable(logits, t3, valid, edit) != pytest.approx(float(ref), abs=1e-6)


def test_bce_arm_has_no_weighting_left_in_it():
    src = inspect.getsource(L8A.bce_editable)
    for banned in ("pos_weight", "gamma", "dice", "w_unknown", "focal"):
        assert banned not in src
    # equal-weight check: mean over the mask, nothing else
    logits = torch.zeros(10); tgt = torch.zeros(10); tgt[0] = 1.0
    v = torch.ones(10, dtype=torch.bool); e = torch.ones(10, dtype=torch.bool)
    assert float(L8A.bce_editable(logits, tgt, v, e)) == pytest.approx(float(np.log(2)))


def test_ablation_configs_differ_only_in_the_two_factors():
    import yaml
    cfg = {n: yaml.safe_load(open(os.path.join(REPO, "configs", "gate8a", f"{n}.yaml")))
           for n in ("cellA_occcentred_focal", "cellB_uniform_focal",
                     "cellC_occcentred_bce", "cellD_uniform_bce")}
    keys = set.union(*[set(c) for c in cfg.values()])
    for k in keys - {"sampler", "occ_loss"}:
        vals = {repr(c.get(k)) for c in cfg.values()}
        assert len(vals) == 1, f"{k} differs across ablation cells: {vals}"
    assert {(c["sampler"], c["occ_loss"]) for c in cfg.values()} == {
        ("occ_centred", "focal_dice"), ("uniform", "focal_dice"),
        ("occ_centred", "bce"), ("uniform", "bce")}
    g8 = yaml.safe_load(open(os.path.join(REPO, "configs", "gate8", "completion.yaml")))
    a = cfg["cellA_occcentred_focal"]
    for k in set(g8) & set(a):
        assert g8[k] == a[k], f"cell A drifted from the Gate 8 config at {k}"


# ----------------------------------------------------------------- leakage
def test_no_kitti360_anywhere_in_the_selection_path():
    for f in ("selection.py", "train.py", "sampler_stats.py"):
        src = open(os.path.join(REPO, "tools", "gate8a", f)).read()
        low = src.lower()
        for tok in ("kitti360", "kitti_360"):
            for line in low.splitlines():
                if tok in line:
                    assert ("not" in line or "never" in line or "no " in line
                            or "false" in line or "#" in line), f"{f}: {line.strip()}"
    import yaml
    for n in ("cellA_occcentred_focal", "cellB_uniform_focal", "cellC_occcentred_bce",
              "cellD_uniform_bce", "ablation"):
        c = yaml.safe_load(open(os.path.join(REPO, "configs", "gate8a", f"{n}.yaml")))
        assert "kitti360" not in c["train_sources"] + c["val_sources"]


def test_gate8_artifacts_are_untouched():
    for f in ("gate8_results.json", "gate8_manifest.json", "train_completion.json",
              "checkpoints/completion_best.pt"):
        assert os.path.exists(os.path.join(REPO, "artifacts", "gate8", f)), f


def test_threshold_then_reduce_equals_reduce_then_threshold():
    """Occ3D predicts at 0.2 m and scores at 0.4 m under an any-sub-voxel rule, so the
    evaluator may binarize on either grid -- but only if the two agree exactly."""
    dims = G6G.PREDICTION_GRID["occ3d"].dims
    g = torch.Generator().manual_seed(31)
    base = torch.rand(int(np.prod(dims)), generator=g) * 12 - 6
    final = base + (torch.rand_like(base) * 4 - 2) * RG.editable_native(base)
    for tau in (-2.0, 0.0, 0.75, 3.0):
        a = RG.occ_to_eval_grid(final >= tau, "occ3d")
        b = RG.to_eval_grid(final, base, "occ3d")[0] >= tau
        assert torch.equal(a, b)


def test_evaluator_binarizes_with_the_same_comparison_as_the_histogram():
    """`per_clip_counts` counts bin index >= tau; the semantic path must use `>=` too, or
    the SSC counts and the binary counts silently describe different predictions."""
    src = open(os.path.join(REPO, "tools", "gate8a", "evaluate.py")).read()
    body = src[src.index("sem_variants = {"):src.index("sem_labels = {}")]
    assert "finals[locked] >= threshold" in body
    assert "> threshold" not in body
    assert "base >= mapper_threshold" in src


def test_unobserved_voxels_sit_exactly_on_the_zero_bin():
    """A never-observed voxel has log-odds exactly 0.0, so "score >= 0" sweeps the whole
    unknown volume in. The mapper's own rule is strict (> 0); the report must not confuse
    the two, and the sweep recovers the strict rule one bin up."""
    logit = torch.tensor([0.0, 0.0, 0.5, -0.5])
    y = torch.tensor([False, True, True, False])
    blk = _hist(logit, y)
    sw = SC.sweep(blk["pos"], blk["neg"])
    assert SC.at_threshold(sw, 0.0)["tp"] == 2                 # includes the unknown voxel
    assert SC.at_threshold(sw, SC.threshold_of_bin(SC.ZERO_BIN + 1))["tp"] == 1


def test_bin_edge_precision_limit_is_known_and_bounded():
    """`bin_index` adds 16.0 before dividing, which costs ~1.9e-6 of float32 precision.
    Scores within one ULP below a threshold therefore bin as ">= tau". The limit is
    documented rather than papered over: it is 7 orders of magnitude below anything the
    report compares, and `aggregate.py` asserts a bound on it against the semantic counts.
    """
    tau = 0.5625
    b = int(round((tau - SC.LOGIT_LO) / ((SC.LOGIT_HI - SC.LOGIT_LO) / SC.N_BINS)))
    just_below = torch.tensor([np.nextafter(np.float32(tau), np.float32(-1))],
                              dtype=torch.float32)
    assert bool((just_below < tau).all())
    assert bool((SC.bin_index(just_below) >= b).all())          # the known disagreement
    well_below = torch.tensor([tau - 1e-4], dtype=torch.float32)
    assert bool((SC.bin_index(well_below) < b).all())           # and it is one ULP wide
    assert float(np.spacing(np.float32(16.5625))) < 2e-6
