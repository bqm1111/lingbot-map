"""The frozen geometry, the small semantic-completion head, the supervision set, the two
trivial baselines and the per-voxel metrics. Everything the training, source and transfer
scripts share, so they cannot drift apart.

Gradient flow, stated once
--------------------------
``FrozenGeometry`` loads the adopted Gate 8C-1 checkpoint, sets ``requires_grad=False`` on
every parameter and runs the whole network -- trunk, occupancy head and the original
semantic head -- under ``torch.no_grad``. The new head reads the detached decoder feature
``d1`` (24 channels) concatenated with the 32-channel causal input. No path exists from the
semantic loss to any frozen parameter or to the occupancy logits; ``verify.py`` checks the
parameter digests and the occupancy output before and after training anyway.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.datasets import union_vocab as V8
from core.model.net import CompletionUNet, load_checkpoint
from core.supervision.targets import N_INPUT_CHANNELS

U = V8.U


class SemanticCompletionHead(nn.Module):
    """Two padded 3x3x3 convolutions and a 1x1 classifier: ~77 k parameters, receptive
    field 5 voxels (1 m) on top of the trunk's. Same normalisation and padding as the
    trunk so the boundary convention is the adopted one."""

    def __init__(self, c_trunk: int = 24, c_in: int = N_INPUT_CHANNELS, width: int = 32,
                 n_sem: int = U, padding_mode: str = "replicate"):
        super().__init__()
        k = dict(kernel_size=3, padding=1, padding_mode=padding_mode)
        self.c_trunk, self.c_in, self.width, self.n_sem = c_trunk, c_in, width, n_sem
        self.padding_mode = padding_mode
        self.body = nn.Sequential(nn.Conv3d(c_trunk + c_in, width, **k), nn.GroupNorm(8, width),
                                  nn.GELU(), nn.Conv3d(width, width, **k),
                                  nn.GroupNorm(8, width), nn.GELU())
        self.out = nn.Conv3d(width, n_sem, 1)

    def forward(self, trunk: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(torch.cat([trunk, x], 1)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FrozenGeometry:
    """The adopted checkpoint, frozen. ``forward`` returns detached fp32 tensors."""

    def __init__(self, path: str, device):
        comp = load_checkpoint(path, device)
        self.net: CompletionUNet = comp.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.path = path
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(occ_residual [B,1,...], existing sem logits [B,U,...], trunk d1 [B,24,...])``
        under the adopted bf16 inference convention."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ_res, sem_log, trunk = self.net(x, return_trunk=True)
        return occ_res.float(), sem_log.float(), trunk.float()

    def state_digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for k, v in self.net.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()


def key_mask(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """currently unobserved -> future-observed -> LiDAR-occupied -> teacher target present."""
    return (~d["observed"]) & d["fut_observed"] & (d["gt_occ"] > 0) & d["gt_valid"] & d["fut_valid"]


def observed_mask(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The contrast population: map-observed, LiDAR-occupied, with a future teacher target."""
    return d["observed"] & (d["gt_occ"] > 0) & d["gt_valid"] & d["fut_valid"]


def loss_kl(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) averaged over the given voxels. ``logits/target_p: [n, U]``."""
    logq = F.log_softmax(logits.float(), dim=1)
    t = target_p.clamp_min(1e-6)
    return (t * (t.log() - logq)).sum(1).mean()


# ----------------------------------------------------------------------------- baselines
def b1_nearest_index(semw_grid: np.ndarray, max_dist: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """For every voxel: flat index of the nearest map voxel carrying teacher evidence
    (``sem_w > 0``) and the Euclidean distance to it in voxels. Nothing but the causal map
    is read. ``scipy.ndimage.distance_transform_edt`` on the complement of the evidence
    mask gives both in one pass."""
    from scipy import ndimage
    src = semw_grid > 0
    if not src.any():
        n = src.size
        return np.full(n, -1, np.int64), np.full(n, np.inf, np.float32)
    dist, ind = ndimage.distance_transform_edt(~src, return_indices=True)
    flat = np.ravel_multi_index((ind[0], ind[1], ind[2]), src.shape).reshape(-1)
    return flat.astype(np.int64), dist.reshape(-1).astype(np.float32)


def b1_predict(x: torch.Tensor, rows: torch.Tensor, b0: torch.Tensor, max_dist: float
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """B1 on the voxels ``rows`` of one full-grid input ``x [Cin, X, Y, Z]``: the probability
    vector of the nearest evidence-carrying voxel, or B0 when it is farther than
    ``max_dist``. Returns ``(probs [n, U], distance [n])``."""
    semw = x[6].cpu().numpy()                       # semw / 20, positive iff sem_w > 0
    near, dist = b1_nearest_index(semw, max_dist)
    near = torch.from_numpy(near).to(x.device); dist = torch.from_numpy(dist).to(x.device)
    probs_all = x[7:7 + U].reshape(U, -1).T          # [n_all, U] map class probabilities
    r = rows
    src = near[r]
    p = torch.where((src >= 0).unsqueeze(1), probs_all[src.clamp_min(0)], b0.expand(len(r), U))
    far = dist[r] > max_dist
    p = torch.where(far.unsqueeze(1), b0.expand(len(r), U), p)
    return p, dist[r]


# ------------------------------------------------------------------------------- metrics
COS_BINS = np.linspace(0.0, 1.0, 21)


def cosine(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(p.float(), t.float(), dim=1, eps=1e-8)


class SetAccumulator:
    """Per-voxel metrics over one voxel population, accumulated across anchors.

    Keeps per-anchor sums so paired bootstrap over anchors can be run afterwards, plus a
    [U, U] argmax confusion (rows = teacher, cols = prediction), a cosine histogram and the
    second moment of predictions and targets for the collapse diagnostic.
    """

    def __init__(self, name: str):
        self.name = name
        self.anchors = []            # (anchor_id, group, n, sum_cos, sum_kl, n_top1, sum_cos_b0_pred, sum_cos_b0_tgt)
        self.conf = np.zeros((U, U), np.int64)
        self.cos_hist = np.zeros(len(COS_BINS) - 1, np.int64)
        self.pred_sum = np.zeros(U, np.float64); self.pred_sq = 0.0
        self.tgt_sum = np.zeros(U, np.float64); self.tgt_sq = 0.0
        self.n = 0

    @torch.no_grad()
    def add(self, anchor_id, group, p: torch.Tensor, t: torch.Tensor, b0: torch.Tensor):
        n = int(p.shape[0])
        if n == 0:
            self.anchors.append((str(anchor_id), str(group), 0, 0.0, 0.0, 0, 0.0, 0.0))
            return
        c = cosine(p, t)
        logq = torch.log(p.clamp_min(1e-6)); tt = t.clamp_min(1e-6)
        kl = (tt * (tt.log() - logq)).sum(1)
        pa, ta = p.argmax(1), t.argmax(1)
        top1 = int((pa == ta).sum())
        cb0p = cosine(p, b0.expand_as(p)); cb0t = cosine(t, b0.expand_as(t))
        self.anchors.append((str(anchor_id), str(group), n, float(c.sum()), float(kl.sum()),
                             top1, float(cb0p.sum()), float(cb0t.sum())))
        idx = (ta * U + pa).cpu().numpy()
        self.conf += np.bincount(idx, minlength=U * U).reshape(U, U)
        self.cos_hist += np.histogram(c.cpu().numpy(), bins=COS_BINS)[0]
        pn = p.double(); tn = t.double()
        self.pred_sum += pn.sum(0).cpu().numpy(); self.pred_sq += float((pn * pn).sum())
        self.tgt_sum += tn.sum(0).cpu().numpy(); self.tgt_sq += float((tn * tn).sum())
        self.n += n

    def summary(self) -> Dict[str, object]:
        a = np.asarray([x[2:] for x in self.anchors], np.float64)
        n = max(self.n, 1)
        tp = np.diag(self.conf).astype(np.float64)
        sup = self.conf.sum(1).astype(np.float64); predn = self.conf.sum(0).astype(np.float64)
        den = sup + predn - tp
        iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
        present = sup > 0
        pm = self.pred_sum / n; tm = self.tgt_sum / n
        return {"n_voxels": int(self.n), "n_anchors": len(self.anchors),
                "cosine_mean": float(a[:, 1].sum() / n), "kl_mean": float(a[:, 2].sum() / n),
                "top1_agreement": float(a[:, 3].sum() / n),
                "cosine_to_b0_pred": float(a[:, 4].sum() / n),
                "cosine_to_b0_target": float(a[:, 5].sum() / n),
                "miou_present_classes": float(np.nanmean(iou[present])) if present.any() else None,
                "per_class": {V8.UNION[u]: {"support": int(sup[u]), "predicted": int(predn[u]),
                                            "iou": (None if np.isnan(iou[u]) else float(iou[u])),
                                            "recall": (float(tp[u] / sup[u]) if sup[u] else None)}
                              for u in range(U)},
                "pred_variance": float(self.pred_sq / n - (pm * pm).sum()),
                "target_variance": float(self.tgt_sq / n - (tm * tm).sum()),
                "pred_mean_vector": pm.tolist(), "target_mean_vector": tm.tolist(),
                "cosine_hist_bins": COS_BINS.tolist(), "cosine_hist": self.cos_hist.tolist()}

    def per_anchor(self) -> np.ndarray:
        """Structured rows for the bootstrap: id, group, n, sum_cos, sum_kl, n_top1."""
        return np.asarray([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in self.anchors],
                          dtype=[("id", "U64"), ("group", "U64"), ("n", "i8"),
                                 ("sum_cos", "f8"), ("sum_kl", "f8"), ("n_top1", "i8")])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, field: str, n_boot: int = 2000,
                     block: int = 20, by_group: bool = False, seed: int = 0) -> Dict[str, float]:
    """Paired difference of a ratio-of-sums metric (``sum_field / n``) between two
    conditions scored on identical per-anchor voxel sets. Resamples contiguous blocks of
    ``block`` anchors, or whole groups when ``by_group`` and >= 5 groups exist."""
    assert len(a) == len(b) and (a["n"] == b["n"]).all(), "populations differ"
    n = a["n"].astype(np.float64)
    da, db = a[field].astype(np.float64), b[field].astype(np.float64)
    if by_group:
        groups = np.unique(a["group"])
        if len(groups) >= 5:
            units = [np.flatnonzero(a["group"] == g) for g in groups]
        else:
            units = [np.arange(i, min(i + block, len(a))) for i in range(0, len(a), block)]
    else:
        units = [np.arange(i, min(i + block, len(a))) for i in range(0, len(a), block)]
    rng = np.random.default_rng(seed)
    point = (da.sum() - db.sum()) / max(n.sum(), 1)
    ds = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        ix = np.concatenate([units[k] for k in pick])
        ds.append((da[ix].sum() - db[ix].sum()) / max(n[ix].sum(), 1))
    ds = np.asarray(ds)
    return {"delta": float(point), "ci95": [float(np.percentile(ds, 2.5)),
                                              float(np.percentile(ds, 97.5))],
            "p_positive": float((ds > 0).mean()), "n_units": len(units)}


__all__ = ["SemanticCompletionHead", "FrozenGeometry", "key_mask", "observed_mask", "loss_kl",
           "b1_predict", "b1_nearest_index", "SetAccumulator", "paired_bootstrap", "cosine",
           "U"]
