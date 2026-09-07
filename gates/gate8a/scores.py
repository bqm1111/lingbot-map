"""Threshold-free scoring: histogram accumulators and the metrics derived from them.

Gate 8 reported one operating point (``final_logodds > 0``) and therefore could not
separate "the completion ranks voxels badly" from "the completion ranks voxels fine and
is thresholded badly". Gate 8A stores the *continuous* final occupancy log-odds instead,
but storing 2.1-5.1 M floats per anchor for 3 100 anchors is 40 GB, so the scores are
accumulated into a **per-anchor histogram** over log-odds bins.

Everything the brief asks for is an exact function of that histogram: AP, PR curve,
AUROC, and IoU / precision / recall / predicted density at *every* threshold, including
per-anchor counts for the paired bootstrap -- no metric is ever recomputed from an
already-binarized prediction. Brier score and ECE are accumulated separately and exactly
(a running sum of squared error, and a 20-bin reliability table).

Bin edges are chosen so that ``0.0`` -- the frozen decision boundary of both the mapper
and Gate 8's completion -- is exactly a bin boundary, so "threshold 0" is representable
without rounding.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

LOGIT_LO, LOGIT_HI, N_BINS = -16.0, 16.0, 1024
N_CAL = 20                        #: reliability-diagram bins over probability
_W = (LOGIT_HI - LOGIT_LO) / N_BINS
assert abs((0.0 - LOGIT_LO) / _W - round((0.0 - LOGIT_LO) / _W)) < 1e-12
ZERO_BIN = int(round((0.0 - LOGIT_LO) / _W))       #: bin whose lower edge is logit 0


def bin_edges() -> np.ndarray:
    return LOGIT_LO + _W * np.arange(N_BINS + 1)


def bin_index(logit: torch.Tensor) -> torch.Tensor:
    return torch.clamp(((logit - LOGIT_LO) / _W).floor().long(), 0, N_BINS - 1)


def threshold_of_bin(b: int) -> float:
    """The logit threshold ``tau`` such that "bin index >= b" means "logit >= tau"."""
    return float(LOGIT_LO + _W * b)


class ScoreAccumulator:
    """Per-anchor histograms for one method on one region."""

    def __init__(self):
        self.pos, self.neg = [], []
        self.cal_n, self.cal_p, self.cal_y = [], [], []
        self.sse, self.n_scored = [], []
        self.clip_id, self.group = [], []

    def add(self, logit: torch.Tensor, gt: torch.Tensor, region: torch.Tensor,
            clip_id: str, group: str) -> None:
        m = region
        lg = logit[m]
        y = gt[m]
        b = bin_index(lg)
        n = int(m.sum())
        pos = torch.bincount(b[y], minlength=N_BINS)
        tot = torch.bincount(b, minlength=N_BINS)
        self.pos.append(pos.cpu().numpy().astype(np.int32))
        self.neg.append((tot - pos).cpu().numpy().astype(np.int32))
        p = torch.sigmoid(lg.float())
        cb = torch.clamp((p * N_CAL).long(), 0, N_CAL - 1)
        self.cal_n.append(torch.bincount(cb, minlength=N_CAL).cpu().numpy().astype(np.int64))
        self.cal_p.append(torch.bincount(cb, weights=p.double(),
                                         minlength=N_CAL).cpu().numpy())
        self.cal_y.append(torch.bincount(cb, weights=y.double(),
                                         minlength=N_CAL).cpu().numpy())
        self.sse.append(float(((p - y.float()) ** 2).sum()))
        self.n_scored.append(n)
        self.clip_id.append(clip_id)
        self.group.append(group)

    def block(self) -> Dict[str, np.ndarray]:
        return {"pos": np.stack(self.pos), "neg": np.stack(self.neg),
                "cal_n": np.stack(self.cal_n), "cal_p": np.stack(self.cal_p),
                "cal_y": np.stack(self.cal_y),
                "sse": np.asarray(self.sse, np.float64),
                "n_scored": np.asarray(self.n_scored, np.int64),
                "clip_id": np.asarray(self.clip_id), "group": np.asarray(self.group)}


# --------------------------------------------------------------------------- #
# metrics, all derived from the pooled histogram
# --------------------------------------------------------------------------- #
def sweep(pos: np.ndarray, neg: np.ndarray) -> Dict[str, np.ndarray]:
    """TP/FP/FN and derived rates at every bin boundary (descending score)."""
    p = np.asarray(pos, np.float64); n = np.asarray(neg, np.float64)
    if p.ndim > 1:
        p, n = p.sum(0), n.sum(0)
    # cum[b] = count of voxels whose bin index >= b   (b = 0 .. N_BINS)
    tp = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
    fp = np.concatenate([np.cumsum(n[::-1])[::-1], [0.0]])
    n_pos, n_tot = p.sum(), p.sum() + n.sum()
    fn = n_pos - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0)
        rec = tp / max(n_pos, 1e-12)
        iou = tp / np.maximum(tp + fp + fn, 1e-12)
        fpr = fp / max(n_tot - n_pos, 1e-12)
    return {"tau": np.array([threshold_of_bin(b) for b in range(N_BINS + 1)]),
            "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "iou": iou,
            "fpr": fpr, "density": (tp + fp) / max(n_tot, 1e-12),
            "n_pos": n_pos, "n_tot": n_tot}


def average_precision(sw: Dict[str, np.ndarray]) -> float:
    """AP = sum over tie groups of  (R_k - R_{k+1}) * P_k, ties handled as one group."""
    r, p = sw["recall"], sw["precision"]
    return float(np.sum((r[:-1] - r[1:]) * p[:-1]))


def auroc(sw: Dict[str, np.ndarray]) -> float:
    """Trapezoidal ROC area; the trapezoid gives ties the usual 1/2 credit."""
    x, y = sw["fpr"][::-1], sw["recall"][::-1]
    return float(np.trapz(y, x))


def ece(cal_n: np.ndarray, cal_p: np.ndarray, cal_y: np.ndarray) -> float:
    n = np.asarray(cal_n, np.float64).sum(0) if cal_n.ndim > 1 else np.asarray(cal_n, np.float64)
    sp = np.asarray(cal_p, np.float64).sum(0) if cal_p.ndim > 1 else np.asarray(cal_p, np.float64)
    sy = np.asarray(cal_y, np.float64).sum(0) if cal_y.ndim > 1 else np.asarray(cal_y, np.float64)
    tot = n.sum()
    if tot == 0:
        return float("nan")
    ok = n > 0
    return float(np.sum(n[ok] / tot * np.abs(sy[ok] / n[ok] - sp[ok] / n[ok])))


def reliability(cal_n, cal_p, cal_y):
    n = np.asarray(cal_n, np.float64).sum(0); sp = np.asarray(cal_p, np.float64).sum(0)
    sy = np.asarray(cal_y, np.float64).sum(0)
    ok = n > 0
    conf = np.full(N_CAL, np.nan); acc = np.full(N_CAL, np.nan)
    conf[ok] = sp[ok] / n[ok]; acc[ok] = sy[ok] / n[ok]
    return {"n": n, "confidence": conf, "accuracy": acc}


def summarize(blk: Dict[str, np.ndarray], n_curve: int = 64) -> Dict[str, object]:
    """Every threshold-free number the brief asks for, from one stored block."""
    sw = sweep(blk["pos"], blk["neg"])
    prev = float(sw["n_pos"] / max(sw["n_tot"], 1))
    ap = average_precision(sw)
    j = int(np.argmax(sw["iou"]))
    brier = float(blk["sse"].sum() / max(blk["n_scored"].sum(), 1))
    step = max(1, (N_BINS + 1) // n_curve)
    return {
        "n_voxels": int(sw["n_tot"]), "n_occupied": int(sw["n_pos"]), "prevalence": prev,
        "average_precision": ap, "ap_over_prevalence": ap / prev if prev > 0 else float("nan"),
        "auroc": auroc(sw), "brier": brier,
        "ece": ece(blk["cal_n"], blk["cal_p"], blk["cal_y"]),
        "reliability": {k: v.tolist() for k, v in
                        reliability(blk["cal_n"], blk["cal_p"], blk["cal_y"]).items()},
        "at_zero": at_threshold(sw, 0.0),
        "best_iou": {"threshold": float(sw["tau"][j]), "iou": float(sw["iou"][j]),
                     "precision": float(sw["precision"][j]), "recall": float(sw["recall"][j]),
                     "density": float(sw["density"][j])},
        "pr_curve": {"recall": sw["recall"][::step].tolist(),
                     "precision": sw["precision"][::step].tolist(),
                     "threshold": sw["tau"][::step].tolist()},
        "threshold_sweep": {"threshold": sw["tau"][::step].tolist(),
                            "iou": sw["iou"][::step].tolist(),
                            "precision": sw["precision"][::step].tolist(),
                            "recall": sw["recall"][::step].tolist(),
                            "density": sw["density"][::step].tolist()},
    }


def at_threshold(sw: Dict[str, np.ndarray], tau: float) -> Dict[str, float]:
    b = int(np.clip(round((tau - LOGIT_LO) / _W), 0, N_BINS))
    n_pos = max(sw["n_pos"], 1e-12)
    return {"threshold": float(sw["tau"][b]), "tp": int(sw["tp"][b]), "fp": int(sw["fp"][b]),
            "fn": int(sw["fn"][b]), "iou": float(sw["iou"][b]),
            "precision": float(sw["precision"][b]), "recall": float(sw["recall"][b]),
            "density": float(sw["density"][b]),
            "pred_over_gt": float((sw["tp"][b] + sw["fp"][b]) / n_pos)}


def per_clip_counts(blk: Dict[str, np.ndarray], tau: float) -> np.ndarray:
    """``[n_clip, 3]`` of TP/FP/FN at ``tau`` -- the paired-bootstrap input, taken from
    the stored continuous scores rather than from any binarized prediction."""
    b = int(np.clip(round((tau - LOGIT_LO) / _W), 0, N_BINS))
    pos, neg = np.asarray(blk["pos"], np.int64), np.asarray(blk["neg"], np.int64)
    tp = pos[:, b:].sum(1); fp = neg[:, b:].sum(1); fn = pos.sum(1) - tp
    return np.stack([tp, fp, fn], 1)


__all__ = ["ScoreAccumulator", "sweep", "average_precision", "auroc", "ece", "reliability",
           "summarize", "at_threshold", "per_clip_counts", "bin_edges", "bin_index",
           "threshold_of_bin", "LOGIT_LO", "LOGIT_HI", "N_BINS", "N_CAL", "ZERO_BIN"]
