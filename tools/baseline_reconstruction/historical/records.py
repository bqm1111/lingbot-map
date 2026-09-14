"""Locate and parse the historical Gate 8C-1 results as they exist on disk.

Nothing here is computed from a target dataset: every number is read from an evaluation
JSON, a log line, a frozen manifest or a report that already exists. The parsers keep the
original metric names (``binary_iou`` of the ``semantic.completion`` block is what the gate
printed as ``SC_IoU``), so no remembered value can leak in.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from tools.baseline_reconstruction import config as C

#: the four model variants of the ceiling investigation, by the manifest that selects them
VARIANTS = {
    "prefix": "frozen_manifest.json",          # shipped headline before the fix (zeros padding, old targets)
    "padfix": "frozen_manifest_padfix.json",   # replicate padding only, old targets (ablation)
    "sky": "frozen_manifest_sky.json",         # replicate padding + open-sky-is-FREE -- ADOPTED
    "sky2": "frozen_manifest_sky2.json",       # + 15x15 neighbourhood roof -- rejected
}


@dataclass(frozen=True)
class HistoricalResult:
    """One (variant, dataset, protocol, seed) evaluation, as recorded by ``core/run/eval_target.py``."""
    variant: str
    dataset: str
    protocol: str
    seed: int
    sc_iou: float               # semantic.completion.binary_iou  (printed as SC_IoU)
    precision: float            # semantic.completion.binary_precision
    recall: float               # semantic.completion.binary_recall
    ssc_miou: float             # semantic.completion.ssc_miou (the model carried semantics)
    auroc: float                # methods.completion.full.auroc (threshold-free)
    mapper_only_iou: float      # baselines.mapper_native.binary_iou (raw map, no completion)
    n_clips: int
    threshold: float            # the source-selected per-seed final-log-odds threshold
    checkpoint: str
    source_file: str
    source_mtime: str

    @property
    def sc_iou_pct(self) -> float:
        return 100.0 * self.sc_iou


def _mtime(path: str) -> str:
    return _dt.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(timespec="seconds")


def parse_eval_json(path: str, variant: str) -> HistoricalResult:
    d = json.load(open(path))
    comp = d["semantic"]["completion"]
    return HistoricalResult(
        variant=variant, dataset=d["dataset"], protocol=d["mode"], seed=int(d["seed"]),
        sc_iou=float(comp["binary_iou"]), precision=float(comp["binary_precision"]),
        recall=float(comp["binary_recall"]), ssc_miou=float(comp["ssc_miou"]),
        auroc=float(d["methods"]["completion"]["full"]["auroc"]),
        mapper_only_iou=float(d["baselines"]["mapper_native"]["binary_iou"]),
        n_clips=int(d["n_clips"]), threshold=float(d["threshold"]), checkpoint=d["checkpoint"],
        source_file=os.path.relpath(path, C.REPO_ROOT), source_mtime=_mtime(path))


def load_variant_results(paths: C.HistoricalPaths = C.HistoricalPaths()) -> List[HistoricalResult]:
    """Every eval JSON of the adopted sky model and of the shipped pre-fix model."""
    out = []
    for p in sorted(glob.glob(os.path.join(paths.sky_eval_dir, "eval_*_seed*.json"))):
        out.append(parse_eval_json(p, "sky"))
    for p in sorted(glob.glob(os.path.join(paths.prefix_eval_dir, "eval_*_seed*.json"))):
        out.append(parse_eval_json(p, "prefix"))
    return out


def select(results: List[HistoricalResult], variant: str, dataset: str, protocol: str) -> List[HistoricalResult]:
    return sorted([r for r in results if (r.variant, r.dataset, r.protocol) == (variant, dataset, protocol)],
                  key=lambda r: r.seed)


# --------------------------------------------------------------------------- #
# log lines: an independent record of the same runs
# --------------------------------------------------------------------------- #
_RESULT_LINE = re.compile(r"RESULT (?:(?P<ds>occ3d) )?seed(?P<seed>\d) SC_IoU=(?P<iou>[\d.]+) P=(?P<p>[\d.]+) R=(?P<r>[\d.]+)")


def parse_result_logs(log_dir: str) -> Dict[str, Dict[int, float]]:
    """``{dataset: {seed: SC_IoU %}}`` from the ``RESULT`` lines the sky-model runs printed."""
    out: Dict[str, Dict[int, float]] = {}
    for p in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        name = os.path.basename(p)
        if not (name.startswith("eval_seed") or name.startswith("occ3d_sky_seed")):
            continue
        for line in open(p):
            m = _RESULT_LINE.search(line)
            if m:
                ds = m.group("ds") or "semantickitti"
                out.setdefault(ds, {})[int(m.group("seed"))] = float(m.group("iou"))
    return out


# --------------------------------------------------------------------------- #
# manifests and checkpoints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestSeed:
    seed: int
    config: str
    checkpoint: str
    checkpoint_sha256_recorded: str
    checkpoint_sha256_on_disk: Optional[str]
    occupancy_threshold: float
    padding_mode_in_checkpoint: Optional[str]
    input_channels_in_checkpoint: Optional[int]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest_seeds(manifest_path: str) -> List[ManifestSeed]:
    import torch
    man = json.load(open(manifest_path))
    out = []
    for s, fz in sorted(man["seeds"].items()):
        ck = os.path.join(C.REPO_ROOT, fz["checkpoint"])
        on_disk = sha256_of(ck) if os.path.exists(ck) else None
        pad = cin = None
        if on_disk:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            pad, cin = meta.get("padding_mode", "zeros"), int(meta.get("cin", -1))
        out.append(ManifestSeed(int(s), fz["config"], fz["checkpoint"], fz["checkpoint_sha256"], on_disk,
                                float(fz["occupancy_threshold"]), pad, cin))
    return out


def manifest_summary(manifest_path: str) -> Dict:
    man = json.load(open(manifest_path))
    return {"path": os.path.relpath(manifest_path, C.REPO_ROOT), "gate": man.get("gate"),
            "frozen_at": man.get("frozen_at"), "data_root": man.get("data_root"),
            "code_commit": man.get("code_commit"), "rule": man.get("rule"),
            "seeds": [asdict(x) for x in read_manifest_seeds(manifest_path)]}


# --------------------------------------------------------------------------- #
# numbers that survive only in prose (their JSON records were overwritten)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProseOnlyNumber:
    label: str
    dataset: str
    value_pct: float
    source: str
    protocol: str
    comparability_to_canonical: str


PROSE_ONLY: List[ProseOnlyNumber] = [
    ProseOnlyNumber("sky model, OccAny SSCMetrics, 161 shared frames, seeds 0/1/2", "semantickitti", 22.83,
                    "artifacts/gate8c1/ceiling_fix.md (One model, both benchmarks; median of 22.61/23.79/22.83)",
                    "OccAny's stored labels on 161 of 163 frames, scored by OccAny SSCMetrics.get_score_completion",
                    "partially comparable: subset of frames and OccAny's label volume (matches ours on 1 of 161 frames)"),
    ProseOnlyNumber("sky model, OccAny SSCMetrics, 882 shared samples, seed 0", "occ3d", 33.28,
                    "artifacts/gate8c1/ceiling_fix.md (One model, both benchmarks)",
                    "882 of 1182 Occ3D anchors shared with OccAny's output, OccAny SSCMetrics",
                    "partially comparable: subset of anchors, single seed"),
    ProseOnlyNumber("sky model, matched forward (non-causal) input, seed 0", "occ3d", 30.70,
                    "artifacts/gate8c1/ceiling_fix.md", "occany_fwd offsets 0,+2,+4,+6,+8 (non-causal)",
                    "different protocol: non-causal window"),
    ProseOnlyNumber("sky2 (rejected neighbourhood rule), seed 0", "occ3d", 28.12,
                    "artifacts/gate8c1/ceiling_fix.md addendum; artifacts/gate8c1/occany_headtohead.json (overwritten by this run)",
                    "882 shared samples, OccAny SSCMetrics", "different model: rejected variant"),
]

__all__ = ["VARIANTS", "HistoricalResult", "parse_eval_json", "load_variant_results", "select",
           "parse_result_logs", "ManifestSeed", "sha256_of", "read_manifest_seeds", "manifest_summary",
           "ProseOnlyNumber", "PROSE_ONLY"]
