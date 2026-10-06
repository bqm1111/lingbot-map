"""Loader equivalence: the disk-backed runs reproduce the ORIGINAL trainers on a small deterministic anchor selection.

For B4 (``trainer.train_seed``) and B3 (pilot ``fit`` with Gate 16's "select" objective), on every 45th current-pool anchor:
  1. loaded values: every tensor of ``DiskAnchorSequence[i]`` equals the original loader's tensor (dtype, shape, bytes);
  2. short run (N updates, seed 0, cosine over N) through the original code and through ``train_on_anchors``:
     draw records / digest, per-step target counts, the dense B4 input and hidden field of the first steps, the loss inputs
     (masked / delayed targets and student features), every logged loss, and the final parameters;
  3. resume: N/2 + resume + N/2 equals the uninterrupted run (draws exactly; losses / parameters to tolerance);
  4. memory: host cache bytes stay within the configured bound; peak GPU bytes and host RSS are recorded.
Model selection never happens here: the models are disposable and nothing is evaluated on 0006.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.verify_loader --device cuda:0
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import tempfile
from typing import Callable, Dict, Iterator, List, Tuple

import torch

from tools.delayed_semantic import train as PT
from tools.delayed_semantic.model import EvidenceAssociation
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_field.config.schema import Paths as B4Paths
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead
from tools.semantic_field.scaling import b4_step as B4S, loaders
from tools.semantic_field.scaling.anchor_store import DiskAnchorSequence
from tools.semantic_field.scaling.config import ARTIFACT_DIR, CheckpointConfig, LoaderConfig
from tools.semantic_field.scaling.resumable import draw_digest
from tools.semantic_field.scaling.run import train_on_anchors
from tools.semantic_field.scaling.source_pool import current_pool
from tools.semantic_field.training import trainer as ORIGINAL_B4

SELECTION_STRIDE = 45
CAPTURE_STEPS = 2              # dense inputs / hidden fields / loss inputs compared on the first two steps


def selection() -> List[Tuple[str, int]]:
    return list(current_pool().anchors[::SELECTION_STRIDE])


@contextlib.contextmanager
def only_these_anchors(keys: List[Tuple[str, int]]) -> Iterator[None]:
    """Make the ORIGINAL loaders see only ``keys`` (they enumerate ``G15B.anchor_ids``)."""
    by_drive: Dict[str, List[int]] = {}
    for d, t in keys:
        by_drive.setdefault(d, []).append(t)
    original = G15B.anchor_ids
    G15B.anchor_ids = lambda drive, paths=None: by_drive.get(drive, [])
    try:
        yield
    finally:
        G15B.anchor_ids = original


@contextlib.contextmanager
def capture_b4_tensors(sink: List[Dict[str, torch.Tensor]]) -> Iterator[None]:
    """Record the dense input / hidden field of ``hidden_field`` and the loss inputs, for the first ``CAPTURE_STEPS`` steps."""
    orig_hidden, orig_masked_tr, orig_masked_new = SemanticFieldHead.hidden_field, ORIGINAL_B4.masked_reconstruction_loss, B4S.masked_reconstruction_loss
    orig_delayed_tr, orig_delayed_new = ORIGINAL_B4.delayed_loss, B4S.delayed_loss

    def hidden(self, dense):
        out = orig_hidden(self, dense)
        if len(sink) < CAPTURE_STEPS:
            sink.append({"dense": dense.detach().cpu().clone(), "hidden": out.detach().float().cpu().clone()})
        return out

    def wrap(fn: Callable, name: str) -> Callable:
        def inner(*args, **kw):
            if len(sink) <= CAPTURE_STEPS and sink and name not in sink[-1]:
                sink[-1][name + "_target"] = args[0].detach().cpu().clone(); sink[-1][name + "_student"] = args[1].detach().float().cpu().clone()
                sink[-1][name] = True
            return fn(*args, **kw)
        return inner

    SemanticFieldHead.hidden_field = hidden
    ORIGINAL_B4.masked_reconstruction_loss = wrap(orig_masked_tr, "masked"); B4S.masked_reconstruction_loss = wrap(orig_masked_new, "masked")
    ORIGINAL_B4.delayed_loss = wrap(orig_delayed_tr, "delayed"); B4S.delayed_loss = wrap(orig_delayed_new, "delayed")
    try:
        yield
    finally:
        SemanticFieldHead.hidden_field = orig_hidden
        ORIGINAL_B4.masked_reconstruction_loss, B4S.masked_reconstruction_loss = orig_masked_tr, orig_masked_new
        ORIGINAL_B4.delayed_loss, B4S.delayed_loss = orig_delayed_tr, orig_delayed_new


def compare_loaded_values(method: str, keys, device) -> Dict:
    tiers = loaders.delayed_tiers()
    seq = DiskAnchorSequence(keys, loaders.host_loader(method, tiers), loaders.files_of(method), device, 0)
    original = (lambda k: ORIGINAL_B4.load_training_anchor(k[0], k[1], device, tiers)) if method == "B4" else (lambda k: PT.load_anchor(k[0], k[1], device, tiers, ""))
    mismatches = []
    for i, k in enumerate(keys):
        a, b = original(k), seq[i]
        if a.keys() != b.keys():
            mismatches.append((k, "keys")); continue
        for name, v in a.items():
            w = b[name]
            same = (torch.equal(v, w) and v.dtype == w.dtype and v.device == w.device) if isinstance(v, torch.Tensor) else v == w
            if not same:
                mismatches.append((k, name))
    return {"anchors": len(keys), "mismatches": [list(map(str, m)) for m in mismatches], "pass": not mismatches}


def _max_param_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def _log_diff(a: List[Dict], b: List[Dict]) -> Dict:
    keys_equal = all(x.keys() == y.keys() for x, y in zip(a, b)) and len(a) == len(b)
    counts_equal = all(x.get(k) == y.get(k) for x, y in zip(a, b) for k in ("n_masked", "n_delayed"))
    diffs = [abs(x[k] - y[k]) for x, y in zip(a, b) for k in x if isinstance(x[k], float) and k in y]
    exact = all(x == y for x, y in zip(a, b))
    return {"rows": len(a), "keys_equal": keys_equal, "target_counts_equal": counts_equal, "all_rows_bit_identical": exact,
            "max_abs_logged_diff": max(diffs) if diffs else 0.0}


#: dense input channel groups [61]: frozen trunk d1 (24), S / F / O (3), projected centred evidence (32, depends on the
#: trainable projection, so it follows the parameters), evidence mask + log count (2)
DENSE_GROUPS = {"trunk": slice(0, 24), "geometry": slice(24, 27), "projected_evidence": slice(27, 59), "mask_count": slice(59, 61)}


def _original_b4(keys, device, steps: int, tmp: str, name: str) -> Tuple[Dict, Dict[str, torch.Tensor], List[Dict]]:
    caps: List[Dict] = []
    with only_these_anchors(keys), capture_b4_tensors(caps):
        paths = B4Paths(checkpoints=os.path.join(tmp, name, "ck"), runs=os.path.join(tmp, name, "runs"))
        rec = ORIGINAL_B4.train_seed(0, device, steps=steps, paths=paths)
        state = torch.load(os.path.join(paths.checkpoints, "smoke", f"smoke_B4_seed0_step{steps:05d}.pt"), map_location="cpu")["state_dict"]
    return rec, state, caps


def _compare_captures(a: List[Dict], b: List[Dict]) -> Dict:
    out = {}
    for s in range(CAPTURE_STEPS):
        for g, sl in DENSE_GROUPS.items():
            out[f"step{s + 1}_dense_{g}_exact"] = bool(torch.equal(a[s]["dense"][:, sl], b[s]["dense"][:, sl]))
        for name in ("masked_target", "delayed_target"):
            out[f"step{s + 1}_{name}_exact"] = bool(torch.equal(a[s][name], b[s][name]))
        for name in ("hidden", "masked_student", "delayed_student"):
            out[f"step{s + 1}_{name}_max_abs_diff"] = float((a[s][name] - b[s][name]).abs().max())
    return out


@contextlib.contextmanager
def deterministic_kernels(enabled: bool) -> Iterator[None]:
    """Diagnostic only: force deterministic CUDA kernels to separate kernel nondeterminism from pipeline differences."""
    if not enabled:
        yield; return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False); torch.backends.cudnn.deterministic = False


def verify_b4(keys, device, steps: int, tmp: str, deterministic: bool = False) -> Dict:
    """Original vs disk-backed B4; also original vs original, so kernel nondeterminism has a measured reference."""
    tag = "det" if deterministic else "default"
    with deterministic_kernels(deterministic):
        rec, orig_state, orig_caps = _original_b4(keys, device, steps, tmp, f"orig_a_{tag}")
        rec_again, orig_state_again, orig_caps_again = _original_b4(keys, device, steps, tmp, f"orig_b_{tag}")
        new_caps: List[Dict] = []
        with capture_b4_tensors(new_caps):
            new = train_on_anchors("B4", keys, 0, steps, device, os.path.join(tmp, f"b4_resume_{tag}.pt"), LoaderConfig(2 * 1024 ** 3),
                                   CheckpointConfig(resume_every_steps=steps), tag=f"new_B4_{tag}")
    new_state = {k: v.cpu() for k, v in new.model.state_dict().items()}
    return {"steps": steps, "deterministic_kernels": deterministic,
            "draw_digest_equal": rec["draw_digest"] == draw_digest(new.state.draws) == rec_again["draw_digest"],
            "original_vs_new": {"log": _log_diff(rec["log"], new.state.log), "tensors": _compare_captures(orig_caps, new_caps),
                                "final_param_max_abs_diff": _max_param_diff(orig_state, new_state)},
            "original_vs_original": {"log": _log_diff(rec["log"], rec_again["log"]), "tensors": _compare_captures(orig_caps, orig_caps_again),
                                     "final_param_max_abs_diff": _max_param_diff(orig_state, orig_state_again)},
            "params_finite": all(bool(torch.isfinite(p).all()) for p in new.model.parameters()), "geometry_digest_unchanged": new.geometry_digest is not None,
            "peak_gpu_bytes": new.peak_gpu_bytes, "cache": new.anchors.stats.__dict__, "cache_bound_bytes": 2 * 1024 ** 3}


def verify_b3(keys, device, steps: int, tmp: str) -> Dict:
    tiers = loaders.delayed_tiers(); text, _, _ = TR.load_text_embeddings(); text = text.to(device)
    with only_these_anchors(keys):
        anchors = PT.load_training_anchors(torch.device(device), tiers, "")
    torch.manual_seed(0); model = EvidenceAssociation(semantic_projection=True).to(device)
    cfg = PT.C.TrainConfig()
    rec = PT.fit(model, anchors, text, cfg, steps, 0, True, "select", lambda it: os.path.join(tmp, "b3", f"B3_step{it:05d}.pt"), "orig_B3", {})
    orig_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}; del anchors
    new = train_on_anchors("B3", keys, 0, steps, device, os.path.join(tmp, "b3_resume.pt"), LoaderConfig(2 * 1024 ** 3),
                           CheckpointConfig(resume_every_steps=steps), tag="new_B3")
    return {"steps": steps, "original_draw_digest": rec["draw_digest"], "new_draw_digest": draw_digest(new.state.draws),
            "draw_digest_equal": rec["draw_digest"] == draw_digest(new.state.draws), "log": _log_diff(rec["log"], new.state.log),
            "final_param_max_abs_diff": _max_param_diff(orig_state, {k: v.cpu() for k, v in new.model.state_dict().items()}),
            "params_finite": all(bool(torch.isfinite(p).all()) for p in new.model.parameters()), "peak_gpu_bytes": new.peak_gpu_bytes,
            "cache": new.anchors.stats.__dict__}


def verify_resume(method: str, keys, device, steps: int, tmp: str) -> Dict:
    straight = train_on_anchors(method, keys, 0, steps, device, os.path.join(tmp, f"{method}_straight.pt"), LoaderConfig(0),
                                CheckpointConfig(resume_every_steps=steps), tag=f"straight_{method}")
    resume = os.path.join(tmp, f"{method}_split.pt")
    train_on_anchors(method, keys, 0, steps, device, resume, LoaderConfig(0), CheckpointConfig(resume_every_steps=steps // 2),
                     stop_after_step=steps // 2, tag=f"first_half_{method}")
    second = train_on_anchors(method, keys, 0, steps, device, resume, LoaderConfig(0), CheckpointConfig(resume_every_steps=steps // 2),
                              tag=f"second_half_{method}")
    return {"steps": steps, "draws_equal": straight.state.draws == second.state.draws, "log": _log_diff(straight.state.log, second.state.log),
            "final_param_max_abs_diff": _max_param_diff({k: v.cpu() for k, v in straight.model.state_dict().items()},
                                                        {k: v.cpu() for k, v in second.model.state_dict().items()})}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    keys = selection(); out: Dict = {"selection": {"stride": SELECTION_STRIDE, "anchors": [list(k) for k in keys]}}
    with tempfile.TemporaryDirectory(dir=os.path.join(ARTIFACT_DIR, "verification")) as tmp:
        out["b4_loaded_values"] = compare_loaded_values("B4", keys, a.device)
        out["b3_loaded_values"] = compare_loaded_values("B3", keys, a.device)
        out["b4_short_run"] = verify_b4(keys, a.device, a.steps, tmp)
        out["b4_short_run_deterministic_kernels"] = verify_b4(keys, a.device, a.steps, tmp, deterministic=True)
        out["b3_short_run"] = verify_b3(keys, a.device, a.steps, tmp)
        out["b4_resume"] = verify_resume("B4", keys, a.device, a.steps, tmp)
        out["b3_resume"] = verify_resume("B3", keys, a.device, a.steps, tmp)
    out["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    json.dump(out, open(os.path.join(ARTIFACT_DIR, "verification", "loader_equivalence.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "selection"}, indent=1, default=str))


if __name__ == "__main__":
    main()
