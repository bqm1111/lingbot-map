#!/usr/bin/env python
"""Gate 8D Phase 1: freeze the protocol before any target or weight exists.

Writes ``artifacts/gate8d/protocol.json`` with the protocol, its source hash, and the
hashes of the Gate 8C-1 baseline this experiment is measured against. Once a Gate 8D
checkpoint exists, ``assert_unchanged`` refuses to let training proceed if the protocol
source has moved -- the design cannot be edited in response to a result.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART_ROOT, REPO_ROOT                                          # noqa: E402
from gates.gate8d import protocol as P                                                 # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8d")
PROTOCOL_JSON = os.path.join(ART, "protocol.json")
BASELINE = {
    "manifest": "artifacts/gate8c1/frozen_manifest_sky.json",
    "checkpoints": ["artifacts/gate8c1/checkpoints/sky_seed0_last.pt",
                    "artifacts/gate8c1/checkpoints/sky_seed1_last.pt",
                    "artifacts/gate8c1/checkpoints/sky_seed2_last.pt"],
    "reported": {"semantickitti_sc_iou_seeds": [22.61, 23.79, 22.83],
                 "occ3d_causal_seed0": 33.28, "occ3d_matched_forward_seed0": 30.70,
                 "rejected_variant": "neighbourhood height propagation (sky2) -- "
                                     "Occ3D 28.12, must remain rejected"}}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assert_unchanged() -> None:
    """Raise if the protocol moved after the first Gate 8D checkpoint was written."""
    if not os.path.exists(PROTOCOL_JSON):
        raise RuntimeError("gate8d protocol is not frozen: run freeze_protocol.py first")
    frozen = json.load(open(PROTOCOL_JSON))
    ck = glob.glob(os.path.join(ART, "checkpoints", "*.pt"))
    now = P.source_digest()
    if now != frozen["protocol"]["protocol_source_sha256"] and ck:
        raise RuntimeError(
            f"gate8d/protocol.py changed after {len(ck)} checkpoint(s) were written "
            f"({frozen['protocol']['protocol_source_sha256'][:12]} -> {now[:12]}). "
            "The protocol is frozen; delete the checkpoints deliberately or revert.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    if os.path.exists(PROTOCOL_JSON) and not a.force:
        assert_unchanged()
        print(f"already frozen: {PROTOCOL_JSON}")
        return 0
    base = dict(BASELINE)
    base["hashes"] = {}
    for rel in [base["manifest"]] + base["checkpoints"]:
        p = os.path.join(REPO_ROOT, rel)
        base["hashes"][rel] = sha256(p) if os.path.exists(p) else "MISSING"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
    except Exception:                                                   # noqa: BLE001
        commit = "unknown"
    out = {"gate": P.GATE, "frozen_at": __import__("datetime").datetime.now()
           .astimezone().isoformat(timespec="seconds"),
           "code_commit": commit, "protocol": P.as_dict(), "baseline_gate8c1": base,
           "claim": ("KITTI-360-only training with no target-domain optimisation, "
                     "fine-tuning, threshold calibration or checkpoint selection. "
                     "SemanticKITTI and Occ3D have been inspected during earlier "
                     "development, so they are NOT historically untouched.")}
    with open(PROTOCOL_JSON, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print(f"froze {PROTOCOL_JSON}")
    print(f"  protocol sha256 {out['protocol']['protocol_source_sha256'][:16]}")
    for k, v in base["hashes"].items():
        print(f"  {k}: {v[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
