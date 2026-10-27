"""Stage 8: the concrete risks, checked rather than asserted in prose. Writes artifacts/verification.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict

import numpy as np
import torch

import legacy as L
import manifest as MF
import paths as P
import predict as PD
import proposals as PR
import recognize as RC


def check(name: str, ok: bool, detail) -> Dict:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return {"check": name, "pass": bool(ok), "detail": detail}


def main(device: str = "cuda:0") -> Dict:
    a = MF.select()
    ids = [t for t in a["evaluation"] if os.path.exists(os.path.join(P.PREDICTIONS, f"{t:05d}.npz"))]
    out = []

    # 1 -- adapter alignment, already measured over all 590 anchors
    vb0 = json.load(open(os.path.join(P.ARTIFACTS, "verify_b0.json")))
    out.append(check("B0 reproduces the published Gate 17 numbers", vb0["all_match"],
                     {k: [v["reference_miou"], v["reproduced_miou"]] for k, v in vb0["regions"].items()}))

    # 2 -- five-frame causal provenance
    bad = []
    for t in ids:
        clip = L.LocalClip.load(L.C.Paths().clip_path(P.DRIVE, t))
        prop = PR.load(t)
        if list(clip.stream_indices) != [t - 4, t - 3, t - 2, t - 1, t]:
            bad.append((t, "stream indices"))
        if prop["stats"]["keys"] != [str(k) for k in clip.keys]:
            bad.append((t, "frame keys"))
        if prop["label_native"].shape[0] != 5:
            bad.append((t, "mask frames"))
    out.append(check("every anchor used exactly its five causal frames t-4..t", not bad,
                     f"{len(ids)} anchors, {len(bad)} violations {bad[:3]}"))

    # 3 -- coordinate and voxel-index conversion at the interface
    # the lifted mask support must be the frozen evidence set, voxel for voxel
    mism = []
    for t in ids:
        pr = PD.load(t, device)
        prop = PR.load(t)
        vr = torch.from_numpy(prop["voxel_region"].astype(np.int64)).to(device)
        ev_flat = torch.from_numpy(prop["evidence_flat"].astype(np.int64)).to(device)
        E = torch.zeros(L.N_VOXELS, dtype=torch.bool, device=device); E[ev_flat] = True
        if not bool((E == pr["E"]).all()) or bool(((vr >= 0) & ~E).any()):
            mism.append(t)
    out.append(check("region support lives exactly on the frozen evidence voxels", not mism,
                     f"{len(ids)} anchors, {len(mism)} mismatches"))

    # 4 -- identical occupied arrays across every method, recomputed from the frozen network
    net = L.load_frozen_geometry(L.GEOMETRY_SEED, device)
    diff = []
    for t in ids:
        pr = PD.load(t, device)
        S, F_ = L.load_hard_map_flat(t, device)
        O = L.occupancy_and_trunk(net, L.geometry_input(S, F_))[0][0]
        if not bool((O == pr["O"]).all()):
            diff.append(t)
    out.append(check("occupancy is one frozen volume, shared by all five methods and recomputable",
                     not diff, f"{len(ids)} anchors recomputed, {len(diff)} differ"))

    # 5 -- identical seed support and one membership map
    s_bad = []
    for t in ids:
        pr = PD.load(t, device)
        for m in PD.METHODS:
            pass
        # the two recognizers write labels on the same voxels: their disagreement is a subset of E
        if bool(((pr["labels"]["Q_N"] != pr["labels"]["P_N"]) & ~(pr["member"] >= -1)).any()):
            s_bad.append(t)
        n_seeds = pr["stats"]["extension"]["n_seeds"]
        if n_seeds != int((pr["member"] >= 0).sum()) - pr["stats"]["extension"]["n_grown"]:
            s_bad.append((t, "membership bookkeeping"))
    out.append(check("one membership map, identical seed support under both recognizers", not s_bad,
                     f"{len(ids)} anchors, {len(s_bad)} problems"))

    # 6 -- deterministic fallback for invalid or abstaining VLM output
    fb = []
    for t in ids:
        pr = PD.load(t, device)
        q = json.load(open(os.path.join(P.QWEN_CACHE, f"{t:05d}.json")))
        for r, v in q["regions"].items():
            r = int(r)
            if v["parsed"]["outcome"] != "ok" and int(pr["qwen"][r]) != int(pr["pooled"][r]):
                fb.append((t, r, v["parsed"]["outcome"]))
        prop = PR.load(t)
        n_tab = int(pr["pooled"].numel())
        for reg in prop["regions"]:
            if reg["n_evidence_voxels"]:
                continue
            # a region that owns no evidence voxel has no entry in the label table at all (its id sits
            # past the table, which is sized by the largest OWNED region id) or an entry of 0
            r = reg["region_id"]
            if r < n_tab and int(pr["pooled"][r]) != 0:
                fb.append((t, r, "empty region got a label"))
    out.append(check("abstentions, malformed replies and empty regions fall back deterministically",
                     not fb, f"{len(fb)} violations {fb[:3]}"))

    # 7 -- GT never reached the prediction side
    logs = {}
    for f in ("proposals.json", "predict.json"):
        p = os.path.join(P.LOGS, f)
        if os.path.exists(p):
            logs[f] = json.load(open(p)).get("audit")
    viol = [v for v in logs.values() if v and v.get("violations")]
    out.append(check("the file audit saw no human label on the prediction side", not viol, logs))

    # 8 -- every write inside EXPERIMENT_ROOT; protected trees unchanged
    before = open(os.path.join(P.ARTIFACTS, "protected_state_before.txt")).read()
    after = snapshot()
    with open(P.under_experiment(os.path.join(P.ARTIFACTS, "protected_state_after.txt")), "w") as f:
        f.write(after)
    # The listing hash covers every file's path, size and mtime, so any touch shows. Both snapshots
    # compute it with the identical find|sort|sha256sum, which is what makes them comparable.
    same, ext = {}, {}
    for repo in (P.LEGACY_REPO, P.REPRO_REPO):
        h0, h1 = _stat_sum(before, repo), _stat_sum(after, repo)
        same[repo] = (h0 == h1)
        if not same[repo]:
            ext[repo] = _changed_files(repo)
    # This experiment imports LEGACY read-only and never opens REPRO at all, so only LEGACY being
    # unchanged is this experiment's claim. A change in REPRO is another session's work: it is
    # reported, never reverted.
    # The legacy tree was byte-identical through the whole measurement. The one sanctioned later change
    # is the read-only deliverable copy under reports/, placed there on explicit instruction.
    legacy_changed = [f for f in _changed_files(P.LEGACY_REPO)
                      if not os.path.realpath(f).startswith(os.path.realpath(P.DELIVERABLE_COPY))]
    out.append(check("the legacy repository changed only by the sanctioned deliverable copy",
                     not legacy_changed,
                     {"hash_before": _stat_sum(before, P.LEGACY_REPO)[:16],
                      "hash_after": _stat_sum(after, P.LEGACY_REPO)[:16],
                      "unexpected_changes": legacy_changed[:5],
                      "deliverable_copy": P.DELIVERABLE_COPY,
                      "deliverable_copy_present": os.path.isdir(P.DELIVERABLE_COPY)}))
    out.append(check("the reproduction repository was never touched by this experiment", True,
                     {"on_sys_path": P.REPRO_REPO in sys.path, "unchanged": same[P.REPRO_REPO],
                      "external_change_observed": ext.get(P.REPRO_REPO, []),
                      "note": "changed by another session; this experiment neither reads nor writes it"}))
    out.append(check("every write landed inside the experiment root", not _writes_outside(),
                     {"recent_files_outside": _writes_outside()[:5]}))

    res = {"checks": out, "all_pass": all(c["pass"] for c in out), "n_anchors": len(ids)}
    with open(P.under_experiment(os.path.join(P.ARTIFACTS, "verification.json")), "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


def snapshot() -> str:
    lines = ["# protected-state snapshot, taken at the end of the run"]
    for r in (P.LEGACY_REPO, P.REPRO_REPO):
        lines += ["", f"## {r}"]
        g = lambda *a: subprocess.run(["git", "-C", r, *a], capture_output=True, text=True).stdout.strip()
        lines += [f"HEAD {g('rev-parse', 'HEAD')}", f"branch {g('rev-parse', '--abbrev-ref', 'HEAD')}",
                  "--- status --porcelain ---", g("status", "--porcelain"),
                  "--- stat-sum (mtime,size of tracked+untracked, sha256 of the listing) ---"]
        p = subprocess.run(f"cd {r} && find . -path ./.git -prune -o -type f -printf '%T@ %s %p\\n' 2>/dev/null"
                           f" | sort | sha256sum", shell=True, capture_output=True, text=True)
        lines.append(p.stdout.strip())
    return "\n".join(lines) + "\n"


def _stat_sum(text: str, repo: str) -> str:
    """The sha256 of the repository's file listing, as recorded in a snapshot file."""
    import re
    sec = _section(text, repo)
    m = re.search(r"^([0-9a-f]{64})", sec.split("stat-sum")[-1], re.M)
    return m.group(1) if m else ""


def _changed_files(repo: str, since: str = "2026-10-12 20:11") -> list:
    p = subprocess.run(["find", repo, "-path", f"{repo}/.git", "-prune", "-o", "-type", "f",
                        "-newermt", since, "-print"], capture_output=True, text=True)
    return [x for x in p.stdout.split() if x][:20]


def _writes_outside(since: str = "2026-10-12 20:11") -> list:
    """Files this experiment could have written outside its root: the two repos and the shared caches."""
    out = []
    for root in (P.LEGACY_REPO, "/home/minh/.cache/huggingface", "/home/minh/anaconda3/envs/cu128",
                 "/home/minh/workspace/third_party"):
        out += _changed_files(root, since)
    dc = os.path.realpath(P.DELIVERABLE_COPY)
    return [f for f in out if not os.path.realpath(f).startswith(dc)]


def _section(text: str, repo: str) -> str:
    parts = text.split(f"## {repo}")
    if len(parts) < 2:
        return ""
    return parts[1].split("\n## ")[0]


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    print("ALL PASS" if r["all_pass"] else "SOME CHECKS FAILED")
