#!/usr/bin/env python
"""Build the semantically broad TartanAir training corpus for the controlled study.

    python tools/build_broad_tartanair_corpus.py --out /media/SSD1/.../corpus

Exactly 3,347 unlabeled RGB frames — matching the original narrow corpus size — drawn
evenly from six semantically distinct TartanAir environments.  Frames are fetched from
the official TartanAir V2 HuggingFace mirror (``theairlabcmu/tartanair2``) using HTTP
range requests, so only the ~140 members of each chosen slice are transferred rather
than the multi-gigabyte environment archive.  ``ForestEnv`` is already present locally
and is read from disk.

Selection is fixed a priori from the requested semantic categories.  No frame is chosen
using any evaluation signal, no semantic label or taxonomy is consulted, and no
target-dataset (KITTI / SemanticKITTI / ScanNet / Replica / Speira) path may appear.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

HF_BASE = "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/{env}/{diff}/image_lcam_front.zip"
LOCAL_FOREST = "/media/minh/TartanAir/dataset/ForestEnv"

#: Directory fragments that must never appear in a training source path.
FORBIDDEN = ("kitti", "semantickitti", "scannet", "replica", "speira")

#: Frames taken per trajectory slice, in order; sums to 3,347 across the six blocks.
@dataclass
class Slice:
    """One contiguous run of frames from a single trajectory."""

    traj: str
    count: int
    #: Which contiguous block of the trajectory to take (0 = first, 1 = second, ...).
    block: int = 0


@dataclass
class EnvSpec:
    """One environment's contribution to the corpus."""

    name: str
    category: str
    difficulty: str
    slices: List[Slice]
    local_root: Optional[str] = None
    #: Frames skipped between kept frames in the source trajectory.
    stride: int = 2

    @property
    def total(self) -> int:
        return sum(s.count for s in self.slices)


#: The corpus. Six categories, ~558 frames each, 3,347 total.
CORPUS: List[EnvSpec] = [
    EnvSpec("ModularNeighborhood", "road / neighbourhood", "Data_hard",
            [Slice("P000", 140), Slice("P003", 140), Slice("P006", 139), Slice("P010", 139)]),
    EnvSpec("ModernCityDowntown", "urban street / vehicles", "Data_hard",
            [Slice("P000", 140), Slice("P001", 140), Slice("P005", 139), Slice("P006", 139)]),
    EnvSpec("AbandonedFactory", "industrial / factory", "Data_hard",
            [Slice("P000", 140), Slice("P001", 140), Slice("P003", 139), Slice("P006", 139)]),
    EnvSpec("Office", "indoor office", "Data_hard",
            [Slice("P000", 140), Slice("P004", 140), Slice("P005", 139), Slice("P006", 139)]),
    EnvSpec("OldTownFall", "old town / buildings", "Data_easy",
            [Slice("P000", 140), Slice("P001", 140, block=0), Slice("P001", 139, block=1),
             Slice("P002", 139)]),
    EnvSpec("ForestEnv", "vegetation / forest", "Data_hard",
            [Slice("P000", 140), Slice("P002", 139), Slice("P003", 139), Slice("P009", 139)],
            local_root=LOCAL_FOREST),
]

TARGET_TOTAL = 3347


def scene_name(env: EnvSpec, sl: Slice) -> str:
    suffix = f"{sl.traj}b{sl.block}" if sl.block else sl.traj
    return f"broad_{env.name}_{suffix}"


def pick_window(names: Sequence[str], sl: Slice, stride: int) -> List[str]:
    """Deterministic contiguous, strided window of ``sl.count`` frames.

    Blocks are laid out back-to-back from the start of the trajectory, so two slices of
    the same trajectory never overlap.
    """
    span = sl.count * stride
    start = sl.block * span
    if start + span > len(names):
        start = max(len(names) - span, 0)
    chosen = names[start : start + span : stride][: sl.count]
    if len(chosen) < sl.count:
        raise ValueError(
            f"trajectory has {len(names)} frames; cannot take {sl.count} at stride {stride} "
            f"from block {sl.block}"
        )
    return chosen


def check_path_is_clean(path: str) -> None:
    low = path.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise ValueError(f"forbidden dataset fragment {bad!r} in source path: {path}")


def plan_remote(env: EnvSpec) -> Tuple[str, List[Tuple[Slice, List[str]]]]:
    """Resolve which archive members each slice needs (one central-directory read)."""
    from remotezip import RemoteZip

    url = HF_BASE.format(env=env.name, diff=env.difficulty)
    check_path_is_clean(url)
    with RemoteZip(url) as z:
        by_traj: Dict[str, List[str]] = {}
        for m in z.namelist():
            if m.endswith(".png"):
                by_traj.setdefault(m.split("/")[2], []).append(m)
    for key in by_traj:
        by_traj[key].sort()

    plan: List[Tuple[Slice, List[str]]] = []
    for sl in env.slices:
        if sl.traj not in by_traj:
            raise KeyError(f"{env.name}: trajectory {sl.traj} not in archive")
        plan.append((sl, pick_window(by_traj[sl.traj], sl, env.stride)))
    return url, plan


def _fetch_slice(url: str, env: EnvSpec, sl: Slice, members: List[str],
                 out_root: str) -> List[Dict[str, object]]:
    """Download one slice.  Each call owns its connection, so slices run in parallel."""
    from remotezip import RemoteZip

    dest = os.path.join(out_root, scene_name(env, sl))
    os.makedirs(dest, exist_ok=True)

    def _needs(m: str) -> bool:
        t = os.path.join(dest, os.path.basename(m))
        return not os.path.exists(t) or os.path.getsize(t) == 0

    todo = [m for m in members if _needs(m)]
    if todo:
        with RemoteZip(url) as z:
            for member in todo:
                target = os.path.join(dest, os.path.basename(member))
                with z.open(member) as src, open(target + ".part", "wb") as fh:
                    shutil.copyfileobj(src, fh)
                os.replace(target + ".part", target)
    out = []
    for member in members:
        target = os.path.join(dest, os.path.basename(member))
        out.append({
            "environment": env.name, "category": env.category,
            "difficulty": env.difficulty, "trajectory": sl.traj, "block": sl.block,
            "scene": scene_name(env, sl),
            "frame_index": int("".join(c for c in os.path.basename(member) if c.isdigit())),
            "source_path": f"{url}!{member}", "local_path": os.path.abspath(target),
        })
    print(f"    {scene_name(env, sl)}: {len(members)} frames ({len(todo)} fetched)", flush=True)
    return out


def fetch_remote_all(envs: List[EnvSpec], out_root: str, workers: int) -> List[Dict[str, object]]:
    """Fetch every remote slice concurrently.

    Per-member range requests are latency-bound rather than bandwidth-bound, so the
    only way to make this fast is to keep many requests in flight at once.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = []
    for env in envs:
        url, plan = plan_remote(env)
        print(f"  planned {env.name}/{env.difficulty}: {len(plan)} slices", flush=True)
        for sl, members in plan:
            jobs.append((url, env, sl, members))

    records: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_slice, u, e, s_, m, out_root) for u, e, s_, m in jobs]
        for fut in as_completed(futures):
            records.extend(fut.result())
    return records


def fetch_local(env: EnvSpec, out_root: str, records: List[Dict[str, object]]) -> None:
    """Take the chosen frames from the already-extracted local environment."""
    for sl in env.slices:
        src_dir = os.path.join(env.local_root, env.difficulty, sl.traj, "image_lcam_front")
        check_path_is_clean(src_dir)
        # Zero-byte files exist in this mirror and must not enter the corpus.
        names = sorted(p for p in glob.glob(os.path.join(src_dir, "*.png")) if os.path.getsize(p) > 0)
        if not names:
            raise FileNotFoundError(f"no usable frames under {src_dir}")
        chosen = pick_window(names, sl, env.stride)
        dest = os.path.join(out_root, scene_name(env, sl))
        os.makedirs(dest, exist_ok=True)
        for path in chosen:
            target = os.path.join(dest, os.path.basename(path))
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                shutil.copyfile(path, target)
            records.append({
                "environment": env.name, "category": env.category,
                "difficulty": env.difficulty, "trajectory": sl.traj, "block": sl.block,
                "scene": scene_name(env, sl),
                "frame_index": int("".join(c for c in os.path.basename(path) if c.isdigit())),
                "source_path": os.path.abspath(path), "local_path": os.path.abspath(target),
            })
        print(f"    {scene_name(env, sl)}: {len(chosen)} frames (local)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="corpus output directory")
    ap.add_argument("--manifest", required=True, help="where to write the frame manifest JSON")
    ap.add_argument("--workers", type=int, default=16, help="concurrent slice downloads")
    args = ap.parse_args()

    planned = sum(e.total for e in CORPUS)
    if planned != TARGET_TOTAL:
        raise SystemExit(f"corpus plan sums to {planned}, expected {TARGET_TOTAL}")

    os.makedirs(args.out, exist_ok=True)
    records: List[Dict[str, object]] = []
    for env in CORPUS:
        if env.local_root:
            print(f"[{env.category}] {env.name}/{env.difficulty} -> {env.total} frames (local)")
            fetch_local(env, args.out, records)
    remote = [e for e in CORPUS if not e.local_root]
    if remote:
        print(f"fetching {len(remote)} remote environments with {args.workers} workers")
        records.extend(fetch_remote_all(remote, args.out, args.workers))

    # ---- verification -------------------------------------------------- #
    assert len(records) == TARGET_TOTAL, f"got {len(records)} records, expected {TARGET_TOTAL}"
    locals_ = [r["local_path"] for r in records]
    assert len(set(locals_)) == TARGET_TOTAL, "duplicate frames in the manifest"
    for r in records:
        check_path_is_clean(str(r["source_path"]))
        check_path_is_clean(str(r["local_path"]))
        if not os.path.exists(str(r["local_path"])) or os.path.getsize(str(r["local_path"])) == 0:
            raise SystemExit(f"missing or empty extracted frame: {r['local_path']}")

    per_env: Dict[str, int] = {}
    for r in records:
        per_env[str(r["environment"])] = per_env.get(str(r["environment"]), 0) + 1

    digest = hashlib.sha256()
    for r in sorted(records, key=lambda x: str(x["local_path"])):
        digest.update(str(r["source_path"]).encode())

    payload = {
        "total_frames": len(records),
        "target_total": TARGET_TOTAL,
        "unique_frames": len(set(locals_)),
        "per_environment": per_env,
        "per_scene": {s: sum(1 for r in records if r["scene"] == s)
                      for s in sorted({str(r["scene"]) for r in records})},
        "categories": {e.name: e.category for e in CORPUS},
        "stride": {e.name: e.stride for e in CORPUS},
        "source": "TartanAir V2 (HuggingFace theairlabcmu/tartanair2) + local ForestEnv mirror",
        "forbidden_fragments_checked": list(FORBIDDEN),
        "uses_semantic_labels": False,
        "uses_target_taxonomy": False,
        "selected_using_evaluation_signal": False,
        "manifest_sha256": digest.hexdigest(),
        "frames": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\ncorpus OK: {len(records)} frames, {len(payload['per_scene'])} scenes")
    for k, v in sorted(per_env.items()):
        print(f"  {k:<22}{v:>5}")
    print(f"manifest sha256 {payload['manifest_sha256'][:16]} -> {args.manifest}")


if __name__ == "__main__":
    main()
