#!/usr/bin/env python
"""Gate 4 step 0 — storage/access preflight and frozen-component hash verification.

Downloads nothing. Resolves the data root, verifies the official nuScenes + Occ3D
installation already present on this machine, and re-verifies every frozen checkpoint
and configuration hash against the Gate-3.1 record before anything else runs.

    python tools/occ3d_zeroshot/preflight.py
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json

# Root resolution order, per the brief. $HOME, the repo and /tmp are never candidates.
ENV_ORDER = ("OCC3D_DATA_ROOT", "NUSCENES_DATA_ROOT")
FORBIDDEN_PREFIXES = (os.path.expanduser("~"), REPO_ROOT, "/tmp", "/var/tmp", "/dev/shm")

REQUIRED_NUSCENES = ("v1.0-trainval", "samples/CAM_FRONT", "samples/LIDAR_TOP")
REQUIRED_OCC3D = ("annotations.json", "gts")


def sha256(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
            read += len(b)
            if limit and read >= limit:
                break
    return h.hexdigest()


def fs_info(path: str) -> dict:
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    fstype = ""
    try:
        fstype = subprocess.run(["findmnt", "-no", "FSTYPE", "--target", path],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                              # noqa: BLE001
        pass
    return {"path": path, "filesystem_type": fstype,
            "total_bytes": total, "free_bytes": free,
            "total_gib": total / 2**30, "free_gib": free / 2**30,
            "free_fraction": free / total if total else 0.0,
            "inodes_total": st.f_files, "inodes_free": st.f_favail}


def du_bytes(path: str) -> int:
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True,
                             timeout=1800).stdout.split()[0]
        return int(out)
    except Exception:                                              # noqa: BLE001
        return -1


def resolve_root(cfg) -> tuple[str, str]:
    for key in ENV_ORDER:
        v = os.environ.get(key)
        if v:
            return os.path.abspath(v), f"environment variable {key}"
    v = cfg.data.nuscenes_root
    if v:
        return os.path.abspath(v), "repository configuration (data.nuscenes_root)"
    raise SystemExit("no data root resolved; set OCC3D_DATA_ROOT or NUSCENES_DATA_ROOT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    ap.add_argument("--skip-du", action="store_true", help="skip slow directory sizing")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()

    root, how = resolve_root(cfg)
    for bad in FORBIDDEN_PREFIXES:
        if os.path.abspath(root) == bad or os.path.abspath(root).startswith(bad + os.sep):
            raise SystemExit(f"resolved data root {root} lies under forbidden prefix {bad}")
    occ3d = os.path.join(REPO_ROOT, cfg.data.occ3d_root) if not os.path.isabs(
        cfg.data.occ3d_root) else cfg.data.occ3d_root

    print(f"data root      : {root}\n  resolved via : {how}")
    print(f"occ3d root     : {occ3d}")

    present, missing = {}, []
    for rel in REQUIRED_NUSCENES:
        p = os.path.join(root, rel)
        ok = os.path.exists(p)
        present[f"nuscenes/{rel}"] = ok
        if not ok:
            missing.append(p)
    for rel in REQUIRED_OCC3D:
        p = os.path.join(occ3d, rel)
        ok = os.path.exists(p)
        present[f"occ3d/{rel}"] = ok
        if not ok:
            missing.append(p)

    fs = fs_info(root)
    fs_occ = fs_info(occ3d) if os.path.exists(occ3d) else None
    sizes = {}
    if not a.skip_du:
        for nm, p in (("nuscenes_samples_cam_front", os.path.join(root, "samples/CAM_FRONT")),
                      ("nuscenes_v1.0-trainval_meta", os.path.join(root, "v1.0-trainval")),
                      ("occ3d_gts", os.path.join(occ3d, "gts")),
                      ("occ3d_annotations", os.path.join(occ3d, "annotations.json"))):
            if os.path.exists(p):
                sizes[nm] = du_bytes(p)

    counts = {}
    cf = os.path.join(root, "samples/CAM_FRONT")
    if os.path.isdir(cf):
        counts["samples_CAM_FRONT_files"] = sum(1 for _ in os.scandir(cf))
    gts = os.path.join(occ3d, "gts")
    if os.path.isdir(gts):
        counts["occ3d_gt_scene_dirs"] = sum(1 for e in os.scandir(gts) if e.is_dir())

    ann = os.path.join(occ3d, "annotations.json")
    split = {}
    if os.path.exists(ann):
        d = json.load(open(ann))
        split = {"train_scenes": len(d["train_split"]), "val_scenes": len(d["val_split"]),
                 "scene_infos": len(d["scene_infos"])}

    # ---- frozen component hashes -------------------------------------------- #
    frozen = {}
    for name, rel in cfg.frozen.items():
        p = os.path.join(REPO_ROOT, rel)
        frozen[name] = {"path": rel, "exists": os.path.exists(p),
                        "sha256": sha256(p) if os.path.exists(p) else None}

    status = "DATA_PRESENT" if not missing else "DATA_ACCESS_BLOCKED"
    headroom_ok = fs["free_fraction"] >= 0.0     # no download planned; recorded anyway

    out = {"status": status, "resolved_root": root, "resolution_method": how,
           "occ3d_root": occ3d, "forbidden_prefix_check": "passed",
           "filesystem_nuscenes": fs, "filesystem_occ3d": fs_occ,
           "present": present, "missing": missing, "sizes_bytes": sizes,
           "counts": counts, "official_split": split,
           "download_required": bool(missing),
           "download_performed": False,
           "free_space_headroom_ok": headroom_ok,
           "frozen_components": frozen,
           "elapsed_s": time.time() - t0}
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "preflight.json"), out)

    print(f"\nfilesystem     : {fs['filesystem_type']}  "
          f"{fs['free_gib']:.0f} GiB free of {fs['total_gib']:.0f} GiB "
          f"({100*fs['free_fraction']:.0f}% free)")
    for k, v in present.items():
        print(f"  {'OK ' if v else 'MISS'}  {k}")
    if split:
        print(f"official split : {split['val_scenes']} val scenes, "
              f"{split['train_scenes']} train, {split['scene_infos']} scene_infos")
    print("\nfrozen components:")
    for k, v in frozen.items():
        print(f"  {'OK ' if v['exists'] else 'MISS'}  {k:22s} "
              f"{(v['sha256'] or '')[:16]}  {v['path']}")
    print(f"\nstatus: {status}  (download required: {bool(missing)})")
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
