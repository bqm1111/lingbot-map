#!/usr/bin/env python
"""Fixed Gate 8C-1 audit. No training, selection, prediction correction, or sweeps.

Stages: prepare (freeze IDs/recount historical summaries), infer (recover missing
spatial predictions), analyze (independent labels/counts/bins), integrity (actual
evaluator forward path with perturbed scoring data). All writes stay in OUT.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "artifacts/gate8c1"
OUT = ROOT / "artifacts/gate8c1_original_audit"
OCC = Path("/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred")
MAN = json.loads((OLD / "frozen_manifest.json").read_text())
DS = ("semantickitti", "occ3d")
METHODS = ("mapper", "dilation", "seed0", "seed1", "seed2")
OFFSETS = {"past5": (-4, -3, -2, -1, 0), "occany_fwd": (0, 2, 4, 6, 8)}
BIN_EDGES = {"height": [-np.inf, 0, 1, 2, 3, np.inf],
             "range": [0, 10, 20, 30, 40, np.inf],
             "boundary": [0, .4, 1, 2, np.inf],
             "gt_distance": [0, .4, 1, 2, 5, np.inf]}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metric(c):
    tp, fp, fn, tn = map(int, c)
    div = lambda a, b: a / b if b else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, valid=tp+fp+fn+tn,
                gt_occupied=tp+fn, pred_occupied=tp+fp,
                iou=div(tp, tp+fp+fn), precision=div(tp, tp+fp),
                recall=div(tp, tp+fn), fpr=div(fp, fp+tn),
                prevalence=div(tp+fn, tp+fp+fn+tn), volume_ratio=div(tp+fp, tp+fn))


def count(pred, gt, valid):
    # Independent explicit binary population; no metric/preprocessing helper.
    return np.array([np.count_nonzero(valid & pred & gt),
                     np.count_nonzero(valid & pred & ~gt),
                     np.count_nonzero(valid & ~pred & gt),
                     np.count_nonzero(valid & ~pred & ~gt)], dtype=np.int64)


def target(ds, rec, corrected=False):
    if ds == "semantickitti":
        import yaml
        base = ROOT / "data/kitti/dataset/sequences" / rec["sequence"] / "voxels" / f"{rec['frame_ids'][-1]:06d}"
        raw = np.fromfile(str(base)+".label", dtype="<u2").reshape(256, 256, 32)
        invalid = np.unpackbits(np.fromfile(str(base)+".invalid", dtype=np.uint8), bitorder="big").reshape(raw.shape).astype(bool)
        lm = yaml.safe_load(Path("/home/minh/workspace/OccAny/occany/datasets/semantic_kitti.yaml").read_text())["learning_map"]
        assert set(np.unique(raw)).issubset(lm), "Unmapped raw label: do not clamp it"
        lut = np.array([lm.get(i, 0) for i in range(65536)], dtype=np.uint16)
        label = lut[raw].astype(np.int32)
        valid = ~invalid
        if corrected:
            valid &= (raw == 0) | (label != 0)
        label[~valid] = 255
        return label, (label != 0) & valid, valid
    p = Path("/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval") / rec["anchor_gt_path"]
    with np.load(p) as z:
        label = z["semantics"].astype(np.int32)
        assert label.shape == (200, 200, 16)
        assert set(np.unique(label)).issubset(set(range(18)) | {255})
        valid = z["mask_camera"].astype(bool) & (label != 255)
    valid[:100] = False  # Official single-camera half-grid convention; not a frustum.
    label[~valid] = 255
    return label, (label != 17) & valid, valid


def historical(ds, mode, seed):
    with np.load(OLD / f"scores_{ds}_{mode}_seed{seed}_completion.npz") as z:
        tau = MAN["seeds"][str(seed)]["occupancy_threshold"]
        j = int((tau + 16) * 32)
        assert j == (tau + 16) * 32
        tp = z["full_pos"][:, j:].sum(1, dtype=np.int64)
        fp = z["full_neg"][:, j:].sum(1, dtype=np.int64)
        fn = z["full_pos"][:, :j].sum(1, dtype=np.int64)
        tn = z["full_neg"][:, :j].sum(1, dtype=np.int64)
        ids = z["full_clip_id"].tolist()
    return ids, np.stack([tp, fp, fn, tn], 1)


def prepare():
    from gates.gate8 import sources as S
    from gates.gate6.grids import PREDICTION_GRID, EVAL_GRID
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "samples.json").exists():
        raise RuntimeError("Sample freeze exists; preserve it")
    refs, samples, summary = {}, {}, {}
    def record(p, expected=None):
        p = Path(p)
        if not p.is_absolute(): p = ROOT / p
        refs[str(p)] = {"exists": p.exists()}
        if p.is_file():
            refs[str(p)].update(bytes=p.stat().st_size, sha256=sha(p))
        if expected:
            refs[str(p)].update(expected_sha256=expected, matches=refs[str(p)].get("sha256") == expected)
    for f in ("frozen_manifest.json", "selection.json", "report.md", "gate8c1_results.json"):
        record(OLD / f)
    for p, h in {**MAN["code_hashes"], **MAN["config_hashes"]}.items(): record(p, h)
    for fz in MAN["seeds"].values():
        record(fz["checkpoint"], fz["checkpoint_sha256"])
        assert refs[str(ROOT / fz["checkpoint"])]["matches"]
    # Preserve the code actually used for this audit, explicitly separate from the
    # unavailable dirty-worktree snapshot referenced by the original manifest.
    files = list((ROOT / "gates" / "gate8").glob("*.py")) + [ROOT / p for p in (
        "tools/gate8c1/eval_target.py", "gates/gate6/targets.py", "gates/gate6/grids.py",
        "gates/gate8a/regions.py", "gates/gate8a/scores.py", "gates/gate7b/replay.py",
        "gates/gate7b/streams.py", "lingbot_map/models/gct_stream.py")]
    for p in files:
        record(p)
        dest = OUT / "code_used" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for ds in DS:
        segs = S.segments(ds, str(ROOT))
        byid = {f.gt_ref["clip_id"]: (seg, i, f) for seg in segs for i, f in enumerate(seg.frames) if f.gt_ref}
        samples[ds], summary[ds] = {}, {}
        oname = "kitti" if ds == "semantickitti" else "nuscenes"
        ocfiles = sorted((OCC / f"OccAny_5frames_{oname}512_rot60_vpi10_fwd3_sTrans2").glob("*/voxel_predictions.pkl"))
        occmap = {p.parent.name.split("_", 1)[1]: str(p) for p in ocfiles}
        for seg in segs:
            record(S.stream_path(ds, seg.name)); record(S.scale_path(ds, seg.name))
        for mode in OFFSETS:
            ids, _ = historical(ds, mode, 0)
            rows = []
            for cid in ids:
                seg, i, f = byid[cid]
                used = [i + o for o in OFFSETS[mode]]
                assert min(used) >= 0 and max(used) < len(seg)
                key = f"{f.gt_ref['frame_ids'][-1]:06d}" if ds == "semantickitti" else f.gt_ref["sample_tokens"][-1]
                with np.load(S.scale_path(ds, seg.name)) as z: scale = float(np.exp(np.median(z["log_s"][:5])))
                row = dict(clip_id=cid, segment=seg.name, anchor_index=i, anchor_order=f.order,
                           gt_ref=f.gt_ref, used_indices=used, input_keys=[seg.frames[j].key for j in used],
                           input_images=[seg.frames[j].path for j in used], input_orders=[seg.frames[j].order for j in used],
                           stream_prefix_keys=[x.key for x in seg.frames[:max(used)+1]],
                           scale_frame_keys=[x.key for x in seg.frames[:5]], scale=scale,
                           stream_cache=S.stream_path(ds, seg.name), scale_cache=S.scale_path(ds, seg.name),
                           T_cam_to_grid=np.asarray(f.T_cam_to_grid).tolist(), occany=occmap.get(key))
                rows.append(row)
            excluded = sorted(set(byid) - set(ids))
            samples[ds][mode] = dict(rows=rows, excluded_original_anchors=excluded,
                verification_ids=[ids[j] for j in sorted(set([0, len(ids)//2, len(ids)-1]))],
                common_occany_ids=[r["clip_id"] for r in rows if r["occany"]],
                missing_occany_ids=[r["clip_id"] for r in rows if not r["occany"]],
                n_occany_available=len(ocfiles))
            summary[ds][mode] = {}
            for seed in range(3):
                ids2, c = historical(ds, mode, seed)
                assert ids == ids2
                np.savez_compressed(OUT / f"historical_{ds}_{mode}_seed{seed}.npz", clip_id=ids2, counts=c)
                summary[ds][mode][str(seed)] = metric(c.sum(0))
                for stem in ("scores", "counts"):
                    record(OLD / f"{stem}_{ds}_{mode}_seed{seed}_completion.npz")
            summary[ds][mode]["seed_summary"] = {k: {"mean": float(np.mean([summary[ds][mode][str(s)][k] for s in range(3)])), "median": float(np.median([summary[ds][mode][str(s)][k] for s in range(3)])), "sample_sd": float(np.std([summary[ds][mode][str(s)][k] for s in range(3)], ddof=1))} for k in ("iou", "precision", "recall", "fpr", "prevalence", "volume_ratio")}
        summary[ds]["grids"] = {k: dict(dims=list(g.dims), origin=list(g.origin), voxel_size=g.voxel_size) for k, g in [("prediction", PREDICTION_GRID[ds]), ("evaluation", EVAL_GRID[ds])]}
    dump(OUT / "samples.json", samples)
    dump(OUT / "historical_metrics.json", summary)
    dump(OUT / "artifact_references.json", refs)
    (OUT / "git_status_before.txt").write_text(subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True))
    print({ds: {m: len(samples[ds][m]["rows"]) for m in OFFSETS} for ds in DS}, flush=True)


def reduce_binary(p, ds):
    from gates.gate8a.regions import occ_to_eval_grid
    from gates.gate6.grids import EVAL_GRID
    return occ_to_eval_grid(p, ds).cpu().numpy().reshape(EVAL_GRID[ds].dims)


def infer(ds, mode, device, verify_only=False):
    import torch
    from gates.gate8 import sources as S, vocab as V8
    from gates.gate8.feed import CachedFeed
    from gates.gate8.net import load_checkpoint
    from gates.gate6.grids import PREDICTION_GRID
    from tools.gate8c1.eval_target import window_map
    from gates.voxel_gate.voxels import dilate
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    doc = json.loads((OUT / "samples.json").read_text())[ds][mode]
    rows = doc["rows"]
    if verify_only: rows = [r for r in rows if r["clip_id"] in doc["verification_ids"]]
    out = OUT / "predictions" / ds / mode; out.mkdir(parents=True, exist_ok=True)
    nets = [load_checkpoint(str(ROOT / MAN["seeds"][str(s)]["checkpoint"]), device) for s in range(3)]
    assert all(n.net.n_params() == 986114 for n in nets)
    segs = {s.name: s for s in S.segments(ds, str(ROOT))}
    feed, last = None, None
    into = V8.into_matrix(ds); grid = PREDICTION_GRID[ds]
    start = time.time()
    for n, row in enumerate(rows):
        path = out / (row["clip_id"] + ".npz")
        if path.exists(): continue
        seg = segs[row["segment"]]; i = row["anchor_index"]
        if last != seg.name: feed, last = CachedFeed(seg, device), seg.name
        m, used = window_map(feed, seg, i, OFFSETS[mode], row["scale"], into, into.shape[0], device)
        P = feed.pose[i].copy(); P[:3, 3] *= row["scale"]
        q = m.query(grid, P @ np.linalg.inv(np.asarray(row["T_cam_to_grid"])))
        q["age"] = torch.where(q["last_time"] >= 0, (i-q["last_time"]).float(), torch.full_like(q["last_time"], -1).float())
        native = q["occupied"] & (q["sem_w"] > 0)
        preds = [reduce_binary(native, ds), reduce_binary(dilate(native.reshape(grid.dims), 2).reshape(-1), ds)]
        for s, net in enumerate(nets):
            fin, _ = net.raw(q, grid, pad_z=0)
            preds.append(reduce_binary(fin >= MAN["seeds"][str(s)]["occupancy_threshold"], ds))
        # Only unmasked predictions are serialized; labels have not been loaded.
        np.savez_compressed(path, **{name: np.packbits(pred.reshape(-1)) for name, pred in zip(METHODS, preds)})
        del m, q, fin
        if (n+1) % 25 == 0 or verify_only:
            print(ds, mode, n+1, "/", len(rows), "seconds", round(time.time()-start, 1), flush=True)


def analyze(ds, mode, verify_only=False):
    from scipy.ndimage import distance_transform_edt, maximum_filter
    from gates.gate6 import targets as old_targets
    from gates.gate6.grids import EVAL_GRID
    from gates.gate8c1.occany_eval import official_sc_counts
    doc = json.loads((OUT / "samples.json").read_text())[ds][mode]
    rows = doc["rows"]
    if verify_only: rows = [r for r in rows if r["clip_id"] in doc["verification_ids"]]
    grid = EVAL_GRID[ds]; shape = tuple(grid.dims); size = int(np.prod(shape))
    coords = np.indices(shape, dtype=np.float32)
    xyz = [(coords[a]+.5)*grid.voxel_size+grid.origin[a] for a in range(3)]
    boundary = np.minimum.reduce([np.minimum((coords[a]+.5)*grid.voxel_size, (shape[a]-.5-coords[a])*grid.voxel_size) for a in range(3)])
    values = dict(height=xyz[2], range=np.hypot(xyz[0], xyz[1]), boundary=boundary)
    static = {k: np.searchsorted(BIN_EDGES[k][1:-1], v, side="right").ravel() for k, v in values.items()}
    hist = {s: dict(zip(*historical(ds, mode, s))) for s in range(3)}
    totals, binned, changes, checks, countrows, occrows = {}, {}, {}, [], [], []
    corrected_rows, populations = [], []
    suffix = "_verify" if verify_only else ""
    for n, row in enumerate(rows):
        cid = row["clip_id"]; path = OUT / "predictions" / ds / mode / (cid+".npz")
        with np.load(path) as z: preds = {m: np.unpackbits(z[m], count=size).reshape(shape).astype(bool) for m in METHODS}
        label, gt, valid = target(ds, row["gt_ref"])
        clabel, cgt, cvalid = target(ds, row["gt_ref"], corrected=True)
        populations.append(dict(clip_id=cid, historical_valid=int(valid.sum()), corrected_valid=int(cvalid.sum()), excluded_falsely_free=int((valid & ~cvalid).sum())))
        ol, ov = old_targets.semantic_target(ds, row["gt_ref"], str(ROOT))
        assert np.array_equal(label, ol) and np.array_equal(valid, ov), cid
        # Distance is to the centre of a scored occupied voxel, not to a physical surface.
        if mode == "past5":
            d = distance_transform_edt(~gt, sampling=grid.voxel_size) if gt.any() else np.full(shape, np.inf)
            bins = dict(static, gt_distance=np.searchsorted(BIN_EDGES["gt_distance"][1:-1], d, side="right").ravel())
        for name, pred in preds.items():
            c = count(pred, gt, valid)
            totals[name] = totals.get(name, np.zeros(4, np.int64)) + c
            countrows.append(dict(clip_id=cid, method=name, **metric(c)))
            corrected_rows.append(dict(clip_id=cid, method=name, **metric(count(pred, cgt, cvalid))))
            if name.startswith("seed"):
                s = int(name[-1]); hc = hist[s][cid]
                checks.append(dict(clip_id=cid, seed=s, delta_from_histogram=(c-hc).tolist()))
                added = pred & ~preds["mapper"]
                removed = ~pred & preds["mapper"]
                masks = dict(added_tp=added & gt, added_fp=added & ~gt,
                             retained_tp=pred & preds["mapper"] & gt,
                             removed_fp=removed & ~gt, removed_tp=removed & gt)
                cc = {k: int(np.count_nonzero(v & valid)) for k, v in masks.items()}
                for k, v in cc.items(): changes.setdefault(name, {}).setdefault(k, 0); changes[name][k] += v
            if mode == "past5":
                masks4 = [valid & pred & gt, valid & pred & ~gt, valid & ~pred & gt, valid & ~pred & ~gt]
                for dim, bi in bins.items():
                    cb = np.stack([np.bincount(bi[x.ravel()], minlength=len(BIN_EDGES[dim])-1) for x in masks4], 1)
                    key = (name, dim)
                    binned[key] = binned.get(key, np.zeros_like(cb)) + cb
            if cid in doc["verification_ids"]:
                official = official_sc_counts(np.where(pred, 1, grid.empty_class).astype(np.uint8), label.astype(np.uint8), 20 if ds == "semantickitti" else 18, grid.empty_class)
                assert np.array_equal(c[:3], official), (cid, name)
        if mode == "occany_fwd" and row["occany"]:
            with open(row["occany"], "rb") as f: oc = pickle.load(f)
            assert np.array_equal(oc["voxel_label"], clabel), (cid, "OccAny labels")
            tau = "2.5" if ds == "semantickitti" else "1.1"
            op = np.asarray(oc[f"render_recon_gen_recon{tau}_gen{tau}"]) != grid.empty_class
            both = {"occany": op, **{k:v for k,v in preds.items() if k.startswith("seed")}}
            for name, pred in both.items():
                for pool in (False, True):
                    pp = maximum_filter(pred, size=3, mode="constant", cval=0) if pool else pred
                    c = count(pp, cgt, cvalid)
                    occrows.append(dict(clip_id=cid, method=name+("_pool_then_mask" if pool else "_raw"), **metric(c)))
        if (n+1) % 100 == 0: print("count", ds, mode, n+1, flush=True)
    def csvwrite(name, items):
        with (OUT/name).open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
    csvwrite(f"counts_{ds}_{mode}{suffix}.csv", countrows)
    csvwrite(f"corrected_counts_{ds}_{mode}{suffix}.csv", corrected_rows)
    csvwrite(f"populations_{ds}_{mode}{suffix}.csv", populations)
    if occrows: csvwrite(f"occany_counts_{ds}{suffix}.csv", occrows)
    if binned:
        br = [dict(method=name, dimension=dim, lo=str(BIN_EDGES[dim][i]), hi=str(BIN_EDGES[dim][i+1]), **metric(c)) for (name,dim), cs in binned.items() for i,c in enumerate(cs)]
        csvwrite(f"bins_{ds}{suffix}.csv", br)
    result = dict(n_samples=len(rows), metrics={k:metric(v) for k,v in totals.items()}, transitions=changes,
                  labels_and_masks_equal_all=True, histogram_max_abs_delta=int(np.max(np.abs([c["delta_from_histogram"] for c in checks]))),
                  histogram_sum_abs_delta=np.abs([c["delta_from_histogram"] for c in checks]).sum(0).tolist())
    dump(OUT/f"summary_{ds}_{mode}{suffix}.json", result)
    dump(OUT/f"verification_{ds}_{mode}{suffix}.json", checks)
    print(json.dumps(result), flush=True)


def integrity(ds, device):
    """Execute the existing evaluator up to unmasked output, with scorer perturbations."""
    import torch
    import tools.gate8c1.eval_target as E
    from gates.gate8.net import load_checkpoint
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4); torch.cuda.set_device(device)
    net = load_checkpoint(str(ROOT / MAN["seeds"]["0"]["checkpoint"]), device)
    original_target = E.G6T.semantic_target
    recorded = []
    class Done(Exception): pass
    class Capture:
        def raw(self, q, grid):
            fin, probs = net.raw(q, grid, pad_z=0)
            digest = lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
            recorded.append(dict(final=digest(fin), probabilities=digest(probs), occupancy=digest(fin >= MAN["seeds"]["0"]["occupancy_threshold"]),
                                 editable=digest(q["logodds"].abs() < 2),
                                 query={k:digest(v) for k,v in q.items()}))
            raise Done()
    old_load=E.load_checkpoint; E.load_checkpoint=lambda *args: Capture()
    try:
        for variant in ("original", "all_free_all_valid", "all_occupied_half_valid"):
            def changed(dataset, rec, root):
                label, valid=original_target(dataset, rec, root)
                if variant == "all_free_all_valid":
                    label[:]=E.G6G.EVAL_GRID[ds].empty_class; valid[:]=True
                if variant == "all_occupied_half_valid":
                    label[:]=1; valid[:]=True; valid[::2]=False; label[~valid]=255
                return label,valid
            E.G6T.semantic_target=changed
            torch.manual_seed(0); np.random.seed(0)
            try: E.run(ds, "past5", 0, device, limit=1, art=str(OUT/"integrity_unused"))
            except Done: pass
    finally: E.G6T.semantic_target=original_target; E.load_checkpoint=old_load
    dump(OUT/f"integrity_{ds}.json", dict(actual_entry_point="tools.gate8c1.eval_target.run", variants=["original", "all_free_all_valid", "all_occupied_half_valid"], identical=all(x==recorded[0] for x in recorded), records=recorded))
    assert len(recorded)==3 and all(x==recorded[0] for x in recorded)
    print("forward label/mask perturbation identical", ds, flush=True)


def streamcheck(ds, device):
    """Small foundation replay: appended future images must not change a prefix."""
    import torch
    from gates.gate7b import replay
    from gates.gate8 import sources as S
    # Activate the existing environment's build executables; install nothing.
    os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/local/cuda/bin:" + os.environ["PATH"]
    torch.set_num_threads(4); torch.cuda.set_device(device); torch.manual_seed(0)
    seg=S.segments(ds, str(ROOT))[0]
    model, loaded=replay.build_model(device)
    images=replay.load_images([f.path for f in seg.frames[:11]])
    k=replay.keyframe_interval_for(len(seg))
    a=replay.replay_segment(model, images[:9], k, device)
    b=replay.replay_segment(model, images[:11], k, device)
    z=np.load(S.stream_path(ds, seg.name))
    keys=("pred_depth", "pred_depth_conf", "pred_pose_c2w", "pred_K", "pose_enc")
    def comparison(x,y):
        delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
        return dict(equal=bool(np.array_equal(x,y)), different_elements=int(np.count_nonzero(delta)), max_abs=float(delta.max()), mean_abs=float(delta.mean()))
    result=dict(segment=seg.name, images=[f.path for f in seg.frames[:11]], keyframe_interval=k,
                checkpoint_load=loaded, normalization=bool(model.pred_normalization),
                prefix_9_vs_11={key:comparison(a[key],b[key][:9]) for key in keys},
                replay_vs_cache={key:comparison(a[key],z[key][:9]) for key in keys})
    dump(OUT/f"streamcheck_{ds}.json", result)
    print(json.dumps(result), flush=True)


def provenance():
    """Hash consumed artifacts and verify OccAny image/calibration provenance."""
    import ast
    import importlib.util
    from PIL import Image
    from gates.gate6 import frames as F
    from gates.gate8 import sources as S
    from gates.gate7b.scale import frame_candidate
    from gates.gate7b.depth import unpack_mask
    samples=json.loads((OUT/"samples.json").read_text())
    official=Path("/home/minh/workspace/OccAny")
    spec=importlib.util.spec_from_file_location("audit_cropping",official/"occany/utils/cropping.py")
    cropping=importlib.util.module_from_spec(spec); spec.loader.exec_module(cropping)
    tree=ast.parse((official/"occany/utils/helpers.py").read_text())
    cropnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="crop_resize_if_necessary")
    scope={"np":np,"Image":Image,"cropping":cropping}
    exec(compile(ast.Module(body=[cropnode],type_ignores=[]),"official_crop", "exec"),scope)
    paths=set(); evidence=[]; scale_checks=[]
    for ds in DS:
        segs=S.segments(ds,str(ROOT))
        for seg in segs:
            with np.load(S.scale_path(ds,seg.name)) as z:
                assert np.isfinite(z['log_s'][:5]).all()
            for f in seg.frames:
                paths.add(Path(S.trident_path(ds,f.key)))
                paths.add(Path(f.path))
            paths.update(Path(S.moge_path(ds,f.key)) for f in seg.frames[:5])
        allrows=samples[ds]['past5']['rows']
        for row in allrows:
            rec=row['gt_ref']
            if ds=='semantickitti':
                base=ROOT/'data/kitti/dataset/sequences'/rec['sequence']/'voxels'/f"{rec['frame_ids'][-1]:06d}"
                paths.update([Path(str(base)+'.label'),Path(str(base)+'.invalid')])
            else:
                paths.add(Path('/media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval')/rec['anchor_gt_path'])
                paths.add(Path(F.lingbot_cache_path(ds,rec['clip_id'],str(ROOT))))
        for row in samples[ds]['occany_fwd']['rows']:
            if row['occany']: paths.add(Path(row['occany']))
        # First/middle/last forward IDs were fixed by prepare before fresh scoring.
        for cid in samples[ds]['occany_fwd']['verification_ids']:
            row=next(r for r in samples[ds]['occany_fwd']['rows'] if r['clip_id']==cid)
            if not row['occany']: continue
            oc=pickle.load(open(row['occany'],'rb'))
            if ds=='semantickitti':
                _,K=S._sk_calib(str(ROOT),'08'); Ks=[K]*5
            else:
                # Native intrinsics are constant within these five CAM_FRONT images.
                with np.load(F.lingbot_cache_path(ds,cid,str(ROOT))) as z: K=z['K_native'][-1]
                Ks=[K]*5
            images=oc['estimated_input_images']; diffs=[]; kdiff=[]
            for i,(impath,K) in enumerate(zip(row['input_images'],Ks)):
                im=Image.open(impath).convert('RGB')
                processed,_,kp=scope['crop_resize_if_necessary'](im,np.zeros((im.height,im.width),np.float32),K,(images.shape[2],images.shape[1]))
                arr=np.asarray(processed)
                delta=abs(arr.astype(float)-images[i].astype(float))
                diffs.append(dict(mean_abs=float(delta.mean()),max_abs=float(delta.max())))
                kdiff.append(float(abs(kp-oc['gt_input_intrinsics'][i]).max()))
            evidence.append(dict(dataset=ds,clip_id=cid,input_images=row['input_images'],input_orders=row['input_orders'],pixel_differences=diffs,intrinsics_max_abs=kdiff,extrinsic_max_abs=float(abs(np.asarray(row['T_cam_to_grid'])-oc['T_cam_to_voxel']).max())))
        for seg in [segs[0]]:
            with np.load(S.stream_path(ds,seg.name)) as stream, np.load(S.scale_path(ds,seg.name)) as scales:
                for i in range(5):
                    with np.load(S.moge_path(ds,seg.frames[i].key)) as z:
                        c=frame_candidate(z['moge_depth'],unpack_mask(z['moge_mask'],z['mask_shape']),stream['pred_depth'][i],stream['pred_depth_conf'][i])
                    scale_checks.append(dict(dataset=ds,key=seg.frames[i].key,saved_log_s=float(scales['log_s'][i]),recomputed_log_s=c['log_s']))
    paths.update(official/p for p in ['checkpoints/occany.pth','occany/datasets/semantic_kitti.yaml','occany/datasets/semantic_kitti_io.py','occany/datasets/kitti.py','occany/datasets/nuscenes.py','occany/datasets/eval_helper.py','occany/metrics/ssc.py','occany/utils/helpers.py','extract_output_occany.py','compute_metrics_from_saved_voxels.py','occany/must3r_inference.py'])
    for p in (ROOT/'artifacts/gate7b').glob('*.json'):
        if p.name.startswith(('stream_','scale_candidates_','moge_b_cache_')): paths.add(p)
    paths.update([ROOT/'artifacts/scale_gate/manifests/val.jsonl',ROOT/'manifests/occ3d_zeroshot/val.jsonl',ROOT/'data/kitti/dataset/sequences/08/calib.txt',ROOT/'data/kitti/dataset/sequences/08/times.txt'])
    source_listing={}
    for kind,key in [('targets','source_dataset_hashes'),('samples','sample_hashes')]:
        for drive in [3,7,10,6]:
            base=Path('/media/SSD1/MINH_DATASETS/lingbot_gate8c1')/kind/f'2013_05_28_drive_{drive:04d}_sync'
            h=hashlib.sha256(); files=sorted(base.glob('*.npz')); total=0
            for p in files:
                n=p.stat().st_size; total+=n; h.update(p.name.encode()); h.update(str(n).encode())
            actual=dict(n_files=len(files),total_bytes=total,listing_sha256=h.hexdigest())
            frozen=MAN[key][f'{kind}_{drive:04d}']
            source_listing[f'{kind}_{drive:04d}']=dict(current=actual,frozen=frozen,matches=actual==frozen)
    with (OUT/'consumed_artifact_hashes.jsonl').open('w') as f:
        for i,p in enumerate(sorted(paths)):
            row=dict(path=str(p),exists=p.exists())
            if p.is_file(): row.update(bytes=p.stat().st_size,sha256=sha(p))
            f.write(json.dumps(row)+'\n')
            if (i+1)%1000==0: print('hashed',i+1,'/',len(paths),flush=True)
    dump(OUT/'input_integrity.json',dict(occany_image_checks=evidence,scale_checks=scale_checks,source_listings=source_listing))


def finish():
    """Summarize the fixed evidence and draw predetermined examples from saved arrays."""
    from collections import defaultdict
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image
    from gates.gate6.grids import EVAL_GRID
    from gates.gate6 import metrics as G6M, vocab as G6V
    from gates.gate8c1.occany_eval import official_sc_counts
    allresults={}
    evaluator_checks=[]; baseline_checks=[]
    read=lambda p:list(csv.DictReader(p.open()))
    for ds in DS:
        result={}; allresults[ds]=result
        for mode in OFFSETS:
            result[mode]={}
            for convention,prefix in [('historical',''),('corrected','corrected_')]:
                rows=read(OUT/f'{prefix}counts_{ds}_{mode}.csv')
                grouped=defaultdict(list)
                for r in rows: grouped[r['method']].append(r)
                metrics={}
                for name,rs in grouped.items():
                    c=np.array([[int(r[k]) for k in ('tp','fp','fn','tn')] for r in rs])
                    metrics[name]=dict(pooled=metric(c.sum(0)),mean_sample_iou=float(np.mean([float(r['iou']) for r in rs])))
                seedvals=[metrics[f'seed{s}']['pooled'] for s in range(3)]
                metrics['seed_aggregation']={k:dict(mean=float(np.mean([r[k] for r in seedvals])),median=float(np.median([r[k] for r in seedvals])),sample_sd=float(np.std([r[k] for r in seedvals],ddof=1))) for k in ('iou','precision','recall','fpr','prevalence','volume_ratio')}
                metrics['pooled_all_seeds']=metric(np.array([[r[k] for k in ('tp','fp','fn','tn')] for r in seedvals]).sum(0))
                result[mode][convention]=metrics
            ocrows=OUT/f'occany_counts_{ds}.csv'
            if mode=='occany_fwd' and ocrows.exists():
                grouped=defaultdict(list)
                for r in read(ocrows): grouped[r['method']].append(r)
                result[mode]['occany_common']={k:dict(n_samples=len(rs),**metric(np.array([[int(r[f]) for f in ('tp','fp','fn','tn')] for r in rs]).sum(0))) for k,rs in grouped.items()}
        samples=json.loads((OUT/'samples.json').read_text())[ds]['past5']
        grid=EVAL_GRID[ds]; dims=tuple(grid.dims); origin=np.array(grid.origin); vs=grid.voxel_size
        vocab=G6V.load(ds)
        for cid in samples['verification_ids']:
            row=next(r for r in samples['rows'] if r['clip_id']==cid)
            label,gt,valid=target(ds,row['gt_ref'])
            with np.load(OUT/'predictions'/ds/'past5'/(cid+'.npz')) as z:
                for method in METHODS:
                    pred=np.unpackbits(z[method],count=np.prod(dims)).reshape(dims).astype(bool)
                    pl=np.where(pred,1,grid.empty_class).astype(np.int32)
                    c=count(pred,gt,valid)
                    old=G6M.clip_counts(pl,None,label,valid,vocab.labels,grid.empty_class,[],[],cid,row['segment'],method)
                    oc=official_sc_counts(pl,label,20 if ds=='semantickitti' else 18,grid.empty_class)
                    assert (old.btp,old.bfp,old.bfn)==tuple(c[:3])==oc
                    assert old.n_valid==int(valid.sum()) and old.n_gt_occupied==int(gt.sum())
                    evaluator_checks.append(dict(dataset=ds,clip_id=cid,method=method,all_equal=True,counts=c.tolist()))
        for mode in OFFSETS:
            rs=read(OUT/f'counts_{ds}_{mode}.csv')
            with np.load(OLD/f'bincounts_{ds}_{mode}_seed0.npz') as old:
                for method,key in [('mapper','pred_mapper_native'),('dilation','pred_mapper_dilate')]:
                    ours={r['clip_id']:np.array([int(r[k]) for k in ('tp','fp','fn')]) for r in rs if r['method']==method}
                    delta=np.array([ours[cid]-c for cid,c in zip(old['clip_id'],old[key])])
                    assert not delta.any()
                    baseline_checks.append(dict(dataset=ds,mode=mode,method=method,n_samples=len(ours),exact=True))
        for cid in samples['verification_ids'][:2]:
            row=next(r for r in samples['rows'] if r['clip_id']==cid)
            with np.load(OUT/'predictions'/ds/'past5'/(cid+'.npz')) as z:
                mapper=np.unpackbits(z['mapper'],count=np.prod(dims)).reshape(dims).astype(bool)
                pred=np.unpackbits(z['seed0'],count=np.prod(dims)).reshape(dims).astype(bool)
            _,gt,valid=target(ds,row['gt_ref'])
            addtp=pred & ~mapper & gt & valid; addfp=pred & ~mapper & ~gt & valid
            fig=plt.figure(figsize=(15,8),layout='constrained')
            ax=fig.add_subplot(231); ax.imshow(Image.open(row['input_images'][-1])); ax.set_title('Anchor RGB (last of five past inputs)'); ax.axis('off')
            for slot,vols,title in [(232,[(mapper,'#65717d')],'Incremental mapper: input geometry'),(233,[(addtp,'#13a56b'),(addfp,'#e44842')],'Completion additions: correct / false')]:
                ax=fig.add_subplot(slot,projection='3d')
                for vol,col in vols:
                    pts=np.argwhere(vol & valid); pts=pts[::max(1,int(np.ceil(len(pts)/10000)))]; pts=(pts+.5)*vs+origin
                    if len(pts): ax.scatter(*pts.T,s=1,c=col,alpha=.4,rasterized=True)
                ax.set(xlim=(origin[0],origin[0]+dims[0]*vs),ylim=(origin[1],origin[1]+dims[1]*vs),zlim=(origin[2],origin[2]+dims[2]*vs),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title)
                ax.set_box_aspect((2,2,.7)); ax.view_init(22,-58)
            categories=np.zeros(dims,np.uint8); categories[~valid]=1
            categories[gt & valid]=2; categories[mapper & valid]=3; categories[addtp]=4; categories[addfp]=5
            cmap=ListedColormap(['white','#dddddd','#508ad3','#65717d','#13a56b','#e44842'])
            # Fixed physical slices: y=0, x=10, z=2 m. Exactly one voxel thick.
            for slot,axis,pos,labels in [(234,1,0,('x','z')),(235,0,10,('y','z')),(236,2,2,('x','y'))]:
                j=int(np.clip(np.floor((pos-origin[axis])/vs),0,dims[axis]-1)); arr=np.take(categories,j,axis=axis)
                axes=[a for a in range(3) if a!=axis]; ext=[origin[axes[0]],origin[axes[0]]+dims[axes[0]]*vs,origin[axes[1]],origin[axes[1]]+dims[axes[1]]*vs]
                ax=fig.add_subplot(slot); ax.imshow(arr.T,origin='lower',extent=ext,cmap=cmap,vmin=0,vmax=5,interpolation='nearest',aspect='auto')
                ax.set(xlabel=labels[0]+' (m)',ylabel=labels[1]+' (m)',title=f"{'xyz'[axis]}={(j+.5)*vs+origin[axis]:.1f} m; {vs:.1f} m slice")
            fig.suptitle(f'{ds} | {cid} | original seed 0, threshold -0.125\nGreen: added TP; red: added FP; slate: mapper; blue: GT occupancy; grey: ignored. 3D points are subsampled; slices are not.')
            fig.savefig(OUT/f'example_{ds}_{cid}.png',dpi=140); plt.close(fig)
    paired=[]
    for seed in range(3):
        a,b=[allresults[ds]['past5']['historical'][f'seed{seed}']['pooled'] for ds in DS]
        precision=lambda p,r,f: p*r/(p*r+(1-p)*f)
        paired.append(dict(seed=seed,sk_precision=a['precision'],occ3d_precision=b['precision'],
            change_only_prevalence=precision(b['prevalence'],a['recall'],a['fpr']),
            then_change_recall=precision(b['prevalence'],b['recall'],a['fpr']),
            then_change_fpr=precision(b['prevalence'],b['recall'],b['fpr']),
            log_odds_gap_components=dict(prevalence=float(np.log((b['prevalence']/(1-b['prevalence']))/(a['prevalence']/(1-a['prevalence'])))),recall=float(np.log(b['recall']/a['recall'])),fpr=float(np.log(a['fpr']/b['fpr'])))))
    allresults['precision_accounting']=paired
    dump(OUT/'results.json',allresults)
    dump(OUT/'evaluator_equivalence.json',dict(deterministic_examples=evaluator_checks,historical_baselines=baseline_checks))
    print('wrote results.json and four deterministic figures',flush=True)


def mechanism(device):
    """Uniform unknown input diagnostic, and visibility of the fixed real outputs.

    Synthetic outputs are never substituted into any benchmark prediction.
    This distinguishes an input-independent spatial response from observed geometry;
    overlap with real errors does not by itself prove their causal mechanism.
    """
    import torch
    from gates.gate8.net import load_checkpoint
    from gates.gate6.grids import PREDICTION_GRID, EVAL_GRID, voxelize
    from gates.gate8 import vocab as V8
    torch.set_num_threads(4); torch.cuda.set_device(device)
    samples=json.loads((OUT/'samples.json').read_text())
    records=[]; synth_profiles=[]; overlap=[]; boundary_checks=[]
    for ds in DS:
        grid=PREDICTION_GRID[ds]; n=int(np.prod(grid.dims))
        q={k:torch.zeros(n,device=device) for k in ['sem_w','observed','logodds','w_free','n_obs']}
        q['age']=torch.full((n,),-1.,device=device); q['sem']=torch.zeros(n,V8.U,device=device)
        synth=[]
        for s in range(3):
            net=load_checkpoint(str(ROOT/MAN['seeds'][str(s)]['checkpoint']),device)
            fin,_=net.raw(q,grid,pad_z=0)
            p=reduce_binary(fin>=MAN['seeds'][str(s)]['occupancy_threshold'],ds); synth.append(p)
            eg=EVAL_GRID[ds]
            for zi in range(eg.dims[2]):
                synth_profiles.append(dict(dataset=ds,seed=s,z_m=(zi+.5)*eg.voxel_size+eg.origin[2],predicted=int(p[:,:,zi].sum()),total=int(np.prod(p.shape[:2]))))
            del net,fin
        del q
        # An explicit boundary test of the production voxelizer: two outside points
        # must be rejected, and cannot accumulate at the two corners.
        o=np.array(grid.origin); upper=o+np.array(grid.dims)*grid.voxel_size
        idx,keep=voxelize(np.stack([o-grid.voxel_size,upper+grid.voxel_size,o+.5*grid.voxel_size]),grid)
        assert keep.tolist()==[False,False,True] and idx.tolist()==[[0,0,0]]
        boundary_checks.append(dict(dataset=ds,keep=keep.tolist(),accepted_indices=idx.tolist()))
        dims=tuple(EVAL_GRID[ds].dims); z=(np.arange(dims[2])+.5)*EVAL_GRID[ds].voxel_size+EVAL_GRID[ds].origin[2]
        bins=np.searchsorted(BIN_EDGES['height'][1:-1],z,side='right')
        accum={(s,b):np.zeros(6,np.int64) for s in range(3) for b in range(5)}
        over=np.zeros((3,4),np.int64)
        for row in samples[ds]['past5']['rows']:
            _,gt,valid=target(ds,row['gt_ref'])
            with np.load(OUT/'predictions'/ds/'past5'/(row['clip_id']+'.npz')) as f:
                mp=np.unpackbits(f['mapper'],count=np.prod(dims)).reshape(dims).astype(bool)
                for s in range(3):
                    p=np.unpackbits(f[f'seed{s}'],count=np.prod(dims)).reshape(dims).astype(bool)
                    fp=p & ~gt & valid; added=p & ~mp & valid
                    over[s]+=np.array([fp.sum(),(fp & synth[s]).sum(),(added & ~gt).sum(),(added & ~gt & synth[s]).sum()])
                    for b in range(5):
                        m=bins==b
                        accum[s,b]+=np.array([p[:,:,m].sum(),(p & valid)[:,:,m].sum(),valid[:,:,m].sum(),valid[:,:,m].size,(added & gt)[:,:,m].sum(),(added & ~gt)[:,:,m].sum()])
        for (s,b),vals in accum.items():
            records.append(dict(dataset=ds,seed=s,lo=str(BIN_EDGES['height'][b]),hi=str(BIN_EDGES['height'][b+1]),**dict(zip(['unmasked_pred_occupied','scored_pred_occupied','valid','all_voxels','added_tp','added_fp'],map(int,vals)))))
        for s,vals in enumerate(over):overlap.append(dict(dataset=ds,seed=s,**dict(zip(['fp','fp_matching_uniform_unknown','added_fp','added_fp_matching_uniform_unknown'],map(int,vals)))))
    for name,rows in [('height_visibility',records),('uniform_unknown_profile',synth_profiles)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(OUT/'mechanism.json',dict(uniform_unknown_overlap=overlap,out_of_grid_checks=boundary_checks))
    print('wrote fixed synthetic-input and mask-scope diagnostics',flush=True)


def seal():
    """Validate saved accounting and freeze the report's reproducibility references."""
    import subprocess
    import torch
    samples=json.loads((OUT/'samples.json').read_text())
    timestamps={}; checks=[]; old_pool={}; checkpoints={}
    for seed,frozen in MAN['seeds'].items():
        ck=torch.load(ROOT/frozen['checkpoint'],map_location='cpu',weights_only=False)
        checkpoints[seed]=dict(frozen=frozen,metadata={k:v for k,v in ck.items() if k!='state_dict'},
                              actual_sha256=sha(ROOT/frozen['checkpoint']))
        assert checkpoints[seed]['actual_sha256']==frozen['checkpoint_sha256']
    for ds in DS:
        timestamps[ds]={}; old_pool[ds]={}
        times=np.loadtxt(ROOT/'data/kitti/dataset/sequences/08/times.txt') if ds=='semantickitti' else None
        for mode,expected in zip(OFFSETS, (163,161) if ds=='semantickitti' else (1182,882)):
            doc=samples[ds][mode]; ids=[r['clip_id'] for r in doc['rows']]
            assert len(ids)==len(set(ids))==expected
            stamps=[]
            for row in doc['rows']:
                if ds=='semantickitti':
                    ts=times[row['input_orders']].tolist(); ref=float(times[row['anchor_order']]); units='seconds since sequence start'; factor=1
                else:
                    ts=[int(Path(p).stem.rsplit('__',1)[1])*1000 for p in row['input_images']]
                    ref=row['anchor_order']; units='Unix nanoseconds'; factor=1e9
                stamps.append(dict(clip_id=row['clip_id'],units=units,input_exposures=ts,
                                   reference_timestamp=ref,offsets_seconds=[(x-ref)/factor for x in ts]))
            timestamps[ds][mode]=stamps
            for prefix in ('','corrected_'):
                rows=list(csv.DictReader((OUT/f'{prefix}counts_{ds}_{mode}.csv').open()))
                assert len(rows)==expected*len(METHODS)
                for method in METHODS:
                    rs=[r for r in rows if r['method']==method]
                    assert [r['clip_id'] for r in rs]==ids
                    for r in rs:
                        c=[int(r[k]) for k in ('tp','fp','fn','tn')]
                        assert sum(c)==int(r['valid']) and c[0]+c[2]==int(r['gt_occupied']) and c[0]+c[1]==int(r['pred_occupied'])
                checks.append(dict(dataset=ds,mode=mode,convention=prefix or 'historical',samples=expected,accounting_valid=True))
            old_pool[ds][mode]={}
            for seed in range(3):
                with np.load(OLD/f'bincounts_{ds}_{mode}_seed{seed}.npz') as z:
                    c=z['pred_completion_occany_pool'].sum(0)
                    old_pool[ds][mode][str(seed)]=dict(tp=int(c[0]),fp=int(c[1]),fn=int(c[2]),iou=float(c[0]/c.sum()))
    dump(OUT/'input_timestamps.json',timestamps)
    dump(OUT/'reproduction_provenance.json',dict(checkpoints=checkpoints,historical_mask_then_pool=old_pool,
        validation=checks,python=sys.version,torch=torch.__version__,numpy=np.__version__))
    for name,repo in [('project',ROOT),('occany',Path('/home/minh/workspace/OccAny'))]:
        for label,args in [('diff',['diff','--binary']),('status',['status','--short','--branch'])]:
            (OUT/f'{name}_{label}_final.txt').write_bytes(subprocess.check_output(['git',*args],cwd=repo))
    shutil.copy2(__file__,OUT/'code_used/tools/gate8c1/original_audit.py')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='audit_file_hashes.json']
    files.append(ROOT/'reports/gate8c1_original_audit.md')
    dump(OUT/'audit_file_hashes.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
    print('validated all sample counts; sealed',len(files),'audit files',flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "infer", "analyze", "integrity", "streamcheck", "provenance", "finish", "mechanism", "seal"])
    p.add_argument("--dataset", choices=DS, default="semantickitti")
    p.add_argument("--mode", choices=list(OFFSETS), default="past5")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--verify-only", action="store_true")
    a=p.parse_args()
    if a.stage=="prepare": prepare()
    elif a.stage=="infer": infer(a.dataset,a.mode,a.device,a.verify_only)
    elif a.stage=="analyze": analyze(a.dataset,a.mode,a.verify_only)
    elif a.stage=="integrity": integrity(a.dataset,a.device)
    elif a.stage=="streamcheck": streamcheck(a.dataset,a.device)
    elif a.stage=="provenance": provenance()
    elif a.stage=="mechanism": mechanism(a.device)
    elif a.stage=="seal": seal()
    else: finish()
