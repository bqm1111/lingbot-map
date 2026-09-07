#!/usr/bin/env python
"""Genuine one-frame-at-a-time runtime: live LingBot direct mode, live MoGe on the five
anchor frames, the incremental map update, the dense export, the completion network, and
-- on the same frames -- **online Trident-H** in its own environment, so the end-to-end
latency is measured rather than inferred from a cache.

Two settings are reported separately, as the brief requires: cached-teacher and
end-to-end with Trident executed online. Trident timing is read from the file written by
``tools/gate8/time_trident.py`` (Trident environment); if absent, the end-to-end number is
marked unavailable rather than estimated.

    python tools/gate8/runtime_audit.py --source kitti360 --frames 120 --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, load_config, write_json                  # noqa: E402
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri                  # noqa: E402
from moge_gauge.calibrated import CalibratedMoGe                                      # noqa: E402
from moge_gauge.calibration import calibrated_fov_x_deg                               # noqa: E402
from gates.gate6 import grids as G6G                                                        # noqa: E402
from gates.gate7b import replay, scale as SC, depth as D7                                   # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                           # noqa: E402
from gates.gate8.mapper import IncrementalMapper, FrameInput                                # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8")


class LiveLingBot:
    """Per-frame direct-mode calls, exactly the sequence ``inference_streaming`` makes."""

    def __init__(self, model, device, n_scale=5):
        self.m, self.dev, self.n_scale = model, device, n_scale
        self.buf, self.started, self.i = [], False, 0

    @torch.inference_mode()
    def push(self, img):                                       # img: [3, H, W] cpu
        outs = []
        if not self.started:
            self.buf.append(img)
            if len(self.buf) < self.n_scale:
                return []
            x = torch.stack(self.buf)[None].to(self.dev)
            self.m.clean_kv_cache()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.m.forward(x, num_frame_for_scale=self.n_scale,
                                   num_frame_per_block=self.n_scale, causal_inference=True)
            self.started = True
            outs = [self._unpack(o, k, x.shape[-2:]) for k in range(self.n_scale)]
            self.i = self.n_scale
            return outs
        x = img[None, None].to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.m.forward(x, num_frame_for_scale=self.n_scale, num_frame_per_block=1,
                               causal_inference=True)
        self.i += 1
        return [self._unpack(o, 0, x.shape[-2:])]

    def _unpack(self, o, k, hw):
        pe = o["pose_enc"][0, k:k + 1].float().cpu()
        extr, intr = pose_encoding_to_extri_intri(pe.unsqueeze(0), tuple(hw))
        c2w = np.eye(4); c2w[:3, :4] = extr[0, 0].double().numpy()
        # keep the frozen outputs on the mapper's device; the live path must not
        # round-trip through the CPU or the mapper receives mixed-device tensors
        return {"depth": o["depth"][0, k, ..., 0].float(),
                "conf": o["depth_conf"][0, k].float(),
                "K": intr[0, 0].float().cpu().numpy().astype(np.float64), "c2w": c2w}


def _default_device() -> str:
    """First GPU of the Gate-8 pool. ``GATE8_GPUS`` (default "1 2 3") reserves
    GPU 0 for other users; ``GATE8_DEVICE`` overrides outright."""
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="kitti360")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); torch.cuda.init()
    ds = S.DATASET_OF[a.source]
    seg = S.segments(a.source, REPO_ROOT)[0]
    frames = seg.frames[:a.frames]
    model, _ = replay.build_model(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    into = V8.into_matrix(ds)
    mapper = IncrementalMapper(dev, sem_into=into)
    net = None
    if a.checkpoint:
        from gates.gate8.net import load_checkpoint
        net = load_checkpoint(a.checkpoint, dev)
    live = LiveLingBot(model, dev)
    T = {k: [] for k in ("image_load", "lingbot", "moge", "scale", "map_step", "sem_load",
                         "query", "net", "cached_stream_compare")}
    with np.load(S.stream_path(a.source, seg.name)) as z:
        cached_dep = z["pred_depth"][:a.frames].astype(np.float32)
    K_native = None
    with np.load(S.stream_path(a.source, seg.name)) as z:
        pw = int(z["proc_hw"][1])
    from PIL import Image
    with Image.open(frames[0].path) as im:
        native_w = im.width
    # calibrated FOV needs the native fx: read it as the evaluator does for this source
    if a.source == "kitti360":
        import glob
        p = sorted(glob.glob("/media/SSD1/MINH_DATASETS/lingbot_gate5_2/cache_lingbot/*.npz"))[0]
        with np.load(p) as z:
            fx = float(z["K_native"][0, 0])
    else:
        fx = float(frames[0].K_native[0, 0])
    fov = calibrated_fov_x_deg(fx * (pw / native_w), pw)
    torch.cuda.reset_peak_memory_stats(dev)
    dep_err = []
    for i, f in enumerate(frames):
        t = time.time(); img = replay.load_images([f.path])[0]; T["image_load"].append(time.time() - t)
        t = time.time(); outs = live.push(img); torch.cuda.synchronize(); T["lingbot"].append(time.time() - t)
        for j, o in enumerate(outs):
            idx = i - (len(outs) - 1) + j
            log_s = float("nan")
            if idx < 5:
                t = time.time(); mo = moge.infer_calibrated(img.numpy() if len(outs) == 1 else
                                                            replay.load_images([frames[idx].path])[0].numpy(),
                                                            fov_x_deg=fov)
                torch.cuda.synchronize(); T["moge"].append(time.time() - t)
                t = time.time()
                c = SC.frame_candidate(mo.depth_z, mo.mask, o["depth"].cpu().numpy(),
                                       o["conf"].cpu().numpy())
                log_s = c["log_s"]; T["scale"].append(time.time() - t)
            t = time.time()
            p = S.trident_path(a.source, frames[idx].key)
            sem = None
            if os.path.exists(p):
                with np.load(p) as z:
                    sem = torch.from_numpy(z["probs"].astype(np.float32)).permute(1, 2, 0)
            T["sem_load"].append(time.time() - t)
            fi = FrameInput(idx, o["depth"], o["conf"], o["K"], o["c2w"], log_s, sem_probs=sem)
            t = time.time(); mapper.step(fi); torch.cuda.synchronize(); T["map_step"].append(time.time() - t)
            dep_err.append(float(np.abs(o["depth"].cpu().numpy() - cached_dep[idx]).mean()))
        if mapper.scale_state.frozen and i % 10 == 0 and f.T_cam_to_grid is not None:
            G = G6G.PREDICTION_GRID[ds]
            P = live.__dict__.get("last_c2w", None)
            P = outs[-1]["c2w"].copy(); P[:3, 3] *= mapper.scale_state.scale
            Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid))
            t = time.time(); q = mapper.query(G, Tgw); torch.cuda.synchronize(); T["query"].append(time.time() - t)
            if net is not None:
                q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(), torch.full_like(q["last_time"], -1).float())
                t = time.time(); net.complete(q, G); torch.cuda.synchronize(); T["net"].append(time.time() - t)
    stat = lambda x: {"median_ms": 1e3 * float(np.median(x)), "p95_ms": 1e3 * float(np.percentile(x, 95)), "n": len(x)} if x else None
    tri = os.path.join(ART, "trident_online_timing.json")
    tri = json.load(open(tri)) if os.path.exists(tri) else None
    per_frame_cached = sum(np.median(T[k]) for k in ("image_load", "lingbot", "map_step", "sem_load") if T[k])
    res = {"source": a.source, "n_frames": len(frames), "components": {k: stat(v) for k, v in T.items()},
           "live_vs_cached_depth_mean_abs_err": float(np.mean(dep_err)),
           "map_rows_final": len(mapper.table), "map_bytes_final": mapper.table.memory_bytes(),
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "per_frame_ms_cached_teacher": 1e3 * per_frame_cached,
           "per_frame_ms_end_to_end": (1e3 * (per_frame_cached - np.median(T["sem_load"]))
                                       + tri["median_ms"]) if tri else None,
           "trident_online": tri,
           "incremental": "map state is updated per frame; the dense grid is an export (query), "
                          "measured separately; no history is replayed",
           "note": "the frozen five-frame MoGe gauge runs on the first five frames only"}
    write_json(os.path.join(ART, f"runtime_{a.source}.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "components"}, indent=1))
    for k, v in res["components"].items():
        if v: print(f"  {k:24s} median {v['median_ms']:8.1f} ms  p95 {v['p95_ms']:8.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
