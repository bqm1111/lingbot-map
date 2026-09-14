#!/usr/bin/env python
"""Sample Trident-H directly in image space at every verified correspondence (Steps 5-7, 10).

For a pair (A, B, X) the 1024-d unit token grid of each frame is sampled bilinearly at the
projected pixel with the same ``align_corners=False`` rule the official code uses to
interpolate its logits to native resolution (``F.interpolate`` on the token grid and
``grid_sample`` at pixel centres coincide; asserted once per run). The per-view coarse
classification is softmax(40 f . text) of the sampled feature; the cached SAM-refined
lattice probabilities are sampled at the same pixel for the reference path.

Controls, all built from the same (A, B) correspondence set: C1 a different landmark of
the set (random), C2 a different landmark 0.3-1.5 m away at similar height and camera
depth, C3 a different landmark whose per-view class in B equals X's class in A, and C0 a
landmark of an earlier, unrelated (A', B') set of the drive. Pixel-neighbourhood
perturbations of the B sample for every 20th pair. Only scalars are stored.

    python tools/direct_correspondence_audit/sample_features.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from collections import OrderedDict
import numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_splits as SRC, sources as S                            # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = DC.NATIVE_HW


class Frames:
    def __init__(self, src, key_of, dev, cap=40):
        self.src, self.key_of, self.dev, self.cap = src, key_of, dev, cap; self.d = OrderedDict()

    def get(self, native):
        if native in self.d:
            self.d.move_to_end(native); return self.d[native]
        key = self.key_of[native]
        with np.load(os.path.join(DC.TOKEN_DIR, self.src, f"{key}.npz")) as z:
            tok = torch.from_numpy(z["tokens"].astype(np.float32)).to(self.dev).reshape(*DC.TOKEN_HW, -1).permute(2, 0, 1)[None]
        with np.load(S.trident_path(self.src, key)) as z:
            ref = torch.from_numpy(z["probs"].astype(np.float32)).to(self.dev)[None]
        self.d[native] = (tok, ref)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return self.d[native]


def sample(img, u, v):
    """bilinear, align_corners=False, at native pixel coordinates (u, v) -> [n, C]."""
    g = torch.stack([(u + 0.5) / W * 2 - 1, (v + 0.5) / H * 2 - 1], -1)[None, None]       # [1,1,n,2]
    return F.grid_sample(img, g.to(img.dtype), mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def classify(f, text):
    return torch.softmax(DC.LOGIT_SCALE * F.normalize(f, dim=1) @ text.T, 1)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=1)


def js(p, q):
    p = p.clamp_min(1e-6); q = q.clamp_min(1e-6); m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(1) + 0.5 * (q * (q / m).log()).sum(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    DC.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    src = SRC.source_of(a.drive); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    key_of = {int(f.key.split("_")[-1]): f.key for f in seg.frames}
    with np.load(DC.TEXT_PATH) as z:
        text = torch.from_numpy(z["text"].astype(np.float32)).to(dev)
    frames = Frames(src, key_of, dev)
    # sampling rule == official interpolation rule (checked on the first frame)
    tok0, _ = frames.get(next(iter(key_of)))
    vv, uu = torch.meshgrid(torch.arange(0, H, 7, device=dev, dtype=torch.float32), torch.arange(0, W, 11, device=dev, dtype=torch.float32), indexing="ij")
    s1 = sample(tok0, uu.reshape(-1), vv.reshape(-1)); s2 = F.interpolate(tok0, size=(H, W), mode="bilinear", align_corners=False)[0][:, ::7, ::11].reshape(tok0.shape[1], -1).T
    rule_dev = float((s1 - s2).abs().max()); assert rule_dev < 1e-4, rule_dev
    files = sorted(glob.glob(os.path.join(DC.PAIR_DIR, a.drive, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    out_dir = os.path.join(DC.ROW_DIR, a.drive); os.makedirs(out_dir, exist_ok=True)
    reservoir = None; rng = np.random.default_rng(20260913); t0 = time.time(); n_done = 0; n_rows = 0
    fields = ["native_A", "native_B", "landmark", "offset", "baseline_m", "angle_deg", "z_A", "z_B", "resid_A", "resid_B", "strict",
              "u_A", "v_A", "u_B", "v_B", "cls_A", "cls_B", "conf_A", "conf_B", "cls_ref_A", "cls_ref_B",
              "cos_same", "pcos_same", "js_same", "top1_same", "refcos_same", "reftop1_same",
              "cos_C0", "pcos_C0", "top1_C0", "cos_C1", "pcos_C1", "top1_C1", "cos_C2", "pcos_C2", "top1_C2", "d_C2", "cos_C3", "pcos_C3", "top1_C3",
              "pert_1tok", "pert_2px", "pert_5px", "cos_C2_sameview", "pcos_C2_sameview", "top1_C2_sameview"]
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                P = {k: z[k] for k in z.files}
            fa = int(P["native_A"]); op = DC.row_path(a.drive, fa)
            if os.path.exists(op):
                continue
            tokA, refA = frames.get(fa)
            uA = torch.from_numpy(P["u_A"]).to(dev); vA = torch.from_numpy(P["v_A"]).to(dev)
            fA_all = sample(tokA, uA, vA); pA_all = classify(fA_all, text); rA_all = sample(refA, uA, vA)
            Xw = P["X_world"]; rows = P["rows"]; out = []
            for fb in np.unique(rows[:, 1]).astype(int):
                R = rows[rows[:, 1] == fb]; li = R[:, 0].astype(int); n = len(li)
                if n < 5:
                    continue
                tokB, refB = frames.get(fb)
                uB = torch.from_numpy(R[:, 2]).float().to(dev); vB = torch.from_numpy(R[:, 3]).float().to(dev)
                fB = sample(tokB, uB, vB); pB = classify(fB, text); rB = sample(refB, uB, vB)
                fA = fA_all[li]; pA = pA_all[li]; rA = rA_all[li]
                cA, cB = pA.argmax(1), pB.argmax(1)
                same = {"cos": cos(fA, fB), "pcos": cos(pA, pB), "js": js(pA, pB), "top1": (cA == cB).float(), "refcos": cos(rA, rB), "reftop1": (rA.argmax(1) == rB.argmax(1)).float()}
                # C1: different landmark, same (A,B) set
                p1 = rng.permutation(n); p1 = np.where(p1 == np.arange(n), np.roll(p1, 1), p1) if n > 1 else p1
                # C2: local physical point
                X = Xw[li]; tree = cKDTree(X); zB = R[:, 4]
                nb = tree.query_ball_point(X, DC.LOCAL_R[1]); p2 = np.full(n, -1); d2 = np.full(n, np.nan)
                for i, cands in enumerate(nb):
                    c = [j for j in cands if j != i and np.linalg.norm(X[j] - X[i]) >= DC.LOCAL_R[0] and abs(X[j, 2] - X[i, 2]) < DC.LOCAL_DZ and abs(zB[j] - zB[i]) < DC.LOCAL_DEPTH_REL * zB[i]]
                    if c:
                        p2[i] = c[rng.integers(len(c))]; d2[i] = np.linalg.norm(X[p2[i]] - X[i])
                # C3: same class (B's class equals A's class of X), different landmark
                cA_np, cB_np = cA.cpu().numpy(), cB.cpu().numpy(); p3 = np.full(n, -1)
                for k in np.unique(cA_np):
                    ii = np.flatnonzero(cA_np == k); jj = np.flatnonzero(cB_np == k)
                    for i in ii:
                        cand = jj[jj != i]
                        if len(cand):
                            p3[i] = cand[rng.integers(len(cand))]
                # C0: unrelated earlier set
                if reservoir is not None and len(reservoir[0]) >= 100:
                    r0 = rng.integers(len(reservoir[0]), size=n); f0, p0 = reservoir[0][r0], reservoir[1][r0]
                else:
                    f0 = p0 = None
                def ctl(perm, valid):
                    idx = torch.from_numpy(np.where(valid, perm, 0)).to(dev); vm = torch.from_numpy(valid).to(dev)
                    c_ = cos(fA, fB[idx]); pc_ = cos(pA, pB[idx]); t_ = (cA == cB[idx]).float()
                    nanv = torch.full_like(c_, float("nan"))
                    return torch.where(vm, c_, nanv), torch.where(vm, pc_, nanv), torch.where(vm, t_, nanv)
                c1 = ctl(p1, np.ones(n, bool)); c2 = ctl(p2, p2 >= 0); c3 = ctl(p3, p3 >= 0)
                # same-view local control: f_A(X) vs f_A(X') -- the feature's own spatial resolution, no view change
                idx2 = torch.from_numpy(np.where(p2 >= 0, p2, 0)).to(dev); vm2 = torch.from_numpy(p2 >= 0).to(dev); nanv = torch.full((n,), float("nan"), device=dev)
                sv = (torch.where(vm2, cos(fA, fA[idx2]), nanv), torch.where(vm2, cos(pA, pA[idx2]), nanv), torch.where(vm2, (cA == cA[idx2]).float(), nanv))
                if f0 is not None:
                    c0 = (cos(fA, f0), cos(pA, p0), (cA == p0.argmax(1)).float())
                else:
                    c0 = tuple(torch.full((n,), float("nan"), device=dev) for _ in range(3))
                # perturbations of the B sample (every 20th pair)
                pert = torch.full((n, 3), float("nan"), device=dev); sub = torch.arange(0, n, DC.PERTURB_EVERY, device=dev)
                if len(sub):
                    for k_, (nm, px) in enumerate(DC.PERTURB_PX.items()):
                        vals = []
                        for du, dv in ((px, 0), (-px, 0), (0, px), (0, -px)):
                            vals.append(cos(fB[sub], sample(tokB, uB[sub] + du, vB[sub] + dv)))
                        pert[sub, k_] = torch.stack(vals, 1).mean(1)
                cols = [torch.full((n,), float(fa), device=dev), torch.full((n,), float(fb), device=dev), torch.from_numpy(R[:, 0]).float().to(dev),
                        torch.from_numpy(R[:, 9]).float().to(dev), torch.from_numpy(R[:, 7]).float().to(dev), torch.from_numpy(R[:, 8]).float().to(dev),
                        torch.from_numpy(P["z_A"][li]).to(dev), torch.from_numpy(R[:, 4]).float().to(dev), torch.from_numpy(P["resid_A"][li]).to(dev),
                        torch.from_numpy(R[:, 5]).float().to(dev), torch.from_numpy(R[:, 6]).float().to(dev),
                        uA[li], vA[li], uB, vB, cA.float(), cB.float(), pA.max(1).values, pB.max(1).values, rA.argmax(1).float(), rB.argmax(1).float(),
                        same["cos"], same["pcos"], same["js"], same["top1"], same["refcos"], same["reftop1"],
                        *c0, *c1, c2[0], c2[1], c2[2], torch.from_numpy(d2).float().to(dev), *c3, pert[:, 0], pert[:, 1], pert[:, 2], *sv]
                out.append(torch.stack(cols, 1).cpu().numpy().astype(np.float32))
                # reservoir for C0: keep a random subset of this set's B features
                keep = rng.choice(n, min(n, 200), replace=False)
                reservoir = ((fB[keep], pB[keep]) if reservoir is None else (torch.cat([reservoir[0], fB[keep]])[-4000:], torch.cat([reservoir[1], pB[keep]])[-4000:]))
            arr = np.concatenate(out) if out else np.zeros((0, len(fields)), np.float32)
            np.savez_compressed(op, rows=arr, fields=np.asarray(fields), sampling_rule_max_dev=np.float32(rule_dev))
            n_done += 1; n_rows += len(arr)
            if n_done % 25 == 0:
                print(f"  [{a.drive[17:21]}] {n_done} source frames, {n_rows:,} rows, {(time.time()-t0)/n_done:.1f}s each", flush=True)
    DC.write_json(os.path.join(DC.ART, "sampling", f"{a.drive}.json"),
                  {"drive": a.drive, "n_source_frames": n_done, "n_rows": n_rows, "sampling_rule_max_dev_vs_official_interpolation": rule_dev,
                   "fields": fields, "controls": {"C0": "landmark of an earlier unrelated (A,B) set of the drive (reservoir of 4000)", "C1": "different landmark, same (A,B) set",
                                                  "C2": f"different landmark, {DC.LOCAL_R[0]}-{DC.LOCAL_R[1]} m away, |dz| < {DC.LOCAL_DZ} m, |dz_cam|/z < {DC.LOCAL_DEPTH_REL}",
                                                  "C3": "different landmark whose per-view class in B equals X's class in A"},
                   "perturbations": DC.PERTURB_PX, "seconds": time.time() - t0, "firewall": audit.summary(), "env": DC.environment()})
    print(f"[{a.drive}] {n_done} source frames, {n_rows:,} rows, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
