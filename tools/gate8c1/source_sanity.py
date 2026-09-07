#!/usr/bin/env python
"""Bounded source-only sanity experiment: can the Gate 8C-1 pipeline learn KITTI-360?

phase1  Score the original seed-0 checkpoint on ALL 590 eligible drive-0006 IDs under
        three fixed conditions: mapper only, mapper + the original 0.4 m dilation, and
        Gate 8C-1 completion with zero padding at tau = -0.125.
phase2  Deterministically freeze 16 training clips from drives 0003/0007/0010, train a
        fresh completion head for exactly 2,000 steps with the original objective and
        optimizer, then score the final head on the same 590 IDs.

KITTI-360 only. The whole run executes inside the Gate 8C-1 FileAudit, which raises the
moment a SemanticKITTI / Occ3D / nuScenes / SSCBench-label path is opened.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'artifacts/gate8c1_original_audit/code_used'
PREV = ROOT / 'artifacts/gate8c1_padding_source_probe'
OUT = ROOT / 'artifacts/gate8c1_source_sanity'
CKPT = ROOT / 'artifacts/gate8c1/checkpoints/seed0_last.pt'
MANIFEST = ROOT / 'artifacts/gate8c1/frozen_manifest.json'
DATA = Path('/media/SSD1/MINH_DATASETS/lingbot_gate8c1')
VAL_DRIVE = '2013_05_28_drive_0006_sync'
TRAIN_DRIVES = ('2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync',
                '2013_05_28_drive_0010_sync')
TAU = -0.125
DILATION_RADIUS = 2          # 2 voxels x 0.2 m = the original fixed 0.4 m dilation
N_CLIPS = 16
N_STEPS = 2000
LOG_EVERY = 100
CONDITIONS = ('mapper', 'mapper_dilate', 'completion')
sys.path[:0] = [str(REF), str(ROOT), str(ROOT / 'gates')]  # gates/ was the repo root before the 2026-09-07 move; order preserves REF precedence

import numpy as np
import torch
import yaml
from gate8 import net as reference_net, targets as reference_targets
from gates.gate8c1 import sources as SRC
from sscbench_kitti360.audit import FileAudit
from gates.voxel_gate.voxels import dilate


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvwrite(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def ratio(a, b):
    return float(a / b) if b else None


def regions(dims):
    distance = np.full(dims, np.inf)
    for axis, size in enumerate(dims):
        d = np.minimum(np.arange(size) + .5, size - np.arange(size) - .5) * .2
        shape = [1, 1, 1]; shape[axis] = size
        distance = np.minimum(distance, d.reshape(shape))
    return {'all': np.ones(dims, bool), 'boundary': distance < .4, 'interior': distance >= 2.0}


def labels(z, dims):
    n = int(np.prod(dims))
    return (np.unpackbits(z['occ_packed'], count=n).reshape(dims).astype(bool),
            np.unpackbits(z['valid_packed'], count=n).reshape(dims).astype(bool))


def counts(pred, gt, valid, reg):
    out = {}
    for name, region in reg.items():
        mask = valid & region
        tp = int((pred & gt & mask).sum()); fp = int((pred & ~gt & mask).sum())
        fn = int((~pred & gt & mask).sum()); tn = int((~pred & ~gt & mask).sum())
        out[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    return out


def derive(c):
    tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
    return dict(c, valid=tp + fp + fn + tn, gt_occupied=tp + fn, pred_occupied=tp + fp,
                free=fp + tn, sc_iou=ratio(tp, tp + fp + fn), precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), fpr=ratio(fp, fp + tn),
                pred_over_gt_occupied=ratio(tp + fp, tp + fn),
                occupied_prevalence=ratio(tp + fn, tp + fp + fn + tn))


def load_eligible():
    """The existing frozen source manifest: all 590 eligible drive-0006 IDs."""
    sel = json.loads((PREV / 'selection.json').read_text())
    rows = sel['eligible']
    assert len(rows) == 590, f'expected 590 eligible drive-0006 IDs, found {len(rows)}'
    assert [r['sample_id'] for r in rows] == sel['eligible_ids']
    return rows


@torch.no_grad()
def score_all(net, rows, device, tag, progress=25):
    """Phase-1 scoring of every eligible drive-0006 ID under the three fixed conditions."""
    per_case, fp_z, pooled = [], {c: np.zeros(32, np.int64) for c in CONDITIONS}, {}
    acc = {c: {r: dict(tp=0, fp=0, fn=0, tn=0) for r in ('all', 'boundary', 'interior')}
           for c in CONDITIONS}
    start = time.time()
    for n, row in enumerate(rows):
        dims = tuple(row['dims']); reg = regions(dims)
        with np.load(row['input_path'], allow_pickle=False) as z:
            dense = reference_targets.unpack_sample(z, device)
        with np.load(row['target_path'], allow_pickle=False) as z:
            gt, valid = labels(z, dims)
        inp = dense['input']; base = dense['base_logodds']
        semw_positive = inp[6] > 0                     # channel 6 is semw.clamp(20)/20
        mapper = ((base > 0.0) & semw_positive)        # gate7b L_OCCUPIED_AT = 0.0
        preds = {'mapper': mapper.cpu().numpy(),
                 'mapper_dilate': dilate(mapper, DILATION_RADIUS).cpu().numpy()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            residual, _sem = net(inp[None])
        final = reference_net.apply_residual(base, residual[0, 0].float())
        assert torch.isfinite(final).all()
        preds['completion'] = (final >= TAU).cpu().numpy()
        rec = dict(sample_id=row['sample_id'], native_frame=row['native_frame'])
        for cond, p in preds.items():
            cc = counts(p, gt, valid, reg)
            for r in acc[cond]:
                for k in acc[cond][r]:
                    acc[cond][r][k] += cc[r][k]
            for r, v in cc.items():
                rec.update({f'{cond}_{r}_{k}': v[k] for k in ('tp', 'fp', 'fn', 'tn')})
            fp_z[cond] += (p & ~gt & valid).sum(axis=(0, 1)).astype(np.int64)
        per_case.append(rec)
        del dense, inp, base, final, residual
        if (n + 1) % progress == 0:
            print(f'  [{tag}] {n + 1}/{len(rows)}  {time.time() - start:.0f}s', flush=True)
    for cond in CONDITIONS:
        pooled[cond] = {r: derive(v) for r, v in acc[cond].items()}
        pooled[cond]['fp_by_z'] = fp_z[cond].tolist()
    return pooled, per_case


def base_config(device):
    cfg = yaml.safe_load((ROOT / 'configs/gate8c1/seed0.yaml').read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert sha(CKPT) == manifest['seeds']['0']['checkpoint_sha256'], 'checkpoint hash'
    assert manifest['seeds']['0']['occupancy_threshold'] == TAU, 'threshold'
    assert cfg['train_drives'] == list(TRAIN_DRIVES) and cfg['val_drive'] == VAL_DRIVE
    files = [CKPT, MANIFEST, Path(__file__), PREV / 'selection.json',
             ROOT / 'configs/gate8c1/seed0.yaml', ROOT / 'gates/gate8c1/data.py',
             ROOT / 'gates/gate8c1/sources.py', ROOT / 'tools/gate8a/train.py',
             ROOT / 'tools/gate8/train.py', ROOT / 'gates/gate8a/losses.py',
             ROOT / 'gates/gate8a/sampler.py', ROOT / 'gates/gate8a/regions.py',
             ROOT / 'gates/voxel_gate/voxels.py', ROOT / 'sscbench_kitti360/audit.py']
    files += sorted((REF / 'gate8').glob('*.py'))
    return cfg, dict(
        experiment='gate8c1_source_sanity', device=device,
        question='Is poor SemanticKITTI performance a broken supervision/training pipeline '
                 'or a model that learns KITTI-360 but does not generalise?',
        checkpoint=str(CKPT), checkpoint_sha256=sha(CKPT), threshold=TAU,
        train_drives=list(TRAIN_DRIVES), val_drive=VAL_DRIVE,
        supervision='approved raw-LiDAR targets under '
                    f'{DATA}/targets (occupied endpoints, free ray interiors, unobserved and '
                    'conflicting voxels ignored); SSCBench _1_1.npy labels are never read',
        conditions={'mapper': 'base_logodds > 0 (gate7b L_OCCUPIED_AT) AND semantic weight > 0',
                    'mapper_dilate': f'Chebyshev dilation of mapper by {DILATION_RADIUS} voxels = 0.4 m',
                    'completion': 'apply_residual(base, net(input)) >= -0.125, zero padding, pad_z=0'},
        mapper_definition_deviation='The target-domain audit used q["occupied"] & (sem_w > 0), where '
                                    'occupied also drops MoGe-only voxels with fewer than 2 MoGe '
                                    'confirmations. The cached source samples do not store n_lb/n_moge, '
                                    'so that term cannot be reproduced here and is omitted.',
        grid=dict(dims=[256, 256, 32], voxel_size_m=.2, origin_m=[0, -25.6, -2], axes='XYZ',
                  reduction=None, pad_z=0),
        masks='gt_valid from the raw-LiDAR target; applied only when counting',
        boundary='centre distance to any face <0.4 m', interior='centre distance to every face >=2 m',
        precision='float32 inputs; CUDA bfloat16 autocast; float32 final logits',
        residual_rule='base + residual * (abs(base) < 2)',
        phase2=dict(n_clips=N_CLIPS, steps=N_STEPS, log_every=LOG_EVERY,
                    selection='per drive, np.linspace over the sorted sample list; shares 6/5/5 '
                              'across drives 0003/0007/0010; recorded before any training',
                    init='torch.manual_seed(0), random.seed(0), np.random.seed(0), then '
                         'CompletionUNet(width=24, padding_mode="zeros")',
                    optimizer='AdamW lr 1e-3 wd 0.01, CosineAnnealingLR over 2000 steps, '
                              'grad clip 1.0, batch 4, crop 128x128x32, sampler uniform, '
                              'loss focal_dice + 0.5 semantic KL (configs/gate8c1/seed0.yaml)',
                    checkpoint_selection='none; the step-2000 head is used',
                    fit_criterion=dict(
                        occupancy_loss_relative_decrease_min=.90,
                        occupancy_loss_initial='mean of the masked occupancy loss over steps 1-10',
                        occupancy_loss_final='mean of the masked occupancy loss over steps 1991-2000',
                        pooled_training_sc_iou_min=.80,
                        pooled_training_sc_iou='pooled over the 16 full 256x256x32 training clips '
                                               'at tau=-0.125, valid voxels only')),
        known_limitation='frozen_manifest code_hashes list gate8/net.py 4cbfced7...; the recovered '
                         'source is 56852e4a... The audit established the difference is optional '
                         'padding variants defaulting to zeros, and this checkpoint carries no '
                         'padding_mode key, so zero padding is selected.',
        commands=dict(phase1=shlex.join([sys.executable, 'tools/gate8c1/source_sanity.py',
                                         'phase1', '--device', device]),
                      phase2=shlex.join([sys.executable, 'tools/gate8c1/source_sanity.py',
                                         'phase2', '--device', device])),
        artifact_hashes={str(p): sha(p) for p in files})


def phase1(device):
    OUT.mkdir(exist_ok=True)
    cfg, meta = base_config(device)
    rows = load_eligible()
    for row in rows:
        assert sha(row['input_path']) == row['input_sha256'], row['sample_id']
        assert sha(row['target_path']) == row['target_sha256'], row['sample_id']
    assert torch.cuda.is_available(), 'CUDA unavailable; no inference performed'
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = False
    write_json('config.json', meta)
    (OUT / 'git_status_before.txt').write_bytes(
        subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    write_json('environment.json', dict(python=sys.version, torch=torch.__version__,
        numpy=np.__version__, device=device, gpu=torch.cuda.get_device_name(device),
        executable=sys.executable, time_unix=time.time(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark))
    write_json('evaluated_ids.json', dict(drive=VAL_DRIVE, n=len(rows),
        ids=[r['sample_id'] for r in rows],
        input_sha256={r['sample_id']: r['input_sha256'] for r in rows},
        target_sha256={r['sample_id']: r['target_sha256'] for r in rows}))
    net = reference_net.load_checkpoint(str(CKPT), device).net
    assert net.n_params() == 986114
    pooled, per_case = score_all(net, rows, device, 'phase1')
    write_json('phase1_pooled.json', pooled)
    csvwrite('phase1_per_case.csv', per_case)
    csvwrite('phase1_fp_by_z.csv', [dict(condition=c, z_index=z, z_m=-2 + (z + .5) * .2,
                                         fp=pooled[c]['fp_by_z'][z]) for c in CONDITIONS
                                    for z in range(32)])
    csvwrite('phase1_pooled.csv', [dict(condition=c, region=r, **pooled[c][r])
                                   for c in CONDITIONS for r in ('all', 'boundary', 'interior')])
    for c in CONDITIONS:
        a = pooled[c]['all']
        print(f'{c:15s} IoU {100*a["sc_iou"]:6.2f}  P {100*a["precision"]:6.2f}  '
              f'R {100*a["recall"]:6.2f}  FPR {100*a["fpr"]:6.3f}  '
              f'pred/gt {a["pred_over_gt_occupied"]:.3f}', flush=True)


def select_clips():
    shares = {TRAIN_DRIVES[0]: 6, TRAIN_DRIVES[1]: 5, TRAIN_DRIVES[2]: 5}
    chosen = []
    for drive in TRAIN_DRIVES:
        files = sorted((DATA / 'samples' / drive).glob('*.npz'))
        idx = np.linspace(0, len(files) - 1, shares[drive]).round().astype(int)
        assert len(set(idx.tolist())) == shares[drive], drive
        for i in idx:
            f = files[int(i)]
            t = DATA / 'targets' / drive / f.name
            chosen.append(dict(drive=drive, sample_id=f.stem, index=int(i), n_in_drive=len(files),
                               input_path=str(f), target_path=str(t),
                               input_sha256=sha(f), target_sha256=sha(t)))
    assert len(chosen) == N_CLIPS
    return chosen


@torch.no_grad()
def training_fit(net, clips, device):
    """Pooled SC IoU over the 16 full training clips at tau = -0.125."""
    net.eval()
    tp = fp = fn = tn = 0
    for c in clips:
        with np.load(c['input_path'], allow_pickle=False) as z:
            d = reference_targets.unpack_sample(z, device)
        with np.load(c['target_path'], allow_pickle=False) as z:
            gt, valid = labels(z, tuple(d['gt_occ'].shape))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            residual, _ = net(d['input'][None])
        pred = (reference_net.apply_residual(d['base_logodds'],
                                             residual[0, 0].float()) >= TAU).cpu().numpy()
        tp += int((pred & gt & valid).sum()); fp += int((pred & ~gt & valid).sum())
        fn += int((~pred & gt & valid).sum()); tn += int((~pred & ~gt & valid).sum())
        del d, residual
    net.train()
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sc_iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                pred_over_gt_occupied=ratio(tp + fp, tp + fn))


def phase2(device):
    assert (OUT / 'phase1_pooled.json').exists(), 'run phase1 first'
    cfg, meta = base_config(device)
    clips = select_clips()
    write_json('phase2_training_clips.json', dict(
        n=len(clips), rule=meta['phase2']['selection'], clips=clips,
        recorded_before_training=True))
    print('FROZEN 16 training clips:', [c['sample_id'] for c in clips], flush=True)
    assert torch.cuda.is_available()
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    # original seed-0 initialization procedure
    seed = int(cfg['seed'])
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    from gates.gate8c1.data import DriveSamples
    from tools.gate8a.train import step_loss
    train = DriveSamples(list(TRAIN_DRIVES), cfg['crop'], dev, seed, None,
                         sampler=cfg['sampler'], centre_on_occ_p=cfg['centre_on_occ_p'])
    keep = {c['input_path'] for c in clips}
    train.by_drive = {d: [f for f in fs if f in keep] for d, fs in train.by_drive.items()}
    train._reindex()
    assert sorted(train.files) == sorted(keep), 'restricted training set mismatch'
    net = reference_net.CompletionUNet(width=cfg['width'], padding_mode='zeros').to(dev)
    assert net.n_params() == 986114
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS)
    curve, occ_losses, t0 = [], [], time.time()
    for it in range(1, N_STEPS + 1):
        b = train.batch([train.draw() for _ in range(cfg['batch_size'])])
        net.train()
        loss, m = step_loss(net, b, cfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
        opt.step(); sched.step()
        occ = cfg['w_focal'] * m['focal'] + cfg['w_dice'] * m['dice']
        occ_losses.append(occ)
        if it % LOG_EVERY == 0 or it == 1:
            fit = training_fit(net, clips, dev)
            row = dict(step=it, lr=sched.get_last_lr()[0], loss=loss.item(),
                       occupancy_loss=occ, focal=m['focal'], dice=m['dice'], sem_kl=m['sem_kl'],
                       **{f'train_{k}': v for k, v in fit.items()})
            curve.append(row)
            print(f'  [{it}/{N_STEPS}] occ_loss {occ:.4f}  pooled train IoU '
                  f'{100*fit["sc_iou"]:.2f}%  TP {fit["tp"]} FP {fit["fp"]} FN {fit["fn"]}  '
                  f'pred/gt {fit["pred_over_gt_occupied"]:.3f}  '
                  f'{(time.time()-t0)/it:.2f}s/it', flush=True)
    initial = float(np.mean(occ_losses[:10])); final = float(np.mean(occ_losses[-10:]))
    fit = training_fit(net, clips, dev)
    criterion = dict(occupancy_loss_initial=initial, occupancy_loss_final=final,
                     occupancy_loss_relative_decrease=ratio(initial - final, initial),
                     occupancy_loss_step1=occ_losses[0],
                     pooled_training_sc_iou=fit['sc_iou'],
                     loss_criterion_passed=bool(initial and (initial - final) / initial >= .90),
                     iou_criterion_passed=bool(fit['sc_iou'] >= .80))
    criterion['fits_training_clips'] = bool(criterion['loss_criterion_passed']
                                            and criterion['iou_criterion_passed'])
    reference_net.save_checkpoint(net, str(OUT / 'phase2_overfit_head.pt'),
                                  dict(config=cfg, steps=N_STEPS, clips=[c['sample_id'] for c in clips],
                                       note='16-clip source overfit sanity head; not a deliverable model'))
    csvwrite('phase2_training_curve.csv', curve)
    rows = load_eligible()
    net.eval()
    pooled, per_case = score_all(net, rows, device, 'phase2-val')
    write_json('phase2_val_pooled.json', pooled)
    csvwrite('phase2_val_per_case.csv', per_case)
    write_json('phase2_result.json', dict(
        clips=[c['sample_id'] for c in clips], steps=N_STEPS, final_training_fit=fit,
        criterion=criterion, seconds=time.time() - t0,
        checkpoint_sha256=sha(OUT / 'phase2_overfit_head.pt')))
    a = pooled['completion']['all']
    print(f'PHASE2 fit={criterion["fits_training_clips"]} train IoU {100*fit["sc_iou"]:.2f}%  '
          f'loss {initial:.4f}->{final:.4f}  |  drive-0006 completion IoU {100*a["sc_iou"]:.2f}%  '
          f'P {100*a["precision"]:.2f} R {100*a["recall"]:.2f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['phase1', 'phase2'])
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    SRC.assert_no_target_access(list(TRAIN_DRIVES) + [VAL_DRIVE, str(DATA), str(OUT)])
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        (phase1 if args.stage == 'phase1' else phase2)(args.device)
    write_json(f'{args.stage}_file_access_check.json',
               dict(audit.summary(), stage=args.stage,
                    note='every open, np.load and np.fromfile in this stage was intercepted; a '
                         'SemanticKITTI / Occ3D / nuScenes / SSCBench-label path would have raised',
                    n_target_domain_paths_opened=len(audit.violations)))
    print('file access check: opened', len(audit.opened), 'paths,',
          len(audit.violations), 'violations', flush=True)
