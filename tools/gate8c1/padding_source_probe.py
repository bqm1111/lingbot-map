#!/usr/bin/env python
"""One frozen, source-only A/B padding probe. No training or post-processing.

prepare freezes eligible IDs, three cases, hashes and criteria before inference.
run performs exactly A/B on those cases and one unknown control per input shape.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'artifacts/gate8c1_original_audit/code_used'
OUT = ROOT / 'artifacts/gate8c1_padding_source_probe'
DATA = Path('/media/SSD1/MINH_DATASETS/lingbot_gate8c1')
DRIVE = '2013_05_28_drive_0006_sync'
CKPT = ROOT / 'artifacts/gate8c1/checkpoints/seed0_last.pt'
MANIFEST = ROOT / 'artifacts/gate8c1/frozen_manifest.json'
TAU = -0.125
sys.path[:0] = [str(REF), str(ROOT), str(ROOT / 'gates')]  # gates/ was the repo root before the 2026-09-07 move; order preserves REF precedence

import numpy as np
import torch
from gate8 import net as reference_net, targets as reference_targets


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvwrite(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def ratio(a, b):
    return float(a / b) if b else None


def regions(dims):
    # XYZ voxel-centre distance; no coordinate rounding or mask-dependent region.
    distance = np.full(dims, np.inf)
    for axis, size in enumerate(dims):
        d = np.minimum(np.arange(size) + .5, size - np.arange(size) - .5) * .2
        shape = [1, 1, 1]; shape[axis] = size
        distance = np.minimum(distance, d.reshape(shape))
    return {'all': np.ones(dims, bool), 'boundary': distance < .4,
            'interior': distance >= 2.0}


def labels(z, dims):
    n = int(np.prod(dims))
    return (np.unpackbits(z['occ_packed'], count=n).reshape(dims).astype(bool),
            np.unpackbits(z['valid_packed'], count=n).reshape(dims).astype(bool))


def module_configuration(net):
    attrs = ('padding_mode', 'padding', 'stride', 'dilation', 'kernel_size',
             'output_padding', 'groups', 'num_groups', 'eps')
    return {name: dict(type=type(layer).__name__, training=layer.training,
                      **{k: getattr(layer, k) for k in attrs if hasattr(layer, k)})
            for name, layer in net.named_modules()}


def prepare():
    if OUT.exists():
        raise RuntimeError(f'Preserve existing experiment directory: {OUT}')
    OUT.mkdir()
    manifest = json.loads(MANIFEST.read_text())
    assert sha(CKPT) == manifest['seeds']['0']['checkpoint_sha256']
    assert manifest['seeds']['0']['occupancy_threshold'] == TAU
    for module in (reference_net, reference_targets):
        assert Path(module.__file__).is_relative_to(REF)
    frozen_hashes = json.loads((REF.parent / 'audit_file_hashes.json').read_text())
    for path in (REF / 'gate8').glob('*.py'):
        assert sha(path) == frozen_hashes[str(path.relative_to(ROOT))]['sha256']
    eligible, excluded = [], []
    required = {'dims', 't', 'native_frame', 'input_frames', 'target_frames', 'rows',
                'logodds', 'w_free', 'n_obs', 'age', 'observed', 'sem_rows', 'sem_p',
                'sem_w', 'gt_occ', 'gt_valid', 'fut_observed', 'fut_rows', 'fut_p'}
    for path in sorted((DATA / 'samples' / DRIVE).glob('*.npz')):
        target = DATA / 'targets' / DRIVE / path.name
        try:
            with np.load(path, allow_pickle=False) as z, np.load(target, allow_pickle=False) as t:
                assert required.issubset(z.files), 'missing input fields'
                dims = tuple(int(v) for v in z['dims'])
                assert dims == tuple(t['dims']) == (256, 256, 32), 'unexpected source grid'
                assert int(z['t']) == int(t['stream_index']) == int(path.stem)
                assert int(z['native_frame']) == int(t['native_frame'])
                assert np.array_equal(z['gt_occ'], t['occ_packed']), 'occupied targets differ'
                assert np.array_equal(z['gt_valid'], t['valid_packed']), 'valid masks differ'
                gt, valid = labels(t, dims)
                assert valid.any() and not (gt & ~valid).any(), 'unusable labels'
                for key in ('logodds', 'w_free', 'n_obs', 'age', 'observed', 'sem_p', 'sem_w'):
                    assert np.isfinite(z[key]).all(), f'nonfinite {key}'
                reg = regions(dims)
                eligible.append(dict(sample_id=path.stem, input_path=str(path), target_path=str(target),
                    input_sha256=sha(path), target_sha256=sha(target), dims=list(dims),
                    native_frame=int(z['native_frame']), target_frames_used=t['frames_used'].tolist(),
                    input_frames=z['input_frames'].tolist(),
                    populations={k: dict(valid=int((valid & r).sum()),
                        occupied=int((gt & valid & r).sum()), free=int((~gt & valid & r).sum()))
                        for k, r in reg.items()}))
        except (OSError, ValueError, KeyError, AssertionError) as e:
            excluded.append(dict(sample_id=path.stem, reason=str(e)))
        if len(eligible) and len(eligible) % 100 == 0:
            print('checked source inputs', len(eligible), flush=True)
    if len(eligible) < 3:
        write_json('blocker.json', dict(reason='Fewer than three usable source input/target pairs', excluded=excluded))
        raise RuntimeError('Insufficient usable source artifacts; no inference performed')
    indices = [0, len(eligible) // 2, len(eligible) - 1]
    selected = [eligible[i] for i in indices]
    model = reference_net.load_checkpoint(str(CKPT), 'cpu').net
    affected = [name for name, layer in model.named_modules()
                if isinstance(layer, torch.nn.Conv3d) and any(layer.padding)]
    assert all(dict(model.named_modules())[name].padding_mode == 'zeros' for name in affected)
    files = [CKPT, MANIFEST, Path(__file__), ROOT / 'tools/gate8c1/build_samples.py',
             ROOT / 'tools/gate8c1/build_targets.py', ROOT / 'sscbench_kitti360/adapter.py',
             ROOT / 'gates/gate8/net.py', ROOT / 'gates/gate8/targets.py', ROOT / 'gates/gate8/vocab.py']
    files += list((REF / 'gate8').glob('*.py'))
    selection = dict(eligible_ids=[r['sample_id'] for r in eligible], eligible=eligible,
                     excluded=excluded, chosen_indices=indices, chosen_ids=[r['sample_id'] for r in selected],
                     order='lexicographic five-digit sample ID; middle is zero-based len//2',
                     eligibility='Existing readable full-volume input and matching intended raw-LiDAR target; finite input fields; nonempty valid mask')
    config = dict(checkpoint=str(CKPT), threshold=TAU, drive=DRIVE,
        input_source='Existing full validation volumes, no crops resampled',
        target_source='Existing raw-LiDAR targets, matching embedded sample masks; current bytes only, historical bytes not established',
        conditions={'A': 'zeros', 'B': 'replicate'}, affected_layers=affected,
        module_configuration=module_configuration(model), parameter_count=model.n_params(),
        voxel_size_m=.2, origin_m=[0, -25.6, -2], axes='XYZ', pad_z=0,
        input_channels=32, input_encoding='preserved gate8.targets.unpack_sample',
        unknown_encoding='channel 3 (unobserved)=1; all other encoded channels=0; raw age=-1 encodes as zero',
        residual_rule='base + residual * (abs(base) < 2)', decision='final >= -0.125',
        precision='float32 inputs; CUDA bfloat16 autocast; float32 final logits',
        evaluation=True, postprocessing=None, boundary='centre distance to any face <0.4 m',
        interior='centre distance to every face >=2 m', undefined_rates='JSON null; CSV empty; report undefined',
        criteria=dict(uniform_boundary_relative_reduction_min=.50,
                      real_boundary_fpr_relative_reduction_min=.25, tp_retention_min=.95,
                      iou_must_not_decrease=True, minimum_pooled_boundary_free_voxels=1000,
                      sufficiency_note='Conservative descriptive minimum fixed before inference, not a statistical power claim; zero baseline boundary FP is inconclusive'),
        randomness_seed=0, cudnn_benchmark=False, commands=dict(
            prepare=shlex.join([sys.executable, 'tools/gate8c1/padding_source_probe.py', 'prepare']),
            run=shlex.join([sys.executable, 'tools/gate8c1/padding_source_probe.py', 'run', '--device', 'cuda:2'])),
        artifact_hashes={str(p): sha(p) for p in files})
    write_json('selection.json', selection); write_json('config.json', config)
    (OUT / 'git_status_before.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    print('FROZEN', selection['chosen_ids'], 'of', len(eligible), 'eligible; affected layers', affected, flush=True)


def metrics(pred, gt, valid, reg):
    rows = {}
    for name, region in reg.items():
        mask = valid & region
        tp = int((pred & gt & mask).sum()); fp = int((pred & ~gt & mask).sum())
        fn = int((~pred & gt & mask).sum()); tn = int((~pred & ~gt & mask).sum())
        rows[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn, valid=tp+fp+fn+tn,
                          occupied=tp+fn, free=fp+tn, iou=ratio(tp, tp+fp+fn), fpr=ratio(fp, fp+tn))
    return rows


def run(device):
    if (OUT / 'run_started.json').exists():
        raise RuntimeError('This frozen experiment already started; do not add inference runs')
    cfg = json.loads((OUT / 'config.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    selected = [r for r in selection['eligible'] if r['sample_id'] in selection['chosen_ids']]
    assert len(selected) == 3 and cfg['threshold'] == TAU
    for path, expected in cfg['artifact_hashes'].items():
        assert sha(path) == expected, path
    for row in selected:
        assert sha(row['input_path']) == row['input_sha256']
        assert sha(row['target_path']) == row['target_sha256']
    # Availability is checked before recording a run, so sandbox denial consumes no case.
    assert torch.cuda.is_available(), 'CUDA unavailable; no inference performed'
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = False
    model = reference_net.load_checkpoint(str(CKPT), device).net
    layers = dict(model.named_modules()); original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before = module_configuration(model)
    write_json('run_started.json', dict(time_unix=time.time(), command=shlex.join([sys.executable, *sys.argv]),
        config_sha256=sha(OUT/'config.json'), selection_sha256=sha(OUT/'selection.json'),
        python=sys.version, torch=torch.__version__, numpy=np.__version__, device=device,
        gpu=torch.cuda.get_device_name(device), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark))
    cases, controls, profiles, invariants = {}, {}, [], []

    @torch.no_grad()
    def pair(case_id, inp, base):
        input_hash = tensor_sha(inp); base_hash = tensor_sha(base); predictions = {}
        for condition, mode in cfg['conditions'].items():
            try:
                for name in cfg['affected_layers']: layers[name].padding_mode = mode
                actual = module_configuration(model)
                expected = {k: dict(v) for k, v in before.items()}
                for name in cfg['affected_layers']: expected[name]['padding_mode'] = mode
                assert actual == expected
                assert all(torch.equal(v.cpu(), original_state[k]) for k,v in model.state_dict().items())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    residual, semantic_logits = model(inp[None])
                final = reference_net.apply_residual(base, residual[0, 0].float())
                assert torch.isfinite(final).all()
                locked = base.abs() >= 2
                assert torch.equal(final[locked], base[locked])
                predictions[condition] = (final >= TAU).cpu().numpy()
                invariants.append(dict(case_id=case_id, condition=condition,
                    input_sha256=input_hash, base_sha256=base_hash, final_sha256=tensor_sha(final),
                    weights_unchanged=all(torch.equal(v.cpu(), original_state[k]) for k,v in model.state_dict().items()),
                    input_unchanged=tensor_sha(inp)==input_hash, base_unchanged=tensor_sha(base)==base_hash,
                    only_requested_padding_attributes_changed=True, locked_logits_unchanged=True))
                del residual, semantic_logits, final
            finally:
                for name in cfg['affected_layers']: layers[name].padding_mode = 'zeros'
        assert module_configuration(model) == before
        np.savez_compressed(OUT/f'predictions_{case_id}.npz', dims=list(base.shape),
                            **{k: np.packbits(v.ravel()) for k,v in predictions.items()})
        print('finished A/B', case_id, flush=True)
        return predictions

    def profile(case_id, preds, valid=None):
        for condition, pred in preds.items():
            for z in range(pred.shape[2]):
                row = dict(case_id=case_id, condition=condition, z_index=z, z_m=-2+(z+.5)*.2,
                    unmasked_occupied=int(pred[:,:,z].sum()), all_voxels=int(pred[:,:,z].size),
                    unmasked_fraction=float(pred[:,:,z].mean()), scored_occupied=None, valid=None, scored_fraction=None)
                if valid is not None:
                    row['scored_occupied']=int((pred[:,:,z] & valid[:,:,z]).sum()); row['valid']=int(valid[:,:,z].sum())
                    row['scored_fraction']=ratio(row['scored_occupied'],row['valid'])
                profiles.append(row)

    for row in selected:
        with np.load(row['input_path'], allow_pickle=False) as z:
            dense=reference_targets.unpack_sample(z,'cpu')
        inp=dense['input'].to(device); base=dense['base_logodds'].to(device)
        dims=tuple(row['dims']); assert tuple(inp.shape)==(32,*dims)
        with np.load(row['target_path'], allow_pickle=False) as z: gt,valid=labels(z,dims)
        assert np.array_equal(dense['gt_occ'].numpy().astype(bool),gt)
        assert np.array_equal(dense['gt_valid'].numpy(),valid)
        del dense
        preds=pair(row['sample_id'],inp,base); profile(row['sample_id'],preds,valid)
        cases[row['sample_id']]={c:metrics(p,gt,valid,regions(dims)) for c,p in preds.items()}
        del inp,base
    for dims in sorted({tuple(r['dims']) for r in selected}):
        name='uniform_'+'x'.join(map(str,dims)); inp=torch.zeros((32,*dims),device=device)
        inp[3]=1; base=torch.zeros(dims,device=device)
        preds=pair(name,inp,base); profile(name,preds)
        controls[name]={c:{r:dict(occupied=int((p & m).sum()),voxels=int(m.sum()),
                                occupied_fraction=ratio(int((p & m).sum()),int(m.sum())))
                           for r,m in regions(dims).items()} for c,p in preds.items()}
        del inp,base
    pooled={}
    for condition in cfg['conditions']:
        pooled[condition]={}
        for reg in ('all','boundary','interior'):
            c={k:sum(case[condition][reg][k] for case in cases.values())
               for k in ('tp','fp','fn','tn','valid','occupied','free')}
            pooled[condition][reg]=dict(c,iou=ratio(c['tp'],c['tp']+c['fp']+c['fn']),fpr=ratio(c['fp'],c['free']))
    reductions={name:ratio(v['A']['boundary']['occupied_fraction']-v['B']['boundary']['occupied_fraction'],
                           v['A']['boundary']['occupied_fraction']) for name,v in controls.items()}
    a,b=pooled['A'],pooled['B']; retention=ratio(b['all']['tp'],a['all']['tp'])
    fpr_drop=ratio(a['boundary']['fpr']-b['boundary']['fpr'],a['boundary']['fpr']) if a['boundary']['fpr'] is not None else None
    support=a['boundary']['free'] >= cfg['criteria']['minimum_pooled_boundary_free_voxels'] and a['boundary']['fp']>0
    status=lambda v:'inconclusive' if v is None else ('passed' if v else 'failed')
    practical=dict(boundary_fpr=status(fpr_drop>=.25 if support and fpr_drop is not None else None),
                   tp_retention=status(retention>=.95 if retention is not None else None),
                   pooled_iou=status(b['all']['iou']>=a['all']['iou'] if a['all']['iou'] is not None and b['all']['iou'] is not None else None))
    overall='inconclusive' if not support or 'inconclusive' in practical.values() else ('passed' if all(v=='passed' for v in practical.values()) else 'failed')
    result=dict(real=cases,pooled=pooled,uniform=controls,
                per_case_tp_retention={k:ratio(v['B']['all']['tp'],v['A']['all']['tp']) for k,v in cases.items()},
                pooled_tp_retention=retention,boundary_fpr_relative_reduction=fpr_drop,
                uniform_boundary_relative_reductions=reductions,
                screening=dict(uniform={k:status(v>=.5 if v is not None else None) for k,v in reductions.items()},
                               practical=practical,practical_overall=overall,sufficient_boundary_support=support))
    write_json('results.json',result)
    write_json('invariants.json',dict(conditions=invariants,affected_layers=cfg['affected_layers'],
        final_module_configuration_restored=module_configuration(model)==before,
        parameter_sha256={k:tensor_sha(v) for k,v in original_state.items()},
        recorded_artifact_hashes_still_match=all(sha(p)==h for p,h in cfg['artifact_hashes'].items())))
    rows=[dict(case_id=case,condition=c,region=r,**vals) for case,cs in {**cases,'pooled':pooled}.items()
          for c,regs in cs.items() for r,vals in regs.items()]
    csvwrite('counts.csv',rows); csvwrite('z_profiles.csv',profiles)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    middle=selection['chosen_ids'][1]; names=[*controls,middle]
    fig,axs=plt.subplots(1,len(names),figsize=(5.5*len(names),3.6),layout='constrained',squeeze=False)
    for ax,name in zip(axs[0],names):
        for condition,color in [('A','#c64b42'),('B','#247ab5')]:
            rs=[r for r in profiles if r['case_id']==name and r['condition']==condition]
            z=[r['z_m'] for r in rs]
            ax.plot(z,[100*r['unmasked_fraction'] for r in rs],color=color,label=condition+' unmasked')
            if name==middle:
                ax.plot(z,[np.nan if r['scored_fraction'] is None else 100*r['scored_fraction'] for r in rs],
                        '--',color=color,label=condition+' among valid voxels')
        ax.set(xlabel='Voxel-centre z (m)',ylabel='Predicted occupied (%)',ylim=(-2,102),
               title='Uniform unknown; no GT' if name in controls else f'Middle real case {middle}')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Fixed seed 0: A zero padding; B replicated padding\nFull 256×256×32 input; no spatial post-processing')
    fig.savefig(OUT/'z_profiles.png',dpi=160); plt.close(fig)
    print(json.dumps(result['screening']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    else: run(args.device)
