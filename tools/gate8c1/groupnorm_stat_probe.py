#!/usr/bin/env python
"""One frozen, source-only A/B/C GroupNorm-statistics counterfactual.

A  zero padding, standard GroupNorm (the original Gate 8C-1 inference path).
B  replicated padding, standard GroupNorm (the previous padding probe).
C  replicated padding, GroupNorm normalised with A's captured per-sample,
   per-layer, per-group statistics and A's original affine parameters.

GroupNorm holds no running statistics, so C is an explicit inference-only
counterfactual: it is not a deployable model. Cases, checkpoint, threshold,
masks, grid, channels, residual rule and precision are inherited unchanged
from artifacts/gate8c1_padding_source_probe; nothing is re-selected here.
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
PREV = ROOT / 'artifacts/gate8c1_padding_source_probe'
OUT = ROOT / 'artifacts/gate8c1_groupnorm_stat_probe'
CKPT = ROOT / 'artifacts/gate8c1/checkpoints/seed0_last.pt'
MANIFEST = ROOT / 'artifacts/gate8c1/frozen_manifest.json'
TAU = -0.125
# Previous audit's largest causal completion IoU replay discrepancy was 0.00061
# percentage points; that is the replay tolerance reused here for condition A.
REPLAY_IOU_TOL_PP = 0.001
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
    # XYZ voxel-centre distance; identical definition to the previous probe.
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


def unpackbits(z, key, dims):
    return np.unpackbits(z[key], count=int(np.prod(dims))).reshape(dims).astype(bool)


def module_configuration(net):
    attrs = ('padding_mode', 'padding', 'stride', 'dilation', 'kernel_size',
             'output_padding', 'groups', 'num_groups', 'eps')
    return {name: dict(type=type(layer).__name__, training=layer.training,
                      **{k: getattr(layer, k) for k in attrs if hasattr(layer, k)})
            for name, layer in net.named_modules()}


def metrics(pred, gt, valid, reg):
    rows = {}
    for name, region in reg.items():
        mask = valid & region
        tp = int((pred & gt & mask).sum()); fp = int((pred & ~gt & mask).sum())
        fn = int((~pred & gt & mask).sum()); tn = int((~pred & ~gt & mask).sum())
        rows[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn, valid=tp+fp+fn+tn,
                          occupied=tp+fn, free=fp+tn, iou=ratio(tp, tp+fp+fn), fpr=ratio(fp, fp+tn))
    return rows


class FrozenStatGroupNorm(torch.nn.Module):
    """GroupNorm normalised with externally supplied statistics.

    Holds the ORIGINAL GroupNorm module, so num_groups, eps and the affine
    weight/bias tensors are the identical objects; no parameter is copied,
    reinitialised or changed.
    """

    def __init__(self, gn, mean, var, out_dtype):
        super().__init__()
        self.gn = gn
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('var', var, persistent=False)
        self.out_dtype = out_dtype

    def forward(self, x):
        n, c = x.shape[:2]
        g = self.gn.num_groups
        y = x.float().reshape(n, g, -1)
        y = (y - self.mean[..., None]) * torch.rsqrt(self.var[..., None] + self.gn.eps)
        y = y.reshape(n, c, *x.shape[2:])
        shape = (1, -1) + (1,) * (x.dim() - 2)
        y = y * self.gn.weight.float().reshape(shape) + self.gn.bias.float().reshape(shape)
        return y.to(self.out_dtype)


def set_module(root, name, module):
    parent, _, attr = name.rpartition('.')
    setattr(root.get_submodule(parent) if parent else root, attr, module)


def run(device):
    if OUT.exists():
        raise RuntimeError(f'Preserve existing experiment directory: {OUT}')
    # ---- integrity of everything inherited, before any inference -------------
    manifest = json.loads(MANIFEST.read_text())
    assert sha(CKPT) == manifest['seeds']['0']['checkpoint_sha256'], 'checkpoint hash'
    assert manifest['seeds']['0']['occupancy_threshold'] == TAU, 'threshold'
    for module in (reference_net, reference_targets):
        assert Path(module.__file__).is_relative_to(REF), 'reference sources'
    frozen_hashes = json.loads((REF.parent / 'audit_file_hashes.json').read_text())
    for path in (REF / 'gate8').glob('*.py'):
        assert sha(path) == frozen_hashes[str(path.relative_to(ROOT))]['sha256'], path
    prev_cfg = json.loads((PREV / 'config.json').read_text())
    prev_sel = json.loads((PREV / 'selection.json').read_text())
    prev_res = json.loads((PREV / 'results.json').read_text())
    assert prev_cfg['threshold'] == TAU and prev_cfg['conditions'] == {'A': 'zeros', 'B': 'replicate'}
    for path, expected in prev_cfg['artifact_hashes'].items():
        assert sha(path) == expected, f'previous artifact changed: {path}'
    selected = [r for r in prev_sel['eligible'] if r['sample_id'] in prev_sel['chosen_ids']]
    assert [r['sample_id'] for r in selected] == ['00004', '00889', '01771'], 'frozen cases'
    for row in selected:
        assert sha(row['input_path']) == row['input_sha256'], row['sample_id']
        assert sha(row['target_path']) == row['target_sha256'], row['sample_id']
    assert torch.cuda.is_available(), 'CUDA unavailable; no inference performed'

    OUT.mkdir()
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = False
    model = reference_net.load_checkpoint(str(CKPT), device).net
    layers = dict(model.named_modules())
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    before = module_configuration(model)
    padded = prev_cfg['affected_layers']
    assert all(layers[n].padding_mode == 'zeros' for n in padded)
    gnames = [n for n, m in model.named_modules() if isinstance(m, torch.nn.GroupNorm)]

    cfg = dict(
        experiment='gate8c1_groupnorm_stat_probe',
        hypothesis='GroupNorm-statistics propagation explains why replicated padding suppresses '
                   'the artificial boundary ceiling but destroys useful occupancy',
        checkpoint=str(CKPT), checkpoint_sha256=sha(CKPT), threshold=TAU,
        drive=prev_cfg['drive'], cases=[r['sample_id'] for r in selected],
        inherited_from=str(PREV.relative_to(ROOT)),
        inherited_config_sha256=sha(PREV / 'config.json'),
        inherited_selection_sha256=sha(PREV / 'selection.json'),
        inherited_results_sha256=sha(PREV / 'results.json'),
        conditions={'A': 'zero padding, standard GroupNorm',
                    'B': 'replicated padding, standard GroupNorm',
                    'C': 'replicated padding, GroupNorm normalised with A per-sample per-group '
                         'mean/variance and A affine parameters'},
        validation_condition={'A_selfstat': 'zero padding, frozen-statistic GroupNorm fed A\'s own '
                                            'statistics; must reproduce A'},
        affected_padding_layers=padded, groupnorm_layers=gnames,
        groupnorm_note='GroupNorm has no running statistics; C is an explicit inference-only '
                       'counterfactual, computed over the same channel-group and spatial '
                       'dimensions PyTorch GroupNorm uses, with unbiased=False variance in float32.',
        module_configuration=module_configuration(model), parameter_count=model.n_params(),
        voxel_size_m=.2, origin_m=[0, -25.6, -2], axes='XYZ', pad_z=0, input_channels=32,
        input_encoding='preserved gate8.targets.unpack_sample',
        unknown_encoding='channel 3 (unobserved)=1; all other encoded channels=0; raw age=-1 encodes as zero',
        residual_rule='base + residual * (abs(base) < 2)', decision='final >= -0.125',
        precision='float32 inputs; CUDA bfloat16 autocast; float32 final logits; '
                  'frozen-statistic normalisation in float32 cast back to the dtype standard '
                  'GroupNorm returned in A',
        evaluation=True, postprocessing=None,
        boundary='centre distance to any face <0.4 m', interior='centre distance to every face >=2 m',
        masks='applied only when counting; never reach model inputs or predictions',
        undefined_rates='JSON null; CSV empty; report undefined',
        replay=dict(reference='artifacts/gate8c1_padding_source_probe',
                    tolerance_iou_percentage_points=REPLAY_IOU_TOL_PP,
                    tolerance_source='largest causal completion IoU replay discrepancy recorded in '
                                     'the Gate 8C-1 original audit (0.00061 pp), rounded up'),
        criteria=dict(boundary_fpr_relative_reduction_min=.25, tp_retention_min=.95,
                      iou_must_not_decrease=True),
        randomness_seed=0, cudnn_benchmark=False,
        command=shlex.join([sys.executable, 'tools/gate8c1/groupnorm_stat_probe.py',
                            'run', '--device', device]),
        artifact_hashes={**prev_cfg['artifact_hashes'],
                         str(ROOT / 'tools/gate8c1/groupnorm_stat_probe.py'):
                             sha(ROOT / 'tools/gate8c1/groupnorm_stat_probe.py'),
                         **{str(PREV / n): sha(PREV / n) for n in
                            ('config.json', 'selection.json', 'results.json', 'counts.csv',
                             'predictions_00004.npz', 'predictions_00889.npz',
                             'predictions_01771.npz', 'predictions_uniform_256x256x32.npz')}},
        case_inputs=[{k: r[k] for k in ('sample_id', 'input_path', 'input_sha256', 'target_path',
                                        'target_sha256', 'dims', 'native_frame')} for r in selected])
    write_json('config.json', cfg)
    (OUT / 'git_status_before.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT))
    write_json('run_started.json', dict(time_unix=time.time(), command=cfg['command'],
        config_sha256=sha(OUT / 'config.json'), python=sys.version, torch=torch.__version__,
        numpy=np.__version__, device=device, gpu=torch.cuda.get_device_name(device),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark))

    cases, controls, profiles, invariants, gnrows, replay = {}, {}, [], [], [], []
    captured = {}

    def capture_hooks(store):
        counter = {}
        handles = []
        def make(name):
            def hook(mod, args, output):
                idx = counter.get(name, 0); counter[name] = idx + 1
                x = args[0]
                n, g = x.shape[0], mod.num_groups
                xf = x.detach().float().reshape(n, g, -1)
                store[f'{name}#{idx}'] = dict(mean=xf.mean(-1), var=xf.var(-1, unbiased=False),
                                              in_dtype=str(x.dtype), out_dtype=output.dtype,
                                              shape=list(x.shape))
            return hook
        for name in gnames:
            handles.append(layers[name].register_forward_hook(make(name)))
        return handles

    @torch.no_grad()
    def forward(inp, base, mode, stats=None, store=None):
        """One inference: `mode` is the padding mode; `stats` freezes GroupNorm."""
        handles = capture_hooks(store) if store is not None else []
        try:
            for name in padded:
                layers[name].padding_mode = mode
            if stats is not None:
                for name in gnames:
                    s = stats[f'{name}#0']
                    set_module(model, name, FrozenStatGroupNorm(layers[name], s['mean'], s['var'],
                                                                s['out_dtype']))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                residual, _sem = model(inp[None])
            final = reference_net.apply_residual(base, residual[0, 0].float())
            assert torch.isfinite(final).all()
            locked = base.abs() >= 2
            assert torch.equal(final[locked], base[locked]), 'residual lock violated'
            return final
        finally:
            for h in handles:
                h.remove()
            if stats is not None:
                for name in gnames:
                    set_module(model, name, layers[name])
            for name in padded:
                layers[name].padding_mode = 'zeros'

    def check_invariants(case_id, condition, inp, base, final, input_hash, base_hash):
        assert module_configuration(model) == before, 'module configuration not restored'
        weights_ok = all(torch.equal(v.cpu(), original_state[k]) for k, v in model.state_dict().items())
        invariants.append(dict(case_id=case_id, condition=condition, input_sha256=input_hash,
            base_sha256=base_hash, final_sha256=tensor_sha(final), weights_unchanged=weights_ok,
            input_unchanged=tensor_sha(inp) == input_hash, base_unchanged=tensor_sha(base) == base_hash,
            module_configuration_restored=True, locked_logits_unchanged=True))
        assert weights_ok, 'parameter tensors changed'

    def triple(case_id, inp, base):
        input_hash, base_hash = tensor_sha(inp), tensor_sha(base)
        stats_a, stats_b = {}, {}
        preds, finals = {}, {}

        final = forward(inp, base, 'zeros', store=stats_a)
        check_invariants(case_id, 'A', inp, base, final, input_hash, base_hash)
        preds['A'] = (final >= TAU).cpu().numpy(); finals['A'] = final

        chk = forward(inp, base, 'zeros', stats=stats_a)
        check_invariants(case_id, 'A_selfstat', inp, base, chk, input_hash, base_hash)
        self_max = float((chk - finals['A']).abs().max())
        self_agree = float(((chk >= TAU) == (finals['A'] >= TAU)).float().mean())
        del chk

        final = forward(inp, base, 'replicate', store=stats_b)
        check_invariants(case_id, 'B', inp, base, final, input_hash, base_hash)
        preds['B'] = (final >= TAU).cpu().numpy(); del final

        final = forward(inp, base, 'replicate', stats=stats_a)
        check_invariants(case_id, 'C', inp, base, final, input_hash, base_hash)
        preds['C'] = (final >= TAU).cpu().numpy(); del final, finals

        for key in stats_a:
            a, b = stats_a[key], stats_b[key]
            for grp in range(a['mean'].shape[1]):
                ma, va = float(a['mean'][0, grp]), float(a['var'][0, grp])
                mb, vb = float(b['mean'][0, grp]), float(b['var'][0, grp])
                gnrows.append(dict(case_id=case_id, layer=key, group=grp,
                    mean_A=ma, var_A=va, mean_B=mb, var_B=vb, mean_delta=mb - ma,
                    std_A=va ** .5, std_B=vb ** .5, std_ratio_B_over_A=ratio(vb ** .5, va ** .5),
                    mean_shift_in_A_std=ratio(abs(mb - ma), va ** .5),
                    in_dtype=a['in_dtype'], out_dtype=str(a['out_dtype']),
                    shape='x'.join(map(str, a['shape']))))
        np.savez_compressed(OUT / f'predictions_{case_id}.npz', dims=list(base.shape),
                            **{k: np.packbits(v.ravel()) for k, v in preds.items()})
        print('finished A/B/C', case_id, 'selfstat_max_abs', self_max, flush=True)
        return preds, dict(case_id=case_id, selfstat_max_abs_logit_diff=self_max,
                           selfstat_decision_agreement=self_agree)

    def profile(case_id, preds, valid=None):
        for condition, pred in preds.items():
            for z in range(pred.shape[2]):
                row = dict(case_id=case_id, condition=condition, z_index=z, z_m=-2 + (z + .5) * .2,
                    unmasked_occupied=int(pred[:, :, z].sum()), all_voxels=int(pred[:, :, z].size),
                    unmasked_fraction=float(pred[:, :, z].mean()), scored_occupied=None,
                    valid=None, scored_fraction=None)
                if valid is not None:
                    row['scored_occupied'] = int((pred[:, :, z] & valid[:, :, z]).sum())
                    row['valid'] = int(valid[:, :, z].sum())
                    row['scored_fraction'] = ratio(row['scored_occupied'], row['valid'])
                profiles.append(row)

    selfstat = []
    for row in selected:
        with np.load(row['input_path'], allow_pickle=False) as z:
            dense = reference_targets.unpack_sample(z, 'cpu')
        inp = dense['input'].to(device); base = dense['base_logodds'].to(device)
        dims = tuple(row['dims']); assert tuple(inp.shape) == (32, *dims)
        with np.load(row['target_path'], allow_pickle=False) as z:
            gt, valid = labels(z, dims)
        assert np.array_equal(dense['gt_occ'].numpy().astype(bool), gt)
        assert np.array_equal(dense['gt_valid'].numpy(), valid)
        del dense
        preds, sstat = triple(row['sample_id'], inp, base)
        selfstat.append(sstat)
        profile(row['sample_id'], preds, valid)
        cases[row['sample_id']] = {c: metrics(p, gt, valid, regions(dims)) for c, p in preds.items()}
        with np.load(PREV / f'predictions_{row["sample_id"]}.npz', allow_pickle=False) as z:
            prev_a = unpackbits(z, 'A', dims)
        prev_counts = prev_res['real'][row['sample_id']]['A']
        now = cases[row['sample_id']]['A']
        replay.append(dict(case_id=row['sample_id'],
            voxels_disagreeing_with_previous_A=int((prev_a != preds['A']).sum()),
            bitwise_identical=bool(np.array_equal(prev_a, preds['A'])),
            **{f'{k}_previous': prev_counts['all'][k] for k in ('tp', 'fp', 'fn')},
            **{f'{k}_now': now['all'][k] for k in ('tp', 'fp', 'fn')},
            iou_previous_pp=100 * prev_counts['all']['iou'], iou_now_pp=100 * now['all']['iou'],
            iou_delta_pp=100 * (now['all']['iou'] - prev_counts['all']['iou'])))
        if abs(replay[-1]['iou_delta_pp']) > REPLAY_IOU_TOL_PP:
            write_json('replay_mismatch.json', dict(replay=replay, tolerance_pp=REPLAY_IOU_TOL_PP,
                note='Condition A did not reproduce the previous padding-probe A counts; '
                     'stopped before interpreting any A/B/C comparison.'))
            raise RuntimeError(f'A replay mismatch on {row["sample_id"]}: '
                               f'{replay[-1]["iou_delta_pp"]:.6f} pp > {REPLAY_IOU_TOL_PP} pp')
        del inp, base

    for dims in sorted({tuple(r['dims']) for r in selected}):
        name = 'uniform_' + 'x'.join(map(str, dims))
        inp = torch.zeros((32, *dims), device=device); inp[3] = 1
        base = torch.zeros(dims, device=device)
        preds, sstat = triple(name, inp, base)
        selfstat.append(sstat)
        profile(name, preds)
        controls[name] = {c: {r: dict(occupied=int((p & m).sum()), voxels=int(m.sum()),
                                      occupied_fraction=ratio(int((p & m).sum()), int(m.sum())))
                              for r, m in regions(dims).items()} for c, p in preds.items()}
        with np.load(PREV / f'predictions_{name}.npz', allow_pickle=False) as z:
            prev_a = unpackbits(z, 'A', dims)
        replay.append(dict(case_id=name,
            voxels_disagreeing_with_previous_A=int((prev_a != preds['A']).sum()),
            bitwise_identical=bool(np.array_equal(prev_a, preds['A'])),
            tp_previous=None, fp_previous=None, fn_previous=None,
            tp_now=None, fp_now=None, fn_now=None,
            iou_previous_pp=None, iou_now_pp=None, iou_delta_pp=None))
        del inp, base

    pooled = {}
    for condition in ('A', 'B', 'C'):
        pooled[condition] = {}
        for reg in ('all', 'boundary', 'interior'):
            c = {k: sum(case[condition][reg][k] for case in cases.values())
                 for k in ('tp', 'fp', 'fn', 'tn', 'valid', 'occupied', 'free')}
            pooled[condition][reg] = dict(c, iou=ratio(c['tp'], c['tp'] + c['fp'] + c['fn']),
                                          fpr=ratio(c['fp'], c['free']))
    a = pooled['A']
    retention = {c: ratio(pooled[c]['all']['tp'], a['all']['tp']) for c in ('B', 'C')}
    fpr_drop = {c: ratio(a['boundary']['fpr'] - pooled[c]['boundary']['fpr'], a['boundary']['fpr'])
                for c in ('B', 'C')}
    replay_ok = all(r['iou_delta_pp'] is None or abs(r['iou_delta_pp']) <= REPLAY_IOU_TOL_PP
                    for r in replay)
    status = lambda v: 'inconclusive' if v is None else ('passed' if v else 'failed')
    criteria = dict(
        boundary_fpr=status(fpr_drop['C'] >= .25 if fpr_drop['C'] is not None else None),
        tp_retention=status(retention['C'] >= .95 if retention['C'] is not None else None),
        pooled_iou=status(pooled['C']['all']['iou'] >= a['all']['iou']
                          if None not in (a['all']['iou'], pooled['C']['all']['iou']) else None))
    overall = ('inconclusive' if 'inconclusive' in criteria.values()
               else 'passed' if all(v == 'passed' for v in criteria.values())
               else 'failed' if all(v == 'failed' for v in criteria.values()) else 'mixed')
    summary = {}
    for key in sorted({r['layer'] for r in gnrows}):
        sel = [r for r in gnrows if r['layer'] == key and not r['case_id'].startswith('uniform')]
        shift = [r['mean_shift_in_A_std'] for r in sel if r['mean_shift_in_A_std'] is not None]
        sratio = [r['std_ratio_B_over_A'] for r in sel if r['std_ratio_B_over_A'] is not None]
        summary[key] = dict(
            groups_compared=len(sel), degenerate_zero_variance_groups=len(sel) - len(shift),
            max_abs_mean_shift_in_A_std=max(shift) if shift else None,
            median_abs_mean_shift_in_A_std=float(np.median(shift)) if shift else None,
            min_std_ratio_B_over_A=min(sratio) if sratio else None,
            max_std_ratio_B_over_A=max(sratio) if sratio else None,
            median_std_ratio_B_over_A=float(np.median(sratio)) if sratio else None)
    result = dict(real=cases, pooled=pooled, uniform=controls,
                  per_case_tp_retention={k: {c: ratio(v[c]['all']['tp'], v['A']['all']['tp'])
                                             for c in ('B', 'C')} for k, v in cases.items()},
                  pooled_tp_retention=retention, boundary_fpr_relative_reduction=fpr_drop,
                  replay=replay, replay_within_tolerance=replay_ok,
                  frozen_stat_selfcheck=selfstat, groupnorm_stat_summary=summary,
                  criteria=criteria, overall=overall)
    write_json('results.json', result)
    write_json('invariants.json', dict(conditions=invariants, affected_padding_layers=padded,
        groupnorm_layers=gnames, final_module_configuration_restored=module_configuration(model) == before,
        parameter_sha256={k: tensor_sha(v) for k, v in original_state.items()},
        parameters_identical_across_conditions=all(r['weights_unchanged'] for r in invariants),
        inputs_identical_across_conditions={
            cid: len({r['input_sha256'] for r in invariants if r['case_id'] == cid}) == 1
            for cid in {r['case_id'] for r in invariants}},
        recorded_artifact_hashes_still_match=all(sha(p) == h for p, h in cfg['artifact_hashes'].items())))
    csvwrite('counts.csv', [dict(case_id=case, condition=c, region=r, **vals)
                            for case, cs in {**cases, 'pooled': pooled}.items()
                            for c, regs in cs.items() for r, vals in regs.items()])
    csvwrite('z_profiles.csv', profiles)
    csvwrite('groupnorm_statistics.csv', gnrows)
    csvwrite('uniform_control.csv', [dict(case_id=name, condition=c, region=r, **vals)
                                     for name, cs in controls.items()
                                     for c, regs in cs.items() for r, vals in regs.items()])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'A': '#c64b42', 'B': '#247ab5', 'C': '#3f9457'}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    bars = [('Pooled SC IoU %', [100 * pooled[c]['all']['iou'] for c in 'ABC']),
            ('Boundary FPR %', [100 * pooled[c]['boundary']['fpr'] for c in 'ABC']),
            ('Interior FPR %', [100 * pooled[c]['interior']['fpr'] for c in 'ABC']),
            ('TP retention vs A %', [100., 100 * retention['B'], 100 * retention['C']])]
    x = np.arange(len(bars))
    for i, c in enumerate('ABC'):
        axs[0].bar(x + (i - 1) * .27, [b[1][i] for b in bars], .27, color=colors[c], label=c)
    axs[0].set(xticks=x, ylabel='percent', title='Pooled over the three drive-0006 cases')
    axs[0].set_xticklabels([b[0] for b in bars], fontsize=8)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.2, axis='y')
    control = next(iter(controls))
    for c in 'ABC':
        rs = [r for r in profiles if r['case_id'] == control and r['condition'] == c]
        axs[1].plot([r['z_m'] for r in rs], [100 * r['unmasked_fraction'] for r in rs],
                    color=colors[c], label=c)
        rs = [r for r in profiles if r['case_id'] == '00889' and r['condition'] == c]
        axs[2].plot([r['z_m'] for r in rs], [100 * r['scored_fraction'] for r in rs],
                    color=colors[c], label=c)
    axs[1].set(xlabel='Voxel-centre z (m)', ylabel='Predicted occupied (%)', ylim=(-2, 102),
               title='Uniform unknown control; no GT')
    axs[2].set(xlabel='Voxel-centre z (m)', ylabel='Predicted occupied among valid (%)',
               title='Middle real case 00889')
    for ax in axs[1:]:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Seed 0, tau -0.125. A zero padding + standard GroupNorm; B replicate + standard '
                 'GroupNorm;\nC replicate + A\'s frozen GroupNorm statistics. Raw occupancy, no '
                 'post-processing.')
    fig.savefig(OUT / 'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(criteria=criteria, overall=overall, retention=retention,
                          fpr_drop=fpr_drop, replay_within_tolerance=replay_ok,
                          selfstat=selfstat), indent=1), flush=True)
    if not replay_ok:
        print('REPLAY MISMATCH: condition A did not reproduce the previous A counts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run'])
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    run(args.device)
