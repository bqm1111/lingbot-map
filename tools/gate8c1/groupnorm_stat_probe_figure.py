#!/usr/bin/env python
"""Render the A/B/C comparison figure from the saved probe outputs only.

Read-only: it re-reads results.json and z_profiles.csv and performs no inference.
It exists because the experiment script's own plotting call raised on undefined
scored fractions after every machine-readable output had already been written;
the experiment script is left byte-unchanged so its recorded hash stays valid.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / 'artifacts/gate8c1_groupnorm_stat_probe'
res = json.loads((OUT / 'results.json').read_text())
prof = list(csv.DictReader((OUT / 'z_profiles.csv').open()))
num = lambda v: float(v) if v not in ('', None) else np.nan
colors = {'A': '#c64b42', 'B': '#247ab5', 'C': '#3f9457'}
pooled, ret = res['pooled'], res['pooled_tp_retention']

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4), layout='constrained')
bars = [('Pooled SC IoU %', [100 * pooled[c]['all']['iou'] for c in 'ABC']),
        ('Boundary FPR %', [100 * pooled[c]['boundary']['fpr'] for c in 'ABC']),
        ('Interior FPR %', [100 * pooled[c]['interior']['fpr'] for c in 'ABC']),
        ('TP retention vs A %', [100., 100 * ret['B'], 100 * ret['C']])]
x = np.arange(len(bars))
for i, c in enumerate('ABC'):
    vals = [b[1][i] for b in bars]
    axs[0].bar(x + (i - 1) * .27, vals, .27, color=colors[c], label=c)
    for xi, v in zip(x + (i - 1) * .27, vals):
        axs[0].annotate(f'{v:.1f}', (xi, v), ha='center', va='bottom', fontsize=7)
axs[0].set(xticks=x, ylabel='percent', ylim=(0, 118),
           title='Pooled over drive-0006 cases 00004 / 00889 / 01771')
axs[0].set_xticklabels([b[0] for b in bars], fontsize=8)
axs[0].legend(fontsize=8, ncol=3); axs[0].grid(alpha=.2, axis='y')

control = next(iter(res['uniform']))
for c in 'ABC':
    rs = [r for r in prof if r['case_id'] == control and r['condition'] == c]
    axs[1].plot([num(r['z_m']) for r in rs], [100 * num(r['unmasked_fraction']) for r in rs],
                color=colors[c], label=c)
    rs = [r for r in prof if r['case_id'] == '00889' and r['condition'] == c]
    axs[2].plot([num(r['z_m']) for r in rs], [100 * num(r['scored_fraction']) for r in rs],
                color=colors[c], label=c)
axs[1].set(xlabel='Voxel-centre z (m)', ylabel='Predicted occupied (%)', ylim=(-2, 102),
           title='Uniform unknown control; no ground truth')
axs[2].set(xlabel='Voxel-centre z (m)', ylabel='Predicted occupied among valid (%)',
           title='Middle real case 00889 (undefined slices omitted)')
for ax in axs[1:]:
    ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.suptitle("Seed 0, tau = -0.125.  A: zero padding + standard GroupNorm.  "
             "B: replicate padding + standard GroupNorm.\n"
             "C: replicate padding + A's frozen GroupNorm statistics.  "
             "Raw occupancy, no spatial post-processing.")
fig.savefig(OUT / 'comparison.png', dpi=160)
print('wrote', OUT / 'comparison.png')
