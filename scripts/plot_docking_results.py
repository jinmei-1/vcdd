#!/usr/bin/env python3
"""Merge docking shards, compute enrichment metrics, and draw manuscript figures.

Figures produced (PNG + SVG):
  1. enrichment_curve.png/svg  - % actives found vs % library screened (log x)
  2. score_distribution.png/svg - actives vs decoys Vina score distributions
  3. ef_summary.png/svg        - EF1/5/10% bar chart with 95% CI

Usage: python plot_docking_results.py
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/workspace')
from enrichment_analysis import metrics, bootstrap_ci

OUT = '/workspace/docking_results'
import os
os.makedirs(OUT, exist_ok=True)

# --- merge shards ---
parts = []
for f in sorted(glob.glob('/workspace/docking_out/scores_shard*.csv')):
    parts.append(pd.read_csv(f))
sc = pd.concat(parts, ignore_index=True).drop_duplicates('set_id').dropna(subset=['score'])
sc['score'] = pd.to_numeric(sc['score'], errors='coerce')
sc = sc.dropna(subset=['score'])
print('docked compounds:', len(sc), '| actives:', int(sc.label.sum()), '| decoys:', int((sc.label == 0).sum()))
sc.to_csv(f'{OUT}/dnapkcs_benchmark_vina_scores.csv', index=False)

y = sc.label.astype(int).values
s = -sc.score.values.astype(float)  # higher = better (Vina is lower=better)

# --- metrics ---
m = metrics(y, s)
lo, hi = bootstrap_ci(y, s)
mtab = pd.DataFrame({'metric': list(m), 'value': [round(v, 4) for v in m.values()],
                     'ci95_lo': [round(lo[k], 4) for k in m],
                     'ci95_hi': [round(hi[k], 4) for k in m]})
print(mtab.to_string(index=False))
mtab.to_csv(f'{OUT}/dnapkcs_benchmark_vina_metrics.csv', index=False)

# --- Fig 1: enrichment curve ---
order = np.argsort(-s, kind='mergesort')
yy = y[order]
frac_found = np.concatenate([[0], np.cumsum(yy) / yy.sum()])
frac_screened = np.concatenate([[0], (np.arange(len(yy)) + 1) / len(yy)])
fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=200)
ax.plot(frac_screened * 100, frac_found * 100, color='#0279EE', lw=2, label='Vina (stage 1)')
# random expectation on a log-x enrichment plot is y = x (NOT a corner-to-corner diagonal)
xr = np.logspace(-2, 2, 200)
ax.plot(xr, xr, 'k--', lw=1, label='random')
ax.set_xscale('log'); ax.set_xlim(0.01, 100); ax.set_ylim(0, 105)
ax.set_xlabel('% library screened (log scale)')
ax.set_ylabel('% actives found')
ax.set_title('DNA-PKcs retrospective enrichment')
ax.legend(frameon=False, fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/enrichment_curve.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUT}/enrichment_curve.svg', bbox_inches='tight')
plt.close()

# --- Fig 2: score distribution ---
fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=200)
data = [sc.loc[sc.label == 0, 'score'], sc.loc[sc.label == 1, 'score']]
bp = ax.boxplot(data, tick_labels=['decoys', 'actives'], patch_artist=True, widths=0.55,
                medianprops=dict(color='black'), showfliers=False)
for patch, c in zip(bp['boxes'], ['#9ecae1', '#fb6a4a']):
    patch.set_facecolor(c)
# overlay jittered points (subsample decoys)
rng = np.random.default_rng(0)
for i, d in enumerate(data):
    dd = d if len(d) < 500 else d.sample(500, random_state=0)
    ax.scatter(np.full(len(dd), i + 1) + rng.normal(0, 0.05, len(dd)), dd,
               s=3, alpha=0.25, color='gray', zorder=0)
ax.set_ylabel('Vina score (kcal/mol)')
ax.set_title('Docking score distribution')
plt.tight_layout()
plt.savefig(f'{OUT}/score_distribution.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUT}/score_distribution.svg', bbox_inches='tight')
plt.close()

# --- Fig 3: EF summary bars ---
efs = ['EF1%', 'EF5%', 'EF10%']
vals = [m[k] for k in efs]
errs = [[m[k] - lo[k] for k in efs], [hi[k] - m[k] for k in efs]]
fig, ax = plt.subplots(figsize=(4.0, 3.6), dpi=200)
bars = ax.bar(efs, vals, yerr=errs, capsize=4, color=['#08519c', '#3182bd', '#6baed6'],
              edgecolor='black', error_kw=dict(lw=1.2))
ax.axhline(1.0, color='k', ls='--', lw=1, label='no enrichment (EF=1)')
for b, v, k in zip(bars, vals, efs):
    ax.text(b.get_x() + b.get_width() / 2, hi[k] + 0.35, f'{v:.1f}', ha='center', fontsize=9)
ax.set_ylabel('Enrichment factor')
ax.set_title('Early enrichment (Vina stage 1)')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/ef_summary.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUT}/ef_summary.svg', bbox_inches='tight')
plt.close()

print('figures saved to', OUT)
