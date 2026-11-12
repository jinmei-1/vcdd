#!/usr/bin/env python3
"""
Retrospective enrichment analysis for the vcdd DNA-PKcs benchmark.

Given a scored benchmark (one score per compound, higher = better) and the
actives/decoys labels, computes the standard virtual-screening metrics:

  - ROC AUC
  - LogAUC (semi-log AUC up to 100% decoys; higher weight on early region)
  - EF1% / EF5% / EF10% (enrichment factors)
  - BEDROC (alpha=20, early-recognition metric)
  - 95% bootstrap confidence intervals (1000 resamples)
  - Enrichment curve figure (PNG + SVG)

Usage:
    python enrichment_analysis.py \
        --scores stage1_scores.csv \   # columns: set_id, score (higher=better)
        --labels dnapkcs_benchmark.csv \  # columns: set_id, label (1=active, 0=decoy)
        --outdir enrichment_out \
        --tag stage1_vina

For Vina docking scores (lower = better), pass --invert or pre-multiply by -1.
To compare ranking strategies (docking-only vs +Boltz-2 vs +S_cell), run once
per strategy into the same outdir with different --tag; the script appends to
summary_all_strategies.csv.
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment


def _rdkit_col(y, s):
    order = np.argsort(-np.asarray(s, dtype=float), kind='mergesort')
    return [[float(s[i]), int(y[i])] for i in order]


def roc_auc(y, s):
    return CalcAUC(_rdkit_col(y, s), 1)


def log_auc(y, s, min_fpr=1e-3):
    order = np.argsort(-s, kind='mergesort')
    y = np.asarray(y)[order]
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / N])
    fpr = np.maximum(fpr, min_fpr)
    lx = np.log10(fpr)
    # integrate tpr over log10(fpr) in [log10(min_fpr), 0]
    idx = np.argsort(lx)
    lx, lt = lx[idx], tpr[idx]
    lx, uidx = np.unique(lx, return_index=True)
    lt = lt[uidx]
    span = 0 - np.log10(min_fpr)
    return np.trapezoid(lt, lx) / span


def enrichment_factor(y, s, frac):
    return CalcEnrichment(_rdkit_col(y, s), 1, [frac])[0]


def bedroc(y, s, alpha=20.0):
    return CalcBEDROC(_rdkit_col(y, s), 1, alpha)


def metrics(y, s):
    return {'ROC_AUC': roc_auc(y, s), 'LogAUC': log_auc(y, s),
            'EF1%': enrichment_factor(y, s, 0.01),
            'EF5%': enrichment_factor(y, s, 0.05),
            'EF10%': enrichment_factor(y, s, 0.10),
            'BEDROC_a20': bedroc(y, s, 20.0)}


def bootstrap_ci(y, s, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        stats.append(metrics(y[idx], s[idx]))
    sdf = pd.DataFrame(stats)
    return sdf.quantile(0.025), sdf.quantile(0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help='CSV: set_id, score (higher=better)')
    ap.add_argument('--labels', required=True, help='CSV: set_id, label (1=active, 0=decoy)')
    ap.add_argument('--outdir', default='enrichment_out')
    ap.add_argument('--tag', default='strategy')
    ap.add_argument('--invert', action='store_true', help='multiply scores by -1 (e.g. Vina)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sc = pd.read_csv(args.scores)
    lb = pd.read_csv(args.labels)
    df = lb.merge(sc, on='set_id', how='inner').dropna(subset=['score'])
    if args.invert:
        df['score'] = -df['score']
    y = df.label.astype(int).values
    s = df.score.values.astype(float)
    print(f'{args.tag}: {len(df)} compounds ({y.sum()} actives)')

    m = metrics(y, s)
    lo, hi = bootstrap_ci(y, s)
    out = pd.DataFrame({'metric': list(m), 'value': [round(v, 4) for v in m.values()],
                        'ci95_lo': [round(lo[k], 4) for k in m],
                        'ci95_hi': [round(hi[k], 4) for k in m]})
    out.insert(0, 'strategy', args.tag)
    print(out.to_string(index=False))
    out.to_csv(f'{args.outdir}/metrics_{args.tag}.csv', index=False)
    summ = f'{args.outdir}/summary_all_strategies.csv'
    out.to_csv(summ, mode='a', header=not os.path.exists(summ), index=False)

    # enrichment curve
    import matplotlib
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    matplotlib.rcParams['svg.fonttype'] = 'none'
    import matplotlib.pyplot as plt
    order = np.argsort(-s, kind='mergesort')
    yy = y[order]
    frac_found = np.cumsum(yy) / yy.sum()
    frac_screened = (np.arange(len(yy)) + 1) / len(yy)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.plot(frac_screened * 100, frac_found * 100, color='#0279EE', lw=1.8, label=args.tag)
    ax.plot([0, 100], [0, 100], 'k--', lw=1, label='random')
    ax.set_xscale('log'); ax.set_xlim(0.01, 100); ax.set_ylim(0, 105)
    ax.set_xlabel('% library screened (log scale)')
    ax.set_ylabel('% actives found')
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{args.outdir}/enrichment_curve_{args.tag}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{args.outdir}/enrichment_curve_{args.tag}.svg', bbox_inches='tight')
    print('figure saved')


if __name__ == '__main__':
    main()
