#!/usr/bin/env python3
"""
S_cell weight sensitivity analysis for the FULL stage-3 set (200 compounds).

Reads vcdd stage-3 output (objective_scores.parquet, which stores the
gene_response / target_consistency / specificity components per compound),
recomputes S_cell over a 25-point weight grid, and reports:
  1. Spearman rho of each grid ranking vs the baseline (2:1:1) ranking
  2. Top-10 membership stability (Jaccard overlap with baseline top-10)
  3. Heatmap figure (PNG + SVG)

Usage:
    python scell_weight_sensitivity_full.py \
        --scores dnapkcs_run/stage3/objective_scores.parquet \
        --outdir sensitivity_out

The component columns are expected to be named gene_response,
target_consistency, specificity (adjust COL_* below if different).
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COL_GENE, COL_TARGET, COL_SPEC = 'gene_response', 'target_consistency', 'specificity'
GRID = [0.25, 0.5, 1, 2, 4]
BASELINE = (2.0, 1.0)  # (w_gene, w_target); w_spec fixed at 1


def scell(df, wg, wt, ws=1.0):
    wsum = wg + wt + ws
    return (df[COL_GENE] ** wg * df[COL_TARGET] ** wt * df[COL_SPEC] ** ws) ** (1.0 / wsum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help='stage3 objective_scores.parquet')
    ap.add_argument('--outdir', default='sensitivity_out')
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.scores)
    idcol = next((c for c in ['compound_id', 'compound', 'id', 'mol_id'] if c in df.columns), None)
    if idcol is None:
        idcol = 'compound_id'
        df[idcol] = [f'C{i:05d}' for i in range(len(df))]

    base = scell(df, *BASELINE)
    base_rank = base.rank(ascending=False)
    base_top10 = set(df.loc[base_rank <= 10, idcol])

    rows = []
    rho_mat = np.zeros((len(GRID), len(GRID)))
    for i, wg in enumerate(GRID):
        for j, wt in enumerate(GRID):
            s = scell(df, wg, wt)
            r = s.rank(ascending=False)
            rho = spearmanr(base_rank, r).statistic
            top10 = set(df.loc[r <= 10, idcol])
            jac = len(top10 & base_top10) / len(top10 | base_top10)
            rho_mat[i, j] = rho
            rows.append({'w_gene': wg, 'w_target': wt, 'spearman_rho': round(rho, 4),
                         'top10_jaccard': round(jac, 3),
                         'top10_retained': len(top10 & base_top10)})
    res = pd.DataFrame(rows)
    res.to_csv(f'{args.outdir}/scell_weight_sensitivity_full.csv', index=False)
    print(res.to_string(index=False))
    print(f'\nrho range: {res.spearman_rho.min()} - {res.spearman_rho.max()}')
    print(f'top-10 retention range: {res.top10_retained.min()} - {res.top10_retained.max()} of 10')

    import matplotlib
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    matplotlib.rcParams['svg.fonttype'] = 'none'
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=200)
    im = ax.imshow(rho_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            ax.text(j, i, f'{rho_mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(GRID))); ax.set_xticklabels(GRID)
    ax.set_yticks(range(len(GRID))); ax.set_yticklabels(GRID)
    ax.set_xlabel('w_target (target_consistency)')
    ax.set_ylabel('w_gene (gene_response)')
    ax.set_title('Spearman rho vs baseline weights (2:1:1)')
    bi, bj = GRID.index(BASELINE[0]), GRID.index(BASELINE[1])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor='black', lw=2))
    fig.colorbar(im, ax=ax, label='Spearman rho', shrink=0.85)
    plt.tight_layout()
    plt.savefig(f'{args.outdir}/fig_scell_weight_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{args.outdir}/fig_scell_weight_sensitivity.svg', bbox_inches='tight')
    print('figure saved to', args.outdir)


if __name__ == '__main__':
    main()
