#!/usr/bin/env python
"""
Generate all paper figures from experiment CSVs.

Outputs (paper/figures/):
  fig1_architecture.pdf     -- CNN architecture schematic (text diagram)
  fig2_scaling_curve.pdf    -- R² vs N_train for RF, MLP, CNN
  fig3_h2o_scaling.pdf      -- H2O R² vs N_train (the data-requirement finding)
  fig4_multiseed_bar.pdf    -- Per-molecule R² with ±1σ error bars (CNN vs RF)
  fig5_scatter_grid.pdf     -- Predicted vs true scatter for CNN at 86.8K

Run from project root:
  python paper/generate_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / 'results' / 'paper_experiments'
ENG_DIR = ROOT / 'inara_data' / 'engineered'
OUT_DIR = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    10,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS  = {'cnn': '#2166ac', 'mlp': '#d6604d', 'rf': '#4dac26'}
MARKERS = {'cnn': 'o',       'mlp': 's',       'rf': '^'}
LABELS  = {'cnn': '1D CNN',  'mlp': 'MLP',     'rf': 'Random Forest'}
DEGENERATE = {'NH3', 'H2O'}
MOLECULE_NAMES = ['H2O','CO2','O2','O3','CH4','N2','N2O','CO','H2','H2S','SO2','NH3']
ACTIVE = [m for m in MOLECULE_NAMES if m not in DEGENERATE]

# ── Figure 2: Scaling curve ───────────────────────────────────────────────────
def fig_scaling_curve():
    df = pd.read_csv(RES_DIR / 'scaling_study.csv')
    active_mean = (df[~df['molecule'].isin(DEGENERATE)]
                   .groupby(['n_train', 'model'])['R2'].mean().reset_index())

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for model in ['rf', 'mlp', 'cnn']:
        sub = active_mean[active_mean['model'] == model].sort_values('n_train')
        ax.plot(sub['n_train'], sub['R2'],
                color=COLORS[model], marker=MARKERS[model],
                markersize=6, linewidth=1.8, label=LABELS[model])

    ax.set_xscale('log')
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('Mean $R^2$ (10 active molecules)')
    ax.set_ylim(0.60, 1.01)
    ax.set_xlim(800, 120_000)
    ax.axhline(0.999, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(900, 0.9995, '$R^2 = 0.999$', fontsize=8, color='gray', va='bottom')
    ax.legend(frameon=False)
    ax.set_title('Learning curves: mean $R^2$ vs training set size')

    xticks = [1_000, 5_000, 10_000, 25_000, 50_000, 86_800]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['1K', '5K', '10K', '25K', '50K', '86.8K'])

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_scaling_curve.pdf')
    fig.savefig(OUT_DIR / 'fig2_scaling_curve.png')
    plt.close(fig)
    print('Saved fig2_scaling_curve')


# ── Figure 3: H2O scaling ─────────────────────────────────────────────────────
def fig_h2o_scaling():
    df = pd.read_csv(RES_DIR / 'scaling_study.csv')
    h2o = df[df['molecule'] == 'H2O']

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for model in ['rf', 'mlp', 'cnn']:
        sub = h2o[h2o['model'] == model].sort_values('n_train')
        ax.plot(sub['n_train'], sub['R2'],
                color=COLORS[model], marker=MARKERS[model],
                markersize=6, linewidth=1.8, label=LABELS[model])

    ax.set_xscale('log')
    ax.set_xlabel('Training set size $N$')
    ax.set_ylabel('$R^2$ — H$_2$O')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.99, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(900, 0.993, '$R^2 = 0.99$', fontsize=8, color='gray', va='bottom')
    ax.axvline(50_000, color='#b2182b', linestyle=':', linewidth=1.0, alpha=0.8)
    ax.text(52_000, 0.05, '50K threshold', fontsize=8, color='#b2182b', rotation=90, va='bottom')
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('H$_2$O is learnable — requires $\\geq$50K samples')

    xticks = [1_000, 5_000, 10_000, 25_000, 50_000, 86_800]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['1K', '5K', '10K', '25K', '50K', '86.8K'])

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_h2o_scaling.pdf')
    fig.savefig(OUT_DIR / 'fig3_h2o_scaling.png')
    plt.close(fig)
    print('Saved fig3_h2o_scaling')


# ── Figure 4: Multi-seed bar chart with error bars ────────────────────────────
def fig_multiseed_bar():
    df = pd.read_csv(RES_DIR / 'multiseed_summary.csv')
    active = df[~df['molecule'].isin(DEGENERATE)]

    cnn = active[active['model'] == 'cnn'].set_index('molecule')
    rf  = active[active['model'] == 'rf'].set_index('molecule')

    mols = ACTIVE
    x    = np.arange(len(mols))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    ax.bar(x - w/2, [cnn.loc[m, 'R2_mean'] for m in mols],
           w, yerr=[cnn.loc[m, 'R2_std'] for m in mols],
           color=COLORS['cnn'], alpha=0.85, label=LABELS['cnn'],
           error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

    ax.bar(x + w/2, [rf.loc[m, 'R2_mean'] for m in mols],
           w, yerr=[rf.loc[m, 'R2_std'] for m in mols],
           color=COLORS['rf'], alpha=0.85, label=f'{LABELS["rf"]} (10K cap)',
           error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(mols, rotation=30, ha='right')
    ax.set_ylabel('Test $R^2$ (mean $\\pm$ 1 s.d., 5 seeds)')
    ax.set_ylim(0.55, 1.04)
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(frameon=False)
    ax.set_title('Per-molecule $R^2$: CNN vs Random Forest (5 random seeds)')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_multiseed_bar.pdf')
    fig.savefig(OUT_DIR / 'fig4_multiseed_bar.png')
    plt.close(fig)
    print('Saved fig4_multiseed_bar')


# ── Figure 5: CNN predicted vs true scatter grid ─────────────────────────────
def fig_scatter_grid():
    pred_path = ROOT / 'results' / 'paper_experiments' / 'scaling_study.csv'

    # Use scaling study CNN predictions at 86.8K — re-generate from stored arrays
    # Fall back to existing deep_test_pred.npy from main pipeline if available
    pred_file  = ROOT / 'results' / 'deep_test_pred.npy'
    tgt_file   = ROOT / 'results' / 'test_targets.npy'

    if not pred_file.exists() or not tgt_file.exists():
        print('Skipping fig5: deep_test_pred.npy or test_targets.npy not found')
        return

    pred = np.load(pred_file)   # (N, 12)
    tgt  = np.load(tgt_file)    # (N, 12)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    axes = axes.flatten()

    for i, mol in enumerate(ACTIVE):
        mi   = MOLECULE_NAMES.index(mol)
        ax   = axes[i]
        y, p = tgt[:, mi], pred[:, mi]

        lims = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.scatter(y, p, s=1, alpha=0.15, color=COLORS['cnn'], rasterized=True)
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)

        from sklearn.metrics import r2_score
        r2 = r2_score(y, p)
        ax.set_title(f'{mol}  $R^2$={r2:.4f}', fontsize=8)
        ax.set_xlabel('True $\\log_{{10}}$', fontsize=7)
        ax.set_ylabel('Pred $\\log_{{10}}$', fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplot
    axes[-1].set_visible(False) if len(ACTIVE) < len(axes) else None

    fig.suptitle('CNN predicted vs true log$_{10}$ abundance (test set)', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_scatter_grid.pdf', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig5_scatter_grid.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig5_scatter_grid')


# ── Figure 6: CNN vs MLP delta R² (architecture ablation) ────────────────────
def fig_cnn_mlp_delta():
    df = pd.read_csv(RES_DIR / 'scaling_study.csv')
    full = df[(df['n_train'] == df['n_train'].max()) & (~df['molecule'].isin(DEGENERATE))]

    cnn_r2 = full[full['model'] == 'cnn'].set_index('molecule')['R2']
    mlp_r2 = full[full['model'] == 'mlp'].set_index('molecule')['R2']
    delta   = (cnn_r2 - mlp_r2).reindex(ACTIVE)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ['#2166ac' if v >= 0 else '#d6604d' for v in delta]
    ax.bar(range(len(ACTIVE)), delta * 1000, color=colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(ACTIVE)))
    ax.set_xticklabels(ACTIVE, rotation=30, ha='right')
    ax.set_ylabel('$\\Delta R^2$ (CNN $-$ MLP) $\\times 10^3$')
    ax.set_title('Architecture ablation: CNN advantage over MLP at $N=86{,}800$')

    mean_d = delta.mean()
    ax.axhline(mean_d * 1000, color='gray', linestyle='--', linewidth=0.9)
    ax.text(len(ACTIVE) - 0.5, mean_d * 1000 + 0.03,
            f'mean = {mean_d*1000:.2f}$\\times 10^{{-3}}$',
            ha='right', fontsize=8, color='gray')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig6_cnn_mlp_delta.pdf')
    fig.savefig(OUT_DIR / 'fig6_cnn_mlp_delta.png')
    plt.close(fig)
    print('Saved fig6_cnn_mlp_delta')


if __name__ == '__main__':
    print('Generating paper figures ...\n')
    fig_scaling_curve()
    fig_h2o_scaling()
    fig_multiseed_bar()
    fig_scatter_grid()
    fig_cnn_mlp_delta()
    print(f'\nAll figures saved to {OUT_DIR}')
