#!/usr/bin/env python
"""
Step 5 — Unified Evaluation & Comparison
==========================================
Loads saved test predictions from Steps 3 and 4, re-computes metrics on
the shared test set, and produces a side-by-side comparison report.

Requires results_dir to contain at least one of:
  baseline_test_pred.npy   (from Step 3)
  deep_test_pred.npy       (from Step 4)
  test_targets.npy         (written by whichever model ran first)

Output (results_dir):
  baseline_test_metrics.csv    — re-computed (consistent with saved predictions)
  deep_test_metrics.csv        — re-computed
  model_comparison.csv         — side-by-side Baseline vs DeepModel per molecule

Usage:
  python pipeline/steps/05_evaluate.py [--profile local|hpc]
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path
from src.data_utils import compute_metrics, print_metrics, MOLECULE_NAMES


# Summary-only exclusions for the comparison report.
# H2O is nearly constant in log10 space on this split, and NH3 is constant at -40,
# so both can dominate mean R² without reflecting typical model behavior.
EXCLUDED_FROM_COMPARISON = {'H2O', 'NH3'}


def _load_if_exists(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(path)
    return None


def main() -> None:
    parser = get_parser('Step 5: Unified evaluation and model comparison')
    args = parser.parse_args()

    cfg   = load_config(args.config, args.profile)
    paths = cfg['paths']

    results_dir = resolve_path(paths['results_dir'], args.profile)

    print('=' * 60)
    print('  INARA Pipeline — Step 5: Evaluation & Comparison')
    print('=' * 60)
    print(f'  Profile     : {args.profile}')
    print(f'  Results dir : {results_dir}')
    print()

    # ── Load test targets ─────────────────────────────────────────────────────
    targets_path = results_dir / 'test_targets.npy'
    if not targets_path.exists():
        print(f'ERROR: {targets_path} not found.')
        print('       Run Step 3 or Step 4 first to generate test_targets.npy')
        sys.exit(1)

    mol_test = np.load(targets_path)
    print(f'Test targets  : {mol_test.shape}  ({len(mol_test):,} samples)')

    # ── Load predictions ──────────────────────────────────────────────────────
    bl_pred   = _load_if_exists(results_dir / 'baseline_test_pred.npy')
    deep_pred = _load_if_exists(results_dir / 'deep_test_pred.npy')

    if bl_pred is None and deep_pred is None:
        print('\nERROR: No prediction files found in results_dir.')
        print('       Run Step 3 and/or Step 4 first.')
        sys.exit(1)

    results = {}

    # ── Baseline metrics ──────────────────────────────────────────────────────
    if bl_pred is not None:
        print(f'\nBaseline predictions : {bl_pred.shape}')
        bl_df = compute_metrics(mol_test, bl_pred)
        print_metrics(bl_df, title='Baseline (Random Forest) — Test Metrics')
        bl_df.to_csv(results_dir / 'baseline_test_metrics.csv', index=False)
        results['Baseline_R2'] = bl_df.set_index('molecule')['R2']
    else:
        print('\nBaseline predictions not found — skipping.')

    # ── Deep model metrics ────────────────────────────────────────────────────
    if deep_pred is not None:
        print(f'\nDeep model predictions : {deep_pred.shape}')
        deep_df = compute_metrics(mol_test, deep_pred)
        print_metrics(deep_df, title='1D CNN — Test Metrics')
        deep_df.to_csv(results_dir / 'deep_test_metrics.csv', index=False)
        results['DeepModel_R2'] = deep_df.set_index('molecule')['R2']
    else:
        print('\nDeep model predictions not found — skipping.')

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if len(results) == 2:
        compare = pd.DataFrame({
            'molecule':     MOLECULE_NAMES,
            'Baseline_R2':  [results['Baseline_R2'].get(m, float('nan'))
                             for m in MOLECULE_NAMES],
            'DeepModel_R2': [results['DeepModel_R2'].get(m, float('nan'))
                             for m in MOLECULE_NAMES],
        })
        compare['Delta_R2'] = compare['DeepModel_R2'] - compare['Baseline_R2']

        filtered_compare = compare[~compare['molecule'].isin(EXCLUDED_FROM_COMPARISON)].copy()

        # Append mean row
        filtered_compare = pd.concat([
            filtered_compare,
            pd.DataFrame([{
                'molecule':     'MEAN',
                'Baseline_R2':  filtered_compare['Baseline_R2'].mean(),
                'DeepModel_R2': filtered_compare['DeepModel_R2'].mean(),
                'Delta_R2':     filtered_compare['Delta_R2'].mean(),
            }])
        ], ignore_index=True)

        excluded_text = ', '.join(sorted(EXCLUDED_FROM_COMPARISON))

        print(f'\n{"="*65}')
        print(f'  Model Comparison — Test Set (R², excluding {excluded_text})')
        print(f'{"="*65}')
        print(filtered_compare.to_string(index=False, float_format='%.4f'))
        print(f'{"="*65}')

        compare.to_csv(results_dir / 'model_comparison_full.csv', index=False)
        filtered_compare.to_csv(results_dir / 'model_comparison.csv', index=False)
        print(f'\nSaved comparison → {results_dir}/model_comparison.csv')
        print('  Justification: H2O is nearly constant in log10 space on this split, '
              'and NH3 is constant at -40.0, so they can distort the mean R².')

        # Quick win/loss summary
        wins = (filtered_compare[filtered_compare['molecule'] != 'MEAN']['Delta_R2'] > 0).sum()
        total = len(filtered_compare[filtered_compare['molecule'] != 'MEAN'])
        print(f'\n  Deep model outperforms baseline on {wins}/{total} molecules '
              f'after exclusions')

    print(f'\nAll results saved to: {results_dir}')


if __name__ == '__main__':
    main()
