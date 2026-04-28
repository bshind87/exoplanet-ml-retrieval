#!/usr/bin/env python
"""
Step 3 — Train Baseline (Random Forest)
========================================
Trains one RandomForestRegressor per molecule using PCA-reduced features
produced by Step 2.  The training set is ALWAYS capped at max_train_samples
(default 10 000) regardless of dataset size — this is intentional to keep the
RF a true "baseline" that is fast and resource-light.

Val/test splits are unchanged so metrics are directly comparable with Step 4.

Input  (engineered_dir):
  feat_train.npy, feat_val.npy, feat_test.npy     — PCA features
  molecules_train.npy, molecules_val.npy, molecules_test.npy

Output (results_dir / models_dir):
  baseline_val_metrics.csv
  baseline_test_metrics.csv
  baseline_test_pred.npy
  test_targets.npy
  baseline_rf.joblib            (if --save)

Usage:
  python pipeline/steps/03_train_baseline.py [--profile local|hpc] [--save]
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path
from src.baseline_model import BaselineModel
from src.data_utils import print_metrics


def main() -> None:
    parser = get_parser('Step 3: Train Random Forest baseline (capped at 10k samples)')
    parser.add_argument(
        '--save', action='store_true',
        help='Save trained RF model to models_dir/baseline_rf.joblib',
    )
    parser.add_argument(
        '--max-train-samples', type=int, default=None,
        help='Override baseline.max_train_samples from config',
    )
    args = parser.parse_args()

    cfg      = load_config(args.config, args.profile)
    paths    = cfg['paths']
    baseline = cfg['baseline']

    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    results_dir    = resolve_path(paths['results_dir'],    args.profile)
    models_dir     = resolve_path(paths['models_dir'],     args.profile)
    max_samples    = args.max_train_samples or baseline['max_train_samples']
    seed           = baseline['seed']

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  INARA Pipeline — Step 3: Baseline (Random Forest)')
    print('=' * 60)
    print(f'  Profile          : {args.profile}')
    print(f'  Engineered dir   : {engineered_dir}')
    print(f'  Results dir      : {results_dir}')
    print(f'  Max train samples: {max_samples:,}  (hard cap for baseline)')
    print()

    # ── 1. Load pre-engineered features ──────────────────────────────────────
    print('Loading engineered features ...')
    t0 = time.time()
    feat_train    = np.load(engineered_dir / 'feat_train.npy')
    feat_val      = np.load(engineered_dir / 'feat_val.npy')
    feat_test     = np.load(engineered_dir / 'feat_test.npy')
    mol_train     = np.load(engineered_dir / 'molecules_train.npy')
    mol_val       = np.load(engineered_dir / 'molecules_val.npy')
    mol_test      = np.load(engineered_dir / 'molecules_test.npy')
    print(f'  feat_train : {feat_train.shape}    mol_train : {mol_train.shape}')
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # ── 2. Cap training set ───────────────────────────────────────────────────
    n_train_full = len(feat_train)
    if max_samples < n_train_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_train_full, size=max_samples, replace=False)
        idx.sort()
        feat_train = feat_train[idx]
        mol_train  = mol_train[idx]
        print(f'\nSubsampled training set: {n_train_full:,} → {max_samples:,} '
              f'(val/test unchanged)')
    else:
        print(f'\nUsing full training set ({n_train_full:,} samples)')

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print(f'\nTraining per-molecule Random Forests ...')
    t1 = time.time()
    model = BaselineModel()
    model.fit(feat_train, mol_train, X_val=feat_val, y_val=mol_val, verbose=True)
    print(f'  Training completed in {time.time()-t1:.1f}s')

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    val_df,  _         = model.evaluate(feat_val,  mol_val,  split_name='Validation')
    test_df, test_pred = model.evaluate(feat_test, mol_test, split_name='Test')

    # ── 5. Save results ───────────────────────────────────────────────────────
    val_df.to_csv(results_dir / 'baseline_val_metrics.csv',  index=False)
    test_df.to_csv(results_dir / 'baseline_test_metrics.csv', index=False)
    np.save(results_dir / 'baseline_test_pred.npy', test_pred)
    np.save(results_dir / 'test_targets.npy', mol_test)
    print(f'\nSaved metrics → {results_dir}')

    if args.save:
        model.save(str(models_dir / 'baseline_rf.joblib'))
        print(f'Saved model  → {models_dir}/baseline_rf.joblib')

    # ── 6. Summary ────────────────────────────────────────────────────────────
    mean_r2 = test_df[test_df['molecule'] != 'MEAN']['R2'].mean()
    print(f'\n{"="*40}')
    print(f'  Baseline Test  Mean R² = {mean_r2:.4f}')
    print(f'  Trained on {len(feat_train):,} samples (max_train_samples={max_samples:,})')
    print(f'{"="*40}')


if __name__ == '__main__':
    main()
