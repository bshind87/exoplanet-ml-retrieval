#!/usr/bin/env python
"""
Train and evaluate the Random Forest baseline model on the INARA exoplanet dataset.

Usage:
    python run_baseline.py [--pca-components 300] [--save]

Outputs:
    - Per-molecule R², RMSE, MAE on validation and test sets
    - results/baseline_metrics.csv
    - models/baseline_rf.joblib  (if --save)
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_utils import prepare_data, MOLECULE_NAMES, print_metrics
from src.baseline_model import BaselineModel


def main(pca_components=300, save_model=False, data_dir=None, max_train_samples=None):
    out_tag = Path(data_dir).name if data_dir else 'default'
    results_dir = Path('results') / out_tag
    models_dir  = Path('models')  / out_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print('Loading and preparing data ...')
    t0 = time.time()
    data = prepare_data(val_frac=0.15, test_frac=0.15,
                        random_state=42, pca_components=pca_components,
                        data_dir=data_dir, max_train_samples=max_train_samples)
    print(f'  Train: {len(data["idx_train"])}  Val: {len(data["idx_val"])}  '
          f'Test: {len(data["idx_test"])}  PCA components: {pca_components}')
    print(f'  Data loaded in {time.time()-t0:.1f}s')

    feat_train = data['feat_train']
    feat_val   = data['feat_val']
    feat_test  = data['feat_test']
    mol_train  = data['molecules_train']
    mol_val    = data['molecules_val']
    mol_test   = data['molecules_test']

    # ------------------------------------------------------------------
    # 2. Train per-molecule Random Forests
    # ------------------------------------------------------------------
    print(f'\nTraining per-molecule Random Forests ...')
    t1 = time.time()
    model = BaselineModel()
    model.fit(feat_train, mol_train, X_val=feat_val, y_val=mol_val, verbose=True)
    print(f'  Training completed in {time.time()-t1:.1f}s')

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    val_df,  val_pred  = model.evaluate(feat_val,  mol_val,  split_name='Validation')
    test_df, test_pred = model.evaluate(feat_test, mol_test, split_name='Test')

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    val_df.to_csv(results_dir / 'baseline_val_metrics.csv',  index=False)
    test_df.to_csv(results_dir / 'baseline_test_metrics.csv', index=False)
    np.save(results_dir / 'baseline_test_pred.npy', test_pred)
    np.save(results_dir / 'test_targets.npy', mol_test)
    print(f'\nSaved metrics to results/')

    if save_model:
        model.save(str(models_dir / 'baseline_rf.joblib'))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    mean_r2 = test_df[test_df['molecule'] != 'MEAN']['R2'].mean()
    print(f'\n{"="*40}')
    print(f'  Baseline Test  Mean R² = {mean_r2:.4f}')
    print(f'{"="*40}')
    return test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline RF model')
    parser.add_argument('--pca-components', type=int, default=300,
                        help='Number of PCA components (default: 300)')
    parser.add_argument('--save', action='store_true',
                        help='Save trained models to disk')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to processed data directory '
                             '(default: inara_data/processed)')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Cap training set size (val/test unchanged for fair comparison)')
    args = parser.parse_args()
    main(pca_components=args.pca_components, save_model=args.save,
         data_dir=args.data_dir, max_train_samples=args.max_train_samples)
