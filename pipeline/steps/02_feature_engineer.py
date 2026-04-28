#!/usr/bin/env python
"""
Step 2 — Feature Engineering
=============================
Loads raw processed arrays, creates a reproducible train/val/test split,
fits normalization and PCA on the training set only, then saves all
engineered artifacts to engineered_dir/.

This step MUST run once before Steps 3-5.  Re-running it with the same seed
produces identical splits, so baseline and deep model are always compared on
the same held-out test set.

Input  (processed_dir):
  spectra.npy    (N, 12, 101)  — raw CLIMA profiles
  molecules.npy  (N, 12)       — log10 surface molecular abundances

Output (engineered_dir):
  train_indices.npy             — integer indices into original N samples
  val_indices.npy
  test_indices.npy

  spectra_train.npy  (N_tr, 12, 101)  — Z-normalised, fit on train
  spectra_val.npy    (N_v,  12, 101)
  spectra_test.npy   (N_te, 12, 101)

  molecules_train.npy  (N_tr, 12)     — raw log10 (no scaling applied)
  molecules_val.npy    (N_v,  12)
  molecules_test.npy   (N_te, 12)

  feat_train.npy  (N_tr, pca_components)  — PCA-reduced features for RF
  feat_val.npy    (N_v,  pca_components)
  feat_test.npy   (N_te, pca_components)

  scaler.joblib          — fitted SpectraScaler (per-channel Z-score)
  pca.joblib             — fitted sklearn PCA
  feature_info.json      — shapes, split sizes, pca explained variance, config

Usage:
  python pipeline/steps/02_feature_engineer.py [--config pipeline/config.yaml] [--profile local|hpc]
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path
from src.data_utils import (
    MOLECULE_NAMES,
    SpectraScaler,
    extract_baseline_features,
    split_indices,
)


def main() -> None:
    parser = get_parser('Step 2: Feature Engineering — split + normalise + PCA')
    args = parser.parse_args()

    cfg   = load_config(args.config, args.profile)
    paths = cfg['paths']
    data  = cfg['data']
    model = cfg['model']

    processed_dir  = resolve_path(paths['processed_dir'],  args.profile)
    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    pca_components = model['pca_components']
    val_frac       = data['val_frac']
    test_frac      = data['test_frac']
    seed           = data['seed']

    engineered_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  INARA Pipeline — Step 2: Feature Engineering')
    print('=' * 60)
    print(f'  Profile        : {args.profile}')
    print(f'  Input          : {processed_dir}')
    print(f'  Output         : {engineered_dir}')
    print(f'  PCA components : {pca_components}')
    print(f'  Split          : train={1-val_frac-test_frac:.0%}  '
          f'val={val_frac:.0%}  test={test_frac:.0%}')
    print(f'  Seed           : {seed}')
    print()

    # ── 1. Load raw arrays ────────────────────────────────────────────────────
    t0 = time.time()
    print('Loading raw arrays ...')
    spectra   = np.load(processed_dir / 'spectra.npy')    # (N, 12, 101)
    molecules = np.load(processed_dir / 'molecules.npy')  # (N, 12)
    N = len(spectra)
    print(f'  spectra   : {spectra.shape}  dtype={spectra.dtype}')
    print(f'  molecules : {molecules.shape}  dtype={molecules.dtype}')
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # ── 2. Train / val / test split ───────────────────────────────────────────
    print('\nCreating train/val/test split ...')
    train_idx, val_idx, test_idx = split_indices(N, val_frac, test_frac, seed)
    print(f'  Train : {len(train_idx):>7,}  ({len(train_idx)/N:.1%})')
    print(f'  Val   : {len(val_idx):>7,}  ({len(val_idx)/N:.1%})')
    print(f'  Test  : {len(test_idx):>7,}  ({len(test_idx)/N:.1%})')

    np.save(engineered_dir / 'train_indices.npy', train_idx)
    np.save(engineered_dir / 'val_indices.npy',   val_idx)
    np.save(engineered_dir / 'test_indices.npy',  test_idx)
    print('  Saved split indices.')

    # ── 3. Spectral normalisation (Z-score per channel, fit on train) ─────────
    print('\nFitting spectral Z-normalisation on training set ...')
    t1 = time.time()
    scaler = SpectraScaler()
    spec_train = scaler.fit_transform(spectra[train_idx])
    spec_val   = scaler.transform(spectra[val_idx])
    spec_test  = scaler.transform(spectra[test_idx])
    print(f'  Done in {time.time()-t1:.1f}s')
    print(f'  Train spectra mean={spec_train.mean():.4f}  std={spec_train.std():.4f}')

    np.save(engineered_dir / 'spectra_train.npy', spec_train)
    np.save(engineered_dir / 'spectra_val.npy',   spec_val)
    np.save(engineered_dir / 'spectra_test.npy',  spec_test)
    joblib.dump(scaler, engineered_dir / 'scaler.joblib')
    print('  Saved normalised spectra + scaler.joblib')

    # ── 4. Molecule targets (log10, no scaling — raw values used for metrics) ─
    mol_train = molecules[train_idx]
    mol_val   = molecules[val_idx]
    mol_test  = molecules[test_idx]

    np.save(engineered_dir / 'molecules_train.npy', mol_train)
    np.save(engineered_dir / 'molecules_val.npy',   mol_val)
    np.save(engineered_dir / 'molecules_test.npy',  mol_test)
    print('\nSaved molecule targets (log10, unscaled).')

    # ── 5. PCA features for Random Forest baseline ────────────────────────────
    print(f'\nFitting PCA (n_components={pca_components}) on training spectra ...')
    t2 = time.time()
    feat_train, pca = extract_baseline_features(
        spec_train, pca=None, n_components=pca_components
    )
    feat_val,  _    = extract_baseline_features(spec_val,  pca=pca)
    feat_test, _    = extract_baseline_features(spec_test, pca=pca)

    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f'  Done in {time.time()-t2:.1f}s')
    print(f'  PCA {pca_components} components explain '
          f'{explained_var:.1%} of variance')
    print(f'  feat_train : {feat_train.shape}')

    np.save(engineered_dir / 'feat_train.npy', feat_train)
    np.save(engineered_dir / 'feat_val.npy',   feat_val)
    np.save(engineered_dir / 'feat_test.npy',  feat_test)
    joblib.dump(pca, engineered_dir / 'pca.joblib')
    print('  Saved PCA features + pca.joblib')

    # ── 6. Metadata / provenance ──────────────────────────────────────────────
    info = {
        'n_total':          N,
        'n_train':          len(train_idx),
        'n_val':            len(val_idx),
        'n_test':           len(test_idx),
        'val_frac':         val_frac,
        'test_frac':        test_frac,
        'seed':             seed,
        'pca_components':   pca_components,
        'pca_explained_var': explained_var,
        'spectra_shape':    list(spectra.shape),
        'molecules_shape':  list(molecules.shape),
        'molecule_names':   MOLECULE_NAMES,
        'profile':          args.profile,
        'processed_dir':    str(processed_dir),
        'engineered_dir':   str(engineered_dir),
    }
    with open(engineered_dir / 'feature_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    # ── 7. Data quality checks ────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  Data Quality Report')
    print('─' * 60)

    def _check(name, arr):
        n_nan  = int(np.isnan(arr).sum())
        n_inf  = int(np.isinf(arr).sum())
        n_zero = int((arr == 0).sum())
        status = 'OK' if (n_nan == 0 and n_inf == 0) else 'FAIL'
        print(f'  [{status}] {name:<30s}  shape={str(arr.shape):<18s}'
              f'  NaN={n_nan:>6,}  Inf={n_inf:>6,}  zeros={n_zero:>8,}'
              f'  min={arr.min():>10.4f}  max={arr.max():>10.4f}')
        return n_nan == 0 and n_inf == 0

    checks_ok = True
    checks_ok &= _check('spectra_train',   spec_train)
    checks_ok &= _check('spectra_val',     spec_val)
    checks_ok &= _check('spectra_test',    spec_test)
    checks_ok &= _check('molecules_train', mol_train)
    checks_ok &= _check('molecules_val',   mol_val)
    checks_ok &= _check('molecules_test',  mol_test)
    checks_ok &= _check('feat_train',      feat_train)
    checks_ok &= _check('feat_val',        feat_val)
    checks_ok &= _check('feat_test',       feat_test)

    print('─' * 60)
    print(f'  Target std per molecule (train) — constant targets will show std≈0:')
    for i, mol in enumerate(MOLECULE_NAMES):
        std_val = mol_train[:, i].std()
        flag = '  <-- CONSTANT, skip regression' if std_val < 1e-6 else ''
        print(f'    {mol:<6s}  std={std_val:.4f}{flag}')
    print('─' * 60)

    if checks_ok:
        print('  All quality checks passed.')
    else:
        print('  WARNING: NaN or Inf found in engineered arrays — investigate before training.')

    total = time.time() - t0
    print(f'\nFeature engineering complete in {total:.1f}s')
    print(f'All artifacts saved to: {engineered_dir}')


if __name__ == '__main__':
    main()
