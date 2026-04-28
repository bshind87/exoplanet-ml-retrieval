#!/usr/bin/env python
"""
Paper Experiment: Scaling Study
=================================
Trains RF, MLP, and 1D CNN at multiple training set sizes on a fixed
val/test split to produce a learning curve (R² vs N_train).

This answers the key reviewer question: does the CNN advantage persist
across data regimes, and where does each model saturate?

The PCA transform and SpectraScaler are fit on the FULL training split once
and reused at all scale points — only the number of training samples varies.
MoleculeScaler (target normalisation) is refit on each subsample for correctness.

Reads from  : inara_data/engineered/
Writes to   : results/paper_experiments/scaling_study.csv

Output format (tidy, one row per model × scale × molecule):
  n_train, model, molecule, R2, RMSE, MAE, wall_time_s

Usage:
  python experiments/run_scaling_study.py --profile local
  python experiments/run_scaling_study.py --profile hpc
  python experiments/run_scaling_study.py --profile hpc --models rf mlp cnn
  python experiments/run_scaling_study.py --profile hpc --scales 5000 25000 full
  python experiments/run_scaling_study.py --profile hpc --resume   # skip done rows
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import load_config, resolve_path
from src.data_utils import compute_metrics, MOLECULE_NAMES, MoleculeScaler
from src.baseline_model import BaselineModel
from src.deep_model import (
    CNN1D, SpectralDataset, Trainer, get_device, MOLECULE_HEAD_CONFIGS,
)
from src.mlp_model import MLP


# ── Default scale points (auto-clipped to available training size) ────────────
DEFAULT_SCALES = [1_000, 5_000, 10_000, 25_000, 50_000, 'full']


def _parse_scales(raw: list, n_train_full: int) -> list[int]:
    result = []
    for s in raw:
        if str(s).lower() == 'full':
            result.append(n_train_full)
        else:
            n = int(s)
            if n <= n_train_full:
                result.append(n)
            else:
                print(f'  Skipping scale {n:,}: exceeds available training size '
                      f'({n_train_full:,})')
    # Deduplicate and sort, always include n_train_full as last point
    result = sorted(set(result))
    if n_train_full not in result:
        result.append(n_train_full)
    return result


def _subsample(n: int, n_full: int, *arrays, seed: int = 42):
    """Return subsampled versions of all arrays to n rows."""
    if n >= n_full:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_full, size=n, replace=False)
    idx.sort()
    return tuple(a[idx] for a in arrays)


# ── Per-model training functions ──────────────────────────────────────────────

def train_rf(feat_tr, mol_tr, feat_te, mol_te) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    model = BaselineModel()
    model.fit(feat_tr, mol_tr, verbose=False)
    pred = model.predict(feat_te)
    df = compute_metrics(mol_te, pred)
    return df, time.time() - t0


def train_deep(model_cls, spec_tr, mol_tr, spec_val, mol_val, spec_te, mol_te,
               device, train_cfg, model_cfg) -> tuple[pd.DataFrame, float]:
    t0 = time.time()

    mol_scaler = MoleculeScaler()
    mol_tr_sc  = mol_scaler.fit_transform(mol_tr)
    mol_val_sc = mol_scaler.transform(mol_val)
    mol_te_sc  = mol_scaler.transform(mol_te)

    in_channels = model_cfg['in_channels']
    seq_len     = spec_tr.shape[2]
    epochs      = train_cfg['epochs']
    batch_size  = train_cfg['batch_size']
    lr          = train_cfg['lr']
    wd          = train_cfg['weight_decay']
    patience    = train_cfg['patience']

    nw = 0 if str(device) == 'mps' else 2
    train_ds = SpectralDataset(spec_tr, mol_tr_sc, augment=True,  noise_std=0.01)
    val_ds   = SpectralDataset(spec_val, mol_val_sc, augment=False)
    test_ds  = SpectralDataset(spec_te, mol_te_sc,  augment=False)

    t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=nw, pin_memory=(str(device) == 'cuda'))
    v_loader = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=nw)
    e_loader = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=nw)

    if model_cls is CNN1D:
        model = CNN1D(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=in_channels)
    else:
        model = MLP(head_configs=MOLECULE_HEAD_CONFIGS,
                    in_channels=in_channels, seq_len=seq_len)

    trainer   = Trainer(model, device, lr=lr, weight_decay=wd, patience=patience)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-6
    )

    for epoch in range(1, epochs + 1):
        trainer.train_epoch(t_loader, scheduler=scheduler)
        if trainer.check_early_stop(trainer.eval_epoch(v_loader)):
            break

    trainer.restore_best()
    pred = mol_scaler.inverse_transform(trainer.predict(e_loader))
    df = compute_metrics(mol_te, pred)
    return df, time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scaling study: R² vs N_train for RF, MLP, CNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',  type=str, default=str(ROOT / 'pipeline' / 'config.yaml'))
    parser.add_argument('--profile', type=str, default='local', choices=['local', 'hpc'])
    parser.add_argument('--models',  nargs='+', default=['rf', 'mlp', 'cnn'],
                        choices=['rf', 'mlp', 'cnn'],
                        help='Which models to run (any subset of rf mlp cnn)')
    parser.add_argument('--scales',  nargs='+', default=None,
                        help='Training sizes to evaluate, e.g. 5000 25000 full')
    parser.add_argument('--seed',    type=int, default=42,
                        help='Seed used for subsampling training sets')
    parser.add_argument('--resume',  action='store_true',
                        help='Skip (n_train, model) pairs already in output CSV')
    args = parser.parse_args()

    cfg       = load_config(args.config, args.profile)
    paths     = cfg['paths']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    results_dir    = resolve_path(paths['results_dir'],    args.profile)
    paper_exp_dir  = results_dir.parent / 'paper_experiments'
    paper_exp_dir.mkdir(parents=True, exist_ok=True)

    out_csv = paper_exp_dir / 'scaling_study.csv'

    print('=' * 65)
    print('  Paper Experiment: Scaling Study')
    print('=' * 65)
    print(f'  Profile       : {args.profile}')
    print(f'  Models        : {args.models}')
    print(f'  Output        : {out_csv}')
    print(f'  Resume        : {args.resume}')
    print()

    # ── Load engineered arrays ────────────────────────────────────────────────
    print('Loading engineered data ...')
    feat_train  = np.load(engineered_dir / 'feat_train.npy')
    spec_train  = np.load(engineered_dir / 'spectra_train.npy')
    mol_train   = np.load(engineered_dir / 'molecules_train.npy')
    spec_val    = np.load(engineered_dir / 'spectra_val.npy')
    mol_val     = np.load(engineered_dir / 'molecules_val.npy')
    spec_test   = np.load(engineered_dir / 'spectra_test.npy')
    mol_test    = np.load(engineered_dir / 'molecules_test.npy')
    feat_test   = np.load(engineered_dir / 'feat_test.npy')

    n_train_full = len(spec_train)
    print(f'  Full train: {n_train_full:,}   Val: {len(spec_val):,}   Test: {len(spec_test):,}')

    # ── Resolve scale points ──────────────────────────────────────────────────
    raw_scales = args.scales if args.scales else [str(s) for s in DEFAULT_SCALES]
    scale_points = _parse_scales(raw_scales, n_train_full)
    print(f'  Scale points  : {scale_points}')

    # ── Load already-done rows if resuming ────────────────────────────────────
    done_pairs: set[tuple[int, str]] = set()
    existing_rows: list[dict] = []
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        for _, row in prev.iterrows():
            done_pairs.add((int(row['n_train']), row['model']))
        # Keep molecule-level rows only (exclude MEAN summary rows)
        existing_rows = prev[prev['molecule'] != 'MEAN'].to_dict('records')
        print(f'  Resuming: {len(done_pairs)} (n_train, model) pairs already done')

    device = get_device()
    print(f'  Device        : {device}')
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    all_rows: list[dict] = list(existing_rows)

    for n in scale_points:
        # Subsample training split (val/test unchanged)
        feat_tr_n, spec_tr_n, mol_tr_n = _subsample(
            n, n_train_full, feat_train, spec_train, mol_train, seed=args.seed
        )

        for model_name in args.models:
            if (n, model_name) in done_pairs:
                print(f'  Skipping  n={n:>7,}  model={model_name}  (already done)')
                continue

            print(f'  Training  n={n:>7,}  model={model_name} ...', end='', flush=True)

            if model_name == 'rf':
                df, elapsed = train_rf(feat_tr_n, mol_tr_n, feat_test, mol_test)
            elif model_name == 'mlp':
                df, elapsed = train_deep(
                    MLP, spec_tr_n, mol_tr_n, spec_val, mol_val, spec_test, mol_test,
                    device, train_cfg, model_cfg
                )
            else:  # cnn
                df, elapsed = train_deep(
                    CNN1D, spec_tr_n, mol_tr_n, spec_val, mol_val, spec_test, mol_test,
                    device, train_cfg, model_cfg
                )

            mean_r2 = df[df['molecule'].isin(MOLECULE_NAMES)]['R2'].mean()
            print(f'  mean_R²={mean_r2:.4f}  ({elapsed:.0f}s)')

            for _, row in df[df['molecule'] != 'MEAN'].iterrows():
                all_rows.append({
                    'n_train':     n,
                    'model':       model_name,
                    'molecule':    row['molecule'],
                    'R2':          row['R2'],
                    'RMSE':        row['RMSE'],
                    'MAE':         row['MAE'],
                    'wall_time_s': elapsed,
                })

            # Write after every (n, model) pair so partial runs are usable
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    results = pd.read_csv(out_csv)
    summary = (results[results['molecule'] != 'MEAN']
               .groupby(['n_train', 'model'])['R2']
               .mean()
               .unstack('model')
               .round(4))
    print(f'\n{"="*65}')
    print('  Scaling Study — Mean R² (all 12 molecules)')
    print(f'{"="*65}')
    print(summary.to_string())
    print(f'\nFull results → {out_csv}')


if __name__ == '__main__':
    main()
