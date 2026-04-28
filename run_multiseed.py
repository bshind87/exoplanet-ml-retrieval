#!/usr/bin/env python
"""
Paper Experiment: Multi-Seed Evaluation
=========================================
Trains RF and 1D CNN (and optionally MLP) across multiple random seeds to
quantify metric variance and report mean ± std per molecule.

This addresses the TMLR reviewer requirement for statistical significance:
a single training run does not reveal whether R² differences are meaningful.

Seed controls:
  RF  — random_state for tree building + training-set subsampling (if capped)
  CNN — torch / numpy / random seeds for weight init, shuffle, augmentation
  MLP — same as CNN

Reads from  : inara_data/engineered/
Writes to   : results/paper_experiments/multiseed_results.csv   (per-seed rows)
              results/paper_experiments/multiseed_summary.csv   (mean ± std)

Usage:
  python experiments/run_multiseed.py --profile local
  python experiments/run_multiseed.py --profile hpc
  python experiments/run_multiseed.py --profile hpc --models rf cnn mlp --seeds 42 123 456 789 1337
  python experiments/run_multiseed.py --profile hpc --resume
"""

import argparse
import random
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
from src.baseline_model import BaselineModel, MOLECULE_RF_PARAMS
from src.deep_model import (
    CNN1D, SpectralDataset, Trainer, get_device, MOLECULE_HEAD_CONFIGS,
)
from src.mlp_model import MLP


DEFAULT_SEEDS  = [42, 123, 456, 789, 1337]
RF_TRAIN_CAP   = 10_000   # mirror the main pipeline baseline


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_rf(feat_tr, mol_tr, feat_te, mol_te, seed: int, n_cap: int) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    n_full = len(feat_tr)
    if n_cap < n_full:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_full, size=n_cap, replace=False)
        feat_tr = feat_tr[idx]
        mol_tr  = mol_tr[idx]

    # Build seed-specific params for each molecule RF
    mol_params = {
        mol: {**params, 'random_state': seed}
        for mol, params in MOLECULE_RF_PARAMS.items()
    }
    model = BaselineModel(mol_params=mol_params)
    model.fit(feat_tr, mol_tr, verbose=False)
    pred = model.predict(feat_te)
    df   = compute_metrics(mol_te, pred)
    return df, time.time() - t0


def run_deep(model_cls, spec_tr, mol_tr, spec_val, mol_val,
             spec_te, mol_te, device, train_cfg, model_cfg,
             seed: int) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    _set_seed(seed)

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
    g  = torch.Generator()
    g.manual_seed(seed)

    train_ds = SpectralDataset(spec_tr,  mol_tr_sc,  augment=True,  noise_std=0.01)
    val_ds   = SpectralDataset(spec_val, mol_val_sc, augment=False)
    test_ds  = SpectralDataset(spec_te,  mol_te_sc,  augment=False)

    t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=nw, pin_memory=(str(device) == 'cuda'),
                          generator=g)
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
        description='Multi-seed evaluation for statistical significance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',  type=str, default=str(ROOT / 'pipeline' / 'config.yaml'))
    parser.add_argument('--profile', type=str, default='local', choices=['local', 'hpc'])
    parser.add_argument('--models',  nargs='+', default=['rf', 'cnn'],
                        choices=['rf', 'cnn', 'mlp'],
                        help='Models to evaluate across seeds')
    parser.add_argument('--seeds',   nargs='+', type=int, default=DEFAULT_SEEDS,
                        help='Random seeds')
    parser.add_argument('--rf-cap',  type=int,  default=RF_TRAIN_CAP,
                        help='Training set cap for RF (matches main pipeline)')
    parser.add_argument('--resume',  action='store_true',
                        help='Skip (seed, model) pairs already in output CSV')
    args = parser.parse_args()

    cfg       = load_config(args.config, args.profile)
    paths     = cfg['paths']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    results_dir    = resolve_path(paths['results_dir'],    args.profile)
    paper_exp_dir  = results_dir.parent / 'paper_experiments'
    paper_exp_dir.mkdir(parents=True, exist_ok=True)

    results_csv = paper_exp_dir / 'multiseed_results.csv'
    summary_csv = paper_exp_dir / 'multiseed_summary.csv'

    print('=' * 65)
    print('  Paper Experiment: Multi-Seed Evaluation')
    print('=' * 65)
    print(f'  Profile : {args.profile}')
    print(f'  Models  : {args.models}')
    print(f'  Seeds   : {args.seeds}')
    print(f'  RF cap  : {args.rf_cap:,}')
    print(f'  Output  : {results_csv}')
    print()

    # ── Load engineered data ──────────────────────────────────────────────────
    print('Loading engineered data ...')
    feat_train = np.load(engineered_dir / 'feat_train.npy')
    spec_train = np.load(engineered_dir / 'spectra_train.npy')
    mol_train  = np.load(engineered_dir / 'molecules_train.npy')
    spec_val   = np.load(engineered_dir / 'spectra_val.npy')
    mol_val    = np.load(engineered_dir / 'molecules_val.npy')
    spec_test  = np.load(engineered_dir / 'spectra_test.npy')
    mol_test   = np.load(engineered_dir / 'molecules_test.npy')
    feat_test  = np.load(engineered_dir / 'feat_test.npy')
    print(f'  Train: {spec_train.shape}   Val: {spec_val.shape}   Test: {spec_test.shape}')

    device = get_device()
    print(f'  Device: {device}\n')

    # ── Load already-done pairs if resuming ───────────────────────────────────
    done_pairs: set[tuple[int, str]] = set()
    all_rows: list[dict] = []
    if args.resume and results_csv.exists():
        prev = pd.read_csv(results_csv)
        for _, row in prev.iterrows():
            done_pairs.add((int(row['seed']), row['model']))
        all_rows = prev.to_dict('records')
        print(f'  Resuming: {len(done_pairs)} (seed, model) pairs already done')

    # ── Run experiments ───────────────────────────────────────────────────────
    for seed in args.seeds:
        for model_name in args.models:
            if (seed, model_name) in done_pairs:
                print(f'  Skipping  seed={seed}  model={model_name}  (already done)')
                continue

            print(f'  Training  seed={seed}  model={model_name} ...', end='', flush=True)

            if model_name == 'rf':
                df, elapsed = run_rf(
                    feat_train, mol_train, feat_test, mol_test,
                    seed=seed, n_cap=args.rf_cap
                )
            elif model_name == 'mlp':
                df, elapsed = run_deep(
                    MLP, spec_train, mol_train, spec_val, mol_val,
                    spec_test, mol_test, device, train_cfg, model_cfg, seed=seed
                )
            else:  # cnn
                df, elapsed = run_deep(
                    CNN1D, spec_train, mol_train, spec_val, mol_val,
                    spec_test, mol_test, device, train_cfg, model_cfg, seed=seed
                )

            mean_r2 = df[df['molecule'].isin(MOLECULE_NAMES)]['R2'].mean()
            print(f'  mean_R²={mean_r2:.4f}  ({elapsed:.0f}s)')

            for _, row in df[df['molecule'] != 'MEAN'].iterrows():
                all_rows.append({
                    'seed':     seed,
                    'model':    model_name,
                    'molecule': row['molecule'],
                    'R2':       row['R2'],
                    'RMSE':     row['RMSE'],
                    'MAE':      row['MAE'],
                })

            pd.DataFrame(all_rows).to_csv(results_csv, index=False)

    # ── Build summary (mean ± std per model × molecule) ───────────────────────
    results = pd.read_csv(results_csv)
    mol_only = results[results['molecule'] != 'MEAN']

    summary_rows = []
    for (model_name, mol), grp in mol_only.groupby(['model', 'molecule']):
        summary_rows.append({
            'model':     model_name,
            'molecule':  mol,
            'R2_mean':   grp['R2'].mean(),
            'R2_std':    grp['R2'].std(ddof=1),
            'RMSE_mean': grp['RMSE'].mean(),
            'RMSE_std':  grp['RMSE'].std(ddof=1),
            'n_seeds':   len(grp),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_csv, index=False)

    # ── Print summary table ───────────────────────────────────────────────────
    for model_name in args.models:
        sub = summary[summary['model'] == model_name].copy()
        sub['R2 (mean±std)'] = (sub['R2_mean'].map('{:.4f}'.format)
                                + ' ± ' + sub['R2_std'].map('{:.4f}'.format))
        print(f'\n{"="*55}')
        print(f'  {model_name.upper()} — Test R² across {args.seeds} seeds')
        print(f'{"="*55}')
        print(sub[['molecule', 'R2 (mean±std)']].to_string(index=False))

    print(f'\nPer-seed results → {results_csv}')
    print(f'Summary          → {summary_csv}')


if __name__ == '__main__':
    main()
