#!/usr/bin/env python
"""
Paper Experiment: MLP Baseline
================================
Trains the flat MLP model on the same engineered data as the 1D CNN (Step 4).
Provides an intermediate baseline between Random Forest and CNN to isolate the
contribution of the convolutional inductive bias.

Reads from  : inara_data/engineered/   (output of pipeline Step 2)
Writes to   : results/paper_experiments/

Output files:
  mlp_val_metrics.csv
  mlp_test_metrics.csv
  mlp_test_pred.npy
  mlp_training_history.csv
  mlp_model.pt              (if --save)

Usage:
  python experiments/run_mlp_baseline.py --profile local
  python experiments/run_mlp_baseline.py --profile hpc --save
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path
from src.data_utils import compute_metrics, print_metrics, MOLECULE_NAMES, MoleculeScaler
from src.deep_model import (
    SpectralDataset, Trainer, get_device, MOLECULE_HEAD_CONFIGS,
)
from src.mlp_model import MLP


def main() -> None:
    parser = get_parser('Paper experiment: MLP baseline')
    parser.add_argument('--save',        action='store_true', help='Save model checkpoint')
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--batch-size',  type=int,   default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--patience',    type=int,   default=None)
    args = parser.parse_args()

    cfg       = load_config(args.config, args.profile)
    paths     = cfg['paths']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    results_dir    = resolve_path(paths['results_dir'],    args.profile)
    models_dir     = resolve_path(paths['models_dir'],     args.profile)
    paper_exp_dir  = results_dir.parent / 'paper_experiments'
    paper_exp_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    epochs      = args.epochs     or train_cfg['epochs']
    batch_size  = args.batch_size or train_cfg['batch_size']
    lr          = args.lr         or train_cfg['lr']
    weight_decay = train_cfg['weight_decay']
    patience    = args.patience   or train_cfg['patience']
    in_channels = model_cfg['in_channels']

    device = get_device()
    if str(device) == 'mps':
        torch.mps.empty_cache()
        device_label = 'Apple Silicon MPS'
    elif str(device) == 'cuda':
        device_label = torch.cuda.get_device_name(0)
    else:
        device_label = 'CPU'

    print('=' * 60)
    print('  Paper Experiment: MLP Baseline')
    print('=' * 60)
    print(f'  Profile        : {args.profile}')
    print(f'  Device         : {device}  ({device_label})')
    print(f'  Engineered dir : {engineered_dir}')
    print(f'  Output dir     : {paper_exp_dir}')
    print(f'  epochs={epochs}  batch={batch_size}  lr={lr}  patience={patience}')
    print()

    # ── Load engineered data ──────────────────────────────────────────────────
    print('Loading engineered spectra ...')
    t0 = time.time()
    spec_train = np.load(engineered_dir / 'spectra_train.npy')
    spec_val   = np.load(engineered_dir / 'spectra_val.npy')
    spec_test  = np.load(engineered_dir / 'spectra_test.npy')
    mol_train  = np.load(engineered_dir / 'molecules_train.npy')
    mol_val    = np.load(engineered_dir / 'molecules_val.npy')
    mol_test   = np.load(engineered_dir / 'molecules_test.npy')
    seq_len    = spec_train.shape[2]
    print(f'  Train: {spec_train.shape}   Val: {spec_val.shape}   Test: {spec_test.shape}')
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # ── Target normalisation (Z-score per molecule, fit on train only) ────────
    mol_scaler = MoleculeScaler()
    mol_train_sc = mol_scaler.fit_transform(mol_train)
    mol_val_sc   = mol_scaler.transform(mol_val)
    mol_test_sc  = mol_scaler.transform(mol_test)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    nw = 0 if str(device) == 'mps' else 2
    train_ds = SpectralDataset(spec_train, mol_train_sc, augment=True,  noise_std=0.01)
    val_ds   = SpectralDataset(spec_val,   mol_val_sc,   augment=False)
    test_ds  = SpectralDataset(spec_test,  mol_test_sc,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(str(device) == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=nw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLP(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=in_channels, seq_len=seq_len)
    print(f'\nMLP parameters  : {model.count_parameters():,}')
    print(f'  (CNN1D params for comparison: see models/cnn1d.pt)')

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, device, lr=lr, weight_decay=weight_decay, patience=patience)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-6
    )

    print(f'\nTraining for up to {epochs} epochs (early stop patience={patience}) ...\n')
    best_val = np.inf
    history  = {'train_loss': [], 'val_loss': []}
    t_train  = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, scheduler=scheduler)
        val_loss   = trainer.eval_epoch(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        marker  = '*' if val_loss < best_val else ''
        best_val = min(best_val, val_loss)

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = trainer.optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:3d}/{epochs}  '
                  f'train={train_loss:.5f}  val={val_loss:.5f}  '
                  f'lr={lr_now:.2e}  {marker}')

        if trainer.check_early_stop(val_loss):
            print(f'\n  Early stopping at epoch {epoch} (patience={patience})')
            break

    print(f'\n  Training completed in {time.time()-t_train:.1f}s')
    trainer.restore_best()
    pd.DataFrame(history).to_csv(paper_exp_dir / 'mlp_training_history.csv', index=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_pred  = mol_scaler.inverse_transform(trainer.predict(val_loader))
    test_pred = mol_scaler.inverse_transform(trainer.predict(test_loader))

    val_df  = compute_metrics(mol_val,  val_pred)
    test_df = compute_metrics(mol_test, test_pred)

    print_metrics(val_df,  title='MLP — Validation Metrics')
    print_metrics(test_df, title='MLP — Test Metrics')

    val_df.to_csv(paper_exp_dir  / 'mlp_val_metrics.csv',  index=False)
    test_df.to_csv(paper_exp_dir / 'mlp_test_metrics.csv', index=False)
    np.save(paper_exp_dir / 'mlp_test_pred.npy', test_pred)
    print(f'\nSaved metrics → {paper_exp_dir}')

    if args.save:
        ckpt = models_dir / 'mlp_model.pt'
        torch.save(model.state_dict(), ckpt)
        print(f'Saved model  → {ckpt}')

    mean_r2 = test_df[test_df['molecule'] != 'MEAN']['R2'].mean()
    print(f'\n{"="*40}')
    print(f'  MLP Test  Mean R² = {mean_r2:.4f}')
    print(f'{"="*40}')


if __name__ == '__main__':
    main()
