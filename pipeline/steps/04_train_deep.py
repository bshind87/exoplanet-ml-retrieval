#!/usr/bin/env python
"""
Step 4 — Train Deep Model (1D CNN)
===================================
Trains the 1D CNN using the normalised spectra produced by Step 2.
Uses the full training split (no sample cap).

Device selection:  auto → MPS (Apple Silicon) → CUDA → CPU

Input  (engineered_dir):
  spectra_train.npy, spectra_val.npy, spectra_test.npy
  molecules_train.npy, molecules_val.npy, molecules_test.npy

Output (results_dir / models_dir):
  deep_val_metrics.csv
  deep_test_metrics.csv
  deep_test_pred.npy
  deep_training_history.csv
  cnn1d.pt                        (if --save)

Usage:
  python pipeline/steps/04_train_deep.py [--profile local|hpc] [--save]
  python pipeline/steps/04_train_deep.py --resume models/cnn1d.pt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path
from src.data_utils import compute_metrics, print_metrics, MOLECULE_NAMES, MoleculeScaler
from src.deep_model import (
    CNN1D, SpectralDataset, Trainer, get_device, MOLECULE_HEAD_CONFIGS,
)


def main() -> None:
    parser = get_parser('Step 4: Train 1D CNN deep model')
    parser.add_argument('--save',   action='store_true', help='Save model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    # Allow per-run overrides without editing config
    parser.add_argument('--epochs',       type=int,   default=None)
    parser.add_argument('--batch-size',   type=int,   default=None)
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--patience',     type=int,   default=None)
    args = parser.parse_args()

    cfg      = load_config(args.config, args.profile)
    paths    = cfg['paths']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    engineered_dir = resolve_path(paths['engineered_dir'], args.profile)
    results_dir    = resolve_path(paths['results_dir'],    args.profile)
    models_dir     = resolve_path(paths['models_dir'],     args.profile)

    epochs       = args.epochs     or train_cfg['epochs']
    batch_size   = args.batch_size or train_cfg['batch_size']
    lr           = args.lr         or train_cfg['lr']
    weight_decay = train_cfg['weight_decay']
    patience     = args.patience   or train_cfg['patience']
    in_channels  = model_cfg['in_channels']

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()
    if str(device) == 'mps':
        torch.mps.empty_cache()
        device_label = 'Apple Silicon MPS'
    elif str(device) == 'cuda':
        device_label = torch.cuda.get_device_name(0)
    else:
        device_label = 'CPU'

    print('=' * 60)
    print('  INARA Pipeline — Step 4: Deep Model (1D CNN)')
    print('=' * 60)
    print(f'  Profile        : {args.profile}')
    print(f'  Device         : {device}  ({device_label})')
    print(f'  Engineered dir : {engineered_dir}')
    print(f'  Results dir    : {results_dir}')
    print(f'  epochs={epochs}  batch={batch_size}  lr={lr}  patience={patience}')
    print()

    # ── 1. Load pre-engineered normalised spectra ─────────────────────────────
    print('Loading engineered spectra ...')
    t0 = time.time()
    spec_train = np.load(engineered_dir / 'spectra_train.npy')
    spec_val   = np.load(engineered_dir / 'spectra_val.npy')
    spec_test  = np.load(engineered_dir / 'spectra_test.npy')
    mol_train  = np.load(engineered_dir / 'molecules_train.npy')
    mol_val    = np.load(engineered_dir / 'molecules_val.npy')
    mol_test   = np.load(engineered_dir / 'molecules_test.npy')
    print(f'  Train: {spec_train.shape}   Val: {spec_val.shape}   Test: {spec_test.shape}')
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # ── 2. Scale molecule targets (Z-score per molecule, fit on train only) ─────
    mol_scaler = MoleculeScaler()
    mol_train_scaled = mol_scaler.fit_transform(mol_train)
    mol_val_scaled   = mol_scaler.transform(mol_val)
    mol_test_scaled  = mol_scaler.transform(mol_test)
    print(f'  Target means (log10): {mol_scaler.means.round(2)}')
    print(f'  Target stds  (log10): {mol_scaler.stds.round(2)}')

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    nw = 0 if str(device) == 'mps' else 2   # MPS doesn't support multi-worker DL
    train_ds = SpectralDataset(spec_train, mol_train_scaled, augment=True,  noise_std=0.01)
    val_ds   = SpectralDataset(spec_val,   mol_val_scaled,   augment=False)
    test_ds  = SpectralDataset(spec_test,  mol_test_scaled,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(str(device) == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=nw)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = CNN1D(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=in_channels)
    print(f'\nModel parameters: {model.count_parameters():,}')

    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state)
        print(f'Resumed from {args.resume}')

    # ── 5. Training loop ──────────────────────────────────────────────────────
    trainer = Trainer(model, device, lr=lr, weight_decay=weight_decay,
                      patience=patience)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-6
    )

    print(f'\nTraining for up to {epochs} epochs (early stop patience={patience}) ...\n')
    best_val_loss = np.inf
    history = {'train_loss': [], 'val_loss': []}
    t_train = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, scheduler=scheduler)
        val_loss   = trainer.eval_epoch(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        improved = '*' if val_loss < best_val_loss else ''
        best_val_loss = min(best_val_loss, val_loss)

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = trainer.optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch:3d}/{epochs}  '
                  f'train={train_loss:.5f}  val={val_loss:.5f}  '
                  f'lr={lr_now:.2e}  {improved}')

        if trainer.check_early_stop(val_loss):
            print(f'\n  Early stopping at epoch {epoch} (patience={patience})')
            break

    print(f'\n  Training completed in {time.time()-t_train:.1f}s')
    trainer.restore_best()
    pd.DataFrame(history).to_csv(results_dir / 'deep_training_history.csv', index=False)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    # Predictions are in scaled space — inverse-transform back to log10 before metrics
    val_pred  = mol_scaler.inverse_transform(trainer.predict(val_loader))
    test_pred = mol_scaler.inverse_transform(trainer.predict(test_loader))

    val_df  = compute_metrics(mol_val,  val_pred)
    test_df = compute_metrics(mol_test, test_pred)

    print_metrics(val_df,  title='1D CNN — Validation Metrics')
    print_metrics(test_df, title='1D CNN — Test Metrics')

    # ── 7. Save results ───────────────────────────────────────────────────────
    val_df.to_csv(results_dir  / 'deep_val_metrics.csv',   index=False)
    test_df.to_csv(results_dir / 'deep_test_metrics.csv',  index=False)
    np.save(results_dir / 'deep_test_pred.npy', test_pred)

    # Save test targets only if not already written by Step 3
    targets_path = results_dir / 'test_targets.npy'
    if not targets_path.exists():
        np.save(targets_path, mol_test)

    print(f'\nSaved metrics → {results_dir}')

    if args.save:
        ckpt = models_dir / 'cnn1d.pt'
        torch.save(model.state_dict(), ckpt)
        print(f'Saved model  → {ckpt}')

    mean_r2 = test_df[test_df['molecule'] != 'MEAN']['R2'].mean()
    print(f'\n{"="*40}')
    print(f'  DeepModel Test  Mean R² = {mean_r2:.4f}')
    print(f'{"="*40}')


if __name__ == '__main__':
    main()
