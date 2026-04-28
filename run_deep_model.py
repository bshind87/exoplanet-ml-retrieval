#!/usr/bin/env python
"""
Train and evaluate the deep learning (1D CNN) model on the INARA exoplanet dataset.

Usage:
    python run_deep_model.py [--epochs 150] [--batch-size 32] [--lr 1e-3] [--save]

Training details:
  - Optimizer : AdamW (lr=1e-3, weight_decay=1e-4)
  - Scheduler : CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
  - Loss      : Weighted MSE with per-molecule importance weights
  - Augment   : Gaussian noise on spectra (std=0.01) during training
  - Early stop: patience=30 on validation loss
  - Device    : MPS (Apple Silicon) → CUDA → CPU
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_utils import prepare_data, MOLECULE_NAMES, compute_metrics, print_metrics
from src.deep_model import (
    CNN1D, SpectralDataset, Trainer, get_device, MOLECULE_HEAD_CONFIGS
)


def main(epochs=150, batch_size=32, lr=1e-3, weight_decay=1e-4,
         patience=30, save_model=False, resume=None,
         data_dir=None, in_channels=12):

    out_tag = Path(data_dir).name if data_dir else 'default'
    results_dir = Path('results') / out_tag
    models_dir  = Path('models')  / out_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    import torch
    if str(device) == 'mps':
        # Ensure MPS is fully initialised before training
        torch.mps.empty_cache()
        print(f'Device: {device}  (Apple Silicon MPS — GPU accelerated)')
    elif str(device) == 'cuda':
        print(f'Device: {device}  ({torch.cuda.get_device_name(0)})')
    else:
        print(f'Device: {device}  (CPU — consider installing PyTorch with MPS/CUDA support)')

    # ------------------------------------------------------------------
    # 1. Load and prepare data (spectra only — no PCA needed for DL)
    # ------------------------------------------------------------------
    print('Loading and preparing data ...')
    t0 = time.time()
    data = prepare_data(val_frac=0.15, test_frac=0.15,
                        random_state=42,
                        data_dir=data_dir)
    print(f'  Train: {len(data["idx_train"])}  Val: {len(data["idx_val"])}  '
          f'Test: {len(data["idx_test"])}')
    print(f'  Data loaded in {time.time()-t0:.1f}s')

    spec_train = data['spectra_train']    # (N, 12, 101) for INARA ATMOS
    spec_val   = data['spectra_val']
    spec_test  = data['spectra_test']
    mol_train  = data['molecules_train']  # (N, 12) log10
    mol_val    = data['molecules_val']
    mol_test   = data['molecules_test']

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    train_ds = SpectralDataset(spec_train, mol_train, augment=True,  noise_std=0.01)
    val_ds   = SpectralDataset(spec_val,   mol_val,   augment=False)
    test_ds  = SpectralDataset(spec_test,  mol_test,  augment=False)

    # Adjust num_workers based on device
    nw = 0 if str(device) == 'mps' else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(str(device) == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=nw)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = CNN1D(head_configs=MOLECULE_HEAD_CONFIGS, in_channels=in_channels)
    print(f'\nModel parameters: {model.count_parameters():,}')

    if resume is not None:
        state = torch.load(resume, map_location='cpu')
        model.load_state_dict(state)
        print(f'Resumed from {resume}')

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
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

    # Save training history
    pd.DataFrame(history).to_csv(results_dir / 'deep_training_history.csv', index=False)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    val_pred  = trainer.predict(val_loader)
    test_pred = trainer.predict(test_loader)

    val_df  = compute_metrics(mol_val,  val_pred)
    test_df = compute_metrics(mol_test, test_pred)

    print_metrics(val_df,  title='1D CNN — Validation Metrics')
    print_metrics(test_df, title='1D CNN — Test Metrics')

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    val_df.to_csv(results_dir  / 'deep_val_metrics.csv',   index=False)
    test_df.to_csv(results_dir / 'deep_test_metrics.csv',  index=False)
    np.save(results_dir / 'deep_test_pred.npy', test_pred)
    # save test targets only if not already saved by baseline run
    targets_path = results_dir / 'test_targets.npy'
    if not targets_path.exists():
        np.save(targets_path, mol_test)
    print(f'Saved metrics to results/')

    if save_model:
        ckpt_path = models_dir / 'spectral_resnet.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f'Saved model to {ckpt_path}')

    # ------------------------------------------------------------------
    # 7. Compare with baseline (if available)
    # ------------------------------------------------------------------
    baseline_path = results_dir / 'baseline_test_metrics.csv'
    if baseline_path.exists():
        bl = pd.read_csv(baseline_path)
        compare = pd.merge(
            bl[bl['molecule'] != 'MEAN'][['molecule', 'R2']].rename(columns={'R2': 'Baseline_R2'}),
            test_df[test_df['molecule'] != 'MEAN'][['molecule', 'R2']].rename(columns={'R2': 'DeepModel_R2'}),
            on='molecule'
        )
        compare['Delta_R2'] = compare['DeepModel_R2'] - compare['Baseline_R2']
        compare.loc[len(compare)] = {
            'molecule': 'MEAN',
            'Baseline_R2': compare['Baseline_R2'].mean(),
            'DeepModel_R2': compare['DeepModel_R2'].mean(),
            'Delta_R2': compare['Delta_R2'].mean(),
        }
        print(f'\n{"="*65}')
        print(f'  Model Comparison (Test Set)')
        print(f'{"="*65}')
        print(compare.to_string(index=False, float_format='%.4f'))
        compare.to_csv(results_dir / 'model_comparison.csv', index=False)

    mean_r2 = test_df[test_df['molecule'] != 'MEAN']['R2'].mean()
    print(f'\n{"="*40}')
    print(f'  DeepModel Test  Mean R² = {mean_r2:.4f}')
    print(f'{"="*40}')
    return test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D CNN deep model')
    parser.add_argument('--epochs',       type=int,   default=150,  help='Max epochs (default: 150)')
    parser.add_argument('--batch-size',   type=int,   default=32,   help='Batch size (default: 32)')
    parser.add_argument('--lr',           type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience',     type=int,   default=30,   help='Early stop patience (default: 30)')
    parser.add_argument('--save',         action='store_true',      help='Save trained model')
    parser.add_argument('--resume',       type=str,   default=None, help='Resume from checkpoint path')
    parser.add_argument('--data-dir',     type=str,   default=None,
                        help='Path to processed data directory '
                             '(default: inara_data/processed)')
    parser.add_argument('--in-channels',  type=int,   default=12,
                        help='Input channels: 12 for INARA ATMOS (default), 3 for PSG spectra')
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_model=args.save,
        resume=args.resume,
        data_dir=args.data_dir,
        in_channels=args.in_channels,
    )
