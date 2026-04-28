"""
Deep Learning model: simple 1D CNN backbone + per-molecule output heads.

Architecture (INARA ATMOS — 12 CLIMA channels × 101 altitude levels):
  Input  : (B, 12, 101) — 12 CLIMA channels × 101 altitude levels
  Block 1: Conv1d(12→32,  k=9, s=2, p=4) + BN + ReLU + MaxPool1d(2) → (B, 32, 25)
  Block 2: Conv1d(32→64,  k=7, s=2, p=3) + BN + ReLU + MaxPool1d(2) → (B, 64,  6)
  Block 3: Conv1d(64→128, k=5, s=2, p=2) + BN + ReLU + MaxPool1d(2) → (B, 128, 1)
  Block 4: Conv1d(128→256,k=3, s=1, p=1) + BN + ReLU                → (B, 256, 1)
  Pool   : AdaptiveAvgPool1d(1) + Flatten                            → (B, 256)
  Shared : Dropout(0.25) + FC(256→128) + LayerNorm + ReLU           → (B, 128)
  Heads  : 12 × molecule-specific MLP → scalar log10 abundance

Per-molecule head configs stay lightweight so the model remains simple.
"""

import numpy as np
import torch
import torch.nn as nn

from .data_utils import MOLECULE_NAMES


# ------------------------------------------------------------------
# Per-molecule head configurations
# ------------------------------------------------------------------
# Keep the heads compact so the network stays easy to train and interpret.

MOLECULE_HEAD_CONFIGS = {
    'H2O': {'hidden_dims': [128], 'dropout': 0.20},
    'CO2': {'hidden_dims': [64],   'dropout': 0.15},
    'O2':  {'hidden_dims': [64],   'dropout': 0.15},
    'O3':  {'hidden_dims': [128],  'dropout': 0.20},
    'CH4': {'hidden_dims': [128],  'dropout': 0.20},
    'N2':  {'hidden_dims': [64],   'dropout': 0.15},
    'N2O': {'hidden_dims': [128],  'dropout': 0.20},
    'CO':  {'hidden_dims': [128],  'dropout': 0.20},
    'H2':  {'hidden_dims': [128],  'dropout': 0.20},
    'H2S': {'hidden_dims': [128],  'dropout': 0.25},
    'SO2': {'hidden_dims': [128],  'dropout': 0.25},
    'NH3': {'hidden_dims': [128],  'dropout': 0.25},
}

# Per-molecule loss weights (inverse of log10-space variance across dataset)
# Molecules with high variance are harder to learn; down-weight them slightly
# to prevent domination; up-weight trace molecules to force focus
MOLECULE_LOSS_WEIGHTS = {
    'H2O': 1.5, 'CO2': 1.2, 'O2':  1.2, 'O3':  1.8,
    'CH4': 1.5, 'N2':  1.0, 'N2O': 1.5, 'CO':  1.5,
    'H2':  1.5, 'H2S': 1.8, 'SO2': 2.0, 'NH3': 2.0,
}


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------
class MoleculeHead(nn.Module):
    """Per-molecule output head with configurable depth and dropout."""

    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class CNN1D(nn.Module):
    """
    Simple 1D CNN backbone with 12 per-molecule output heads.

    Input : (B, in_channels, seq_len)  normalised spectra / atmospheric profile
    Output: (B, 12)                    predicted log10 molecular abundances

    in_channels=3   for PSG spectra (3 signal channels, 4378 wavelength pts)
    in_channels=12  for INARA ATMOS (12 CLIMA variables, 101 altitude levels)
    """

    def __init__(self, head_configs=None, in_channels: int = 3):
        super().__init__()
        head_configs = head_configs or MOLECULE_HEAD_CONFIGS
        self.in_channels = in_channels

        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # → (B, 256, 1)

        self.shared = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleDict({
            mol: MoleculeHead(128, cfg['hidden_dims'], cfg['dropout'])
            for mol, cfg in head_configs.items()
        })

    def forward(self, x):
        """x: (B, in_channels, seq_len) → out: (B, 12)"""
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)   # (B, 256)
        x = self.shared(x)             # (B, 128)
        out = torch.stack([self.heads[mol](x) for mol in MOLECULE_NAMES], dim=1)
        return out   # (B, 12)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy aliases kept for any saved checkpoints or old configs.
SpectralCNN    = CNN1D
SpectralResNet = CNN1D


# ------------------------------------------------------------------
# Loss with per-molecule weighting
# ------------------------------------------------------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [MOLECULE_LOSS_WEIGHTS[m] for m in MOLECULE_NAMES]
        self.register_buffer('w', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        # pred, target: (B, 12)
        sq_err = (pred - target) ** 2          # (B, 12)
        weighted = sq_err * self.w.unsqueeze(0) # (B, 12)
        return weighted.mean()


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class Trainer:
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4,
                 patience=30, min_delta=1e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = WeightedMSELoss().to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.wait = 0
        self.best_state = None

    def _batch_forward(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        pred = self.model(X_batch)
        loss = self.criterion(pred, y_batch)
        return loss, pred

    def train_epoch(self, loader, scheduler=None):
        self.model.train()
        total_loss, n = 0.0, 0
        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            loss, _ = self._batch_forward(X_batch, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n += len(X_batch)
        if scheduler is not None:
            scheduler.step()
        return total_loss / n

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss, n = 0.0, 0
        for X_batch, y_batch in loader:
            loss, _ = self._batch_forward(X_batch, y_batch)
            total_loss += loss.item() * len(X_batch)
            n += len(X_batch)
        return total_loss / n

    def check_early_stop(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore_best(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        preds = []
        for X_batch, *_ in loader:
            X_batch = X_batch.to(self.device)
            preds.append(self.model(X_batch).cpu().numpy())
        return np.concatenate(preds, axis=0)


# ------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------
class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, spectra, molecules, augment=False, noise_std=0.01):
        self.spectra   = torch.from_numpy(spectra)     # (N, C, L) float32
        self.molecules = torch.from_numpy(molecules)   # (N, 12)   float32
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        x = self.spectra[idx]
        y = self.molecules[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
