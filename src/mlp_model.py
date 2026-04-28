"""
MLP baseline: flat fully-connected backbone + per-molecule output heads.

Same input/output interface as CNN1D — takes (B, in_channels, seq_len) shaped
normalized CLIMA profiles, flattens internally, and predicts log10 abundances.

This isolates the contribution of the 1D convolutional inductive bias: the
backbone is replaced with fully-connected layers while the training procedure,
loss function, and per-molecule heads remain identical to the CNN.

Architecture:
  Flatten : (B, 12, 101) → (B, 1212)
  FC-1    : Linear(1212→512) + BN + ReLU + Dropout(0.25)
  FC-2    : Linear(512→256)  + BN + ReLU + Dropout(0.25)
  Shared  : Linear(256→128)  + LN + ReLU          (matches CNN shared layer)
  Heads   : 12 × molecule-specific MLP → scalar log10 abundance
"""

import torch
import torch.nn as nn

from .data_utils import MOLECULE_NAMES
from .deep_model import MoleculeHead, MOLECULE_HEAD_CONFIGS


class MLP(nn.Module):
    """
    Flat MLP backbone with 12 per-molecule output heads.

    Input : (B, in_channels, seq_len) — normalised CLIMA profiles
    Output: (B, 12)                   — predicted log10 abundances

    The (B, in_channels, seq_len) signature intentionally matches CNN1D so the
    same DataLoader, Trainer, and evaluation code can be reused without change.
    """

    def __init__(self, head_configs=None, in_channels: int = 12, seq_len: int = 101):
        super().__init__()
        head_configs = head_configs or MOLECULE_HEAD_CONFIGS
        in_features = in_channels * seq_len

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        # Shared projection to 128-dim — identical to CNN1D shared layer
        self.shared = nn.Sequential(
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
        x = self.backbone(x)   # (B, 256)
        x = self.shared(x)     # (B, 128)
        out = torch.stack([self.heads[mol](x) for mol in MOLECULE_NAMES], dim=1)
        return out  # (B, 12)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
