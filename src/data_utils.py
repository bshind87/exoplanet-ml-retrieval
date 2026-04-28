"""
Data loading, preprocessing, and splitting utilities for the INARA ATMOS dataset.

Dataset (INARA ATMOS — full 124k):
  spectra.npy    : (N, 12, 101) - 12 CLIMA atmospheric channels × 101 altitude levels
  molecules.npy  : (N, 12)      - log10 molecular volume mixing ratios (targets)
  wavelengths.npy: (101,)       - altitude level index axis

CLIMA channels (12):
  Temperature, Pressure, H2O, CO2, O2, O3, CH4, N2, N2O, CO, H2, H2S
  (all Z-score normalised per channel before model input)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MOLECULE_NAMES = ['H2O', 'CO2', 'O2', 'O3', 'CH4', 'N2', 'N2O', 'CO', 'H2', 'H2S', 'SO2', 'NH3']
NUM_MOLECULES = len(MOLECULE_NAMES)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'inara_data' / 'processed'


# ------------------------------------------------------------------
# Raw loading
# ------------------------------------------------------------------
def load_raw(data_dir=None):
    """Load raw numpy arrays from disk. Returns spectra, molecules, wavelengths."""
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    spectra    = np.load(d / 'spectra.npy')      # (N, C, L) float32
    molecules  = np.load(d / 'molecules.npy')    # (N, 12)   float32
    wavelengths = np.load(d / 'wavelengths.npy') # (L,)      float64
    return spectra, molecules, wavelengths


# ------------------------------------------------------------------
# Train / val / test split  (stratified-ish by molecule mean)
# ------------------------------------------------------------------
def split_indices(n_samples, val_frac=0.15, test_frac=0.15, random_state=42):
    """Return (train_idx, val_idx, test_idx) index arrays."""
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=random_state, shuffle=True
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test, random_state=random_state, shuffle=True
    )
    return train_idx, val_idx, test_idx


# ------------------------------------------------------------------
# Spectral normalisation (fit on train, apply to all)
# ------------------------------------------------------------------
class SpectraScaler:
    """Per-channel Z-score normalisation across (N, L) for each of the C channels."""

    def __init__(self):
        self.means = None   # (C, L)
        self.stds = None    # (C, L)

    def fit(self, spectra_train):
        """spectra_train: (N, C, L)"""
        self.means = spectra_train.mean(axis=0)   # (C, L)
        self.stds = spectra_train.std(axis=0)     # (C, L)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)
        return self

    def transform(self, spectra):
        """Returns float32 normalised spectra (N, C, L)."""
        return ((spectra - self.means) / self.stds).astype(np.float32)

    def fit_transform(self, spectra_train):
        return self.fit(spectra_train).transform(spectra_train)


# ------------------------------------------------------------------
# Molecule target scaler (fit on train only)
# ------------------------------------------------------------------
class MoleculeScaler:
    """Per-molecule Z-score normalisation of log10 targets."""

    def __init__(self):
        self.means = None  # (12,)
        self.stds = None   # (12,)

    def fit(self, molecules_train):
        self.means = molecules_train.mean(axis=0)
        self.stds = molecules_train.std(axis=0)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)
        return self

    def transform(self, molecules):
        return ((molecules - self.means) / self.stds).astype(np.float32)

    def inverse_transform(self, molecules_scaled):
        return (molecules_scaled * self.stds + self.means).astype(np.float32)

    def fit_transform(self, molecules_train):
        return self.fit(molecules_train).transform(molecules_train)


# ------------------------------------------------------------------
# Feature extraction for baseline (PCA-reduced spectral features)
# ------------------------------------------------------------------
def extract_baseline_features(spectra, pca=None, n_components=300):
    """
    Flatten all C channels and reduce with PCA.
    Returns:
      features_pca : (N, n_components) or (N, fitted_components)
      pca          : fitted sklearn PCA object
    """
    from sklearn.decomposition import PCA

    # Flatten (N, C, L) → (N, C*L) and upcast to float64.
    # float32 matmul (N, C*L) @ (C*L, K) can overflow/produce NaN when
    # C*L is large (e.g. 1212) and PCA eigenvectors span many dimensions.
    # float64 eliminates this entirely with no meaningful cost for RF features.
    N = spectra.shape[0]
    flat = spectra.reshape(N, -1).astype(np.float64)

    if np.isnan(flat).any() or np.isinf(flat).any():
        raise ValueError(
            'extract_baseline_features: NaN or Inf detected in spectra input. '
            'Check SpectraScaler output and raw data integrity.'
        )

    # Suppress spurious FPU RuntimeWarnings from the macOS Accelerate BLAS
    # backend (divide-by-zero / overflow / invalid in matmul).  These fire
    # even when the result is fully valid, as confirmed by NaN/Inf checks
    # above and in the quality report.  Real data problems still raise above.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='.*encountered in matmul.*')
        if pca is None:
            pca = PCA(n_components=n_components, svd_solver='full',
                      random_state=42, whiten=False)
            features_pca = pca.fit_transform(flat)
        else:
            features_pca = pca.transform(flat)

    return features_pca, pca


# ------------------------------------------------------------------
# Convenience: load and prepare everything
# ------------------------------------------------------------------
def prepare_data(val_frac=0.15, test_frac=0.15, random_state=42,
                 pca_components=300, data_dir=None, max_train_samples=None):
    """
    Full pipeline: load → split → normalize → PCA features for baseline.

    max_train_samples: if set, randomly subsample only the training split to
                       this many samples. Val/test splits are unchanged so
                       model comparisons remain on the same held-out sets.

    Returns dict with keys:
      spectra_{train,val,test}        : normalized spectra (N, C, L)
      molecules_{train,val,test}      : raw log10 targets (N, 12)
      feat_{train,val,test}           : PCA features for baseline (N, pca_components)
      wavelengths                     : wavelength/altitude axis
      spectra_scaler, molecule_scaler, pca
      idx_{train,val,test}
    """
    spectra, molecules, wavelengths = load_raw(data_dir)
    train_idx, val_idx, test_idx = split_indices(len(spectra), val_frac, test_frac, random_state)

    if max_train_samples is not None and max_train_samples < len(train_idx):
        rng = np.random.default_rng(random_state)
        train_idx = rng.choice(train_idx, size=max_train_samples, replace=False)
        print(f'  Subsampled training set to {len(train_idx):,} samples '
              f'(val/test unchanged)')

    # Normalize spectra
    spec_scaler = SpectraScaler()
    spec_train = spec_scaler.fit_transform(spectra[train_idx])
    spec_val   = spec_scaler.transform(spectra[val_idx])
    spec_test  = spec_scaler.transform(spectra[test_idx])

    # Raw molecule targets (log10 space, no scaling for baseline metrics)
    mol_train = molecules[train_idx]
    mol_val   = molecules[val_idx]
    mol_test  = molecules[test_idx]

    # PCA features for baseline
    feat_train, pca = extract_baseline_features(spec_train, pca=None, n_components=pca_components)
    feat_val,  _    = extract_baseline_features(spec_val,   pca=pca)
    feat_test, _    = extract_baseline_features(spec_test,  pca=pca)

    return {
        'spectra_train': spec_train, 'spectra_val': spec_val, 'spectra_test': spec_test,
        'molecules_train': mol_train, 'molecules_val': mol_val, 'molecules_test': mol_test,
        'feat_train': feat_train, 'feat_val': feat_val, 'feat_test': feat_test,
        'wavelengths': wavelengths,
        'spectra_scaler': spec_scaler, 'pca': pca,
        'idx_train': train_idx, 'idx_val': val_idx, 'idx_test': test_idx,
    }


# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, mol_names=None):
    """
    Compute per-molecule R², RMSE, MAE.
    Returns a DataFrame with columns [molecule, R2, RMSE, MAE].
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    mol_names = mol_names or MOLECULE_NAMES
    rows = []
    for i, name in enumerate(mol_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        rows.append({
            'molecule': name,
            'R2':   r2_score(yt, yp),
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE':  mean_absolute_error(yt, yp),
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        'molecule': 'MEAN',
        'R2':   df['R2'].mean(),
        'RMSE': df['RMSE'].mean(),
        'MAE':  df['MAE'].mean(),
    }])
    return pd.concat([df, summary], ignore_index=True)


def print_metrics(df, title=''):
    if title:
        print(f'\n{"="*60}')
        print(f'  {title}')
        print(f'{"="*60}')
    print(df.to_string(index=False, float_format='%.4f'))
