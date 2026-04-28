"""
Baseline model: per-molecule Random Forest with PCA-reduced spectral features.

Each of the 12 molecules gets its own RF with intentionally modest hyperparameters
so the method stays a true baseline rather than a high-capacity model.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from .data_utils import MOLECULE_NAMES, compute_metrics, print_metrics


# ------------------------------------------------------------------
# Per-molecule RF hyperparameters
# ------------------------------------------------------------------
# Baseline policy:
#   - keep n_estimators <= 100
#   - keep max_depth shallow
#   - use a small min_samples_leaf to reduce overfitting

MOLECULE_RF_PARAMS = {
    mol: {
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42,
    }
    for mol in MOLECULE_NAMES
}


# ------------------------------------------------------------------
# BaselineModel: collection of per-molecule RF estimators
# ------------------------------------------------------------------
class BaselineModel:
    """
    Trains one RandomForestRegressor per molecule with molecule-specific hyperparameters.
    Input: PCA-reduced spectral features (N, n_components).
    Output: log10 molecular abundances (N, 12).
    """

    def __init__(self, mol_params=None):
        self.mol_params = mol_params or MOLECULE_RF_PARAMS
        self.models = {}   # molecule_name -> fitted RandomForestRegressor

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        X_train : (N, n_features) PCA features
        y_train : (N, 12) log10 targets
        """
        for i, mol in enumerate(MOLECULE_NAMES):
            params = self.mol_params[mol]
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train[:, i])
            self.models[mol] = rf

            if verbose:
                train_r2 = r2_score(y_train[:, i], rf.predict(X_train))
                val_str = ''
                if X_val is not None:
                    val_r2 = r2_score(y_val[:, i], rf.predict(X_val))
                    val_str = f'  val_R²={val_r2:.4f}'
                print(f'  [{i+1:2d}/12] {mol:5s}  train_R²={train_r2:.4f}{val_str}')

        return self

    def predict(self, X):
        """Returns (N, 12) predicted log10 abundances."""
        preds = np.column_stack([self.models[mol].predict(X) for mol in MOLECULE_NAMES])
        return preds.astype(np.float32)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        print(f'Saved baseline models to {path}')

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.models = joblib.load(path)
        return obj

    def feature_importance(self, top_n=10):
        """Return top-n most important PCA components per molecule."""
        result = {}
        for mol in MOLECULE_NAMES:
            imp = self.models[mol].feature_importances_
            top_idx = np.argsort(imp)[::-1][:top_n]
            result[mol] = list(zip(top_idx.tolist(), imp[top_idx].tolist()))
        return result

    def evaluate(self, X, y_true, split_name='Test'):
        y_pred = self.predict(X)
        df = compute_metrics(y_true, y_pred)
        print_metrics(df, title=f'Baseline RF — {split_name} Metrics')
        return df, y_pred
