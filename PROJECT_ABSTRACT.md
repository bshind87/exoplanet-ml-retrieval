# CS6140 Machine Learning – Spring 2026 – Northeastern University San Jose
## Project Title: Exploring ML Models for Detection of Atmospheric Composition in Exoplanets

**Team Members**
- Shantanu Wankhare — wankhare.s@northeastern.edu
- Bhalchandra Shinde — shinde.b@northeastern.edu
- Asad Mulani — mulani.a@northeastern.edu

**Dataset**: INARA ATMOS (NASA FDL / Zorzan et al. 2025)

---

## Overview

This project applies machine learning to exoplanet atmospheric retrieval — the task of inferring the chemical composition of a planet's atmosphere from its climate state. Traditional retrieval methods based on Bayesian sampling are computationally expensive, often requiring hours per planet. As space missions such as the James Webb Space Telescope (JWST) and the planned Habitable Worlds Observatory (HWO) generate data at unprecedented scale, fast and automated retrieval is a scientific necessity.

We trained two models on the INARA ATMOS dataset — a per-molecule Random Forest baseline and a compact 1D CNN — to predict the log₁₀ surface volume mixing ratios of 12 atmospheric molecules from simulated climate profiles. Both models infer all 12 abundances in milliseconds. For the final report, we treat H2O and NH3 as degenerate targets on this split and exclude them from the summary comparison because they are nearly constant and can distort aggregate R². Under that filtered comparison, the 1D CNN is the preferred model.

---

## Dataset

The **INARA ATMOS** dataset (Zorzan et al. 2025) contains synthetic rocky planet atmosphere simulations generated with the CLIMA photochemical-climate model, designed as a machine learning training resource. The full archive contains approximately **3.1 million samples** across ten tar archives. We ran experiments at 25,000 samples locally; full-scale runs are performed on Northeastern Explorer HPC.

**Input — CLIMA profiles**: Each sample is a climate profile of 12 atmospheric variables measured at **101 altitude levels** (surface to top of atmosphere), forming a `(12, 101)` tensor:

| Channel | Variable | Physical meaning |
|---|---|---|
| 0 | `J` | Actinic radiative flux |
| 1 | `P` | Pressure |
| 2 | `ALT` | Altitude (km) |
| 3 | `T` | Temperature |
| 4 | `CONVEC` | Convective flux |
| 5 | `DT` | Temperature tendency |
| 6 | `TOLD` | Previous temperature |
| 7 | `FH2O` | Water vapour flux |
| 8 | `FSAVE` | Saved radiative flux |
| 9 | `FO3` | Ozone flux |
| 10 | `TCOOL` | Radiative cooling rate |
| 11 | `THEAT` | Radiative heating rate |

**Targets — molecular surface abundances**: log₁₀ volume mixing ratios of 12 molecules at the surface layer. Values range from ~0 (major constituent) to −40 (detection floor).

**Processing**: Raw per-sample `.npy.npz` files are extracted from the tar archives by `process_inara.py` and saved as:
- `spectra.npy` — `(N, 12, 101)` CLIMA profiles
- `molecules.npy` — `(N, 12)` log₁₀ surface abundances
- `aux_params.npy` — `(N, 11)` metadata scalars (input fluxes, conditions)
- `wavelengths.npy` — `(101,)` altitude axis in km

**Data split**: 70% train / 15% validation / 15% test (fixed seed 42).

---

## Models

### Baseline — Per-Molecule Random Forest

Each of the 12 molecules gets its own `RandomForestRegressor` trained on PCA-reduced CLIMA features:

- Flatten `(12, 101)` → `(1,212,)`, reduce with PCA to **300 components** (`svd_solver='full'`, `whiten=False`, float64)
- Uniform hyperparameters across all molecules: `n_estimators=100`, `max_depth=8`, `min_samples_leaf=2`, `max_features='sqrt'`
- Intentionally minimal — represents the "fast baseline" condition, always capped at 10,000 training samples

### Deep Model — Compact 1D CNN

A compact 1D CNN backbone with 12 per-molecule output heads operating directly on the `(12, 101)` CLIMA profile:

```
Input  : (B, 12, 101)  — 12 CLIMA channels × 101 altitude levels
Block 1: Conv1d(12→32, k=9, s=2) + BN + ReLU + MaxPool1d(2)
Block 2: Conv1d(32→64, k=7, s=2) + BN + ReLU + MaxPool1d(2)
Block 3: Conv1d(64→128, k=5, s=2) + BN + ReLU + MaxPool1d(2)
Block 4: Conv1d(128→256, k=3, s=1) + BN + ReLU
Pool   : AdaptiveAvgPool1d(1)  → (B, 256)
Shared : Dropout(0.25) + FC(256→128) + LayerNorm + ReLU
Heads  : 12 × molecule-specific MLP → scalar log₁₀ abundance
```

Per-molecule heads stay compact to keep the model simple and robust. The final report excludes H2O and NH3 from aggregate comparison only, because H2O is nearly constant on this split and NH3 is fixed at the detection floor of −40, which would otherwise distort mean R².

Training: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, WeightedMSE loss with per-molecule importance weights, early stopping (patience=30), Gaussian noise augmentation (std=0.01). Targets are Z-score normalised per molecule (MoleculeScaler) before training and inverse-transformed for evaluation.

---

## Results

### Filtered final comparison

| Molecule | RF R² | CNN R² | ΔR² |
|---|---|---|---|---|
| CO₂ | 0.873 | 0.995 | +0.122 |
| O₂ | 0.755 | 0.918 | +0.163 |
| O₃ | 0.798 | 0.861 | +0.063 |
| CH₄ | 0.931 | 0.999 | +0.068 |
| N₂ | 0.766 | 0.873 | +0.107 |
| N₂O | 0.783 | 0.949 | +0.166 |
| CO | 0.944 | 0.999 | +0.056 |
| H₂ | 0.769 | 0.979 | +0.210 |
| H₂S | 0.935 | 0.995 | +0.060 |
| SO₂ | 0.900 | 0.973 | +0.073 |
| **MEAN** | **0.845** | **0.954** | **+0.109** |

**Key observations:**
- The filtered comparison excludes H2O and NH3 because they are degenerate on this split: H2O is nearly constant and NH3 is fixed at the detection floor.
- The 1D CNN outperforms the baseline on all 10 retained molecules.
- The simplified CNN is therefore the final model choice for reporting, while the capped Random Forest remains the baseline reference.

---

## Evaluation

Models are evaluated on a held-out test set (15%) using per-molecule:
- **R²** — fraction of log₁₀ abundance variance explained
- **RMSE** — in log₁₀ units (1 unit = one order of magnitude)
- **MAE** — robust to outliers

---

## Deliverables

1. **EDA notebook** (`notebooks/eda.ipynb`) — statistical analysis of CLIMA profiles, molecular distributions, correlation structure, and dataset split validation
2. **Visualisation notebook** (`notebooks/visualize.ipynb`) — model loading, training curve, R² comparison, scatter plots, residual distributions, RF feature importance, CNN architecture breakdown
3. **Training scripts** — `run_baseline.py`, `run_deep_model.py` with configurable data scale and hyperparameters
4. **Processing pipeline** — `process_inara.py` for extracting CLIMA profiles from raw INARA tar archives at any sample count (5K–124K)
5. **Saved models** — `models/baseline_rf.joblib`, `models/cnn1d.pt`
6. **Final report** — background, methodology, results, and discussion

---

## References

1. Zorzan et al. (2025) — *ApJS* 277:38 — INARA dataset and baseline retrieval model
2. Márquez-Neila, P. et al. (2018) — *Nature Astronomy* 2 — random forest atmospheric retrieval
3. Vasist, M. et al. (2023) — *A&A* — neural posterior estimation with normalising flows
4. Gebhard, T. et al. (2024) — *A&A* — flow matching for full posterior atmospheric retrieval
5. JWST Transiting Exoplanet Community ERS Team (2023) — *Nature* 614:649 — WASP-39 b benchmark
