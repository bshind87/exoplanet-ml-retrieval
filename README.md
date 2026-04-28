# exo-atmos-retrieval

**Deep Learning for Exoplanet Atmospheric Retrieval on the INARA ATMOS Dataset**

A machine learning pipeline that predicts molecular surface abundances in exoplanet atmospheres from CLIMA atmospheric profiles. Three models are compared: a **Random Forest baseline**, a **flat MLP**, and a **compact 1D CNN**.

> **Paper:** *Deep Learning for Exoplanet Atmospheric Retrieval: Scaling, Architecture Ablation, and Statistical Validation on the INARA ATMOS Dataset*
> Bhalchandra Shinde, Sandhya Shinde, Amit Rajput — arXiv preprint

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Running Locally (Mac M5)](#6-running-locally-mac-m5)
7. [Running on HPC (SLURM)](#7-running-on-hpc-slurm)
8. [Running Individual Steps](#8-running-individual-steps)
9. [Paper Experiments](#9-paper-experiments)
10. [Configuration Reference](#10-configuration-reference)
11. [Dashboard — Viewing Results](#11-dashboard--viewing-results)
12. [Results & Output Files](#12-results--output-files)

---

## 1. Project Overview

**Task:** Given a simulated exoplanet's atmospheric CLIMA profile (12 variables × 101 altitude levels), predict the log₁₀ surface volume mixing ratios of 12 molecular species.

**Models:**

| Model | Description | Training samples |
|---|---|---|
| Random Forest (baseline) | Per-molecule RF on PCA-reduced features | ≤ 10,000 (capped) |
| MLP | Flat fully-connected network | Full training split |
| 1D CNN | Compact 1D CNN + per-molecule output heads | Full training split |

**Molecules predicted:** H₂O · CO₂ · O₂ · O₃ · CH₄ · N₂ · N₂O · CO · H₂ · H₂S · SO₂ · NH₃

**Key results (86,800 training samples, 5 seeds):**
- 1D CNN: mean R² = 0.9993 ± <0.0001 on 10 active molecules
- Random Forest: mean R² = 0.8295 ± 0.0053
- CNN and MLP achieve near-identical accuracy; CNN preferred for parameter efficiency (368K vs 963K)
- Performance saturates near 50K samples for both deep models
- H₂O achieves R² = 0.999 at 86.8K samples (previously reported as unlearnable at 25K)

---

## 2. Dataset

**Source:** [INARA ATMOS](https://doi.org/10.3847/1538-4365/ad9230) — Zorzan et al. 2025, ApJS 277:38
**Full size:** 3,112,620 samples · experiments use 124,000 samples

| Array | Shape | Description |
|---|---|---|
| `spectra.npy` | (N, 12, 101) | CLIMA atmospheric profiles — input features |
| `molecules.npy` | (N, 12) | log₁₀ surface molecular mixing ratios — targets |
| `aux_params.npy` | (N, 11) | Input fluxes & conditions metadata |
| `wavelengths.npy` | (101,) | Altitude axis (km, 0 = surface) |

Raw data: 10 tar.gz archives (`dir_0.tar.gz` … `dir_9.tar.gz` + `Dir_alpha.tar.gz`) + `pyatmos_summary.csv`.
Download from NASA FDL / the INARA project.

---

## 3. Project Structure

```
exo-atmos-retrieval/
│
├── src/                           # Core ML modules
│   ├── data_utils.py              # Loading, splitting, normalisation, PCA, metrics
│   ├── baseline_model.py          # Per-molecule Random Forest model
│   ├── deep_model.py              # CNN1D + Trainer + SpectralDataset
│   └── mlp_model.py               # MLP ablation model
│
├── pipeline/
│   ├── config.yaml                # All configuration (paths, hyperparameters, toggles)
│   └── steps/                     # Numbered pipeline steps — run in order
│       ├── config_loader.py
│       ├── 01_extract.py          # Raw archives → processed numpy arrays
│       ├── 02_feature_engineer.py # Split + normalise + PCA → artifacts
│       ├── 03_train_baseline.py   # Random Forest (≤10k samples)
│       ├── 04_train_deep.py       # 1D CNN training
│       └── 05_evaluate.py         # Unified test eval + comparison report
│
├── experiments/                   # Paper experiment scripts
│   ├── run_scaling_study.py       # R² vs N_train for RF / MLP / CNN
│   ├── run_multiseed.py           # 5-seed statistical validation
│   └── run_mlp_baseline.py        # MLP architecture ablation
│
├── jobs/
│   ├── slurm/                     # SLURM HPC job scripts
│   └── local/
│       └── run_pipeline.sh        # Mac / Linux — runs all steps sequentially
│
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── planet_predictions.ipynb   # Interactive predictions
│   ├── source_data_explorer.ipynb # Raw archive exploration
│   └── visualize.ipynb            # Results visualisation
│
├── paper/
│   ├── main.tex                   # LaTeX source
│   ├── main.pdf                   # Compiled paper
│   ├── references.bib             # Bibliography
│   ├── generate_figures.py        # Regenerate all paper figures from CSVs
│   └── figures/                   # PDF + PNG figures
│
├── results/paper_experiments/     # Experiment result CSVs
│   ├── scaling_study.csv
│   ├── multiseed_summary.csv
│   └── multiseed_results.csv
│
├── dashboard.py                   # Streamlit interactive results dashboard
├── run_baseline.py                # Standalone baseline training script
└── run_deep_model.py              # Standalone deep model training script
```

---

## 4. Environment Setup

### Local (Mac Apple Silicon)

```bash
conda create -n inara_env python=3.11 -y
conda activate inara_env

pip install numpy pandas scikit-learn joblib tqdm pyyaml
pip install torch                          # PyTorch 2.x (MPS-enabled on Apple Silicon)
pip install streamlit plotly jupyter

# Verify MPS (Apple Silicon GPU)
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

### HPC (CUDA)

```bash
conda create -n inara_env python=3.11 -y
conda activate inara_env

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn numpy pandas matplotlib
pip install streamlit plotly pyyaml tqdm joblib

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 5. Pipeline Stages

```
[Step 1] Extract          Raw tar.gz archives → processed numpy arrays
              ↓
[Step 2] Feature Engineer  Train/val/test split → Z-normalise spectra → PCA
              ↓                        ↓
[Step 3] Train Baseline   ──────────── │ ──── Random Forest (≤10k samples)
[Step 4] Train Deep       ─────────────┘ ──── 1D CNN (full training set)
              ↓                ↓
[Step 5] Evaluate         Unified test metrics + baseline vs deep comparison
```

Steps 3 and 4 are independent of each other — on HPC they run **in parallel**.

---

## 6. Running Locally (Mac M5)

```bash
# Full pipeline
bash jobs/local/run_pipeline.sh

# Skip extraction (data already processed)
bash jobs/local/run_pipeline.sh --skip-extract

# Skip extraction + feature engineering
bash jobs/local/run_pipeline.sh --skip-extract --skip-feature-eng
```

**Expected runtimes (Mac M5 Pro, 124K dataset):**

| Step | Time |
|---|---|
| Step 1 — Extraction | ~30–45 min |
| Step 2 — Feature Engineering | ~5–10 min |
| Step 3 — Baseline RF (10k) | < 1 min |
| Step 4 — Deep Model (MPS) | ~20–40 min |
| Step 5 — Evaluation | < 1 min |

---

## 7. Running on HPC (SLURM)

Update `pipeline/config.yaml` before submitting:
- `profiles.hpc.*_dir` → your scratch/work directory paths
- `slurm.email` → your email for job notifications
- `slurm.conda_env` → your environment name
- `slurm.code_dir` → where you cloned the repo

```bash
# Submit full pipeline with dependency chain
bash jobs/slurm/submit_pipeline.sh

# Skip steps already completed
bash jobs/slurm/submit_pipeline.sh --skip-extract
bash jobs/slurm/submit_pipeline.sh --skip-extract --skip-feature-eng
```

---

## 8. Running Individual Steps

```bash
# Step 1 — Extract data
python pipeline/steps/01_extract.py --profile local

# Step 2 — Feature engineering
python pipeline/steps/02_feature_engineer.py --profile local

# Step 3 — Train baseline (saves model)
python pipeline/steps/03_train_baseline.py --profile local --save

# Step 4 — Train deep model (saves checkpoint)
python pipeline/steps/04_train_deep.py --profile local --save

# Step 5 — Evaluate and compare
python pipeline/steps/05_evaluate.py --profile local
```

Use `--profile hpc` for HPC runs. All steps accept `--config` to point to an alternative config file.

---

## 9. Paper Experiments

These reproduce the results in the paper. Run after Step 2 (feature engineering) is complete.

```bash
# Scaling study — trains RF/MLP/CNN at 6 training set sizes
python experiments/run_scaling_study.py

# Multi-seed validation — 5 seeds for CNN and RF
python experiments/run_multiseed.py

# MLP ablation — trains MLP at full scale
python experiments/run_mlp_baseline.py

# Regenerate all paper figures from result CSVs
python paper/generate_figures.py
```

Results are written to `results/paper_experiments/`. Figures are written to `paper/figures/`.

---

## 10. Configuration Reference

`pipeline/config.yaml` controls all pipeline behaviour. Key settings:

```yaml
extraction:
  n_samples: 124000   # samples to extract (reduce for quick tests)

baseline:
  max_train_samples: 10000   # RF cap — always enforced

training:
  epochs:       150
  batch_size:   32
  lr:           0.001
  patience:     30
```

See `pipeline/config.yaml` for the full reference including profile paths and SLURM settings.

---

## 11. Dashboard — Viewing Results

```bash
conda activate inara_env
streamlit run dashboard.py
# Open: http://localhost:8501
```

The dashboard reads from `results/processed/` and auto-detects which steps have completed.

---

## 12. Results & Output Files

```
results/paper_experiments/
├── scaling_study.csv         # R² per molecule per model per N_train
├── multiseed_summary.csv     # Mean ± std R² across 5 seeds
└── multiseed_results.csv     # Per-seed raw results

models/
├── baseline_rf.joblib        # Trained Random Forest
├── cnn1d.pt                  # Trained 1D CNN
└── mlp_model.pt              # Trained MLP

paper/figures/
├── fig2_scaling_curve.pdf    # Learning curves
├── fig3_h2o_scaling.pdf      # H₂O data requirement finding
├── fig4_multiseed_bar.pdf    # Per-molecule R² with error bars
├── fig5_scatter_grid.pdf     # Predicted vs true scatter (CNN)
└── fig6_cnn_mlp_delta.pdf    # CNN − MLP ΔR² ablation
```

---

## Citation

If you use this code or build on our results, please cite:

```bibtex
@article{shinde2026exoatmos,
  author  = {Shinde, Bhalchandra and Shinde, Sandhya and Rajput, Amit},
  title   = {Deep Learning for Exoplanet Atmospheric Retrieval: Scaling,
             Architecture Ablation, and Statistical Validation on the
             {INARA} {ATMOS} Dataset},
  year    = {2026},
  note    = {arXiv preprint}
}
```

Dataset citation: Zorzan et al. (2025), *ApJS* 277:38, [doi:10.3847/1538-4365/ad9230](https://doi.org/10.3847/1538-4365/ad9230)

---

## License

Code released under the MIT License. The INARA ATMOS dataset is subject to NASA FDL terms — see the [original dataset](https://doi.org/10.3847/1538-4365/ad9230) for details.
