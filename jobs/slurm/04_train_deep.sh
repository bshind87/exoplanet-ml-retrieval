#!/bin/bash
#SBATCH --job-name=inara_deep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/04_train_deep_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/04_train_deep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@institution.edu
# #SBATCH --account=<your_account>

# ── Environment ──────────────────────────────────────────────────────────────
CODE_DIR=/home/YOUR_USERNAME/exo-atmos-retrieval
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/YOUR_USERNAME/inara/logs

echo "========================================"
echo "  Job: inara_deep   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

# Verify GPU is available
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "GPU detected." \
    || echo "WARNING: nvidia-smi not found — check partition/gres allocation"

cd ${CODE_DIR}

python pipeline/steps/04_train_deep.py \
    --config pipeline/config.yaml \
    --profile hpc \
    --save

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
