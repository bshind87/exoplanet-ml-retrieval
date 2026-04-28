#!/bin/bash
#SBATCH --job-name=inara_extract
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/01_extract_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/01_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@institution.edu
# Uncomment and set if your account requires it:
# #SBATCH --account=<your_account>

# ── Environment ──────────────────────────────────────────────────────────────
CODE_DIR=/home/YOUR_USERNAME/exo-atmos-retrieval
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/YOUR_USERNAME/inara/logs

echo "========================================"
echo "  Job: inara_extract   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

cd ${CODE_DIR}

python pipeline/steps/01_extract.py \
    --config pipeline/config.yaml \
    --profile hpc

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
