#!/bin/bash
#SBATCH --job-name=inara_feat_eng
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/02_feature_engineer_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/02_feature_engineer_%j.err
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
echo "  Job: inara_feat_eng   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

cd ${CODE_DIR}

python pipeline/steps/02_feature_engineer.py \
    --config pipeline/config.yaml \
    --profile hpc

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
