#!/bin/bash
#SBATCH --job-name=inara_multiseed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/exp_multiseed_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/exp_multiseed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@institution.edu

# Trains RF + CNN + MLP at 5 seeds each = 15 training runs.
# CNN/MLP: full training split each run (~86K samples on HPC).
# RF: capped at 10K each run (consistent with main pipeline baseline).
# Use --resume to safely re-submit if the job is preempted.

CODE_DIR=/home/YOUR_USERNAME/exo-atmos-retrieval
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/YOUR_USERNAME/inara/logs

echo "========================================"
echo "  Job: inara_multiseed   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "GPU detected." \
    || echo "WARNING: nvidia-smi not found"

cd ${CODE_DIR}

python experiments/run_multiseed.py \
    --config  pipeline/config.yaml \
    --profile hpc \
    --models  rf cnn mlp \
    --seeds   42 123 456 789 1337 \
    --resume

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
