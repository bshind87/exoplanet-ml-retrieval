#!/bin/bash
#SBATCH --job-name=inara_scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/exp_scaling_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/exp_scaling_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@institution.edu

# Runs all three models (RF, MLP, CNN) at all scale points sequentially.
# With 6 scale points this takes up to 24h on a GPU node.
# Use --resume to safely re-submit if the job is preempted.

CODE_DIR=/home/YOUR_USERNAME/exo-atmos-retrieval
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/YOUR_USERNAME/inara/logs

echo "========================================"
echo "  Job: inara_scaling   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "GPU detected." \
    || echo "WARNING: nvidia-smi not found"

cd ${CODE_DIR}

python experiments/run_scaling_study.py \
    --config  pipeline/config.yaml \
    --profile hpc \
    --models  rf mlp cnn \
    --scales  1000 5000 10000 25000 50000 full \
    --resume

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
