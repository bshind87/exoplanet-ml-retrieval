#!/bin/bash
#SBATCH --job-name=inara_mlp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/exp_mlp_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/exp_mlp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@institution.edu

CODE_DIR=/home/YOUR_USERNAME/exo-atmos-retrieval
CONDA_ENV=inara_env

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

mkdir -p /scratch/YOUR_USERNAME/inara/logs

echo "========================================"
echo "  Job: inara_mlp   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "GPU detected." \
    || echo "WARNING: nvidia-smi not found"

cd ${CODE_DIR}

python experiments/run_mlp_baseline.py \
    --config pipeline/config.yaml \
    --profile hpc \
    --save

EXIT_CODE=$?
echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
