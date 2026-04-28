#!/bin/bash
#SBATCH --job-name=inara_eval
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/YOUR_USERNAME/inara/logs/05_evaluate_%j.out
#SBATCH --error=/scratch/YOUR_USERNAME/inara/logs/05_evaluate_%j.err
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
echo "  Job: inara_eval   ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME   Date: $(date)"
echo "========================================"

cd ${CODE_DIR}

python pipeline/steps/05_evaluate.py \
    --config pipeline/config.yaml \
    --profile hpc

EXIT_CODE=$?

# ── Copy key results back to /home for safe-keeping ──────────────────────────
# /scratch is purged each semester — back up small result CSVs to /home
RESULTS_SCRATCH=/scratch/YOUR_USERNAME/inara/results
RESULTS_HOME=/home/YOUR_USERNAME/inara_results
mkdir -p ${RESULTS_HOME}
cp ${RESULTS_SCRATCH}/*.csv ${RESULTS_HOME}/ 2>/dev/null && \
    echo "Results backed up to ${RESULTS_HOME}" || \
    echo "WARNING: Could not back up results to home"

echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
