#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# INARA Pipeline — SLURM Submission Script
# HPC cluster
#
# Usage:
#   cd /home/YOUR_USERNAME/exo-atmos-retrieval
#   bash jobs/slurm/submit_pipeline.sh [--skip-extract] [--skip-feature-eng]
#
# Step dependency chain:
#
#   [1] extract  →  [2] feature_engineer  →  [3] baseline  ─┐
#                                          →  [4] deep      ─┼→  [5] evaluate
#
#   Steps 3 and 4 run in PARALLEL (both depend on step 2 only).
#   Step 5 waits for both 3 and 4 to succeed.
#
# Flags:
#   --skip-extract      Skip step 1 (processed/ already exists on scratch)
#   --skip-feature-eng  Skip step 2 (engineered/ already exists on scratch)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SKIP_EXTRACT=false
SKIP_FEAT_ENG=false

for arg in "$@"; do
    case $arg in
        --skip-extract)     SKIP_EXTRACT=true ;;
        --skip-feature-eng) SKIP_FEAT_ENG=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Config: read from pipeline/config.yaml ────────────────────────────────────
# These must match pipeline/config.yaml  →  steps section
STEP_EXTRACT=$(python3 -c "
import yaml
with open('${CODE_DIR}/pipeline/config.yaml') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['extract']).lower())
")
STEP_FEAT=$(python3 -c "
import yaml
with open('${CODE_DIR}/pipeline/config.yaml') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['feature_engineer']).lower())
")
STEP_BL=$(python3 -c "
import yaml
with open('${CODE_DIR}/pipeline/config.yaml') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['train_baseline']).lower())
")
STEP_DEEP=$(python3 -c "
import yaml
with open('${CODE_DIR}/pipeline/config.yaml') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['train_deep']).lower())
")
STEP_EVAL=$(python3 -c "
import yaml
with open('${CODE_DIR}/pipeline/config.yaml') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['evaluate']).lower())
")

echo "======================================================="
echo "  INARA Pipeline — SLURM Job Submission"
echo "  Date: $(date)"
echo "======================================================="
echo "  Steps enabled (from config.yaml):"
echo "    extract:          ${STEP_EXTRACT}"
echo "    feature_engineer: ${STEP_FEAT}"
echo "    train_baseline:   ${STEP_BL}"
echo "    train_deep:       ${STEP_DEEP}"
echo "    evaluate:         ${STEP_EVAL}"
echo ""
[[ $SKIP_EXTRACT  == true ]] && echo "  --skip-extract      : overriding extract step"
[[ $SKIP_FEAT_ENG == true ]] && echo "  --skip-feature-eng  : overriding feature_engineer step"
echo "======================================================="

JID_FEAT=""   # job ID for feature_engineer (needed as dependency for steps 3+4)
JID_BL=""
JID_DEEP=""

# ── Step 1: Extract ───────────────────────────────────────────────────────────
JID_EXTRACT=""
if [[ "${STEP_EXTRACT}" == "true" && "${SKIP_EXTRACT}" == "false" ]]; then
    JID_EXTRACT=$(sbatch --parsable "${SCRIPT_DIR}/01_extract.sh")
    echo "Submitted Step 1 (extract)         → Job ID: ${JID_EXTRACT}"
else
    echo "Step 1 (extract) skipped."
fi

# ── Step 2: Feature Engineering ───────────────────────────────────────────────
if [[ "${STEP_FEAT}" == "true" && "${SKIP_FEAT_ENG}" == "false" ]]; then
    if [[ -n "${JID_EXTRACT}" ]]; then
        JID_FEAT=$(sbatch --parsable \
            --dependency=afterok:${JID_EXTRACT} \
            "${SCRIPT_DIR}/02_feature_engineer.sh")
    else
        JID_FEAT=$(sbatch --parsable "${SCRIPT_DIR}/02_feature_engineer.sh")
    fi
    echo "Submitted Step 2 (feature_engineer) → Job ID: ${JID_FEAT}"
else
    echo "Step 2 (feature_engineer) skipped."
fi

# ── Step 3: Baseline (depends on step 2) ──────────────────────────────────────
if [[ "${STEP_BL}" == "true" ]]; then
    if [[ -n "${JID_FEAT}" ]]; then
        JID_BL=$(sbatch --parsable \
            --dependency=afterok:${JID_FEAT} \
            "${SCRIPT_DIR}/03_train_baseline.sh")
    else
        JID_BL=$(sbatch --parsable "${SCRIPT_DIR}/03_train_baseline.sh")
    fi
    echo "Submitted Step 3 (baseline)         → Job ID: ${JID_BL}"
else
    echo "Step 3 (train_baseline) skipped."
fi

# ── Step 4: Deep model (depends on step 2, PARALLEL with step 3) ──────────────
if [[ "${STEP_DEEP}" == "true" ]]; then
    if [[ -n "${JID_FEAT}" ]]; then
        JID_DEEP=$(sbatch --parsable \
            --dependency=afterok:${JID_FEAT} \
            "${SCRIPT_DIR}/04_train_deep.sh")
    else
        JID_DEEP=$(sbatch --parsable "${SCRIPT_DIR}/04_train_deep.sh")
    fi
    echo "Submitted Step 4 (deep model)       → Job ID: ${JID_DEEP}"
else
    echo "Step 4 (train_deep) skipped."
fi

# ── Step 5: Evaluate (waits for BOTH step 3 and 4) ───────────────────────────
if [[ "${STEP_EVAL}" == "true" ]]; then
    DEP=""
    [[ -n "${JID_BL}"   ]] && DEP="afterok:${JID_BL}"
    if [[ -n "${JID_DEEP}" ]]; then
        if [[ -n "${DEP}" ]]; then
            DEP="${DEP}:${JID_DEEP}"
        else
            DEP="afterok:${JID_DEEP}"
        fi
    fi

    if [[ -n "${DEP}" ]]; then
        JID_EVAL=$(sbatch --parsable \
            --dependency=${DEP} \
            "${SCRIPT_DIR}/05_evaluate.sh")
    else
        JID_EVAL=$(sbatch --parsable "${SCRIPT_DIR}/05_evaluate.sh")
    fi
    echo "Submitted Step 5 (evaluate)         → Job ID: ${JID_EVAL}"
else
    echo "Step 5 (evaluate) skipped."
fi

echo ""
echo "======================================================="
echo "  Monitor with:  squeue -u YOUR_USERNAME"
echo "  Logs at:        /scratch/YOUR_USERNAME/inara/logs/"
echo "  Results at:     /scratch/YOUR_USERNAME/inara/results/processed/"
echo "  (Backed up to:  /home/YOUR_USERNAME/inara_results/ after step 5)"
echo "======================================================="
