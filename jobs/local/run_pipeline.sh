#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# INARA Pipeline — Local Runner (Mac M5)
# Runs all enabled pipeline steps sequentially.
# Device: Apple Silicon MPS (auto-detected by PyTorch).
#
# Usage:
#   cd /path/to/inara_vscode
#   bash jobs/local/run_pipeline.sh [--skip-extract] [--skip-feature-eng]
#
# Flags:
#   --skip-extract      Skip step 1 (processed/ already exists)
#   --skip-feature-eng  Skip step 2 (engineered/ already exists)
#
# Each step can also be disabled globally in pipeline/config.yaml:
#   steps:
#     extract: false
#     ...
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CODE_DIR}/pipeline/config.yaml"
PROFILE="local"

SKIP_EXTRACT=false
SKIP_FEAT_ENG=false

for arg in "$@"; do
    case $arg in
        --skip-extract)     SKIP_EXTRACT=true ;;
        --skip-feature-eng) SKIP_FEAT_ENG=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
step_enabled() {
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
print(str(c['steps']['$1']).lower())
"
}

run_step() {
    local STEP_NUM=$1
    local STEP_NAME=$2
    local SCRIPT=$3
    shift 3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step ${STEP_NUM}: ${STEP_NAME}"
    echo "  Started: $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python3 "${SCRIPT}" \
        --config "${CONFIG}" \
        --profile "${PROFILE}" \
        "$@"

    echo "  Finished: $(date '+%H:%M:%S')"
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  INARA ML Pipeline — Local Run (Mac M5 / Apple Silicon MPS)"
echo "  Profile  : ${PROFILE}"
echo "  Config   : ${CONFIG}"
echo "  Started  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${CODE_DIR}"

START_TIME=$SECONDS

# ── Step 1: Extract ───────────────────────────────────────────────────────────
if [[ "$(step_enabled extract)" == "true" && "${SKIP_EXTRACT}" == "false" ]]; then
    run_step 1 "Data Extraction" \
        "${CODE_DIR}/pipeline/steps/01_extract.py"
else
    echo ""
    echo "  Step 1 (extract) skipped."
fi

# ── Step 2: Feature Engineering ───────────────────────────────────────────────
if [[ "$(step_enabled feature_engineer)" == "true" && "${SKIP_FEAT_ENG}" == "false" ]]; then
    run_step 2 "Feature Engineering" \
        "${CODE_DIR}/pipeline/steps/02_feature_engineer.py"
else
    echo ""
    echo "  Step 2 (feature_engineer) skipped."
fi

# ── Step 3: Baseline RF ───────────────────────────────────────────────────────
if [[ "$(step_enabled train_baseline)" == "true" ]]; then
    run_step 3 "Train Baseline (Random Forest)" \
        "${CODE_DIR}/pipeline/steps/03_train_baseline.py" \
        --save
else
    echo ""
    echo "  Step 3 (train_baseline) skipped."
fi

# ── Step 4: Deep Model ────────────────────────────────────────────────────────
if [[ "$(step_enabled train_deep)" == "true" ]]; then
    run_step 4 "Train Deep Model (1D CNN)" \
        "${CODE_DIR}/pipeline/steps/04_train_deep.py" \
        --save
else
    echo ""
    echo "  Step 4 (train_deep) skipped."
fi

# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
if [[ "$(step_enabled evaluate)" == "true" ]]; then
    run_step 5 "Unified Evaluation & Comparison" \
        "${CODE_DIR}/pipeline/steps/05_evaluate.py"
else
    echo ""
    echo "  Step 5 (evaluate) skipped."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pipeline complete!   Total time: ${MINS}m ${SECS}s"
echo "  Results : ${CODE_DIR}/results/processed/"
echo "  Models  : ${CODE_DIR}/models/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
