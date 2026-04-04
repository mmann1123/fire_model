#!/usr/bin/env bash
# Run all fire model types: rebuild panel, train, evaluate, snapshot.
#
# Usage:
#   conda run -n deep_field bash scripts/run_all_models.sh
#   # Or just:
#   bash scripts/run_all_models.sh
#
# Each model overwrites outputs/model/track{A,B}/lr_calibrated.pkl,
# then 03_evaluate.py snapshots the results under --run-id before
# the next model overwrites them.

set -euo pipefail

CONDA="conda run --no-capture-output -n deep_field"
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/run_all_models_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

# ============================================================
# 0. Rebuild panel (with inc_num for GroupKFold)
# ============================================================
#log "=== Step 0: Rebuilding panel ==="
#$CONDA python scripts/01_build_panel.py --force 2>&1 | tee -a "$MASTER_LOG"
#log "Panel build complete."

# ============================================================
# Helper: train + evaluate + snapshot one model
# ============================================================
run_model() {
    local MODEL_TYPE=$1
    local RUN_ID=$2
    local NOTES=$3
    shift 3
    local EXTRA_TRAIN_ARGS=("$@")

    log ""
    log "============================================================"
    log "=== ${RUN_ID}: ${MODEL_TYPE} ==="
    log "============================================================"

    # Train
    log "Training ${MODEL_TYPE}..."
    $CONDA python scripts/02_train_model.py \
        --model-type "$MODEL_TYPE" \
        "${EXTRA_TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$MASTER_LOG"
    log "Training complete."

    # Evaluate + snapshot
    log "Evaluating ${RUN_ID}..."
    $CONDA python scripts/03_evaluate.py \
        --run-id "$RUN_ID" \
        --notes "$NOTES" \
        2>&1 | tee -a "$MASTER_LOG"
    log "Evaluation complete."

    log "=== ${RUN_ID} done ==="
}

# ============================================================
# 1. Baseline logistic regression (no tuning — reference)
# ============================================================
# SKIP — already completed (v4-logreg-baseline snapshot exists)
# run_model "logistic_regression" \
#     "v4-logreg-baseline" \
#     "Baseline LogReg via new modular framework, verify matches v3-matched-ratio"

# ============================================================
# 2. ElasticNet logistic regression (with tuning, StratifiedKFold)
# ============================================================
# SKIP — already completed (v5-elasticnet-fix snapshot exists)
# run_model "elasticnet_logreg" \
#     "v5-elasticnet-fix" \
#     "ElasticNet LogReg with Optuna tuning, manual CV loop (clone bug fixed)" \
#     --tune --n-trials 100 --cv-folds 3

# ============================================================
# 3. LightGBM (with tuning, GroupKFold by INC_NUM)
# ============================================================
# SKIP — already completed (v7-lgbm-groupkfold snapshot exists)
# run_model "lightgbm" \
#     "v7-lgbm-groupkfold" \
#     "LightGBM GPU with Optuna tuning, GroupKFold by INC_NUM" \
#     --tune --n-trials 100 --cv-folds 3 --tune-subsample 0.2

# ============================================================
# 4. Random Forest (with tuning, GroupKFold by INC_NUM)
# ============================================================
# SKIP — already completed (v8-rf-groupkfold snapshot exists)
# run_model "random_forest" \
#     "v8-rf-groupkfold" \
#     "RandomForest with Optuna tuning, GroupKFold by INC_NUM" \
#     --tune --n-trials 30 --cv-folds 3 --tune-subsample 0.2

# ============================================================
# 5. TabNet (with tuning)
# ============================================================
run_model "tabnet" \
    "v9-tabnet" \
    "TabNet GPU with Optuna tuning, 20% subsample" \
    --tune --n-trials 30 --cv-folds 3 --tune-subsample 0.2

# ============================================================
# 6. Ecoregion LogReg (with tuning, StratifiedKFold)
# ============================================================
run_model "ecoregion_logreg" \
    "v10-ecoregion" \
    "Per-ecoregion LogReg with Optuna tuning" \
    --tune --n-trials 20 --cv-folds 3

# ============================================================
# Summary: compare all runs
# ============================================================
log ""
log "============================================================"
log "=== All models complete. Comparing snapshots... ==="
log "============================================================"

$CONDA python -c "
import json
from pathlib import Path

snap_dir = Path('snapshots')
runs = [
    'v4-logreg-baseline',
    'v5-elasticnet-fix',
    'v7-lgbm-groupkfold',
    'v8-rf-groupkfold',
    'v9-tabnet',
    'v10-ecoregion',
]

print(f'{\"Run ID\":<25s} {\"AUC-A\":>8s} {\"AUC-B\":>8s} {\"Delta\":>8s}')
print('-' * 55)

for rid in runs:
    mpath = snap_dir / rid / 'manifest.json'
    if not mpath.exists():
        print(f'{rid:<25s} --- snapshot not found ---')
        continue
    m = json.load(open(mpath))
    auc_a = m.get('trackA_overall_auc', 0)
    auc_b = m.get('trackB_overall_auc', 0)
    delta = auc_a - auc_b
    print(f'{rid:<25s} {auc_a:8.4f} {auc_b:8.4f} {delta:+8.4f}')
" 2>&1 | tee -a "$MASTER_LOG"

log ""
log "Master log: ${MASTER_LOG}"
log "Done."
