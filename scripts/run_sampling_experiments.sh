#!/bin/bash
# Run fire model pipeline for each new sampling strategy.
# Modifies config.yaml strategy key, rebuilds panel, trains, evaluates + auto-snapshots.
set -e

cd /home/mmann1123/extra_space/fire_model
CONFIG=config.yaml

run_experiment() {
    local strategy=$1
    local run_id=$2
    local notes=$3

    echo "============================================================"
    echo "EXPERIMENT: $run_id (strategy=$strategy)"
    echo "============================================================"

    # Switch strategy in config.yaml
    sed -i "s/^  strategy: .*/  strategy: $strategy/" "$CONFIG"
    echo "Config updated: strategy=$strategy"
    grep 'strategy:' "$CONFIG"

    # 1. Build panel
    echo "--- Building panel ---"
    conda run -n deep_field python scripts/01_build_panel.py --force

    # 2. Train
    echo "--- Training ---"
    conda run -n deep_field python scripts/02_train_model.py

    # 3. Evaluate + auto-snapshot
    echo "--- Evaluating + snapshotting ---"
    conda run -n deep_field python scripts/03_evaluate.py --run-id "$run_id" --notes "$notes"

    echo "DONE: $run_id"
    echo ""
}

# Run the 3 new strategies
run_experiment "matched_ratio" "v3-matched-ratio" \
    "Matched ratio sampling: 10 negatives per positive, random subsample from all valid pixels"

run_experiment "temporal_thin" "v4-temporal-thin" \
    "Temporal thinning: grid spacing 5 + every 3rd month for negatives"

run_experiment "random_subsample" "v5-random-subsample" \
    "Random subsample: 1% of all valid negative pixel-months"

# Restore original strategy
sed -i "s/^  strategy: .*/  strategy: grid_thin/" "$CONFIG"
echo "Config restored to grid_thin"
echo "All experiments complete."
