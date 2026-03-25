#!/bin/bash
# Run all safety experiments on PGR.
# Designed for A100 on Colab. Each run: ~30-90 min.
#
# Usage:
#   bash safety/run_experiments.sh [results_dir]
#
# Total: 3 methods x 3 seeds = 9 runs (~5-14 hours)

# Default to Google Drive on Colab, fallback to local
if [ -d "/content/drive/MyDrive" ]; then
    DEFAULT_DIR="/content/drive/MyDrive/pgr/safety_results"
else
    DEFAULT_DIR="./safety_results"
fi
RESULTS_DIR="${1:-$DEFAULT_DIR}"
GIN_CONFIG="config/online/sac_cond_synther_dmc.gin"
ENV="cheetah-run-v0"
SEEDS="42 123 456"
VEL_THRESHOLD=7.0

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " PGR Safety Experiments"
echo " Results: $RESULTS_DIR"
echo " Env: $ENV, Velocity threshold: $VEL_THRESHOLD"
echo "============================================================"

# ── Experiment A & B: Standard comparison (3 seeds) ──────────────────────────

for SEED in $SEEDS; do
    echo ""
    echo "=== SAC baseline, seed=$SEED ==="
    python safety/online_cost_cond.py \
        --env "$ENV" \
        --seed "$SEED" \
        --gin_config_files config/online/sac.gin \
        --gin_params "redq_sac.disable_diffusion = True" \
        --velocity_threshold "$VEL_THRESHOLD" \
        --results_folder "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/sac_seed${SEED}.log"

    echo ""
    echo "=== PGR, seed=$SEED ==="
    python safety/online_cost_cond.py \
        --env "$ENV" \
        --seed "$SEED" \
        --gin_config_files "$GIN_CONFIG" \
        --gin_params "redq_sac.cond_top_frac = 0.25" \
        --velocity_threshold "$VEL_THRESHOLD" \
        --results_folder "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/pgr_seed${SEED}.log"

    echo ""
    echo "=== PGR+Memory, seed=$SEED ==="
    python safety/online_cost_cond.py \
        --env "$ENV" \
        --seed "$SEED" \
        --gin_config_files "$GIN_CONFIG" \
        --gin_params "redq_sac.cond_top_frac = 0.25" \
        --velocity_threshold "$VEL_THRESHOLD" \
        --use_rare_buffer \
        --use_lagrangian \
        --results_folder "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/pgr_memory_seed${SEED}.log"
done

echo ""
echo "============================================================"
echo " All experiments complete!"
echo " Results in: $RESULTS_DIR"
echo "============================================================"
