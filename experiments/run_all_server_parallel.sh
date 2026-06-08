#!/bin/bash
# Run ALL server experiments in PARALLEL using background jobs.
# Heavier items (pixel PGD, randinit) get their own sessions;
# lighter items run together.
#
# Usage:
#   tmux new-session -d -s dro_batch 'bash experiments/run_all_server_parallel.sh'
#   tmux attach -t dro_batch
#
# Or manually:
#   bash experiments/run_all_server_parallel.sh

set -uo pipefail

PROJECT_DIR="/data/srujan.sai/DRO-FairML"
FEATURE_CACHE="/data/srujan.sai/utkface_features.npz"
DATA_DIR="/data/srujan.sai/UTKFace"
PYTHON="venv/bin/python3"

cd "$PROJECT_DIR"
mkdir -p logs results figures

# --- Session A: Fast items (items 1, 2, 5, 6) ---
echo "[$(date)] Starting fast items..."

# Item 1: tabular lambda diagnostic (CPU, ~10 min)
$PYTHON experiments/run_lambda_diagnostic.py \
    > logs/server_batch_01_lambda_diag.log 2>&1 &
PID1=$!

# Item 2: UTKFace lambda_max cap (GPU, ~20 min)
$PYTHON experiments/run_utkface_extended.py \
    --mode lambda_max_cap --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/server_batch_02_utkface_lmax.log 2>&1 &
PID2=$!

# Item 5: alpha sweep (GPU, ~20 min)
$PYTHON experiments/run_utkface_extended.py \
    --mode alpha_sweep --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/server_batch_05_utkface_alpha.log 2>&1 &
PID5=$!

# Item 6: fairness PGD (GPU, ~60 min)
$PYTHON experiments/run_utkface_extended.py \
    --mode fairness_pgd --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/server_batch_06_utkface_fpgd.log 2>&1 &
PID6=$!

echo "[$(date)] Fast items launched: PIDs $PID1 $PID2 $PID5 $PID6"
wait $PID1 $PID2 $PID5 $PID6
echo "[$(date)] Fast items complete."

# --- Session B: Heavy items (items 3, 4) ---
# These are the slowest; run sequentially to avoid GPU OOM.
echo "[$(date)] Starting heavy items..."

# Item 3: pixel-space PGD (GPU, ~120 min)
$PYTHON experiments/run_utkface_pixel_pgd.py \
    --data_dir "$DATA_DIR" --n_seeds 5 --alphas 0.1 0.2 \
    > logs/server_batch_03_pixel_pgd.log 2>&1

# Item 4: random-init ResNet18 (GPU, ~150 min)
$PYTHON experiments/run_utkface_randinit.py \
    --data_dir "$DATA_DIR" --n_seeds 5 --alphas 0.1 0.2 \
    > logs/server_batch_04_randinit.log 2>&1

echo "[$(date)] Heavy items complete."

# --- Post-processing ---
echo "[$(date)] Generating figures and report..."
$PYTHON experiments/aggregate_all_results.py
$PYTHON experiments/generate_all_figures.py
$PYTHON experiments/check_server_progress.py

echo "[$(date)] ALL DONE."
