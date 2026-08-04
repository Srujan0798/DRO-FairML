#!/bin/bash
# ============================================================================
# UNIFIED SERVER SCRIPT — run EVERYTHING after the DP attack fix
# ============================================================================
# This single script runs:
#   1. Tabular FairnessTargetedPGD re-run (270 runs, ~2-3 hours on CPU)
#   2. UTKFace lambda_max cap (H3 test)
#   3. UTKFace alpha sweep {0.3, 0.4}
#   4. UTKFace FairnessTargetedPGD
#   5. UTKFace pixel-space PGD (H2)
#   6. UTKFace random-init ResNet18 (H1)
#   7. Aggregate results + generate all figures
#
# Usage:
#   ssh flair2.iitgn.ac.in
#   cd /data/srujan.sai/DRO-FairML && git pull
#   bash experiments/run_everything.sh
#
# Or with tmux (recommended):
#   tmux new-session -d -s dro_full 'bash experiments/run_everything.sh'
#   tmux attach -t dro_full
# ============================================================================

set -euo pipefail

PROJECT_DIR="/data/srujan.sai/DRO-FairML"
FEATURE_CACHE="/data/srujan.sai/utkface_features.npz"
DATA_DIR="/data/srujan.sai/UTKFace"
PYTHON="venv/bin/python3"

cd "$PROJECT_DIR"
mkdir -p logs results figures

echo "========================================================================"
echo "DRO-FairML — COMPLETE EXPERIMENT RUN (post DP-attack fix)"
echo "Started: $(date)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# PHASE 1: Tabular FairnessTargetedPGD re-run (270 runs)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 1/7: Tabular FairnessTargetedPGD re-run (270 runs)"
echo "========================================================================"
echo "This regenerates results/fairness_pgd_results.json with the FIXED attack."
echo ""
$PYTHON experiments/run_fairness_pgd.py \
    > logs/run_everything_01_tabular_fpgd.log 2>&1
echo "  -> results/fairness_pgd_results.json"

# ---------------------------------------------------------------------------
# PHASE 2: UTKFace lambda_max cap (H3 test)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 2/7: UTKFace lambda_max cap (H3 test)"
echo "========================================================================"
$PYTHON experiments/run_utkface_extended.py \
    --mode lambda_max_cap --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/run_everything_02_utkface_lmax.log 2>&1
echo "  -> results/utkface_lambda_max_cap.json"

# ---------------------------------------------------------------------------
# PHASE 3: UTKFace alpha sweep {0.3, 0.4}
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 3/7: UTKFace alpha sweep {0.3, 0.4}"
echo "========================================================================"
$PYTHON experiments/run_utkface_extended.py \
    --mode alpha_sweep --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/run_everything_03_utkface_alpha.log 2>&1
echo "  -> results/utkface_alpha_sweep.json"

# ---------------------------------------------------------------------------
# PHASE 4: UTKFace FairnessTargetedPGD
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 4/7: UTKFace FairnessTargetedPGD"
echo "========================================================================"
$PYTHON experiments/run_utkface_extended.py \
    --mode fairness_pgd --feature_cache "$FEATURE_CACHE" --n_seeds 5 \
    > logs/run_everything_04_utkface_fpgd.log 2>&1
echo "  -> results/utkface_fairness_pgd.json"

# ---------------------------------------------------------------------------
# PHASE 5: Pixel-space PGD (H2, HEAVY ~120 min)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 5/7: Pixel-space PGD (H2, HEAVY)"
echo "========================================================================"
$PYTHON experiments/run_utkface_pixel_pgd.py \
    --data_dir "$DATA_DIR" --n_seeds 5 --alphas 0.1 0.2 \
    > logs/run_everything_05_pixel_pgd.log 2>&1
echo "  -> results/utkface_pixel_pgd.json"

# ---------------------------------------------------------------------------
# PHASE 6: Random-init ResNet18 (H1, HEAVIEST ~150 min)
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 6/7: Random-init ResNet18 (H1, HEAVIEST)"
echo "========================================================================"
$PYTHON experiments/run_utkface_randinit.py \
    --data_dir "$DATA_DIR" --n_seeds 5 --alphas 0.1 0.2 \
    > logs/run_everything_06_randinit.log 2>&1
echo "  -> results/utkface_randinit.json"

# ---------------------------------------------------------------------------
# PHASE 7: Aggregation + Figures
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "PHASE 7/7: Aggregate results and generate figures"
echo "========================================================================"

$PYTHON experiments/aggregate_all_results.py \
    > logs/run_everything_07_aggregate.log 2>&1
echo "  -> results/ALL_RESULTS_SUMMARY.md"
echo "  -> results/utkface_all_results.json"

$PYTHON experiments/generate_all_figures.py \
    > logs/run_everything_08_figures.log 2>&1
echo "  -> figures/fig12-fig17.pdf/png"

$PYTHON experiments/check_server_progress.py \
    > logs/run_everything_09_progress.log 2>&1

echo ""
echo "========================================================================"
echo "ALL DONE. Finished: $(date)"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - results/fairness_pgd_results.json       (tabular, FIXED attack)"
echo "  - results/utkface_lambda_max_cap.json     (H3)"
echo "  - results/utkface_alpha_sweep.json        (alpha sweep)"
echo "  - results/utkface_fairness_pgd.json       (FPGD on images)"
echo "  - results/utkface_pixel_pgd.json          (H2)"
echo "  - results/utkface_randinit.json           (H1)"
echo "  - results/ALL_RESULTS_SUMMARY.md          (human-readable report)"
echo "  - figures/fig11-fig17.pdf/png             (publication figures)"
echo ""
