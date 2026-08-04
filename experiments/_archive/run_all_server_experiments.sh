#!/bin/bash
# Master batch script — run ALL remaining experiments on flair2.iitgn.ac.in
# ------------------------------------------------------------------------
# This script runs the 6-item next-week list from TODAY_REPORT.md in the
# optimal order (cheapest / most informative first).
#
# Usage:
#   cd /data/srujan.sai/DRO-FairML
#   git pull
#   bash experiments/run_all_server_experiments.sh
#
# All logs go to logs/server_batch_*.log. Results go to results/.
# ------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/data/srujan.sai/DRO-FairML"
FEATURE_CACHE="/data/srujan.sai/utkface_features.npz"
DATA_DIR="/data/srujan.sai/UTKFace"
PYTHON="venv/bin/python3"

# Sanity checks
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: PROJECT_DIR not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

if [ ! -f "$FEATURE_CACHE" ]; then
    echo "WARNING: UTKFace feature cache not found at $FEATURE_CACHE"
    echo "  Some experiments will fail. Extract features first:"
    echo "    $PYTHON scripts/extract_utkface_features.py"
fi

mkdir -p logs results figures

# ---------------------------------------------------------------------------
# ITEM 1 — Tabular lambda trajectory diagnostic (CPU, ~10 min)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 1/6: Lambda trajectory diagnostic (tabular)"
echo "=============================================="
$PYTHON experiments/run_lambda_diagnostic.py \
    > logs/server_batch_01_lambda_diag.log 2>&1
echo "  -> results/lambda_diagnostic.json"

# ---------------------------------------------------------------------------
# ITEM 2 — UTKFace lambda_max cap (H3 test, ~20 min on GPU)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 2/6: UTKFace lambda_max cap (H3)"
echo "=============================================="
$PYTHON experiments/run_utkface_extended.py \
    --mode lambda_max_cap \
    --feature_cache "$FEATURE_CACHE" \
    --n_seeds 5 \
    > logs/server_batch_02_utkface_lmax.log 2>&1
echo "  -> results/utkface_lambda_max_cap.json"

# ---------------------------------------------------------------------------
# ITEM 5 — UTKFace alpha sweep {0.3, 0.4} (~20 min on GPU)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 5/6: UTKFace alpha sweep {0.3, 0.4}"
echo "=============================================="
$PYTHON experiments/run_utkface_extended.py \
    --mode alpha_sweep \
    --feature_cache "$FEATURE_CACHE" \
    --n_seeds 5 \
    > logs/server_batch_05_utkface_alpha.log 2>&1
echo "  -> results/utkface_alpha_sweep.json"

# ---------------------------------------------------------------------------
# ITEM 6 — FairnessTargetedPGD on UTKFace (~60 min on GPU)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 6/6: FairnessTargetedPGD on UTKFace"
echo "=============================================="
$PYTHON experiments/run_utkface_extended.py \
    --mode fairness_pgd \
    --feature_cache "$FEATURE_CACHE" \
    --n_seeds 5 \
    > logs/server_batch_06_utkface_fpgd.log 2>&1
echo "  -> results/utkface_fairness_pgd.json"

# ---------------------------------------------------------------------------
# ITEM 3 — Pixel-space PGD (H2 test, ~120 min on GPU, HEAVY)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 3/6: Pixel-space PGD (H2, HEAVY)"
echo "=============================================="
$PYTHON experiments/run_utkface_pixel_pgd.py \
    --data_dir "$DATA_DIR" \
    --n_seeds 5 --alphas 0.1 0.2 \
    > logs/server_batch_03_pixel_pgd.log 2>&1
echo "  -> results/utkface_pixel_pgd.json"

# ---------------------------------------------------------------------------
# ITEM 4 — Random-init ResNet18 (H1 test, ~150 min on GPU, HEAVIEST)
# ---------------------------------------------------------------------------
echo "=============================================="
echo "ITEM 4/6: Random-init ResNet18 (H1, HEAVIEST)"
echo "=============================================="
$PYTHON experiments/run_utkface_randinit.py \
    --data_dir "$DATA_DIR" \
    --n_seeds 5 --alphas 0.1 0.2 \
    > logs/server_batch_04_randinit.log 2>&1
echo "  -> results/utkface_randinit.json"

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------
echo "=============================================="
echo "POST-PROCESSING: Plots and aggregation"
echo "=============================================="

# Plot lambda diagnostic (if item 1 produced results)
if [ -f "results/lambda_diagnostic.json" ]; then
    $PYTHON experiments/plot_lambda_diagnostic.py \
        > logs/server_batch_plot_lambda.log 2>&1
    echo "  -> figures/fig11_lambda_diagnostic.pdf"
fi

# Aggregate all UTKFace results into one JSON
$PYTHON -c "
import json, os, glob

files = {
    'baseline': 'results/utkface_results.json',
    'lambda_max_cap': 'results/utkface_lambda_max_cap.json',
    'alpha_sweep': 'results/utkface_alpha_sweep.json',
    'fairness_pgd': 'results/utkface_fairness_pgd.json',
    'pixel_pgd': 'results/utkface_pixel_pgd.json',
    'randinit': 'results/utkface_randinit.json',
}

agg = {}
for key, path in files.items():
    if os.path.exists(path):
        with open(path) as f:
            agg[key] = json.load(f)
        print(f'  Loaded {key}: {len(agg[key])} runs')
    else:
        print(f'  Missing {key}: {path}')

out = 'results/utkface_all_results.json'
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'  -> {out}')
" > logs/server_batch_aggregate.log 2>&1

echo ""
echo "=============================================="
echo "ALL DONE. Check logs/ for per-item logs."
echo "=============================================="
