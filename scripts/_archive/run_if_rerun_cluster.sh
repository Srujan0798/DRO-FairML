#!/usr/bin/env bash
# Cluster job: re-run the IF-attack third of the canonical grid (Agent A, step 5).
#
# The IF metric was degenerate in the original runs (≈1e-10 everywhere) due to a
# threshold-calibration bug in src/evaluation/metrics.py. The metric is now fixed
# (cosine-based, aligned attack/eval k-NN graph). This job regenerates the 180
# missing IF-attack rows (3 datasets × 5 alphas × 6 seeds × 2 methods) and appends
# them to results/canonical_tau1.json.
#
# Each (dataset, alpha, method) config takes ~3-6 min on CPU; the full sweep is
# ~15 h on a single CPU and should run on a cluster node (GPU not required for the
# tabular FairnessTargetedPGD task). run_fairness_pgd.py supports resume: it loads
# existing rows from canonical_tau1.json and only runs missing (dataset,alpha,seed,
# attack,method) keys, so the job is safe to relaunch.
#
# Usage:
#   sbatch run_if_rerun_cluster.sh        # cluster (SLURM)
#   bash run_if_rerun_cluster.sh           # interactive / single node

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Config (matches the canonical grid) ---
Tau=1.0
K_INNER=10          # hardcoded inside run_fairness_pgd.py; do NOT pass --k-inner
EPOCHS=60
PGD_STEPS=20
LAMBDA_INIT=0.0
SEEDS=6
DATASETS="adult credit lsac"
ALPHAS="0.0 0.1 0.2 0.3 0.4"
ATTACKS="if"        # only the missing IF-attack third
METHODS="naive dro"

# --- Python environment ---
if [ -d "venv" ]; then source venv/bin/activate; fi
python3 -c "import torch, numpy, scipy" || { echo "activate your venv first"; exit 1; }

echo "=== IF-attack re-run: datasets=$DATASETS attacks=$ATTACKS alphas=$ALPHAS seeds=$SEEDS ==="
python3 experiments/run_fairness_pgd.py \
    --datasets $DATASETS \
    --attacks $ATTACKS \
    --alphas $ALPHAS \
    --methods $METHODS \
    --n_seeds $SEEDS

echo "=== IF rows now in canonical: ==="
python3 -c "import json;r=json.load(open('results/canonical_tau1.json'));print('total',len(r),'| if-attack',sum(1 for x in r if x['attack']=='if'))"

# --- After the run: regenerate all downstream artifacts ---
echo "=== Regenerating tables + figures + PDFs ==="
python3 experiments/generate_report_tables.py
python3 experiments/canonical_to_all_results.py   # dp panels (existing)
# For IF-attack panels, re-run with attack=if once available:
#   python3 -c "import sys;sys.argv=['x'];exec(open('experiments/canonical_to_all_results.py').read().replace('attack = \"dp\"','attack = \"if\"'))"
python3 experiments/generate_figures.py
( tectonic -X compile paper/main.tex && echo "paper OK" ) || echo "paper build failed"
( tectonic -X compile report/report.tex && echo "report OK" ) || echo "report build failed"

echo "=== DONE ==="
