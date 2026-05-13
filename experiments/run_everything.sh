#!/bin/bash
# Master run script: experiments → ablations → deliverables
# Run this and walk away. Auto-resumes if interrupted.

set -e

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT=1

echo "============================================================"
echo "  DRO-FAIR Master Run Script"
echo "  Adversarial noise | 3 datasets | 5 alphas | 10 seeds"
echo "============================================================"

# 1. Main experiments (parallel per dataset)
echo ""
echo "[1/4] Starting main experiments (3 datasets in parallel)..."
mkdir -p results/full_adult results/full_credit results/full_lsac

for ds in adult credit lsac; do
    if [ -f "results/full_${ds}/all_results.json" ]; then
        echo "  ${ds}: already complete, skipping"
        continue
    fi
    nohup bash -c "cd $(pwd) && python3 -u experiments/run_experiments.py --datasets ${ds} --n_seeds 10 --output_dir results/full_${ds}" > "results/full_${ds}/run.log" 2>&1 &
    echo "  ${ds}: launched (PID $!)"
done

# Wait for all datasets to finish
echo ""
echo "[2/4] Waiting for experiments to complete..."
while true; do
    complete=0
    for ds in adult credit lsac; do
        if [ -f "results/full_${ds}/all_results.json" ]; then
            complete=$((complete + 1))
        fi
    done
    if [ $complete -eq 3 ]; then
        echo "  All datasets complete!"
        break
    fi
    # Show progress every 5 minutes
    python3 experiments/monitor_dashboard.py
    sleep 300
done

# 2. Merge results
echo ""
echo "[3/4] Merging results..."
python3 experiments/merge_split_results.py

# 3. Ablations
echo ""
echo "[4/4] Running ablations..."
python3 experiments/run_ablations.py --output_dir results/ablations > results/ablations/run.log 2>&1 || true

# 4. Generate deliverables
echo ""
echo "[5/4] Generating deliverables..."
python3 experiments/generate_all_deliverables.py

echo ""
echo "============================================================"
echo "  ALL DONE! Check results/ and figures/ for outputs."
echo "============================================================"
