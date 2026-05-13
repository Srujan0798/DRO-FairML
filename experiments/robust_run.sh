#!/bin/bash
# Robust experiment runner with auto-restart and unbuffered output

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Suppress sklearn multiprocessing warnings
export LOKY_MAX_CPU_COUNT=1

LOG="results/robust_run.log"
mkdir -p results

echo "=== Starting robust experiment run at $(date) ===" >> "$LOG"

while true; do
    # Check if checkpoint exists and how many done
    if [ -f results/checkpoint.pkl ]; then
        N=$(python3 -c "import pickle; d=pickle.load(open('results/checkpoint.pkl','rb')); print(len(d.get('completed_keys',[])))" 2>/dev/null || echo 0)
        echo "$(date) - Resuming from checkpoint: $N/150 completed" | tee -a "$LOG"
    else
        echo "$(date) - Starting fresh" | tee -a "$LOG"
        N=0
    fi

    if [ "$N" -ge 150 ]; then
        echo "$(date) - All 150 experiments complete!" | tee -a "$LOG"
        break
    fi

    # Run experiments; if it exits with error, we restart
    echo "$(date) - Launching run_experiments.py..." | tee -a "$LOG"
    python3 -u experiments/run_experiments.py --n_seeds 10 >> "$LOG" 2>&1
    EXIT=$?
    echo "$(date) - Process exited with code $EXIT" | tee -a "$LOG"

    # Brief pause before restart
    sleep 5
done
