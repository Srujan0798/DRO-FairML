#!/usr/bin/env bash
set -u
cd /Users/srujansai/Desktop/DRO-FairML
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ABLATION_WORKERS=0
LOG=logs/a4_sole.log
echo "$(date) SOLE START pid=$$" | tee -a "$LOG"
# refuse if another sole already running
if [ -f logs/SOLE.pid ]; then
  old=$(cat logs/SOLE.pid)
  if kill -0 "$old" 2>/dev/null; then
    echo "$(date) another sole $old alive; exit" | tee -a "$LOG"
    exit 0
  fi
fi
echo $$ > logs/SOLE.pid

TARGET=144
while true; do
  n=$(python3 -c "import json,os; p='results/random_vs_adversarial.json'; print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  echo "$(date) rva=$n/$TARGET" | tee -a "$LOG"
  if [ "$n" -ge "$TARGET" ]; then
    echo "$(date) A4 COMPLETE — chaining rest" | tee -a "$LOG"
    for s in run_a3_lambda.py run_a5_empirical.py run_n5_kinner.py run_a1_knn.py run_a2_tau.py; do
      echo "$(date) START $s" | tee -a "$LOG"
      python3 experiments/$s 0 >>"$LOG" 2>&1 || echo "$(date) FAIL $s" | tee -a "$LOG"
      echo "$(date) END $s" | tee -a "$LOG"
    done
    echo "$(date) ALL DONE" | tee -a "$LOG"
    rm -f logs/SOLE.pid
    exit 0
  fi
  # single sequential A4 only
  python3 experiments/run_a4_rva.py 0 >>"$LOG" 2>&1
  echo "$(date) A4 exit rc=$?" | tee -a "$LOG"
  sleep 3
done
