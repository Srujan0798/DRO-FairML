#!/usr/bin/env bash
set -u
cd /Users/srujansai/Desktop/DRO-FairML
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=logs/keeper.log
echo $$ > logs/SOLE.pid
T=144
while true; do
  n=$(python3 -c "import json,os;p='results/random_vs_adversarial.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
  echo "$(date) n=$n" >> "$LOG"
  if [ "$n" -ge "$T" ]; then
    while read -r s; do
      [ -n "$s" ] || continue
      python3 "$s" 0 >>"$LOG" 2>&1 || true
    done < scripts/chain_list.txt
    rm -f logs/SOLE.pid
    exit 0
  fi
  python3 experiments/job_rva.py 0 >>"$LOG" 2>&1
  sleep 2
done
