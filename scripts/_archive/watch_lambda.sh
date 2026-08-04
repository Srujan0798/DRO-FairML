#!/bin/bash
set -e
LOG="logs/agent_a_status.log"
LAMBDA_LOG="logs/lambda_grid_run.log"
echo "=== WATCHER started for lambda completion $(date) ===" >> $LOG
MAX_MIN=40
for ((i=1; i<=MAX_MIN*2; i++)); do   # check every ~30s up to 40min
  /usr/local/bin/python3 /tmp/count_results.py >> $LOG 2>&1
  LCOUNT=$(python3 -c "
import json
with open('results/lambda_lr_grid.json') as f: d=json.load(f)
print(len(d))
" 2>/dev/null || echo 40)
  echo "  [watch $i] lambda rows: $LCOUNT at $(date)" >> $LOG
  tail -3 $LAMBDA_LOG | sed 's/^/    /' >> $LOG
  if [ "$LCOUNT" -ge 72 ]; then
    echo "=== LAMBDA REACHED 72 at $(date) ===" >> $LOG
    /usr/local/bin/python3 /tmp/count_results.py >> $LOG 2>&1
    break
  fi
  sleep 30
done
echo "Watcher done or timeout." >> $LOG
