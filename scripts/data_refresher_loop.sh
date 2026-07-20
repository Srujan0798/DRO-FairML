#!/bin/bash
# Agent Data-Refresher background loop
LOG="logs/agent_data_refresher.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === AGENT DATA-REFRESHER BG LOOP STARTED ===" >> "$LOG"

get_counts() {
  python3 -c '
import json, sys
try:
  lam = len(json.load(open("results/lambda_lr_grid.json")))
except: lam = -1
try:
  can = len(json.load(open("results/canonical_tau1.json")))
  dss = {}
  for r in json.load(open("results/canonical_tau1.json")): 
    ds = r.get("dataset", "?")
    dss[ds] = dss.get(ds,0)+1
except: can = -1; dss={}
print(lam, can, str(dss))
' 
}

run_refresh() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RUNNING refresh cycle..." >> "$LOG"
  python3 finalize_experiments.py status >> "$LOG" 2>&1
  python3 experiments/analyze_tau1.py >> "$LOG" 2>&1
  python3 experiments/generate_report_tables.py >> "$LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refresh cycle DONE." >> "$LOG"
}

update_status_txts() {
  lam=$1; can=$2; dss=$3; ts=$(date '+%Y-%m-%d %H:%M:%S')
  entry="
=== AGENT DATA-REFRESHER @ $ts ===
lambda: $lam/72 , canonical: $can/540
datasets: $dss
Ran: analyze_tau1.py + generate_report_tables.py + finalize status
"
  echo "$entry" >> ORCHESTRATOR_LIVE_STATUS.txt
  echo "$entry" >> DELIVERABLES_CHECKLIST.txt
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updated status txts with $lam/$can" >> "$LOG"
}

# initial
run_refresh
read lam can dss <<< $(get_counts)
update_status_txts $lam $can "$dss"
last_lam=$lam
last_can=$can

# loop every 5 min, or check more often for change
while true; do
  sleep 60   # poll every min for change detection
  read lam can dss <<< $(get_counts)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Poll: lambda=$lam/72 canonical=$can/540 dss=$dss" >> "$LOG"
  if [ "$lam" != "$last_lam" ] || [ "$can" != "$last_can" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CHANGE DETECTED: $last_lam/$last_can -> $lam/$can" >> "$LOG"
    run_refresh
    update_status_txts $lam $can "$dss"
    last_lam=$lam
    last_can=$can
  fi
  # also full periodic every ~5-10min regardless
  if [ $(( $(date +%s) % 600 )) -lt 60 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PERIODIC full refresh" >> "$LOG"
    run_refresh
    update_status_txts $lam $can "$dss"
  fi
  # stop if complete
  if [ "$lam" -ge 72 ] && [ "$can" -ge 540 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRIDS COMPLETE detected. Exiting refresher." >> "$LOG"
    break
  fi
done
