#!/usr/bin/env bash
# Poll flair2 U1/U2 result JSONs; rsync + summarize each when its target is hit.
# Independent finalization: U2=30 does not wait for U1=90 (and vice versa).
# Does NOT touch run_utkface_* processes.
set -euo pipefail

ROOT="${ROOT:-/Users/srujansai/Desktop/DRO-FairML}"
LOG="${LOG:-$ROOT/logs/u12_puller.log}"
REMOTE="${REMOTE:-flair2}"
RDIR="${RDIR:-/data/srujan.sai/DRO-FairML-run/results}"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$ROOT"

u1_done=0
u2_done=0
last_partial=0

count_remote() {
  local file="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" \
    "python3 -c \"import json,os;p='$RDIR/$file';print(len(json.load(open(p))) if os.path.exists(p) else 0)\"" \
    2>/dev/null || echo err
}

rsync_one() {
  local file="$1"
  rsync -az "$REMOTE:$RDIR/$file" "$ROOT/results/"
}

finalize_u1() {
  rsync_one utkface_flair2.json
  python3 experiments/summarize_utkface_repro.py
  echo "$(date) U1 FINALIZED 90/90 + repro summary" | tee -a "$LOG"
}

finalize_u2() {
  rsync_one utkface_multigroup.json
  python3 experiments/summarize_utkface_multigroup.py
  echo "$(date) U2 FINALIZED 30/30 + multigroup summary" | tee -a "$LOG"
}

echo "$(date) puller start (independent U1/U2 finalize; poll=${POLL_SEC}s)" | tee -a "$LOG"

while true; do
  n1=$(count_remote utkface_flair2.json)
  n2=$(count_remote utkface_multigroup.json)
  echo "$(date) U1=$n1/90 U2=$n2/30 u1_done=$u1_done u2_done=$u2_done" | tee -a "$LOG"

  if [[ "$n1" == "90" && "$u1_done" == "0" ]]; then
    finalize_u1
    u1_done=1
  fi
  if [[ "$n2" == "30" && "$u2_done" == "0" ]]; then
    finalize_u2
    u2_done=1
  fi

  # cheap partial mirror every ~10 min so Mac has crash safety
  now=$(date +%s)
  if (( now - last_partial >= 600 )); then
    rsync_one utkface_flair2.json || true
    rsync_one utkface_multigroup.json || true
    last_partial=$now
    echo "$(date) partial rsync ok U1=$n1 U2=$n2" | tee -a "$LOG"
  fi

  if [[ "$u1_done" == "1" && "$u2_done" == "1" ]]; then
    echo "$(date) BOTH FINALIZED — puller exit" | tee -a "$LOG"
    break
  fi
  sleep "$POLL_SEC"
done
