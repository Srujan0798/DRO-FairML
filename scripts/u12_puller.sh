#!/usr/bin/env bash
# Poll flair2 U1/U2 result JSONs; rsync + summarize each when its target is hit.
# Independent finalization: U2=30 does not wait for U1=90 (and vice versa).
# Does NOT touch run_utkface_* processes.
set -euo pipefail

ROOT="${ROOT:-/Users/srujansai/Desktop/DRO-FairML}"
LOG="${LOG:-$ROOT/logs/u12_puller.log}"
REMOTE="${REMOTE:-flair2}"
RDIR="${RDIR:-/data/srujan.sai/DRO-FairML-run/results}"
RBASE="${RBASE:-/data/srujan.sai/DRO-FairML-run}"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$ROOT"

u1_done=0
u2_done=0
last_partial=0

# One SSH: U1/U2 counts + last cell + job alive flags (cheap).
remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" \
    "python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
R = Path('$RDIR')
def info(name, target):
    p = R / name
    if not p.exists():
        print(f'{name}|0|{target}|none|0')
        return
    d = json.loads(p.read_text())
    if d:
        r = d[-1]
        last = f\"{r.get('attack','?')}/a={r.get('alpha')}/s={r.get('seed')}\"
        t = round(float(r.get('total_time') or 0), 0)
    else:
        last, t = 'empty', 0
    print(f'{name}|{len(d)}|{target}|{last}|{t}')
info('utkface_flair2.json', 90)
info('utkface_multigroup.json', 30)
ps = subprocess.getoutput('ps -eo args')
print('alive|u1=%d|u2=%d' % (
    1 if 'run_utkface_server.py' in ps else 0,
    1 if 'run_utkface_multigroup.py' in ps else 0,
))
PY" 2>/dev/null || echo "err"
}

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
  snap=$(remote_snapshot || true)
  n1=$(echo "$snap" | awk -F'|' '/utkface_flair2/{print $2; exit}')
  n2=$(echo "$snap" | awk -F'|' '/utkface_multigroup/{print $2; exit}')
  last1=$(echo "$snap" | awk -F'|' '/utkface_flair2/{print $4; exit}')
  last2=$(echo "$snap" | awk -F'|' '/utkface_multigroup/{print $4; exit}')
  alive=$(echo "$snap" | awk -F'|' '/^alive/{print; exit}')
  n1=${n1:-err}
  n2=${n2:-err}
  echo "$(date) U1=$n1/90 last=$last1 | U2=$n2/30 last=$last2 | $alive u1_done=$u1_done u2_done=$u2_done" | tee -a "$LOG"

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
