#!/usr/bin/env bash
# Poll flair2 U1/U2/U3 result JSONs; rsync + summarize on count-up and at target.
# Independent finalization per job. Does NOT touch run_utkface_* processes.
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
u3_done=0
last_partial=0
prev_n1=-1
prev_n2=-1
prev_n3=-1

# Resume flags from local files (puller restarts mid-run).
if [[ -f results/utkface_flair2.json ]]; then
  n=$(python3 -c "import json; print(len(json.load(open('results/utkface_flair2.json'))))" 2>/dev/null || echo 0)
  if [[ "$n" == "90" ]]; then u1_done=1; prev_n1=90; fi
fi
if [[ -f results/utkface_multigroup.json ]]; then
  n=$(python3 -c "import json; print(len(json.load(open('results/utkface_multigroup.json'))))" 2>/dev/null || echo 0)
  if [[ "$n" == "30" ]]; then u2_done=1; prev_n2=30; fi
fi
if [[ -f results/utkface_pixel_pgd.json ]]; then
  n=$(python3 -c "import json; print(len(json.load(open('results/utkface_pixel_pgd.json'))))" 2>/dev/null || echo 0)
  if [[ "$n" == "24" ]]; then u3_done=1; prev_n3=24; fi
fi

remote_snapshot() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE" \
    "python3 - <<'PY'
import json, subprocess
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
info('utkface_pixel_pgd.json', 24)
ps = subprocess.getoutput('ps -eo args')
print('alive|u1=%d|u2=%d|u3=%d' % (
    1 if 'run_utkface_server.py' in ps else 0,
    1 if 'run_utkface_multigroup.py' in ps else 0,
    1 if 'run_utkface_pixel_pgd.py' in ps else 0,
))
PY" 2>/dev/null || echo "err"
}

rsync_one() {
  local file="$1"
  rsync -az "$REMOTE:$RDIR/$file" "$ROOT/results/" 2>/dev/null || return 1
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

finalize_u3() {
  rsync_one utkface_pixel_pgd.json
  python3 experiments/summarize_utkface_pixel_pgd.py
  echo "$(date) U3 FINALIZED 24/24 + pixel_pgd summary" | tee -a "$LOG"
}

partial_refresh() {
  local n1="$1" n2="$2" n3="$3" grew=0
  if [[ "$n1" =~ ^[0-9]+$ ]] && (( n1 > prev_n1 )) && [[ "$u1_done" == "0" ]]; then
    rsync_one utkface_flair2.json || true
    python3 experiments/summarize_utkface_repro.py >/dev/null || true
    grew=1
  fi
  if [[ "$n2" =~ ^[0-9]+$ ]] && (( n2 > prev_n2 )) && [[ "$u2_done" == "0" ]]; then
    rsync_one utkface_multigroup.json || true
    python3 experiments/summarize_utkface_multigroup.py >/dev/null || true
    grew=1
  fi
  if [[ "$n3" =~ ^[0-9]+$ ]] && (( n3 > prev_n3 )) && [[ "$u3_done" == "0" ]]; then
    rsync_one utkface_pixel_pgd.json || true
    python3 experiments/summarize_utkface_pixel_pgd.py >/dev/null || true
    grew=1
  fi
  if (( grew )); then
    echo "$(date) count-up rsync+summarize U1=$n1 (was $prev_n1) U2=$n2 (was $prev_n2) U3=$n3 (was $prev_n3)" | tee -a "$LOG"
  fi
  if [[ "$n1" =~ ^[0-9]+$ ]]; then prev_n1=$n1; fi
  if [[ "$n2" =~ ^[0-9]+$ ]]; then prev_n2=$n2; fi
  if [[ "$n3" =~ ^[0-9]+$ ]]; then prev_n3=$n3; fi
}

echo "$(date) puller start (U1/U2/U3; poll=${POLL_SEC}s; flags u1=$u1_done u2=$u2_done u3=$u3_done)" | tee -a "$LOG"

while true; do
  snap=$(remote_snapshot || true)
  n1=$(echo "$snap" | awk -F'|' '/utkface_flair2/{print $2; exit}')
  n2=$(echo "$snap" | awk -F'|' '/utkface_multigroup/{print $2; exit}')
  n3=$(echo "$snap" | awk -F'|' '/utkface_pixel_pgd/{print $2; exit}')
  last1=$(echo "$snap" | awk -F'|' '/utkface_flair2/{print $4; exit}')
  last2=$(echo "$snap" | awk -F'|' '/utkface_multigroup/{print $4; exit}')
  last3=$(echo "$snap" | awk -F'|' '/utkface_pixel_pgd/{print $4; exit}')
  alive=$(echo "$snap" | awk -F'|' '/^alive/{print; exit}')
  n1=${n1:-err}
  n2=${n2:-err}
  n3=${n3:-err}
  echo "$(date) U1=$n1/90 last=$last1 | U2=$n2/30 last=$last2 | U3=$n3/24 last=$last3 | $alive u1_done=$u1_done u2_done=$u2_done u3_done=$u3_done" | tee -a "$LOG"

  if [[ "$n1" == "90" && "$u1_done" == "0" ]]; then
    finalize_u1
    u1_done=1
    prev_n1=90
  fi
  if [[ "$n2" == "30" && "$u2_done" == "0" ]]; then
    finalize_u2
    u2_done=1
    prev_n2=30
  fi
  if [[ "$n3" == "24" && "$u3_done" == "0" ]]; then
    finalize_u3
    u3_done=1
    prev_n3=24
  fi

  if [[ "$u1_done" == "0" || "$u2_done" == "0" || "$u3_done" == "0" ]]; then
    partial_refresh "$n1" "$n2" "$n3" || true
  fi

  now=$(date +%s)
  if (( now - last_partial >= 600 )); then
    rsync_one utkface_flair2.json || true
    rsync_one utkface_multigroup.json || true
    rsync_one utkface_pixel_pgd.json || true
    last_partial=$now
    echo "$(date) partial rsync ok U1=$n1 U2=$n2 U3=$n3" | tee -a "$LOG"
  fi

  if [[ "$u1_done" == "1" && "$u2_done" == "1" && "$u3_done" == "1" ]]; then
    echo "$(date) U1+U2+U3 FINALIZED — puller exit" | tee -a "$LOG"
    break
  fi
  sleep "$POLL_SEC"
done
