#!/usr/bin/env bash
# One-shot flair2 U1/U2 status (read-only). Never kills jobs.
set -euo pipefail
REMOTE="${REMOTE:-flair2}"
RBASE="${RBASE:-/data/srujan.sai/DRO-FairML-run}"

ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd /data/srujan.sai/DRO-FairML-run
./venv_gpu/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
from collections import Counter

def show(path, target):
    p = Path("results") / path
    d = json.loads(p.read_text()) if p.exists() else []
    print(f"== {path}: {len(d)}/{target} ==")
    if not d:
        return
    r = d[-1]
    print(f"  last: attack={r.get('attack')} alpha={r.get('alpha')} seed={r.get('seed')} total_time={r.get('total_time',0):.0f}s")
    c = Counter((x.get("attack"), x.get("alpha")) for x in d)
    for k, v in sorted(c.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        print(f"  {k}: {v}")
    # remaining rough ETA using mean total_time for alpha>=0.2 if available
    ts = [x["total_time"] for x in d if float(x.get("alpha") or 0) >= 0.2 and x.get("total_time")]
    rem = target - len(d)
    if ts and rem > 0:
        mean_t = sum(ts) / len(ts)
        print(f"  ETA ~{rem * mean_t / 3600:.1f} h ({rem} left × {mean_t/60:.0f} min, from α≥0.2 cells)")

show("utkface_flair2.json", 90)
show("utkface_multigroup.json", 30)

ps = subprocess.getoutput("ps -eo pid,etime,pcpu,args")
print("== processes ==")
for line in ps.splitlines():
    if "run_utkface_server.py" in line or "run_utkface_multigroup.py" in line:
        print(" ", line[:200])
PY
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null || true
echo "== logs =="
wc -c logs/u1_utkface_flair2.log logs/u2_multigroup.log 2>/dev/null || true
tail -3 logs/u2_multigroup.log 2>/dev/null || true
REMOTE
