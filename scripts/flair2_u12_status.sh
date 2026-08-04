#!/usr/bin/env bash
# One-shot flair2 U1/U2 status (read-only). Never kills jobs.
set -euo pipefail
REMOTE="${REMOTE:-flair2}"
RBASE="${RBASE:-/data/srujan.sai/DRO-FairML-run}"

ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd /data/srujan.sai/DRO-FairML-run
./venv_gpu/bin/python - <<'PY'
import json, os, subprocess, time
from pathlib import Path
from collections import Counter

def show(path, target):
    p = Path("results") / path
    d = json.loads(p.read_text()) if p.exists() else []
    print(f"== {path}: {len(d)}/{target} ==")
    if not d:
        return
    r = d[-1]
    age = time.time() - p.stat().st_mtime
    print(
        f"  last: attack={r.get('attack')} alpha={r.get('alpha')} seed={r.get('seed')} "
        f"total_time={r.get('total_time',0):.0f}s | json_age={age/60:.1f} min"
    )
    if age > 25 * 60:
        print(f"  WARN: no new row for {age/60:.0f} min (typical cell ~16 min; stall if >>30)")
    c = Counter((x.get("attack"), x.get("alpha")) for x in d)
    for k, v in sorted(c.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        print(f"  {k}: {v}")
    # Prefer recent higher-α timings when available (better ETA for remaining grid).
    ts = [x["total_time"] for x in d if float(x.get("alpha") or 0) >= 0.3 and x.get("total_time")]
    src = "α≥0.3"
    if len(ts) < 2:
        ts = [x["total_time"] for x in d if float(x.get("alpha") or 0) >= 0.2 and x.get("total_time")]
        src = "α≥0.2"
    rem = target - len(d)
    if ts and rem > 0:
        mean_t = sum(ts) / len(ts)
        print(f"  ETA ~{rem * mean_t / 3600:.1f} h ({rem} left × {mean_t/60:.0f} min, from {src} cells n={len(ts)})")
        # If a cell is in flight, estimate when next row lands.
        if age < mean_t:
            print(f"  next row ETA ~{(mean_t - age)/60:.0f} min (if mid-cell)")

show("utkface_flair2.json", 90)
show("utkface_multigroup.json", 30)

ps = subprocess.getoutput("ps -eo pid,etime,pcpu,args")
print("== processes ==")
found = 0
for line in ps.splitlines():
    if "run_utkface_server.py" in line or "run_utkface_multigroup.py" in line:
        print(" ", line[:200])
        found += 1
if not found:
    print("  (none — jobs finished or died)")

print("== U3 images ==")
utk = Path("/data/srujan.sai/UTKFace")
if utk.exists():
    njpg = sum(1 for e in os.scandir(utk) if e.name.endswith(".jpg"))
    tgt = os.readlink(utk) if utk.is_symlink() else "dir"
    print(f"  ready: {utk} -> {tgt}  n_jpg={njpg}")
else:
    print("  MISSING /data/srujan.sai/UTKFace (run flair2_link_utkface_images.sh)")
PY
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null || true
echo "== logs =="
wc -c logs/u1_utkface_flair2.log logs/u2_multigroup.log 2>/dev/null || true
tail -3 logs/u2_multigroup.log 2>/dev/null || true
REMOTE
