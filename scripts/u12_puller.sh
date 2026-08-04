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
  python3 - <<'PY'
import json
from statistics import mean
from pathlib import Path
from collections import defaultdict

rows = json.load(open("results/utkface_multigroup.json"))
lines = [
    "# UTKFace multi-group (5-race) summary",
    "",
    f"rows: {len(rows)}/30",
    "",
    "| α | n | DP_bin N | DP_bin D | wins_bin | DP_multi N | DP_multi D | wins_multi |",
    "|---:|--:|---------:|---------:|---------:|-----------:|-----------:|-----------:|",
]
for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
    cell = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
    if not cell:
        continue
    nb = [r["naive"]["dp_binary"] for r in cell]
    db = [r["dro"]["dp_binary"] for r in cell]
    nm = [r["naive"]["dp_multigroup"] for r in cell]
    dm = [r["dro"]["dp_multigroup"] for r in cell]
    lines.append(
        f"| {a} | {len(cell)} | {mean(nb):.4f} | {mean(db):.4f} | "
        f"{sum(n > d for n, d in zip(nb, db))}/{len(cell)} | "
        f"{mean(nm):.4f} | {mean(dm):.4f} | "
        f"{sum(n > d for n, d in zip(nm, dm))}/{len(cell)} |"
    )

# Which race extremum drives max-min DP (DRO)?
lines += ["", "### DRO group positive rates (mean over seeds, by α)", ""]
for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
    cell = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
    if not cell:
        continue
    rates = defaultdict(list)
    for r in cell:
        for g, v in r["dro"]["group_pos_rates"].items():
            rates[g].append(v)
    means = {g: mean(v) for g, v in rates.items()}
    gmax = max(means, key=means.get)
    gmin = min(means, key=means.get)
    lines.append(
        f"- α={a}: max={gmax} ({means[gmax]:.3f}) min={gmin} ({means[gmin]:.3f}) "
        f"— gap≈{means[gmax]-means[gmin]:.3f}; rates={{{', '.join(f'{k}:{means[k]:.3f}' for k in sorted(means))}}}"
    )

lines += [
    "",
    "Protocol: train DP on binary race (White vs non-White); eval max-min DP on 5 race groups.",
    "REAL ResNet18 features. device=cuda flair2. Not a paper claim until human review.",
]
Path("results/utkface_multigroup_summary.md").write_text("\n".join(lines) + "\n")
print("wrote results/utkface_multigroup_summary.md")
PY
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
