#!/usr/bin/env bash
# Wait for the parallel IF sweep (pid $1) to complete the 540-row grid, then run the
# DETERMINISTIC regeneration (Wilcoxon, tables, figures, PDFs) and capture the real IF
# Wilcoxon summary. Does NOT commit/push and does NOT touch prose — the judgment part
# (interpreting IF results, reconciling claims, meeting brief) is done after this exits.
set -uo pipefail
cd /Users/srujansai/Desktop/DRO-FairML
SWEEP_PID="${1:-10146}"
LOG=logs/monitor_regen.log
: >"$LOG"
echo "monitor start $(date) watching pid $SWEEP_PID" >>"$LOG"

count_if(){ python3 -c "import json;d=json.load(open('results/canonical_tau1.json'));print(sum(1 for r in d if r['attack']=='if'))" 2>/dev/null || echo 0; }

while true; do
  n=$(count_if); echo "$(date +%H:%M:%S) IF=$n/180" >>"$LOG"
  [ "$n" -ge 180 ] && { echo "COMPLETE at $n" >>"$LOG"; break; }
  if ! ps -p "$SWEEP_PID" >/dev/null 2>&1; then
    echo "SWEEP DIED at $n/180 — stopping, not regenerating partial" >>"$LOG"; exit 2
  fi
  sleep 60
done

echo "=== regenerating $(date) ===" >>"$LOG"
python3 experiments/compute_canonical_wilcoxon.py >>"$LOG" 2>&1 || echo "wilcoxon FAILED" >>"$LOG"
python3 experiments/generate_report_tables.py     >>"$LOG" 2>&1 || echo "tables FAILED" >>"$LOG"
make results       >>"$LOG" 2>&1 || echo "make results FAILED" >>"$LOG"
make deliverables  >>"$LOG" 2>&1 || echo "make deliverables FAILED" >>"$LOG"
make paper         >>"$LOG" 2>&1 || echo "make paper FAILED" >>"$LOG"
make report        >>"$LOG" 2>&1 || echo "make report FAILED" >>"$LOG"

# Capture the REAL IF Wilcoxon table (first valid IF results in the project)
python3 - >>"$LOG" 2>&1 <<'PY'
import json, collections
from scipy.stats import wilcoxon
d=json.load(open('results/canonical_tau1.json'))
g=collections.defaultdict(dict)
for r in d: g[(r['dataset'],r['attack'],r['alpha'],r['seed'])][r['method']]=r
out=["Real IF-attack Wilcoxon (DRO better on DP, one-sided), n=6, from canonical_tau1.json",
     "Also shows IF-violation means so the IF-vs-DP tradeoff is visible.",""]
for ds in ['adult','credit','lsac']:
  for a in [0.0,0.1,0.2,0.3,0.4]:
    pr=[(v['naive'],v['dro']) for k,v in g.items() if k[:3]==(ds,'if',a) and 'naive' in v and 'dro' in v]
    if len(pr)<2: continue
    nv=[p[0]['dp_clean'] for p in pr]; dv=[p[1]['dp_clean'] for p in pr]
    ni=[p[0]['if_clean'] for p in pr]; di=[p[1]['if_clean'] for p in pr]
    wins=sum(1 for n,x in zip(nv,dv) if n>x)
    try: p=wilcoxon(nv,dv,alternative='greater').pvalue
    except Exception: p=float('nan')
    out.append(f"{ds:6} if a={a}: DP n={sum(nv)/len(nv):.4f} d={sum(dv)/len(dv):.4f} wins={wins}/{len(pr)} p={p:.4f} | IF n={sum(ni)/len(ni):.4f} d={sum(di)/len(di):.4f}")
open('results/if_wilcoxon_summary.txt','w').write("\n".join(out)+"\n")
print("\n".join(out))
PY
echo "=== monitor done $(date) — regen complete, NOT committed (awaiting judgment pass) ===" >>"$LOG"
