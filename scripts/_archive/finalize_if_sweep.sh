#!/usr/bin/env bash
# Auto-finalize the local IF-attack sweep (PID-driven watcher).
#
# Waits until results/canonical_tau1.json holds all 180 IF-attack rows (i.e. the
# full 540-row grid), OR the sweep process dies. On full completion it regenerates
# all data-derived artifacts, captures the real IF Wilcoxon summary, commits, and
# pushes. It does NOT rewrite prose/narrative claims — those stay for human review.
set -uo pipefail
cd /Users/srujansai/Desktop/DRO-FairML

SWEEP_PID="${1:-29657}"
LOG=logs/finalize_if_sweep.log
mkdir -p logs
echo "=== finalize watcher started $(date) — watching PID $SWEEP_PID ===" >"$LOG"

count_if() { python3 -c "import json;d=json.load(open('results/canonical_tau1.json'));print(sum(1 for r in d if r['attack']=='if'))" 2>/dev/null || echo 0; }

while true; do
  n=$(count_if)
  echo "$(date +%H:%M:%S) IF rows: $n/180" >>"$LOG"
  if [ "$n" -ge 180 ]; then echo "COMPLETE" >>"$LOG"; break; fi
  if ! ps -p "$SWEEP_PID" >/dev/null 2>&1; then
    echo "SWEEP PROCESS $SWEEP_PID DIED at $n/180 rows — stopping watcher, NOT finalizing partial data" >>"$LOG"
    exit 2
  fi
  sleep 120
done

echo "=== regenerating downstream artifacts $(date) ===" >>"$LOG"
python3 experiments/generate_report_tables.py   >>"$LOG" 2>&1
python3 experiments/canonical_to_all_results.py  >>"$LOG" 2>&1
python3 experiments/generate_figures.py          >>"$LOG" 2>&1
tectonic -X compile paper/main.tex               >>"$LOG" 2>&1 && echo "paper OK" >>"$LOG"
tectonic -X compile report/report.tex            >>"$LOG" 2>&1 && echo "report OK" >>"$LOG"

# Capture the REAL IF Wilcoxon numbers (first valid IF results the project has produced)
python3 - >>"$LOG" 2>&1 <<'PY'
import json, collections
from scipy.stats import wilcoxon
d=json.load(open('results/canonical_tau1.json'))
g=collections.defaultdict(dict)
for r in d:
    g[(r['dataset'],r['attack'],r['alpha'],r['seed'])][r['method']]=r
out=["IF-attack Wilcoxon (DRO better on DP, one-sided), n=6, from canonical_tau1.json",""]
for ds in ['adult','credit','lsac']:
  for a in [0.0,0.1,0.2,0.3,0.4]:
    pairs=[(v['naive'],v['dro']) for k,v in g.items() if k[:3]==(ds,'if',a) and 'naive' in v and 'dro' in v]
    if len(pairs)<2: continue
    nv=[p[0]['dp_clean'] for p in pairs]; dv=[p[1]['dp_clean'] for p in pairs]
    nvif=[p[0]['if_clean'] for p in pairs]; dvif=[p[1]['if_clean'] for p in pairs]
    wins=sum(1 for n,x in zip(nv,dv) if n>x)
    try: p=wilcoxon(nv,dv,alternative='greater').pvalue
    except Exception: p=float('nan')
    out.append(f"{ds:6} if a={a}: DP naive={sum(nv)/len(nv):.4f} dro={sum(dv)/len(dv):.4f} wins={wins}/{len(pairs)} p={p:.4f} | IF naive={sum(nvif)/len(nvif):.4f} dro={sum(dvif)/len(dvif):.4f}")
open('results/if_wilcoxon_summary.txt','w').write("\n".join(out)+"\n")
print("\n".join(out))
PY

echo "=== committing + pushing factual artifacts $(date) ===" >>"$LOG"
git add results/canonical_tau1.json results/if_wilcoxon_summary.txt \
        report/sections/*.tex paper/auto_generated/*.tex \
        figures/ report/report.pdf paper/main.pdf >>"$LOG" 2>&1
git commit -q -F - >>"$LOG" 2>&1 <<'MSG'
data: complete IF-attack third (180 rows) — canonical now 540 rows

Local IF sweep finished. canonical_tau1.json now holds all three attacks
(dp/if/combined) at n=6, k_inner=10, tau=1.0. IF metric is the fixed
cosine-based version (was degenerate ~1e-10 pre-fix). Tables, figures and
both PDFs regenerated from the full 540-row grid. Real IF Wilcoxon numbers
captured in results/if_wilcoxon_summary.txt.

NOTE: prose/narrative claims (STATUS.md story, docs/KULDEEP_CORRECTION.md
IF section, paper prose that says "IF pending cluster re-run") are NOT
auto-updated here — they need human review against the new IF numbers.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
git push origin main >>"$LOG" 2>&1 && echo "PUSHED $(date)" >>"$LOG"
echo "=== finalize watcher done $(date) ===" >>"$LOG"
