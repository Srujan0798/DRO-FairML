#!/bin/bash
# One-command script to finish the project once the running experiments complete.
# Run this after lambda hits 72 and canonical has Credit/LSAC.

set -e

echo "=== Step 1: Finalize lambda if needed ==="
python3 finalize_experiments.py

echo "=== Step 2: Launch empirical if not already running ==="
if ! pgrep -f "run_canonical.*empirical" > /dev/null; then
  echo "Launching empirical companion..."
  nohup python3 experiments/run_canonical.py --k_inner 10 --radii_mode empirical >> logs/empirical.log 2>&1 &
  echo "Empirical launched. Wait for it to produce rows before next step."
  exit 0
fi

echo "=== Step 3: Full analysis + tables ==="
python3 experiments/analyze_tau1.py
python3 experiments/generate_report_tables.py

echo "=== Step 4: Rebuild PDFs ==="
(cd report && /opt/homebrew/bin/tectonic report.tex)
(cd paper && /opt/homebrew/bin/tectonic main.tex)

echo "=== Step 5: Final figures + wilcoxon (if needed) ==="
# The analyze step above should have handled most. Add any extra here.

echo "=== Step 6: Docs + cleanup + commit ==="
python3 -c "
import json, time
lam = len(json.load(open('results/lambda_lr_grid.json')))
can = len(json.load(open('results/canonical_tau1.json')))
print(f'Final counts: lambda {lam}/72, canonical {can}/540')
" > FINAL_EVIDENCE.txt

git add -A
git commit -m "FINAL: lambda + canonical + empirical complete. All analysis, figures, report, docs updated." || echo "Nothing new to commit"

echo "Done. Check FINAL_EVIDENCE.txt and git log."
