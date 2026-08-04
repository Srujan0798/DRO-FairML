# What to do the moment a new row is appended (monitors will show the "-> acc=..." or next [N+1] line)

## 1. Quick poll
python3 -c '
import json
print("lambda:", len(json.load(open("results/lambda_lr_grid.json"))), "/72")
print("canonical:", len(json.load(open("results/canonical_tau1.json"))), "/540")
'

## 2. Nice status
python3 finalize_experiments.py status

## 3. If lambda reached 72 (or you want fresh heatmaps/summary)
python3 finalize_experiments.py

## 4. Rebuild report tables + PDFs (safe anytime)
python3 experiments/generate_report_tables.py
(cd report && /opt/homebrew/bin/tectonic report.tex)
(cd paper && /opt/homebrew/bin/tectonic main.tex)

## 5. Commit what makes sense
# Do NOT commit the .json while a run is still in progress on that file.
# Safe to commit:
git add results/tau1_summary.csv results/tau1_wilcoxon.csv figures/ report/sections/ paper/auto_generated/ report/report.pdf paper/main.pdf DELIVERABLES_CHECKLIST.txt ORCHESTRATOR_FINAL_STATUS.md
git commit -m "data: lambda or canonical progress + refreshed tables/figures"

## 6. When canonical has Credit or LSAC rows
# You can re-run analyze_tau1.py to get Credit numbers in the summary (even if not full 540 yet).
# Full canonical 540 + 6 seeds is still the target for the final paper/Wilcoxon.

## Kill safely (only if you really must pause)
# pkill -f "run_lambda_lr_grid"   # last resort
# pkill -f "run_canonical"        # last resort
# The runners are resume-safe.

