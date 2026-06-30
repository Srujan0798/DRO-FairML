#!/bin/bash
# Auto-completion watcher for DRO-FairML
# Runs after canonical + empirical experiments finish
set -e

REPO="/Users/srujansai/Desktop/DRO-FairML"
CD="cd $REPO"

while true; do
  CANONICAL=$(python3 -c "import json; print(len(json.load(open('$REPO/results/canonical_tau1.json'))))" 2>/dev/null || echo 0)
  EMPIRICAL=$(python3 -c "import json; print(len(json.load(open('$REPO/results/canonical_tau1_empirical.json'))))" 2>/dev/null || echo 0)
  echo "[$(date)] Canonical: $CANONICAL/540  Empirical: $EMPIRICAL/270"

  if [ "$CANONICAL" -ge 540 ] && [ "$EMPIRICAL" -ge 270 ]; then
    echo "=== BOTH EXPERIMENTS COMPLETE ==="
    break
  fi

  # Check if watchdogs are alive
  CANONICAL_WD=$(ps aux | grep 'run_canonical.py' | grep -v grep | wc -l)
  EMPIRICAL_WD=$(ps aux | grep 'run_canonical_empirical' | grep -v grep | wc -l)
  if [ "$CANONICAL_WD" -eq 0 ]; then
    echo "WARNING: Canonical watchdog dead! Restarting..."
    nohup bash -c '
      while true; do
        timeout 600 python3 -u experiments/run_canonical.py
        rows=$(python3 -c "import json; print(len(json.load(open(\"results/canonical_tau1.json\"))))" 2>/dev/null || echo 0)
        [ "$rows" -ge 540 ] && break
        sleep 2
      done
    ' > $REPO/logs/canonical_resume.log 2>&1 &
  fi
  if [ "$EMPIRICAL_WD" -eq 0 ]; then
    echo "WARNING: Empirical watchdog dead! Restarting..."
    nohup bash -c '
      while true; do
        timeout 600 python3 -u experiments/run_canonical_empirical.py
        rows=$(python3 -c "import json; print(len(json.load(open(\"results/canonical_tau1_empirical.json\"))))" 2>/dev/null || echo 0)
        [ "$rows" -ge 270 ] && break
        sleep 2
      done
    ' > $REPO/logs/empirical_resume.log 2>&1 &
  fi

  sleep 120
done

# ===== POST-EXPERIMENT PIPELINE =====
echo "=== Phase 1: Wilcoxon significance tests ==="
python3 $REPO/experiments/compute_canonical_wilcoxon.py 2>&1 | tee $REPO/logs/post_wilcoxon.log

echo "=== Phase 2: Tau1 analysis ==="
python3 $REPO/experiments/analyze_tau1.py 2>&1 | tee $REPO/logs/post_analyze.log

echo "=== Phase 3: Report tables ==="
python3 $REPO/experiments/generate_report_tables.py 2>&1 | tee $REPO/logs/post_tables.log

echo "=== Phase 4: All figures ==="
python3 $REPO/experiments/generate_all_figures.py 2>&1 | tee $REPO/logs/post_figures.log

echo "=== Phase 5: Final figures ==="
python3 $REPO/experiments/generate_final_figures.py 2>&1 | tee $REPO/logs/post_final_fig.log

echo "=== Phase 6: Uniform vs empirical plot ==="
python3 $REPO/experiments/plot_uniform_vs_empirical.py 2>&1 | tee $REPO/logs/post_uniform_emp.log

echo "=== Phase 7: Win curves ==="
python3 $REPO/experiments/plot_win_curves_tau1.py 2>&1 | tee $REPO/logs/post_win_curves.log

echo "=== Phase 8: Headline figure ==="
python3 $REPO/experiments/plot_tau1_headline.py 2>&1 | tee $REPO/logs/post_headline.log

echo "=== Phase 9: Build PDFs ==="
tectonic -X compile $REPO/report/report.tex 2>&1 | tee $REPO/logs/post_report_build.log
tectonic -X compile $REPO/paper/main.tex 2>&1 | tee $REPO/logs/post_paper_build.log

echo "=== Phase 10: Tests ==="
python3 -m pytest $REPO/tests/ -q 2>&1 | tee $REPO/logs/post_tests.log

echo "=== Phase 11: Git commit ==="
cd $REPO
git add -A
git commit -m "feat: auto-complete — full canonical + empirical results, analysis, figures, PDFs" || true
git push 2>&1 || echo "Push skipped (no remote or auth)"

echo ""
echo "============================================"
echo "  DRO-FairML PROJECT AUTO-COMPLETE DONE"
echo "============================================"
echo "Summary:"
python3 -c "
import json
c = json.load(open('$REPO/results/canonical_tau1.json'))
e = json.load(open('$REPO/results/canonical_tau1_empirical.json'))
print(f'  Canonical:  {len(c)}/540 rows')
print(f'  Empirical:  {len(e)}/270 rows')
print(f'  Report PDF: report/report.pdf')
print(f'  Paper PDF:  paper/main.pdf')
"