#!/bin/bash
# AUTO RUNNER - Does everything automatically
# Friend just runs this ONE command every day

export PYTHONPATH=/data/srujan.sai/DRO-FairML
REPORT_DIR="/data/srujan.sai/DRO-FairML/REPORTS"
RESULTS_DIR="/data/srujan.sai/DRO-FairML/results"

mkdir -p "$REPORT_DIR" "$RESULTS_DIR"

echo "=========================================="
echo "DRO-FAIR AUTO RUNNER"
echo "Date: $(date)"
echo "=========================================="

# Step 1: Get latest code
echo "[1/4] Getting latest code from GitHub..."
cd /data/srujan.sai/DRO-FairML
git pull origin main 2>&1 | tail -3

# Step 2: Run fairness test
echo ""
echo "[2/4] Running Fairness PGD test..."
python3 scripts/test_fairness_pgd.py > "$RESULTS_DIR/latest_output.txt" 2>&1

# Step 3: Save results
echo ""
echo "[3/4] Saving results..."
RESULTS=$(cat "$RESULTS_DIR/latest_output.txt" | tail -20)
echo "====================" > "$RESULTS_DIR/today.txt"
echo "Date: $(date)" >> "$RESULTS_DIR/today.txt"
echo "====================" >> "$RESULTS_DIR/today.txt"
echo "$RESULTS" >> "$RESULTS_DIR/today.txt"

# Step 4: Commit and push
echo ""
echo "[4/4] Committing and pushing to GitHub..."
git add results/REPORTS/ 2>/dev/null
git add results/*.txt 2>/dev/null
git add REPORTS/ 2>/dev/null
git add -A 2>/dev/null

git commit -m "Auto run $(date +%Y%m%d): Fairness PGD test results" --allow-empty 2>&1 | tail -3
git push origin main 2>&1 | tail -3

echo ""
echo "=========================================="
echo "DONE! Check REPORTS/ for output."
echo "=========================================="
echo ""
echo "To write a report for Srujan, run:"
echo "  echo 'what happened today' >> $REPORT_DIR/notes.txt"