# DRO-FAIR RESEARCH PROJECT - AI AGENT EXECUTION GUIDE
# This file is read by AI - Friend types ONE LINE: "do your work"
# AI must understand FULL context and execute ALL tasks autonomously

================================================================================
SECTION 1: PROJECT CONTEXT AND BACKGROUND
================================================================================

PROJECT NAME: DRO-FAIR (Distributionally Robust Fairness in ML)

RESEARCH PROBLEM:
We are testing adversarial attacks on machine learning models that aim to 
increase unfairness in predictions. Specifically, we built a "FairnessTargetedPGD" 
attack that strategically flips labels to maximize Demographic Parity (DP) 
violation in trained models.

WHY THIS MATTERS:
- Fair ML models should have similar prediction rates across protected groups
- Adversarial attacks can corrupt training data to make models unfair
- We want to understand how bad such attacks can be, to defend against them
- This research extends our course project on DRO-FAIR framework

CURRENT STATUS:
- FairnessTargetedPGD class implemented and tested on local Mac (works!)
- Test results: Adult DP increased from 0.14 to 0.22 (+0.08), Credit from 0.01 to 0.02 (+0.01), LSAC from 0.02 to 0.05 (+0.03)
- UTKFace pipeline designed but not yet run (needs GPU)
- Code committed to GitHub, needs to run on FLAIR2 server

================================================================================
SECTION 2: TECHNICAL DETAILS
================================================================================

FAIRNESS METRIC: Demographic Parity (DP)
- DP = |P(Y=1|A=0) - P(Y=1|A=1)|
- Where Y = prediction, A = protected attribute (e.g., gender)
- Higher DP = more unfair

FAIRNESSTARGETEDPGD ATTACK:
- Computes gradient of DP w.r.t. each sample's label
- Gradient tells which samples, if flipped, would MOST increase unfairness
- Flips top-k samples (k = alpha * n samples)
- Coordinated mode: 70% corruption budget targets minority group
- alpha = 0.2 means 20% of training samples are corrupted

ADVERSARIAL CORRUPTION TYPES:
1. Feature perturbation: Add noise to X (input features)
2. Label flipping: Change y (output labels)
3. Attribute flipping: Change a (protected attributes)

COMPARISON BASELINES:
- Random corruption: Flip randomly
- Heuristic adversarial: Use rules (e.g., flip minority samples 0→1)
- FairnessTargetedPGD: Use exact gradient to maximize unfairness

================================================================================
SECTION 3: CODE STRUCTURE
================================================================================

REPO: https://github.com/Srujan0798/DRO-FairML
BRANCH: main
LOCAL PATH: /data/srujan.sai/DRO-FairML/

DIRECTORY STRUCTURE:
/data/srujan.sai/DRO-FairML/
├── src/
│   ├── corruption/
│   │   └── adversarial.py      # FairnessTargetedPGD class (lines 203-405)
│   ├── data/
│   │   └── datasets.py         # load_adult, load_credit, load_lsac, load_utkface
│   └── models/
│       └── classifier.py       # MLPClassifier neural network
├── scripts/
│   ├── test_fairness_pgd.py   # MAIN EXPERIMENT SCRIPT
│   ├── extract_utkface_features.py  # UTKFace feature extraction (GPU)
│   ├── auto_runner.sh          # Auto-runner script
│   └── setup_server.sh         # Server setup
├── data/raw/                   # Tabular data files
│   ├── adult.data
│   ├── adult.test
│   ├── default_of_credit_card_clients.xls
│   └── lsac.csv
├── results/                    # Output goes here
├── REPORTS/                    # Daily reports go here
└── docs/
    ├── FAIRNESS_PGD_DESIGN.md
    └── UTKFACE_PIPELINE.md

KEY CLASSES:

class FairnessTargetedPGD:
    """Gradient-based fairness attack targeting DP"""
    - __init__(alpha=0.2, target_metric='dp', pgd_steps=5, coordinated=True)
    - compute_dp_gradient(y, a) → returns gradient array
    - corrupt(X, y, a, model=None, device='cpu') → returns corrupted X, y, a, mask

class AdversarialCorruptor:
    """Heuristic-based adversarial corruption (existing, for comparison)"""

class MLPClassifier(nn.Module):
    """Simple MLP for binary classification"""
    - forward(x) → logits
    - predict_proba(x, temperature=1.0) → probabilities
    - predict(x) → binary predictions

FUNCTIONS:
- load_adult(data_dir) → X, y, a, name
- load_credit(data_dir) → X, y, a, name
- load_lsac(data_dir) → X, y, a, name
- load_utkface(data_dir, feature_cache) → X, y, a, name
- get_dataset(name, data_dir, test_size, val_size, random_state) → splits data

================================================================================
SECTION 4: LOGIN AND SERVER DETAILS
================================================================================

JUPYTERHUB ACCESS:
URL: http://flair2.iitgn.ac.in:8000/hub/login
Username: srujan.sai
Password: ss#081

SSH ACCESS (if needed):
Host: 10.0.62.234
Port: 22
User: srujan.sai
Password: ss#081

SERVER DIRECTORIES:
- Code: /data/srujan.sai/DRO-FairML/
- Raw data: /data/srujan.sai/DRO-FairML/data/raw/
- Results: /data/srujan.sai/DRO-FairML/results/
- Reports: /data/srujan.sai/DRO-FairML/REPORTS/
- UTKFace images: /data/srujan.sai/UTKFace/
- Feature cache: /data/srujan.sai/utkface_features.npz

GPU INFO:
- 2x Nvidia L40S GPU, 48GB each
- CUDA available via PyTorch
- OS: Ubuntu 22.04 LTS
- RAM: 128GB

================================================================================
SECTION 5: DAILY WORK EXECUTION PROTOCOL
================================================================================

WHEN FRIEND SAYS: "do your work" or "run experiments" or similar

AI AGENT MUST EXECUTE ALL OF THE FOLLOWING:

STEP 1: LOGIN TO JUPYTERHUB
- Access http://flair2.iitgn.ac.in:8000/hub/login
- Use credentials: srujan.sai / ss#081
- Open Terminal (New → Terminal)

STEP 2: NAVIGATE TO CODE DIRECTORY
cd /data/srujan.sai/DRO-FairML

STEP 3: GET LATEST CODE FROM GITHUB
git pull origin main
If error: git fetch origin main && git reset --hard origin/main

STEP 4: ENSURE PYTHON PACKAGES INSTALLED
pip install numpy torch scikit-learn pandas scikit-image --quiet
Verify: python3 -c "import numpy; import torch; import sklearn; print('OK')"

STEP 5: SET PYTHONPATH
export PYTHONPATH=/data/srujan.sai/DRO-FairML

STEP 6: RUN EXPERIMENTS
python3 scripts/test_fairness_pgd.py

If packages missing: pip install --upgrade pip && pip install numpy torch scikit-learn pandas

STEP 7: SAVE OUTPUT TO REPORTS
mkdir -p /data/srujan.sai/DRO-FairML/REPORTS
mkdir -p /data/srujan.sai/DRO-FairML/results

Save experiment output:
python3 scripts/test_fairness_pgd.py > /data/srujan.sai/DRO-FairML/results/output_$(date +%Y%m%d_%H%M%S).txt 2>&1

Append summary to daily report:
echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
echo "Date: $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
echo "================================" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
tail -30 /data/srujan.sai/DRO-FairML/results/output_*.txt >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt 2>/dev/null || echo "No output yet" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt

STEP 8: COMMIT AND PUSH TO GITHUB
cd /data/srujan.sai/DRO-FairML
git add -A
git commit -m "Auto-run $(date): Fairness PGD experiments"
git push origin main

If push fails due to network: try again after 30 seconds, max 3 attempts

STEP 9: WRITE EXECUTION SUMMARY
echo "Execution complete at $(date)" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt
echo "Experiments run successfully" >> /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt

================================================================================
SECTION 6: EXPERIMENT DETAILS
================================================================================

SCRIPT: scripts/test_fairness_pgd.py

WHAT IT DOES:
1. Loads three datasets: Adult, Credit, LSAC
2. For each dataset:
   - Trains clean model (no corruption) → measures DP
   - Trains with Random corruption → measures DP
   - Trains with Heuristic adversarial → measures DP
   - Trains with FairnessTargetedPGD → measures DP
3. Compares DP values across methods

EXPECTED OUTPUT FORMAT:
============================================================
Dataset: adult, alpha=0.2
============================================================
Train: 29393, Val: 6784, Test: 9045
Protected: a=0: 9558, a=1: 19835

[1/5] Clean baseline... DP=X.XXXX, Acc=X.XXXX (Xs.Xs)
[2/5] Random corruption... DP=X.XXXX, Acc=X.XXXX (Xs.Xs)
[3/5] Heuristic adversarial... DP=X.XXXX, Acc=X.XXXX (Xs.Xs)
[4/5] Fairness-Targeted PGD (gradient)... DP=X.XXXX, Acc=X.XXXX (Xs.Xs)

--- Summary for adult ---
Clean:        DP=X.XXXX, Acc=X.XXXX
Random:       DP=X.XXXX, Acc=X.XXXX
Heuristic:    DP=X.XXXX, Acc=X.XXXX
Grad-PGD:     DP=X.XXXX, Acc=X.XXXX
DP Increase:  Clean→Grad: +X.XXXX

(Repeats for Credit and LSAC)

FINAL SUMMARY:
ADULT:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

CREDIT:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

LSAC:
  Clean:      DP=X.XXXX
  Grad-PGD:   DP=X.XXXX
  Attack effect: +X.XXXX

SUCCESS INDICATORS:
- Grad-PGD DP should be HIGHER than Clean DP (attack works)
- This shows the gradient-based attack successfully increased unfairness

================================================================================
SECTION 7: ERROR HANDLING
================================================================================

IF PACKAGE NOT FOUND:
→ pip install numpy torch scikit-learn pandas scikit-image
→ pip install --upgrade pip

IF GIT PUSH FAILS:
→ Check network: ping github.com
→ Wait 30 seconds and retry
→ If still fails, save results locally and try again next day

IF PYTHON IMPORTS FAIL:
→ Check Python version: python3 --version (should be 3.8+)
→ Try: python3 -c "import sys; print(sys.path)"
→ Ensure PYTHONPATH is set correctly

IF GPU NOT AVAILABLE:
→ PyTorch will fallback to CPU
→ Experiments will still run (just slower)
→ Test script works on CPU

IF SCRIPT HANGS OR TIMEOUT:
→ Kill with Ctrl+C
→ Check for infinite loops
→ Re-run with timeout: timeout 300 python3 scripts/test_fairness_pgd.py

IF DATA FILES MISSING:
→ Check /data/srujan.sai/DRO-FairML/data/raw/
→ Adult needs: adult.data, adult.test
→ Credit needs: default_of_credit_card_clients.xls
→ LSAC needs: lsac.csv
→ If missing, download from UCI or recreate

================================================================================
SECTION 8: AUTO RUNNER SCRIPT
================================================================================

SCRIPT: scripts/auto_runner.sh

This script does everything automatically:
1. Pulls latest code
2. Creates necessary directories
3. Runs experiments
4. Saves output
5. Commits and pushes to GitHub

TO USE:
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh

If this script exists and is executable, prefer using this over manual steps.

================================================================================
SECTION 9: EXPECTED EXPERIMENT RESULTS
================================================================================

Based on local testing (Mac), we expect:

ADULT DATASET:
- Clean DP: ~0.14 (unfairness baseline)
- After Grad-PGD attack: ~0.22 (increased unfairness)
- Attack effect: +0.08

CREDIT DATASET:
- Clean DP: ~0.01 (almost fair)
- After Grad-PGD attack: ~0.02 (slight increase)
- Attack effect: +0.01

LSAC DATASET:
- Clean DP: ~0.02 (fair)
- After Grad-PGD attack: ~0.05 (more unfair)
- Attack effect: +0.03

If results differ significantly, there may be an issue with:
- Random seed (should be 42)
- Data loading (should be from data/raw/)
- Model training (should train for 20 epochs)
- Corruption application (should use alpha=0.2)

================================================================================
SECTION 10: REPORTING FORMAT
================================================================================

Write daily reports in: /data/srujan.sai/DRO-FairML/REPORTS/daily.txt

FORMAT:
================================
DATE: YYYY-MM-DD HH:MM:SS
================================
EXPERIMENT: FairnessTargetedPGD on Adult, Credit, LSAC

RESULTS:
Adult: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
Credit: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX
LSAC: Clean DP=X.XXXX, Corrupted DP=X.XXXX, Effect=+X.XXXX

STATUS: SUCCESS/ERRORS/PARTIAL
NOTES: Any observations or issues

================================

Append to file daily. Do not overwrite previous days.

================================================================================
SECTION 11: UTKFACE PIPELINE (FUTURE WORK)
================================================================================

NOT YET RUN - GPU required, takes ~45 minutes

SCRIPT: scripts/extract_utkface_features.py

STEPS TO RUN (after UTKFace images downloaded):
1. Download UTKFace from Kaggle or GitHub
2. Place in /data/srujan.sai/UTKFace/
3. Run: python3 scripts/extract_utkface_features.py --data-dir /data/srujan.sai/UTKFace --output /data/srujan.sai/utkface_features.npz
4. Wait ~45 minutes
5. Then can run image fairness experiments

CURRENT PRIORITY: Run tabular experiments first (Section 5-6).
UTKFace is secondary.

================================================================================
SECTION 12: GIT WORKFLOW
================================================================================

DAILY GIT COMMANDS:
cd /data/srujan.sai/DRO-FairML

# Get latest
git pull origin main

# After experiments, commit:
git add -A
git commit -m "Friend: experiments on YYYYMMDD"
git push origin main

# Check status:
git status
git log --oneline -5

BRANCH: main (only branch we use)

REMOTE: origin (https://github.com/Srujan0798/DRO-FairML.git)

================================================================================
SECTION 13: END OF EXECUTION
================================================================================

When execution is complete:
1. Results saved to /data/srujan.sai/DRO-FairML/results/
2. Report appended to /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
3. Code committed and pushed to GitHub
4. Execution logged to /data/srujan.sai/DRO-FairML/REPORTS/execution_log.txt

AI Agent should then report to friend:
- What was done
- Key results (DP values)
- Any errors or issues
- What needs attention from Srujan

================================================================================
FRIEND'S ONE LINE PROMPT TO AI:
"Do the DRO-FAIR research work. Repo: https://github.com/Srujan0798/DRO-FairML. Login: http://flair2.iitgn.ac.in:8000/hub/login (user: srujan.sai, pass: ss#081). Server path: /data/srujan.sai/DRO-FairML. Run experiments, save results in REPORTS/, commit and push to GitHub when done. If anything fails, fix it yourself. Only tell me if you truly cannot fix it."

AI READS THIS FILE AND EXECUTES ALL STEPS ABOVE AUTONOMOUSLY.
================================================================================