You are the SUPREME VERIFICATION AGENT for the DRO-FAIR project.
Your mission: BRUTAL, LINE-BY-LINE verification of everything.
This is a 12+ hour verification task. Take as long as needed.

Project: /Users/srujansai/Desktop/DRO-FairML
Repository: git-based, 32+ commits on main branch
Status: Experiments running (98/150 complete at time of this prompt)
Goal: Prove DRO-FAIR works under adversarial corruption (α=0.1, 0.2, 0.3)

================================================================================
CRITICAL CONTEXT - KNOW THIS BEFORE STARTING
================================================================================

THE HYPOTHESIS BEING TESTED:
DRO-FAIR should BEAT Naive-FAIR on demographic parity (DP) violation
when data is adversarially corrupted (α=0.1, 0.2, 0.3).
At α=0.0 (no corruption), they should TIE.

SUCCESS METRIC:
DRO wins if dp_violation_DRO < dp_violation_Naive
Minimum needed: 6/9 wins (α=0.1, 0.2, 0.3 across 3 datasets)
α=0.0 is expected tie, NOT a win or loss

WHAT WE KNOW ABOUT THE CODEBASE:
- Algorithm step order was FIXED from p→θ→λ to θ→λ→p (per paper Algorithm 1)
- Inner gradient uses ∇g NOT λ∇g (verified correct)
- Bias-corrected radii was FIXED per Appendix F
- Dykstra projection is mathematically CORRECT (verified by external review)
- lambda_max=2.0 (NOT 10.0 - 10.0 caused prediction collapse)

================================================================================
PHASE 1: ENVIRONMENT & SETUP VERIFICATION (30 minutes)
================================================================================

1.1 REPOSITORY STATE
Command: cd /Users/srujansai/Desktop/DRO-FairML && git status
Verify:
- On main branch
- No uncommitted changes (or committed changes are intentional)
- Upstream remote exists

1.2 VIRTUAL ENVIRONMENT
Command: ls -la venv/bin/python3
Verify:
- venv exists and is properly set up
- Python 3.14 (or compatible version)
- All packages installed: torch, numpy, pandas, scikit-learn, scipy, matplotlib, pytest

1.3 DIRECTORY STRUCTURE
Verify these directories exist with correct contents:
- src/training/ (dro_fair.py, naive_fair.py, standard_ml.py)
- src/corruption/ (adversarial.py)
- src/data/ (datasets.py)
- src/evaluation/ (metrics.py)
- src/models/ (classifier.py)
- src/utils/ (projections.py)
- experiments/ (12 experiment scripts)
- configs/ (default.yaml)
- tests/ (4 test files)
- data/raw/ (4 raw data files)
- results/ (output directory)

1.4 PYTHON DEPENDENCIES
Command: python3 -c "import torch, numpy, pandas, sklearn, scipy, matplotlib, pytest; print('All imports OK')"
Verify all packages import without errors.

================================================================================
PHASE 2: CORE ALGORITHM VERIFICATION - THE MOST CRITICAL (2 hours)
================================================================================

2.1 src/training/dro_fair.py - THE HEART OF THE PROJECT

READ THIS FILE COMPLETELY. Then verify each of these SPECIFIC things:

A. CLASS STRUCTURE
- DroFairTrainer class exists with __init__, fit, predict methods
- Constructor parameters: alpha, device, lr_theta, lr_lambda, lr_p, lambda_max, tau, beta, k, K_inner, epochs, weight_decay, tau_warmup_epochs

B. ALGORITHM STEP ORDER (CRITICAL - MUST MATCH PAPER)
Find the training loop. Verify this EXACT order:
1. Forward pass with current θ
2. Compute base losses
3. Update θ (outer minimization)
4. Update λ via dual ascent
5. Update p via inner maximization
NOTHING else should happen in between steps 3-5

C. INNER GRADIENT VERIFICATION (CRITICAL)
Find where p gradient is computed.
- Verify it computes gradient of g(p), NOT λ*g(p)
- g(p) is the constraint function (DP or IF violation)
- The inner loop maximizes g(p), so gradient should be of g alone
- If you see lambda_multiplied_by_gradient, that's a BUG

D. BIAS-CORRECTED RADII (FIXED PER APPENDIX F)
Find _compute_radii method. Verify:
- pi_clean[j] = (pi_obs - alpha) / (1 - 2*alpha)  [NOT pi_obs directly]
- This accounts for corruption bias in group proportions
- OLD BUG was: using pi_obs directly without correction

E. TAU WARMUP
Find tau warmup logic:
- If epoch < tau_warmup_epochs: use tau=1.0
- Else: use tau from config (100.0 for α≤0.3)

F. DYKKSTRA PROJECTION CALLS
Find where p_dp_dict and p_if are projected.
- Verify radius = 2 * rho_dp[j] (NOT rho_dp[j])
- Verify project_simplex_l1_ball() is called
- Verify max_iter=500 in tail loop

G. K_INNER LOOP
Find K_inner loop for p updates.
- Default is K_inner=10
- Each iteration does gradient ascent on p

2.2 src/training/naive_fair.py - THE BASELINE

READ THIS FILE COMPLETELY. Then verify:

A. CLASS STRUCTURE
- NaiveFairTrainer class exists
- Same __init__ signature as DroFairTrainer (except alpha)
- fit and predict methods

B. NO DRO LOGIC
- Should NOT have inner maximization on p
- Should NOT have dual ascent on λ
- Just standard ERM training

C. HYPERPARAMETER MATCHING
- Verify lambda_max, epochs, lr_theta, lr_lambda match dro_fair.py
- This ensures fair comparison

2.3 src/corruption/adversarial.py - THE ATTACK

READ THIS FILE COMPLETELY. Then verify:

A. AdversarialCorruptor class
- __init__ parameters: alpha, epsilon, pgd_steps, pgd_step_size, feature_attack, label_flip, attr_flip, coordinated, random_state
- corrupt() method returns X_c, y_c, a_c, corrupt_mask

B. PGD ATTACK (for features)
Find _attack_features method. Verify:
- Uses model gradients when model is provided
- PGD steps loop: for step in range(self.pgd_steps)
- Gradient sign-based perturbation: X_batch + pgd_step_size * torch.sign(grad)
- Epsilon projection: clamp delta to [-epsilon, epsilon]

C. LABEL ATTACK
Find _attack_labels method. Verify:
- Flips y to maximize group disparity
- Uses group-conditional positive rates
- coordinated=True means target minority group more

D. ATTRIBUTE ATTACK
Find _attack_attributes method. Verify:
- Flips a to opposite group
- coordinated=True means minority group targeted more

E. RandomCorruptor class (BASELINE)
- Same interface as AdversarialCorruptor
- Uses random noise, NOT gradient-based
- Random label flips, NOT coordinated

2.4 src/utils/projections.py - THE MATH

READ THIS FILE COMPLETELY. Then verify each projection:

A. project_simplex(v)
- Projects onto probability simplex (sum=1, all >= 0)
- Uses sorted λ trick (Duchi et al.)
- Returns uniform if all elements equal

B. project_l1_ball(v, center, radius)
- Projects onto L1 ball centered at 'center'
- Uses soft-thresholding
- Returns center if radius <= 1e-12

C. project_simplex_l1_ball(v, center, radius, max_iter=100, tol=1e-5)
- Dykstra's alternating projection
- Initialize p=0, q=0 (auxiliary variables)
- Loop: y = simplex(x+p), p = x+p-y, z = l1(y+q), q = y+q-z
- Converge when ||y-z|| < tol
- CRITICAL: Tail loop should run 500 iterations (NOT 50)
- This ensures both constraints are satisfied

2.5 src/evaluation/metrics.py

READ THIS FILE COMPLETELY. Then verify:

A. compute_dp_violation(logits, attributes, k, temperature)
- Group samples by attribute (0 vs 1)
- Compute positive rate per group
- Return |rate_group0 - rate_group1|
- Temperature scaling: sigmoid(logits * temperature)

B. compute_if_violation(embeddings, labels, k)
- Build k-NN graph
- Compare same-label vs different-label neighborhoods
- Return violation metric

C. compute_accuracy(logits, labels)
- Standard accuracy: (predictions == labels).mean()

D. compute_metrics_torch(model, X, y, a, device, temperature, k, gamma)
- Wrapper that calls all metrics
- Returns dict with accuracy, dp_violation, if_violation

2.6 src/data/datasets.py

READ THIS FILE COMPLETELY. Then verify get_dataset():
- 'adult': loads adult.data/test, returns train/val/test splits
- 'credit': loads default_of_credit_card_clients.xls, returns splits
- 'lsac': loads lsac.csv, returns splits
- Each returns: X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dataset_name

2.7 src/models/classifier.py

READ THIS FILE COMPLETELY. Then verify MLPClassifier:
- Inherits from torch.nn.Module
- __init__ takes: input_dim, hidden_dims, dropout
- forward(x) returns logits
- predict(x) returns sigmoid(logits)

================================================================================
PHASE 3: EXPERIMENT PIPELINE VERIFICATION (1 hour)
================================================================================

3.1 experiments/run_experiments.py

READ THIS FILE COMPLETELY. Then verify the pipeline:

A. SEED SETTING
- Sets random.seed, np.random.seed, torch.manual_seed
- Sets cudnn.deterministic=True, cudnn.benchmark=False

B. DATASET LOADING
- Calls get_dataset() correctly for adult/credit/lsac
- Uses random_state=seed for reproducibility

C. MODEL PRETRAINING (for adversarial attack)
- Pretrains model for 15 epochs
- This pretrained model is passed to AdversarialCorruptor

D. CORRUPTION TYPE (CRITICAL)
- MUST use AdversarialCorruptor (NOT RandomCorruptor)
- Check: from src.corruption.adversarial import AdversarialCorruptor
- Verify corruptor = AdversarialCorruptor(alpha=alpha, ...)
- Verify model=pretrained_model passed for gradient-based attack

E. TRAINING
- NaiveFairTrainer on corrupted data
- DroFairTrainer on corrupted data (with alpha parameter)
- Both trained for same epochs (fair comparison)

F. EVALUATION
- Evaluate on CLEAN test data (not corrupted)
- Both models evaluated on same clean test set

G. RESULTS SAVING
- Save to checkpoint.pkl
- Save individual JSONs to results/individual/
- Structure: {dataset}_{alpha}_{seed}.json

3.2 experiments/generate_results.py

READ THIS FILE COMPLETELY. Then verify:

A. Reads all_results.json
B. Aggregates across 10 seeds: mean ± std
C. Creates table1.csv with columns: dataset, alpha, naive_acc, naive_dp, dro_acc, dro_dp, reduction
D. Creates table1.tex LaTeX table
E. Creates reductions.json with percentage improvements

3.3 experiments/validate_results.py

READ THIS FILE COMPLETELY. Then verify:
- Checks if DRO beats Naive in ≥6/9 comparisons
- Only counts α=0.1, 0.2, 0.3 (excludes α=0.0)
- Returns pass/fail

================================================================================
PHASE 4: CONFIGURATION VERIFICATION (30 minutes)
================================================================================

4.1 configs/default.yaml

READ THIS FILE. Verify these EXACT values:
model:
  hidden_dims: [128, 64]
  dropout: 0.1
training:
  lr_theta: 0.001
  lr_lambda: 0.005
  lr_p: 0.005
  lambda_max: 2.0       ← CRITICAL: NOT 10.0
  tau: 100.0
  beta: 5.0
  k: 5
  gamma: 0.0
  K_inner: 10
  epochs: 60
  weight_decay: 0.0001
  tau_warmup_epochs: 10
corruption:
  alpha: [0.0, 0.1, 0.2, 0.3, 0.4]
  epsilon: 0.1
  mode: adversarial
  coordinated: true

4.2 Verify run_experiments.py matches config

Grep run_experiments.py for these values:
- epochs should be 60
- lambda_max should be 2.0
- K_inner should be 10
- tau_warmup_epochs should be 10

================================================================================
PHASE 5: DATA FILES VERIFICATION (30 minutes)
================================================================================

5.1 Check all 4 raw data files exist:
- data/raw/adult.data
- data/raw/adult.test
- data/raw/default_of_credit_card_clients.xls
- data/raw/lsac.csv

5.2 Quick load test (load 100 samples each):
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data.datasets import get_dataset
for ds in ['adult', 'credit', 'lsac']:
    try:
        X_tr, y_tr, a_tr, X_val, y_val, a_val, X_te, y_te, a_te, _ = get_dataset(ds, random_state=42)
        print(f'{ds}: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}')
    except Exception as e:
        print(f'{ds}: ERROR - {e}')
"

================================================================================
PHASE 6: TEST SUITE VERIFICATION (30 minutes)
================================================================================

6.1 Run full test suite:
cd /Users/srujansai/Desktop/DRO-FairML && pytest tests/ -v --tb=short 2>&1

6.2 Verify exactly 32 tests pass:
- test_corruption.py: Tests AdversarialCorruptor and RandomCorruptor
- test_metrics.py: Tests DP, IF, accuracy computation
- test_projections.py: Tests simplex, L1, Dykstra projections
- test_end_to_end.py: Tests full training pipeline

6.3 If any tests fail, report which ones and why.

================================================================================
PHASE 7: RUNNING EXPERIMENT VERIFICATION ( Ongoing - until 150/150 )
================================================================================

7.1 CHECK CURRENT STATUS
ps aux | grep run_experiments | grep -v grep

7.2 CHECK PROGRESS
python3 -c "
import pickle
with open('results/checkpoint.pkl', 'rb') as f:
    d = pickle.load(f)
n = len(d['results'])
print(f'{n}/150 experiments complete')
"

7.3 CHECK FOR NAN/INF
python3 -c "
import pickle
with open('results/checkpoint.pkl', 'rb') as f:
    d = pickle.load(f)
for r in d['results']:
    for m in ['naive', 'dro']:
        for e in ['clean', 'corrupted']:
            for k in ['accuracy', 'dp_violation', 'if_violation']:
                v = r[m][e][k]
                if v != v or abs(v) == float('inf'):  # NaN or Inf check
                    print(f'WARNING: {r[\"dataset\"]} α={r[\"alpha\"]} s={r[\"seed\"]} {m}.{e}.{k}={v}')
print('All values finite: OK' if no_warnings else 'FOUND ISSUES')
"

7.4 MONITOR UNTIL 150/150
Check every 10-15 minutes. Report any issues immediately.

================================================================================
PHASE 8: RESULTS VERIFICATION (After completion)
================================================================================

8.1 VERIFY ALL 150 RESULTS
python3 -c "
import json
with open('results/all_results.json') as f:
    r = json.load(f)
print(f'Total experiments: {len(r)}/150')

# Check all datasets and alphas
from collections import defaultdict
by_ds = defaultdict(lambda: defaultdict(int))
for exp in r:
    by_ds[exp['dataset']][exp['alpha']] += 1

for ds in ['adult', 'credit', 'lsac']:
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        n = by_ds[ds].get(a, 0)
        status = 'OK' if n == 10 else f'ERROR: {n}/10'
        print(f'{ds} α={a}: {status}')
"

8.2 VERIFY TABLE 1 GENERATION
python3 -c "
import pandas as pd
df = pd.read_csv('results/table1.csv')
print(f'Table shape: {df.shape}')
print(f'Expected: 15 rows (3 datasets × 5 alphas)')
print(df.head())
"

8.3 VERIFY REDUCTIONS
python3 -c "
import json
with open('results/reductions.json') as f:
    red = json.load(f)
wins = sum(1 for v in red.values() if v > 0)
print(f'DRO wins: {wins}/9 (expected: 6+/9)')
print(f'Reductions: {red}')
"

8.4 VERIFY STATISTICAL SIGNIFICANCE (CRITICAL)
Run Wilcoxon signed-rank test:
python3 -c "
from scipy.stats import wilcoxon
import json

with open('results/all_results.json') as f:
    results = json.load(f)

print('Statistical Significance Tests (Wilcoxon signed-rank):')
print('=' * 60)
for ds in ['adult', 'credit', 'lsac']:
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
        if len(subset) >= 5:
            naive_dps = [r['naive']['clean']['dp_violation'] for r in subset]
            dro_dps = [r['dro']['clean']['dp_violation'] for r in subset]
            try:
                stat, p = wilcoxon(naive_dps, dro_dps)
                sig = 'YES' if p < 0.05 else 'NO'
                winner = 'DRO' if p < 0.05 and sum(dro_dps)/len(dro_dps) < sum(naive_dps)/len(naive_dps) else 'Naive'
                print(f'{ds} α={alpha}: p={p:.4f} sig={sig} winner={winner}')
            except Exception as e:
                print(f'{ds} α={alpha}: ERROR - {e}')
print('=' * 60)
print('Note: sig=YES only if p<0.05 AND DRO actually won')
"

================================================================================
PHASE 9: FINAL REPORT GENERATION (30 minutes)
================================================================================

9.1 CREATE SUMMARY DOCUMENT
Save to results/VERIFICATION_REPORT.md:

# DRO-FAIR Project - Verification Report
**Date:** [DATE]
**Verifier:** [AGENT NAME]
**Project:** /Users/srujansai/Desktop/DRO-FairML

## Environment
- Python version: [from python3 --version]
- All dependencies installed: YES/NO
- Git status: [clean/dirty]

## Code Quality
- All 33 Python files syntax OK: YES/NO
- All imports resolve: YES/NO
- All 32 tests pass: YES/NO

## Algorithm Correctness
- Step order θ→λ→p: VERIFIED/NOT VERIFIED
- Inner gradient ∇g: VERIFIED/NOT VERIFIED
- Bias-corrected radii: VERIFIED/NOT VERIFIED
- Dykstra projection: VERIFIED/NOT VERIFIED

## Experiments
- Total experiments: X/150
- All values finite (no NaN/Inf): YES/NO
- Table 1 generated: YES/NO

## Results Summary
- DRO wins: X/9
- α=0.0 (expected tie): [results]
- α=0.1, 0.2, 0.3: [results per dataset]
- α=0.4: [results]

## Issues Found
- [List any issues]

## Recommendations
- [Any improvements needed]

================================================================================
SUCCESS CRITERIA CHECKLIST
================================================================================

Mark PASS or FAIL for each:

□ Repository cloned and venv set up
□ All 33 Python files readable with no syntax errors
□ All imports resolve without MissingModule errors
□ src/training/dro_fair.py: step order θ→λ→p
□ src/training/dro_fair.py: inner gradient ∇g (not λ∇g)
□ src/training/dro_fair.py: bias-corrected radii formula
□ src/utils/projections.py: Dykstra with 500-iteration tail loop
□ src/corruption/adversarial.py: AdversarialCorruptor attacks X, Y, A
□ configs/default.yaml: lambda_max=2.0 (not 10.0)
□ 150 experiments run to completion
□ 0 NaN or Inf values in any result
□ table1.csv has 15 rows with correct data
□ DRO wins ≥6/9 comparisons at α=0.1, 0.2, 0.3
□ All 32 tests pass
□ Final report generated

================================================================================
END OF VERIFICATION
================================================================================