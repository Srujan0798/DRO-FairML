# DRO-FAIR — Verification Protocol

A step-by-step protocol for verifying that the project is reproducible and that every claim in the report is supported by the saved data. The protocol is organized into nine phases. Each phase has explicit checks and the commands needed to run them.

A complete pass through all nine phases is a substantive effort (several hours of compute and inspection). The phases are independent enough that they can be partially executed for spot checks.

---

## Phase 1: Environment and Setup

### 1.1 Repository state
```bash
git status         # Should be clean (or only intentional unstaged changes)
git log --oneline -5
```
Verify the working tree is clean and on the `main` branch with an upstream remote configured.

### 1.2 Virtual environment
```bash
ls -la venv/bin/python3
python3 -c "import torch, numpy, pandas, sklearn, scipy, matplotlib, pytest; print('OK')"
```
Verify Python 3.10+ and that all packages import without errors.

### 1.3 Directory structure
The following directories must exist with their expected contents:
- `src/training/` — `dro_fair.py`, `naive_fair.py`, `standard_ml.py`
- `src/corruption/` — `adversarial.py`
- `src/data/` — `datasets.py`
- `src/evaluation/` — `metrics.py`
- `src/models/` — `classifier.py`
- `src/utils/` — `projections.py`
- `experiments/` — see file list in [PROJECT_REFERENCE.md](PROJECT_REFERENCE.md)
- `configs/` — `default.yaml`
- `tests/` — four test files
- `data/raw/` — four raw data files (Adult train/test, Credit xls, LSAC csv)
- `results/`, `figures/`, `report/`

---

## Phase 2: Core Algorithm Verification

This is the most critical phase. Read each source file completely and verify the listed invariants.

### 2.1 `src/training/dro_fair.py`

**Class structure.** `DroFairTrainer` exposes `__init__`, `fit`, `predict`. The constructor accepts: `alpha`, `device`, `lr_theta`, `lr_lambda`, `lr_p`, `lambda_max`, `tau`, `beta`, `k`, `K_inner`, `epochs`, `weight_decay`, `tau_warmup_epochs`.

**Algorithm step order (critical, must match paper).** In `fit()`, the training loop executes (in this exact order):
1. Forward pass with current θ
2. Compute base losses (`L_tilt`, `g_DP`, `g_IF`)
3. Update θ (outer minimization)
4. Update λ via dual ascent
5. Update p via inner maximization

Nothing else happens between steps 3 and 5.

**Inner gradient.** The p-update computes the gradient of `g(p)`, not `λ · g(p)`. Look for the inner loop: gradient should be on `g_DP`/`g_IF` alone.

**Bias-corrected radii.** `_compute_radii()` uses:
```
π_clean[j] = (π_obs[j] − α) / (1 − 2α)   # clipped to [0, 1]
ρ_DP[j]    = α / ((1 − α) · π_clean[j] + α)
ρ_IF       = 2α − α²
```

**Tau warmup.** Loop should select `current_tau = 1.0` while `epoch < tau_warmup_epochs`, otherwise `current_tau = self.tau`.

**Dykstra projection calls.** `project_simplex_l1_ball` is called with radius `2 * rho_dp[j]` (TV → L1), and the tail loop in `projections.py` runs `max_iter = 500`.

**K_inner loop.** Default `K_inner = 10`. Each iteration performs gradient ascent on p followed by a projection.

### 2.2 `src/training/naive_fair.py`

**Class structure.** `NaiveFairTrainer` exposes the same interface as `DroFairTrainer` minus the `alpha` parameter. No inner maximization, no Dykstra projection — uniform weights only.

**No DRO logic.** The `fit()` method must not contain a p-update or a Dykstra call. Dual ascent on λ is permitted (Naive-FAIR is the special case ρ = 0 of the full algorithm).

**Hyperparameter matching.** `lambda_max`, `epochs`, `lr_theta`, `lr_lambda` must match the values used in `DroFairTrainer` for a fair comparison.

### 2.3 `src/corruption/adversarial.py`

**`AdversarialCorruptor` constructor.** Parameters: `alpha`, `epsilon`, `pgd_steps`, `pgd_step_size`, `feature_attack`, `label_flip`, `attr_flip`, `coordinated`, `random_state`. The `corrupt()` method returns `(X_c, y_c, a_c, corrupt_mask)`.

**PGD attack.** `_attack_features` uses model gradients (when a model is provided), performs `pgd_steps` iterations, applies sign-based perturbation `x + step_size · sign(grad)`, and clamps the cumulative delta to `[-epsilon, epsilon]`.

**Label attack.** `_attack_labels` flips labels to maximize the group disparity `|P(ŷ=1|a=1) − P(ŷ=1|a=0)|`. When `coordinated=True`, the flips concentrate on the disparity-maximizing direction.

**Attribute attack.** `_attack_attributes` flips `a` to the opposite group. When `coordinated=True`, the minority group receives ~70% of the attribute flips.

**`RandomCorruptor`.** A genuinely distinct class with the same interface but using random (non-gradient) attacks and uniform random label/attribute flips.

### 2.4 `src/utils/projections.py`

**`project_simplex(v)`.** Projects onto the probability simplex using the sorted-λ algorithm (Duchi et al. 2008). Returns uniform when all elements are equal.

**`project_l1_ball(v, center, radius)`.** Projects onto an L1 ball centered at `center` via soft-thresholding. Returns `center` when `radius < 1e-12`.

**`project_simplex_l1_ball(v, center, radius, max_iter=100, tol=1e-5)`.** Dykstra's alternating projection with auxiliary variables p and q. Loop body: `y = simplex(x+p); p = x+p-y; z = l1(y+q); q = y+q-z`. Converges when `‖y - z‖ < tol`. **The tail loop must run 500 iterations** to ensure both constraints are satisfied within tolerance.

### 2.5 `src/evaluation/metrics.py`

`compute_dp_violation(logits, attributes, k, temperature)` groups by attribute, computes positive rate per group using `sigmoid(logits · temperature)`, and returns `|rate_0 - rate_1|`.

`compute_if_violation(embeddings, labels, k)` builds the k-NN graph and computes the weighted IF violation.

`compute_accuracy(logits, labels)` returns the standard accuracy on hard predictions (threshold 0.5).

`compute_metrics_torch(...)` is the wrapper that returns `{accuracy, dp_violation, if_violation}`.

### 2.6 `src/data/datasets.py`

`get_dataset(name, random_state)` supports `'adult'`, `'credit'`, `'lsac'` and returns a 10-tuple `(X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dataset_name)`. Splits are 80/20 with stratification. The `StandardScaler` is fit on the train split only.

### 2.7 `src/models/classifier.py`

`MLPClassifier` inherits from `torch.nn.Module`. Constructor: `(input_dim, hidden_dims, dropout)`. `forward(x)` returns logits. `predict_proba(x)` returns `sigmoid(logits · temperature)`. The temperature is used by **multiplication**, never division.

---

## Phase 3: Experiment Pipeline

### 3.1 `experiments/run_experiments.py`

- Seeds are set on `random`, `numpy.random`, and `torch`. `torch.backends.cudnn.deterministic = True` and `benchmark = False`.
- `get_dataset(..., random_state=seed)` is called per seed.
- A standard MLP is pretrained for 15 epochs to provide gradients for PGD.
- The corruption type is `AdversarialCorruptor` (not `RandomCorruptor`), with `model=pretrained_model` passed in.
- Both Naive and DRO trainers run for the same number of epochs on the same corrupted training data.
- Evaluation is on the clean test set.
- Results are saved per-(dataset, alpha, seed) as `results/individual/{dataset}_{alpha}_{seed}.json`.

### 3.2 `experiments/generate_results.py`

- Reads `results/all_results.json`.
- Aggregates across the 10 seeds: mean ± SE.
- Writes `table1.csv`, `table1.tex`, `reductions.json`, and `summary_stats.csv`.

### 3.3 `experiments/validate_results.py`

- Loads `results/all_results.json`.
- Runs Wilcoxon signed-rank tests for each (dataset, α) cell at α ∈ {0.1, 0.2, 0.3}.
- Reports DP and IF wins under both the Wilcoxon (p < 0.05) and mean-based criteria.
- Passes when DP wins ≥ 6/9 under Wilcoxon.

---

## Phase 4: Configuration

`configs/default.yaml` must contain (exact values):

```yaml
model:
  hidden_dims: [128, 64]
  dropout: 0.1
training:
  lr_theta: 0.001
  lr_lambda: 0.005
  lr_p: 0.005
  lambda_max: 1.5      # stability adjustment; paper uses 2.0
  tau: 100.0
  beta: 5.0
  k: 5
  gamma: 0.0
  K_inner: 10
  epochs: 60
  weight_decay: 0.0001
  tau_warmup_epochs: 15
corruption:
  alpha: [0.0, 0.1, 0.2, 0.3, 0.4]
  epsilon: 0.1
  mode: adversarial
  coordinated: true
```

Cross-check `experiments/run_experiments.py` for the same values where they appear inline.

---

## Phase 5: Data Files

All four raw data files must be present:
- `data/raw/adult.data`
- `data/raw/adult.test`
- `data/raw/default_of_credit_card_clients.xls`
- `data/raw/lsac.csv`

Quick load test:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data.datasets import get_dataset
for ds in ['adult', 'credit', 'lsac']:
    X_tr, y_tr, a_tr, X_val, y_val, a_val, X_te, y_te, a_te, _ = get_dataset(ds, random_state=42)
    print(f'{ds}: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}')
"
```

---

## Phase 6: Test Suite

```bash
pytest tests/ -v --tb=short
```

Exactly 32 tests should pass across:
- `test_corruption.py` — `AdversarialCorruptor` and `RandomCorruptor`
- `test_metrics.py` — DP, IF, accuracy
- `test_projections.py` — simplex, L1, Dykstra
- `test_end_to_end.py` — full training pipeline

If pytest hangs during plugin collection: `pytest tests/ -p no:hypothesis`.

---

## Phase 7: Experiment Completion

### 7.1 Check process status (when running)
```bash
ps aux | grep run_experiments | grep -v grep
```

### 7.2 Progress
```bash
ls results/individual/*.json | wc -l    # Should reach 150
```

### 7.3 NaN/Inf check
```bash
python3 -c "
import json
import math
results = json.load(open('results/all_results.json'))
bad = 0
for r in results:
    for m in ['naive', 'dro']:
        for e in ['clean', 'corrupted']:
            for k in ['accuracy', 'dp_violation', 'if_violation']:
                v = r[m][e][k]
                if math.isnan(v) or math.isinf(v):
                    print(f'BAD: {r[\"dataset\"]} a={r[\"alpha\"]} s={r[\"seed\"]} {m}.{e}.{k}={v}')
                    bad += 1
print(f'Bad values: {bad}')
"
```

### 7.4 Coverage check
```bash
python3 -c "
import json
from collections import defaultdict
results = json.load(open('results/all_results.json'))
by_ds = defaultdict(lambda: defaultdict(int))
for exp in results:
    by_ds[exp['dataset']][exp['alpha']] += 1
for ds in ['adult', 'credit', 'lsac']:
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        n = by_ds[ds].get(a, 0)
        print(f'{ds} a={a}: {n}/10 {\"OK\" if n == 10 else \"MISSING\"}')
"
```

---

## Phase 8: Results Verification

### 8.1 Table 1
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/table1.csv')
print(df.shape)   # Expect ~60 rows (3 datasets × 5 alphas × Naive/DRO)
print(df.head())
"
```

### 8.2 Reductions
```bash
python3 -c "
import json
red = json.load(open('results/reductions.json'))
wins = sum(1 for r in red if r['eval'] == 'clean' and r['dp_reduction'] > 0 and r['alpha'] in [0.1, 0.2, 0.3])
print(f'DRO mean-based DP wins at a in {{0.1, 0.2, 0.3}}: {wins}/9')
"
```

### 8.3 Statistical significance (Wilcoxon, one-sided)
```bash
python3 -c "
from scipy.stats import wilcoxon
import json
results = json.load(open('results/all_results.json'))
for ds in ['adult', 'credit', 'lsac']:
    for a in [0.1, 0.2, 0.3]:
        sub = [r for r in results if r['dataset']==ds and abs(r['alpha']-a) < 1e-6]
        if len(sub) < 5:
            continue
        n_dp = [r['naive']['clean']['dp_violation'] for r in sub]
        d_dp = [r['dro']['clean']['dp_violation'] for r in sub]
        diff = [n - d for n, d in zip(n_dp, d_dp)]
        try:
            _, p = wilcoxon(diff, alternative='greater')
            mean_win = sum(d_dp)/len(d_dp) < sum(n_dp)/len(n_dp)
            sig_win = (p < 0.05) and mean_win
            print(f'{ds:6s} a={a}: p={p:.4f}  mean_win={mean_win}  sig_win={sig_win}')
        except Exception as e:
            print(f'{ds} a={a}: {e}')
"
```

Cells where the mean favors DRO but `p ≥ 0.05` should be reported as "insufficient evidence", not as wins.

---

## Phase 9: Generate Verification Report

When the full protocol completes, summarize the findings in `results/VERIFICATION_REPORT.md`:

```markdown
# DRO-FAIR Verification Report

Date: [YYYY-MM-DD]
Verifier: [name or process]
Project: /Users/srujansai/Desktop/DRO-FairML

## Environment
- Python: [version]
- Dependencies installed: yes/no
- Git status: clean / dirty

## Code Quality
- All Python files parse without errors: yes/no
- All imports resolve: yes/no
- Tests passing: 32/32 expected

## Algorithm Correctness
- Step order θ → λ → p: verified
- Inner gradient ∇g (not λ∇g): verified
- Bias-corrected radii: verified
- Dykstra tail loop max_iter=500: verified

## Experiments
- Total experiments: 150 / 150
- No NaN/Inf values: yes
- Table 1 generated: yes

## Results Summary
- DP wins (Wilcoxon p<0.05): __ / 9
- IF wins (Wilcoxon p<0.05): __ / 9
- Average accuracy drop: __ pp

## Issues Found
- [list]

## Recommendations
- [list]
```

---

## Success Criteria

Mark each pass/fail:

- [ ] Repository in clean state, venv set up
- [ ] All Python files parse with no syntax errors
- [ ] All imports resolve
- [ ] `dro_fair.py` step order θ → λ → p
- [ ] `dro_fair.py` inner gradient `∇g` (not `λ∇g`)
- [ ] `dro_fair.py` bias-corrected radii
- [ ] `projections.py` Dykstra tail loop runs 500 iterations
- [ ] `adversarial.py` attacks features, labels, and attributes
- [ ] `configs/default.yaml` has `lambda_max=1.5`
- [ ] 150 experiments saved
- [ ] No NaN or Inf in any result
- [ ] `table1.csv` populated correctly
- [ ] DP wins under Wilcoxon ≥ 6/9 at α ∈ {0.1, 0.2, 0.3}
- [ ] All 32 tests pass
- [ ] Verification report generated
