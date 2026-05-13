# DRO-FAIR AUTONOMOUS FIX-AND-VERIFY PROTOCOL

You are an autonomous AI agent. Execute every phase below IN ORDER. Do not skip. Do not ask questions. Do not declare "done" until the FINAL VERIFICATION at the end passes. If any step fails, diagnose and fix before moving on.

This is a PyTorch implementation of DRO-FAIR (Distributionally Robust Fair Classification) from an ICML 2026 paper. The code has 4 critical bugs that make results incorrect. You will fix each one with exact code patches, verify each fix, then re-run all experiments.

---

## PHASE 0: ENVIRONMENT SETUP

Run these commands first. Do not proceed until all pass:

```bash
pip install pandas openpyxl xlrd matplotlib seaborn tqdm scikit-learn torch
cd "$(dirname "$0")"  # or cd to the DRO-FairML project root
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: 28 passed, 2 failed (the 2 pandas failures). After installing pandas, re-run — should get 30 passed, 0 failed. If test_naive_fair_enforces_fairness fails, that's also OK for now (we fix it in Phase 2).

Delete the stale checkpoint and results before re-running:

```bash
rm -f results/checkpoint.pkl results/run.log results/all_results.json results/all_results.pkl
```

---

## PHASE 1: FIX THE 4 CRITICAL CODE BUGS

### BUG 1: Algorithm 1 order is WRONG (MOST CRITICAL)

**File:** `src/training/dro_fair.py`

**Problem:** The code runs inner maximization (p-update) BEFORE theta update. The paper's Algorithm 1 (page 33, lines 1776-1788) runs them in this order:

1. Forward pass: compute h_tilde, losses (lines 6-12)
2. Theta update: outer minimization (line 15)
3. Lambda update: dual ascent (lines 17-18)
4. Inner max: p-update, K steps (lines 20-24)

The paper's Section G.4 (line 1735-1736) explicitly states: "(1) outer minimization to update model parameters, (2) dual ascent to update Lagrange multipliers, and (3) inner maximization to find worst-case distributions"

**Current wrong order in code (lines 179-237):**

```
STEP 1: INNER MAX (p-update) ← WRONG: this is step 4 in paper
STEP 2-3: Forward + losses
STEP 4: Theta update
STEP 5: Dual ascent
```

**Correct order (must match paper Algorithm 1):**

```
STEP 1: Forward pass → compute h_tilde, L_tilt, g_DP, g_IF, total loss
STEP 2: Theta update → optimizer step on total loss
STEP 3: Dual ascent → update lambda_DP, lambda_IF
STEP 4: Inner max → K steps of projected gradient ascent on p weights
```

**EXACT FIX:** Replace the entire training loop body (everything inside `for epoch in range(self.epochs):`) in `src/training/dro_fair.py` with:

```python
        for epoch in range(self.epochs):
            self.model.train()

            # === STEP 1: FORWARD PASS + COMPUTE LOSSES ===
            # Paper Algorithm 1, lines 6-12
            logits = self.model(X_t)
            h_tilde = torch.sigmoid(logits * self.tau)  # line 7: σ(τ·f_θ(x))

            # Classification loss: tilted BCE on raw logits (line 9)
            per_sample_loss = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
            L_tilt = self._compute_tilted_loss(per_sample_loss)

            # DP violation with current p weights (lines 10-11)
            g_dp = self._compute_dp_loss_weighted(h_tilde, a_t, p_dp_dict, group_mask_dict) if self.use_dp else torch.tensor(0.0, device=self.device)

            # IF violation with current p weights (line 12)
            g_if = self._compute_if_loss_weighted(h_tilde, p_if, edge_i, edge_j, edge_dists) if self.use_if else torch.tensor(0.0, device=self.device)

            # Total Lagrangian (line 14)
            total_loss = L_tilt + (lambda_dp * g_dp if self.use_dp else 0.0) + (lambda_if * g_if if self.use_if else 0.0)

            # === STEP 2: UPDATE θ (outer minimization, line 15) ===
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()

            # === STEP 3: DUAL ASCENT (lines 17-18) ===
            with torch.no_grad():
                if self.use_dp:
                    lambda_dp = torch.clamp(lambda_dp + self.lr_lambda * g_dp, 0, self.lambda_max)
                if self.use_if:
                    lambda_if = torch.clamp(lambda_if + self.lr_lambda * g_if, 0, self.lambda_max)

            # === STEP 4: INNER MAXIMIZATION (lines 20-24) ===
            # Paper G.5 line 1747: "With classifier predictions held fixed"
            with torch.no_grad():
                logits_for_p = self.model(X_t)
                h_tilde_for_p = torch.sigmoid(logits_for_p * self.tau)

            with torch.enable_grad():
                for _ in range(self.K_inner):
                    if self.use_dp:
                        for j in [0, 1]:
                            p_j = p_dp_dict[j].clone().detach().requires_grad_(True)
                            p_temp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                            p_temp[j] = p_j
                            dp_loss = self._compute_dp_loss_weighted(h_tilde_for_p, a_t, p_temp, group_mask_dict)
                            dp_loss.backward()
                            if p_j.grad is not None:
                                with torch.no_grad():
                                    p_dp_dict[j] = p_j + self.lr_p * p_j.grad
                                    p_dp_dict[j] = self._project_dp_weights(p_dp_dict[j], self.p_dp_center[j], self.rho_dp[j])

                    if self.use_if:
                        p_if_grad = p_if.clone().detach().requires_grad_(True)
                        if_loss = self._compute_if_loss_weighted(h_tilde_for_p, p_if_grad, edge_i, edge_j, edge_dists)
                        if_loss.backward()
                        if p_if_grad.grad is not None:
                            with torch.no_grad():
                                p_if = p_if_grad + self.lr_p * p_if_grad.grad
                                p_if = self._project_if_weights(p_if, self.p_if_center, self.rho_if)

            history['train_loss'].append(total_loss.item())

            # Validation
            if X_val is not None and (epoch + 1) % 5 == 0:
                from src.evaluation.metrics import compute_metrics_torch
                metrics = compute_metrics_torch(
                    self.model, X_val, y_val, a_val,
                    device=self.device, temperature=self.tau, k=self.k, gamma=self.gamma
                )
                history['val_acc'].append(metrics['accuracy'])
                history['val_dp'].append(metrics['dp_violation'])
                history['val_if'].append(metrics['if_violation'])
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs}: loss={total_loss.item():.4f}, "
                          f"val_acc={metrics['accuracy']:.4f}, val_dp={metrics['dp_violation']:.4f}, "
                          f"val_if={metrics['if_violation']:.4f}, "
                          f"lambda_dp={lambda_dp.item():.2f}, lambda_if={lambda_if.item():.2f}")
```

Also fix the docstring of the `fit` method to say:

```python
    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train DRO-FAIR (Algorithm 1) — full-batch implementation.

        Paper Algorithm 1 ordering (Page 33, lines 13-24):
        1. Forward pass: compute h_tilde, losses
        2. θ update (outer minimization)
        3. λ dual ascent
        4. Inner maximization: update p (K steps)
        """
```

**VERIFY after this fix:**

```bash
python3 -m pytest tests/test_end_to_end.py -v --tb=short
```

All tests that were passing before must still pass.

---

### BUG 2: tau=1 for training, but paper says tau=100

**File:** `src/training/dro_fair.py` (default parameter), `experiments/run_experiments.py` (hardcoded), `experiments/run_ablations.py` (hardcoded)

**Problem:** The code uses `tau=1.0` for training everywhere. The paper says (line 1795): "τ=100 for temperature". Lines 1809-1811: "we use temperature-scaled predictions h̃(x) = σ(τ·f_θ(x)) with τ=100 to produce near-binary outputs" for TRAINING. Line 1797-1799: "τ=100 works well for moderate corruption (α≤0.3), while reducing to τ=1 at α=0.4 on Adult".

The comment in the code says "Use τ=1 for training to maintain gradient flow" — this is a deviation from the paper that causes DRO to fail on Credit at α≥0.2.

**Why tau=100 works despite gradient concerns:** With tau=100, h_tilde is near-binary, making DP/IF violations clearly visible during training. With tau=1, h_tilde hovers around 0.5, making DP violation appear small even when the model is unfair. The paper's authors tested this and got good results. The gradient still flows through BCE (which uses raw logits), and through the Lagrangian terms via the chain rule. The key insight: gradient of sigmoid(100*x) at x=0 is 25 (large!), and the fairness signal is much sharper.

**EXACT FIX 1 — `src/training/dro_fair.py` line 28:** Change default tau from 1.0 to 100.0:

```python
# OLD:
                 lr_p=5e-3, lambda_max=10.0, tau=1.0, beta=5.0, k=5, gamma=0.0,
# NEW:
                 lr_p=5e-3, lambda_max=10.0, tau=100.0, beta=5.0, k=5, gamma=0.0,
```

**EXACT FIX 2 — `src/training/naive_fair.py` line 27:** Change default tau from 1.0 to 100.0:

```python
# OLD:
                 lambda_max=10.0, tau=1.0, k=5, gamma=0.0,
# NEW:
                 lambda_max=10.0, tau=100.0, k=5, gamma=0.0,
```

**EXACT FIX 3 — `experiments/run_experiments.py`:** Remove `tau_train = 1.0` and use proper temperature. Find these lines:

```python
    # Use tau=1 for training to maintain gradient flow through h_tilde.
    # tau=100 is only for evaluation (sharp fairness metrics).
    tau_train = 1.0
    tau_eval = get_temperature(alpha)
```

Replace with:

```python
    # Paper lines 1795, 1809-1811: tau=100 for training AND evaluation.
    # tau=100 for alpha<=0.3, tau=1 for alpha=0.4 (line 1797-1799).
    tau = get_temperature(alpha)
```

Then update all references: every place that uses `tau_train` should use `tau` instead, and every place that uses `tau_eval` should use `tau` instead. Specifically in the trainer constructors:

```python
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
        tau=tau, k=5, gamma=0.0,          # was tau_train
        epochs=15, weight_decay=1e-4
    )
```

And:

```python
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
        tau=tau, beta=5.0, k=5, gamma=0.0,  # was tau_train
        K_inner=10, epochs=15, weight_decay=1e-4
    )
```

**EXACT FIX 4 — `experiments/run_ablations.py`:** Same change. Find:

```python
    tau_train = 1.0
```

Replace with:

```python
    tau = get_temperature(alpha)
```

And update all trainer constructors to use `tau=tau` instead of `tau=tau_train`.

**EXACT FIX 5 — `src/models/classifier.py` line 26:** The `predict_proba` method uses DIVIDE, inconsistent with everything else using MULTIPLY:

```python
# OLD:
    def predict_proba(self, x, temperature=1.0):
        """Return soft predictions using sigmoid with temperature."""
        logits = self.forward(x)
        return torch.sigmoid(logits / temperature)
# NEW:
    def predict_proba(self, x, temperature=1.0):
        """Return soft predictions using sigmoid with temperature scaling."""
        logits = self.forward(x)
        return torch.sigmoid(logits * temperature)
```

**VERIFY:**

```bash
python3 -m pytest tests/test_end_to_end.py::test_tau_multiply_not_divide -v
```

Must pass.

---

### BUG 3: predict_proba divide vs multiply (covered above in Bug 2 Fix 5)

Already handled.

---

### BUG 4: Lambda initialization — start at 0.0, not 1.0

**File:** `src/training/dro_fair.py`, `src/training/naive_fair.py`

**Problem:** Both trainers initialize `lambda_dp = 1.0` and `lambda_if = 1.0`. This means the model penalizes fairness violations heavily from epoch 1, before it has even learned to classify. The paper says "λ_DP, λ_IF ← λ_0" without specifying λ_0, but standard Lagrangian optimization starts multipliers at 0 and lets dual ascent raise them as violations are detected. Starting at 1.0 creates an accuracy-fairness conflict from the very first gradient step.

**EXACT FIX — `src/training/dro_fair.py`:** Find:

```python
        lambda_dp = torch.tensor(1.0, device=self.device)
        lambda_if = torch.tensor(1.0, device=self.device)
```

Replace with:

```python
        lambda_dp = torch.tensor(0.0, device=self.device)
        lambda_if = torch.tensor(0.0, device=self.device)
```

**EXACT FIX — `src/training/naive_fair.py`:** Same change:

```python
# OLD:
        lambda_dp = torch.tensor(1.0, device=self.device)
        lambda_if = torch.tensor(1.0, device=self.device)
# NEW:
        lambda_dp = torch.tensor(0.0, device=self.device)
        lambda_if = torch.tensor(0.0, device=self.device)
```

This lets the model learn accuracy first, then gradually enforce fairness as violations accumulate through dual ascent. This is standard practice for Lagrangian constrained optimization.

---

### BUG 5: Epochs too low for meaningful training

**File:** `experiments/run_experiments.py`

**Problem:** Pretraining uses 15 epochs and fair training uses 15 epochs. The paper implies significantly more training (60s on A100 with 12x overhead vs 5s standard training).

**EXACT FIX:** In `experiments/run_experiments.py`, change all `epochs=15` to `epochs=30` for fair training:

```python
# For pretraining (standard ML), keep 15 epochs:
    pretrainer = StandardMLTrainer(model_pretrained, device=device, epochs=15, lr=1e-3)

# For Naive-FAIR, increase to 30:
    trainer_naive = NaiveFairTrainer(
        ...
        epochs=30, weight_decay=1e-4    # was 15
    )

# For DRO-FAIR, increase to 30:
    trainer_dro = DroFairTrainer(
        ...
        epochs=30, weight_decay=1e-4    # was 15
    )
```

Same in `experiments/run_ablations.py` — the ablation trainers already use `epochs=30`, so no change needed there.

---

## PHASE 2: FIX TESTS

### TEST FIX 1: Install pandas (fixes 2 test failures)

```bash
pip install pandas openpyxl xlrd
python3 -m pytest tests/test_end_to_end.py::test_lsac_is_real_not_synthetic -v
python3 -m pytest tests/test_end_to_end.py::test_scaler_no_leakage -v
```

Both must now pass (or fail for a non-pandas reason).

### TEST FIX 2: Update test defaults for tau=100

After changing default tau to 100, some tests that create trainers with default params will now use tau=100 instead of tau=1. Check if any tests break. If they do, explicitly pass `tau=1.0` to those test trainers (since test synthetic data is small and tau=100 makes gradients vanish on tiny data).

Specifically, in `tests/test_end_to_end.py`, for the tests that create DroFairTrainer or NaiveFairTrainer for small synthetic data (n=200, d=5), explicitly pass `tau=1.0`:

```python
# In test_dro_fair_runs_without_error:
    trainer = DroFairTrainer(
        model, alpha=0.2, device='cpu', epochs=5,
        K_inner=10, lr_p=5e-3, tau=1.0, beta=5.0, k=3,  # explicit tau=1.0 for small test data
        use_dp=True, use_if=True
    )

# In test_naive_fair_runs_without_error:
    trainer = NaiveFairTrainer(
        model, device='cpu', epochs=5,
        tau=1.0, k=3  # explicit tau=1.0
    )
```

Do this for ALL test functions that create trainers. The tests are testing algorithmic correctness on tiny data, not hyperparameter tuning. Always pass `tau=1.0` explicitly in tests.

### TEST FIX 3: Add NaN/inf guard to training loop

In `src/training/dro_fair.py`, after `history['train_loss'].append(total_loss.item())`, add:

```python
            # Guard against silent NaN/inf corruption
            if not np.isfinite(total_loss.item()):
                raise RuntimeError(f"Training diverged at epoch {epoch+1}: loss={total_loss.item()}")
```

Do the same in `src/training/naive_fair.py` after `history['train_loss'].append(total_loss.item())`.

### TEST FIX 4: Add algorithm-order verification test

Add this new test to `tests/test_end_to_end.py`:

```python
def test_algorithm_order_matches_paper():
    """Algorithm 1: theta update BEFORE inner max (paper lines 15, 20-24)."""
    import inspect
    from src.training.dro_fair import DroFairTrainer
    source = inspect.getsource(DroFairTrainer.fit)

    # Find positions of key operations
    theta_pos = source.find('opt_theta.step()')
    inner_max_pos = source.find('INNER MAXIMIZATION')

    assert theta_pos > 0, "opt_theta.step() not found in fit()"
    assert inner_max_pos > 0, "INNER MAXIMIZATION comment not found in fit()"
    assert theta_pos < inner_max_pos, \
        f"θ update (pos {theta_pos}) must come BEFORE inner max (pos {inner_max_pos}) per paper Algorithm 1"
```

### FULL TEST VERIFICATION

```bash
python3 -m pytest tests/ -v --tb=short
```

**Expected:** ALL tests pass (30/30 or 31/31 with the new test). Zero failures.

If any test fails, FIX IT BEFORE PROCEEDING. Do not move to Phase 3 with failing tests.

---

## PHASE 3: RUN EXPERIMENTS

### STEP 1: Delete old results

The old results were computed with wrong algorithm order and tau=1. They are INVALID.

```bash
rm -rf results/checkpoint.pkl results/all_results.json results/all_results.pkl results/run.log
rm -rf results/summary_stats.csv results/table1_latex.tex results/table1_results.csv
rm -rf results/runtimes.json results/reductions.json results/ablation_full.json
rm -rf figures/
```

### STEP 2: Smoke test (1 seed, 1 dataset, 1 alpha)

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from experiments.run_experiments import run_single_experiment
r = run_single_experiment('adult', 0.2, seed=0, verbose=True)
print()
print('=== SMOKE TEST RESULTS ===')
print(f'Naive: Acc={r[\"naive\"][\"clean\"][\"accuracy\"]:.4f} DP={r[\"naive\"][\"clean\"][\"dp_violation\"]:.4f} IF={r[\"naive\"][\"clean\"][\"if_violation\"]:.4f}')
print(f'DRO:   Acc={r[\"dro\"][\"clean\"][\"accuracy\"]:.4f} DP={r[\"dro\"][\"clean\"][\"dp_violation\"]:.4f} IF={r[\"dro\"][\"clean\"][\"if_violation\"]:.4f}')
dro_wins_dp = r['dro']['clean']['dp_violation'] < r['naive']['clean']['dp_violation']
print(f'DRO wins on DP: {dro_wins_dp}')
print(f'DRO acc reasonable (>0.70): {r[\"dro\"][\"clean\"][\"accuracy\"] > 0.70}')
"
```

**Expected output criteria:**
- DRO accuracy > 0.70 (reasonable, not degenerate)
- DRO DP violation < Naive DP violation (DRO should win on fairness)
- No NaN or inf values
- Both methods produce non-trivial predictions (not all same class)

**If DRO does NOT beat Naive on DP:** Something is still wrong. Re-read the code, compare to Algorithm 1 again. Common issues:
- p weights not actually changing (check projection radius)
- Gradient not flowing through fairness terms
- tau too aggressive for this data (try tau=10 as middle ground)

**If accuracy drops below 0.65:** Training is unstable. Try:
- Reducing lr_theta to 5e-4
- Increasing pretraining epochs to 20
- Reducing tau to 10 (still > 1 but not 100)

### STEP 3: 3-seed validation

```bash
python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
from experiments.run_experiments import run_single_experiment
for ds in ['adult', 'credit', 'lsac']:
    results = []
    for seed in range(3):
        r = run_single_experiment(ds, 0.2, seed=seed, verbose=False)
        results.append(r)
    naive_dp = np.mean([r['naive']['clean']['dp_violation'] for r in results])
    dro_dp = np.mean([r['dro']['clean']['dp_violation'] for r in results])
    naive_acc = np.mean([r['naive']['clean']['accuracy'] for r in results])
    dro_acc = np.mean([r['dro']['clean']['accuracy'] for r in results])
    print(f'{ds:6s}: Naive Acc={naive_acc:.3f} DP={naive_dp:.4f} | DRO Acc={dro_acc:.3f} DP={dro_dp:.4f} | DRO DP win: {dro_dp < naive_dp}')
"
```

**Expected:** DRO beats Naive on DP for at least 2/3 datasets at alpha=0.2.

### STEP 4: Full experiment run

Only proceed here if smoke test and 3-seed validation look correct.

```bash
python3 experiments/run_experiments.py
```

This runs 150 experiments (3 datasets x 5 alphas x 10 seeds). Takes 1-3 hours on CPU.

### STEP 5: Validate results

```bash
python3 -c "
import json, numpy as np
results = json.load(open('results/all_results.json'))
print(f'Total experiments: {len(results)}')
print()
print(f'{\"Dataset\":8s} {\"Alpha\":6s} {\"Naive Acc\":10s} {\"Naive DP\":10s} {\"DRO Acc\":10s} {\"DRO DP\":10s} {\"DRO wins\":10s}')
print('-' * 64)
for ds in ['adult', 'credit', 'lsac']:
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
        subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
        if not subset: continue
        n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in subset])
        d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in subset])
        n_ac = np.mean([r['naive']['clean']['accuracy'] for r in subset])
        d_ac = np.mean([r['dro']['clean']['accuracy'] for r in subset])
        n_se = np.std([r['dro']['clean']['dp_violation'] for r in subset]) / np.sqrt(len(subset))
        win = 'YES' if d_dp < n_dp else 'no'
        print(f'{ds:8s} {alpha:6.1f} {n_ac:10.4f} {n_dp:10.4f} {d_ac:10.4f} {d_dp:10.4f} {win:10s}')
        # Degeneracy check
        if n_se == 0:
            print(f'  WARNING: SE=0 for {ds} alpha={alpha} — possible degeneracy!')
"
```

**PASS CRITERIA (all must hold):**
1. 150 experiments completed
2. DRO beats Naive on DP at alpha=0.1, 0.2, 0.3 for majority of datasets
3. DRO accuracy > 0.65 for all datasets (no collapse)
4. No SE = 0.0000 across 10 seeds
5. At alpha=0.0: both methods similar (within 0.02 DP)

**FAIL CRITERIA (investigate if any hold):**
- DRO accuracy < 0.60 on any dataset → training instability
- DRO DP > Naive DP at alpha=0.2 → algorithm not working
- Any SE = 0.0000 → degenerate predictions (all same class)

---

## PHASE 4: GENERATE ALL OUTPUTS

```bash
python3 experiments/generate_results.py
```

This creates:
- `results/summary_stats.csv`
- `results/table1_results.csv`
- `results/table1_latex.tex`
- `results/reductions.json`
- `figures/main_results.png`
- `figures/test_time_eval.png`

### Run ablations

```bash
python3 experiments/run_ablations.py
```

### Run theory verification

```bash
python3 experiments/verify_theory.py
```

Must print "ALL THEORETICAL VERIFICATIONS PASSED".

---

## PHASE 5: FINAL VERIFICATION CHECKLIST

Run EVERY check below. ALL must pass. This is what a professor will verify.

### Check 1: All tests pass

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: 0 failures.

### Check 2: Algorithm 1 matches paper

Read `src/training/dro_fair.py` and verify the training loop order is:
1. Forward pass (h_tilde, losses)
2. θ update (opt_theta.step())
3. λ dual ascent
4. Inner max (K steps on p)

This matches paper Algorithm 1 lines 13-24.

### Check 3: tau is correct

```bash
python3 -c "
from src.training.dro_fair import DroFairTrainer
from src.training.naive_fair import NaiveFairTrainer
from src.models.classifier import MLPClassifier
# Check defaults
m = MLPClassifier(5)
t = DroFairTrainer(m, alpha=0.2)
print(f'DRO default tau: {t.tau}')  # Should be 100.0
t2 = NaiveFairTrainer(m)
print(f'Naive default tau: {t2.tau}')  # Should be 100.0
assert t.tau == 100.0, f'DRO tau should be 100.0, got {t.tau}'
assert t2.tau == 100.0, f'Naive tau should be 100.0, got {t2.tau}'
print('tau check PASSED')
"
```

### Check 4: Results exist and are valid

```bash
python3 -c "
import json, os
assert os.path.exists('results/all_results.json'), 'No results!'
results = json.load(open('results/all_results.json'))
assert len(results) == 150, f'Expected 150, got {len(results)}'
print(f'Results: {len(results)} experiments')

# Check no degeneracy
import numpy as np
for ds in ['adult', 'credit', 'lsac']:
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
        subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
        for method in ['naive', 'dro']:
            accs = [r[method]['clean']['accuracy'] for r in subset]
            se = np.std(accs) / np.sqrt(len(accs))
            assert se > 0, f'DEGENERATE: {ds} {alpha} {method} has SE=0'
            assert np.mean(accs) > 0.5, f'BAD: {ds} {alpha} {method} acc={np.mean(accs):.3f}'
print('All quality checks PASSED')
"
```

### Check 5: DRO beats Naive under corruption

```bash
python3 -c "
import json, numpy as np
results = json.load(open('results/all_results.json'))
wins = 0
tests = 0
for ds in ['adult', 'credit', 'lsac']:
    for alpha in [0.1, 0.2, 0.3]:
        subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
        n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in subset])
        d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in subset])
        tests += 1
        if d_dp < n_dp:
            wins += 1
            print(f'  {ds} alpha={alpha}: DRO DP={d_dp:.4f} < Naive DP={n_dp:.4f} ✓')
        else:
            print(f'  {ds} alpha={alpha}: DRO DP={d_dp:.4f} >= Naive DP={n_dp:.4f} ✗')
print(f'DRO wins {wins}/{tests} comparisons')
assert wins >= 6, f'DRO should win at least 6/9 DP comparisons, won {wins}'
print('DRO effectiveness check PASSED')
"
```

### Check 6: Deliverables exist

```bash
python3 -c "
import os
files = [
    'results/all_results.json',
    'results/summary_stats.csv',
    'results/table1_results.csv',
    'results/table1_latex.tex',
    'figures/main_results.png',
]
for f in files:
    assert os.path.exists(f), f'Missing: {f}'
    print(f'  ✓ {f}')
print('All deliverables present')
"
```

### Check 7: Theory verification

```bash
python3 experiments/verify_theory.py 2>&1 | tail -5
```

Must contain "ALL THEORETICAL VERIFICATIONS PASSED".

### Check 8: Reproducibility (same seed = same result)

```bash
python3 << 'PYEOF'
import sys, numpy as np; sys.path.insert(0, '.')
from experiments.run_experiments import run_single_experiment
r1 = run_single_experiment('adult', 0.2, seed=42, verbose=False)
r2 = run_single_experiment('adult', 0.2, seed=42, verbose=False)
acc1 = r1['dro']['clean']['accuracy']
acc2 = r2['dro']['clean']['accuracy']
dp1 = r1['dro']['clean']['dp_violation']
dp2 = r2['dro']['clean']['dp_violation']
print(f'Run 1: acc={acc1:.6f} dp={dp1:.6f}')
print(f'Run 2: acc={acc2:.6f} dp={dp2:.6f}')
assert abs(acc1 - acc2) < 1e-5, f'NOT REPRODUCIBLE: acc diff={abs(acc1-acc2)}'
assert abs(dp1 - dp2) < 1e-5, f'NOT REPRODUCIBLE: dp diff={abs(dp1-dp2)}'
print('Reproducibility check PASSED')
PYEOF
```

### Check 9: No NaN/inf in results

```bash
python3 << 'PYEOF'
import json, numpy as np
results = json.load(open('results/all_results.json'))
nan_count = 0
for r in results:
    for method in ['naive', 'dro']:
        for ev in ['clean', 'corrupted']:
            for metric in ['accuracy', 'dp_violation', 'if_violation']:
                v = r[method][ev][metric]
                if not np.isfinite(v):
                    print(f'NaN/inf: {r["dataset"]} a={r["alpha"]} s={r["seed"]} {method}/{ev}/{metric}={v}')
                    nan_count += 1
assert nan_count == 0, f'{nan_count} NaN/inf values found!'
print('NaN/inf check PASSED: all 150*2*2*3 = 1800 values are finite')
PYEOF
```

### Check 10: Convergence (training loss decreases)

```bash
python3 << 'PYEOF'
import sys, numpy as np; sys.path.insert(0, '.')
from experiments.run_experiments import run_single_experiment
# Run with verbose to capture loss history
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.corruption.adversarial import AdversarialCorruptor

X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset('adult', random_state=0)
input_dim = X_train.shape[1]

# Quick train to check convergence
model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
trainer = DroFairTrainer(model, alpha=0.2, device='cpu', epochs=20, K_inner=10, tau=100.0, k=5)

warm = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
from src.training.standard_ml import StandardMLTrainer
pre = StandardMLTrainer(warm, device='cpu', epochs=10, lr=1e-3)
pre.fit(X_train, y_train, verbose=False)

corr = AdversarialCorruptor(alpha=0.2, random_state=0)
X_c, y_c, a_c, _ = corr.corrupt(X_train, y_train, a_train, model=warm, device='cpu')

hist = trainer.fit(X_c, y_c, a_c, verbose=False)
losses = hist['train_loss']

first_5 = np.mean(losses[:5])
last_5 = np.mean(losses[-5:])
print(f'Loss first 5 epochs avg: {first_5:.4f}')
print(f'Loss last 5 epochs avg: {last_5:.4f}')
print(f'Decreased: {last_5 < first_5}')
# Allow small increase (fairness penalty can raise total loss) but not explosion
assert last_5 < first_5 * 2.0, f'Loss exploded: {first_5:.4f} -> {last_5:.4f}'
assert np.all(np.isfinite(losses)), 'NaN/inf in loss history!'
print('Convergence check PASSED')
PYEOF
```

### Check 11: Statistical significance (DRO vs Naive paired test)

```bash
python3 << 'PYEOF'
import json, numpy as np
from scipy import stats

results = json.load(open('results/all_results.json'))

print('=== PAIRED SIGN TEST: DRO DP < Naive DP ===')
for ds in ['adult', 'credit', 'lsac']:
    for alpha in [0.2, 0.3]:
        subset = [r for r in results if r['dataset'] == ds and r['alpha'] == alpha]
        if len(subset) < 5:
            continue
        # Paired comparison: same seed, same data split
        naive_dps = [r['naive']['clean']['dp_violation'] for r in subset]
        dro_dps = [r['dro']['clean']['dp_violation'] for r in subset]
        diffs = [n - d for n, d in zip(naive_dps, dro_dps)]  # positive = DRO wins
        wins = sum(1 for d in diffs if d > 0)
        n = len(diffs)
        mean_diff = np.mean(diffs)
        # Sign test: under H0 (no difference), wins ~ Binomial(n, 0.5)
        p_value = 1 - stats.binom.cdf(wins - 1, n, 0.5)
        print(f'  {ds:6s} a={alpha}: DRO wins {wins}/{n} seeds, mean_diff={mean_diff:+.4f}, p={p_value:.4f}')
print('Statistical significance check COMPLETE')
PYEOF
```

### Check 12: Runtime overhead

```bash
python3 -c "
import json
r = json.load(open('results/runtimes.json'))
print(f'Naive avg: {r[\"naive_mean\"]:.1f}s')
print(f'DRO avg: {r[\"dro_mean\"]:.1f}s')
print(f'Overhead: {r[\"overhead\"]:.1f}x')
assert r['overhead'] > 1.5, f'DRO should be slower than Naive due to inner max loop'
print('Runtime check PASSED')
"
```

---

## PAPER REFERENCE NUMBERS (for comparison)

These are from the paper's Table 1 with RANDOM corruption. Your results use ADVERSARIAL corruption (harder), so exact match is NOT expected. But DRO should still beat Naive on fairness.

```
Paper Table 1 (random corruption, clean test eval):

ADULT:
  α=0.0  Naive: Acc=.814 DP=.008 IF=.043  |  DRO: Acc=.813 DP=.005 IF=.043
  α=0.2  Naive: Acc=.833 DP=.168 IF=.067  |  DRO: Acc=.795 DP=.028 IF=.045
  α=0.3  Naive: Acc=.828 DP=.229 IF=.077  |  DRO: Acc=.786 DP=.065 IF=.052

CREDIT:
  α=0.0  Naive: Acc=.817 DP=.020 IF=.035  |  DRO: Acc=.817 DP=.017 IF=.034
  α=0.2  Naive: Acc=.815 DP=.020 IF=.036  |  DRO: Acc=.782 DP=.003 IF=.007
  α=0.3  Naive: Acc=.813 DP=.024 IF=.035  |  DRO: Acc=.788 DP=.009 IF=.012

LSAC:
  α=0.0  Naive: Acc=.907 DP=.006 IF=.012  |  DRO: Acc=.908 DP=.008 IF=.014
  α=0.2  Naive: Acc=.901 DP=.010 IF=.015  |  DRO: Acc=.897 DP=.014 IF=.005
  α=0.3  Naive: Acc=.900 DP=.006 IF=.018  |  DRO: Acc=.891 DP=.026 IF=.006
```

Key observations from the paper:
- DRO accuracy is 1-5% lower than Naive (accuracy-fairness tradeoff)
- DRO DP is MUCH lower than Naive at α=0.2, 0.3 (that's the whole point)
- At α=0.0, both methods are similar (no corruption to be robust against)
- LSAC has ~90% accuracy (high base rate for bar passage)

With adversarial corruption, you should see:
- Larger DP violations for Naive (adversarial is harder than random)
- DRO should still reduce DP relative to Naive
- Accuracy may be slightly lower than paper (adversarial hurts more)

---

## MATHEMATICAL GROUND TRUTH

These formulas are from the paper. The code MUST implement them EXACTLY.

**Algorithm 1, line 7:** h̃_i = σ(τ · f_θ(x_i^c))
- `torch.sigmoid(logits * self.tau)` — MULTIPLY by tau

**Algorithm 1, line 9:** L_tilt = β · log(1/|M| · Σ exp(ℓ_i/β))
- `beta * (logsumexp(loss/beta) - log(m))` — numerically stable via logsumexp

**Algorithm 1, line 10:** h̄_j = Σ_{i:a_i=j} p̃_{j,i} · h̃_i
- Weighted group mean using importance weights p

**Algorithm 1, line 11:** g_DP = |h̄_1 - h̄_0|
- `torch.abs(rate_1 - rate_0)`

**Algorithm 1, line 12:** g_IF = 1/(n-1) · Σ_i Σ_{j∈N(i)} (p̃_i^IF + p̃_j^IF)/2 · (|h̃_i - h̃_j| - d_ij - γ)_+
- Divide by (n-1), NOT by number of edges
- Weight by (p_i + p_j)/2
- ReLU activation: max(0, |h_i - h_j| - d_ij - gamma)

**Eq. 16 — Uncertainty sets:**
- U_j = {p̃ ∈ Δ_{n_j} : ||p̃ - p̂_{n_j}||_1 ≤ 2·ρ_{DP,j}}
- L1 radius = 2·ρ (not ρ)

**Theorem 4.2 — DP radii:**
- ρ_{DP,j} = α / ((1-α)·π_j + α)

**Theorem 4.3 — IF radius:**
- ρ_IF = 2α - α²

**Projection:** Dykstra's alternating projection onto simplex ∩ L1-ball

---

## IF SOMETHING GOES WRONG

**DRO accuracy collapses (< 0.60):**
- tau=100 may be too aggressive. Try tau=50 or tau=10.
- Increase pretraining epochs to 20
- Reduce lr_lambda to 1e-3 (slower Lagrangian growth)
- Increase lambda_max to 20 (allow more fairness enforcement headroom)

**DRO doesn't beat Naive on DP:**
- Check K_inner is 10 (not 1 or 0)
- Check rho values are positive (print them during training)
- Check p weights are actually changing (print before/after inner max)
- Verify projection radius is 2*rho (not rho)

**Constant predictions (all same class):**
- Reduce tau (try 10, then 50, then 100)
- Increase lr_theta
- Add batch normalization to the model
- Check that class weights in training data aren't extreme

**Tests fail after code changes:**
- Run pytest IMMEDIATELY after each code change
- If a test fails, fix it BEFORE proceeding
- Tests using small synthetic data (n=200) should use tau=1.0 explicitly

---

## EXECUTION ORDER SUMMARY

```text
 1. pip install pandas openpyxl xlrd matplotlib seaborn scipy
 2. Fix Bug 1: Algorithm order in dro_fair.py (theta->lambda->inner max)
 3. Fix Bug 2: tau=100 defaults in dro_fair.py, naive_fair.py
 4. Fix Bug 2 cont: tau in run_experiments.py, run_ablations.py
 5. Fix Bug 2 cont: predict_proba in classifier.py (divide -> multiply)
 6. Fix Bug 4: lambda init 1.0 -> 0.0 in dro_fair.py, naive_fair.py
 7. Fix Bug 5: epochs 15 -> 30 in run_experiments.py
 8. Add NaN/inf guard to both training loops
 9. Fix tests: explicit tau=1.0 in test constructors
10. Add new test: test_algorithm_order_matches_paper
11. Run: python3 -m pytest tests/ -v   (ALL PASS or stop)
12. Delete old results (checkpoint.pkl, all_results.json, etc.)
13. Smoke test: 1 seed adult alpha=0.2
14. If smoke fails -> diagnose -> retry from step 2
15. 3-seed validation: adult/credit/lsac alpha=0.2
16. If validation fails -> diagnose -> retry
17. Full run: python3 experiments/run_experiments.py
18. Generate outputs: python3 experiments/generate_results.py
19. Run ablations: python3 experiments/run_ablations.py
20. Run theory: python3 experiments/verify_theory.py
21. Run FINAL VERIFICATION checklist (all 12 checks):
      Check 1:  All tests pass
      Check 2:  Algorithm order correct
      Check 3:  tau=100 defaults
      Check 4:  Results exist (150 experiments)
      Check 5:  DRO beats Naive (6/9 DP wins)
      Check 6:  Deliverables exist
      Check 7:  Theory verification
      Check 8:  Reproducibility (same seed = same output)
      Check 9:  No NaN/inf in results
      Check 10: Convergence (loss decreases)
      Check 11: Statistical significance (paired sign test)
      Check 12: Runtime overhead
22. If all 12 pass -> DONE. If any fail -> fix and re-verify.
```

DO NOT SKIP STEPS. DO NOT DECLARE DONE EARLY. EXECUTE SEQUENTIALLY.
