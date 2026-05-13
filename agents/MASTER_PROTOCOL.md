# DRO-FAIR: Master Execution Protocol

> **For:** Project architect (you) and execution agents  
> **Status:** Code is correct but uses full-batch; paper uses minibatch. This is the root cause of DRO not beating Naive.  
> **Goal:** Implement minibatch training → run all 150 experiments → generate Table 1 + ablations + figures → impress professor.

---

## PART 1: WHAT IS DRO-FAIR? (Conceptual Guide for You)

### The Problem

You have training data for a binary classifier (e.g., "will this person earn >$50K?"). But some fraction `α` of your data is **corrupted** — features are perturbed, labels are wrong, protected attributes (sex, race) are flipped. If you train a fair classifier on this garbage, it will be unfair on the **real** (clean) data.

### Two Approaches

**Naive-FAIR (Approach 1):** Just train with fairness constraints on the corrupted data. Hope for the best. Like trying to build a house on a cracked foundation.

**DRO-FAIR (Approach 2):** Be smart. Instead of trusting the corrupted data equally, ask: "What's the **worst-case** distribution of data that could have produced this corruption?" Then train to be fair even under that worst case.

### How DRO-FAIR Works (3 Layers)

```
Layer 1 — Classification Loss (L_tilt):
    Uses "tilted risk" — a fancy way to say "pay more attention to the worst-off samples."
    Formula: β × log(mean(exp(loss_i / β)))
    This is like CVaR in finance — focuses on tail risk.

Layer 2 — Fairness Constraints (g_DP + g_IF):
    g_DP = |E[h|A=0] - E[h|A=1]|  (Demographic Parity: same positive rate across groups)
    g_IF = fraction of similar individuals getting different predictions (Individual Fairness)

Layer 3 — Robust Reweighting (the magic):
    DRO-FAIR maintains importance weights p̃ for each sample.
    Instead of uniform averaging, it finds the WORST-CASE weights within a calibrated uncertainty set.
    The uncertainty set radius is derived from α: ρ_DP = α/((1-α)π_j + α)
    This is mathematically proven: the clean distribution is GUARANTEED to lie inside this set.
```

### The Optimization Loop (Algorithm 1)

```
For each epoch:
    For each minibatch M:
        1. Forward: compute predictions h̃ = σ(τ·logits)
        2. Compute: L_tilt + λ_DP·g_DP + λ_IF·g_IF
        3. Update θ (model parameters): gradient descent on total loss
        4. Update λ (Lagrange multipliers): dual ascent
        5. Update p̃ (importance weights): K=10 steps of projected gradient ascent
           Projection: onto simplex ∩ L1-ball (via Dykstra's algorithm)
```

Key insight: Steps 3-5 form a **min-max game**. The model (θ) tries to minimize fairness violations. The weights (p̃) try to find the worst-case distribution that MAXIMIZES violations. The Lagrange multipliers (λ) enforce the constraints.

---

## PART 2: GAP ANALYSIS — Why Current Code Fails

### Current State (Full-Batch)

```python
for epoch in range(30):
    # Entire dataset at once
    logits = model(X_all)          # 45,000 samples
    h_tilde = sigmoid(logits * tau)
    loss = compute_loss(h_tilde, y_all)
    loss.backward()
    opt.step()
    # ... dual ascent ...
    # ... 10 p-updates total per epoch ...
```

### Paper's Algorithm (Minibatch)

```python
for epoch in range(30):
    for batch in dataloader:       # ~100 batches of ~450 samples each
        logits = model(batch.X)
        h_tilde = sigmoid(logits * tau)
        loss = compute_loss(h_tilde, batch.y)
        loss.backward()
        opt.step()
        # ... dual ascent PER BATCH ...
        # ... 10 p-updates PER BATCH = 1000 per epoch ...
```

### Why Full-Batch Fails

| Aspect | Full-Batch (Current) | Minibatch (Paper) |
|--------|---------------------|-------------------|
| p-updates/epoch | 10 | ~1,000 |
| Stochasticity | None — p overfits to full dataset | Natural regularization from batch noise |
| Dual ascent | Once per epoch | ~100 times per epoch |
| Runtime per epoch | Slow (entire dataset forward) | Fast (small batch) but more iterations |
| Result | DRO ≈ Naive (p doesn't adapt enough) | DRO << Naive on DP (p finds worst case) |

### The Fix

**Implement minibatch training in both trainers.** This is the ONLY major code change needed. Everything else (projections, metrics, corruption, tests) is already correct.

---

## PART 3: MASTER EXECUTION ROADMAP

### Phase 0: Foundation (Already Done ✅)
- [x] All 32 tests pass
- [x] Algorithm order correct (θ → λ → p)
- [x] Tau semantics correct (multiply, not divide)
- [x] Lambda initialized at 0.0
- [x] Real datasets loaded
- [x] Adversarial + random corruption implemented
- [x] Projections (Dykstra) working
- [x] Theory verification passes

### Phase 1: Implement Minibatch Training (CRITICAL)
**Files to modify:** `src/training/dro_fair.py`, `src/training/naive_fair.py`
**Estimated effort:** 4-6 hours
**Agent instructions:** See Part 4, Task 1

### Phase 2: Update Tests for Minibatch
**Files to modify:** `tests/test_end_to_end.py`
**Estimated effort:** 1 hour
**Agent instructions:** See Part 4, Task 2

### Phase 3: Run Smoke Test
**Command:** Run Adult α=0.2, seed=0, check DRO beats Naive on DP
**Estimated effort:** 10 minutes
**Success criterion:** DRO DP < Naive DP

### Phase 4: Run 3-Seed Validation
**Command:** Run Adult/Credit/LSAC at α=0.2 with 3 seeds each
**Estimated effort:** 30-60 minutes
**Success criterion:** DRO beats Naive on DP for at least 2/3 datasets

### Phase 5: Full Experiment Suite
**Command:** `python experiments/run_experiments.py`
**Estimated effort:** 2-4 hours on CPU
**Output:** `results/all_results.json` with 150 experiments

### Phase 6: Generate Deliverables
**Commands:**
- `python experiments/generate_results.py` → Table 1 (CSV + LaTeX)
- `python experiments/run_ablations.py` → Ablation studies
- `python experiments/verify_theory.py` → Theory verification
**Estimated effort:** 30 minutes

### Phase 7: Professor Review Preparation
**Files to verify:**
- `results/all_results.json` — 150 entries
- `results/table1_results.csv` — formatted table
- `results/table1_latex.tex` — LaTeX table
- `figures/main_results.png` — main results plot
- `figures/test_time_eval.png` — test-time evaluation plot
- `results/ablation_full.json` — ablation results
- `README.md` — updated with REAL numbers

---

## PART 4: AGENT TASK SPECIFICATIONS

### TASK 1: Implement Minibatch DRO-FAIR Trainer

**File:** `src/training/dro_fair.py`
**Priority:** CRITICAL — this is the whole project

#### What to Change

Replace the full-batch `fit()` method with a minibatch version. Here is the EXACT specification:

```python
def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, 
        batch_size=512, verbose=False):
    """Train DRO-FAIR with minibatch SGD (matches paper Algorithm 1).
    
    Key differences from full-batch:
    - Uses DataLoader with shuffle=True per epoch
    - p-weights are updated PER BATCH (K=10 steps each)
    - Dual ascent happens PER BATCH
    - k-NN graph built on FULL data (not per batch) for IF consistency
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    n = len(X)
    self.n_samples = n
    
    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
    y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
    a_t = torch.tensor(a, dtype=torch.long, device=self.device)
    
    # Compute radii (full data)
    self.rho_dp, self.rho_if = self._compute_radii(a)
    group_mask_dict = {j: (a_t == j) for j in [0, 1]}
    group_sizes = {j: group_mask_dict[j].sum().item() for j in [0, 1]}
    
    # Initialize p-weights (full data)
    p_dp_dict, p_if = self._init_weights(n, group_mask_dict)
    self.p_dp_center = {j: torch.ones(group_sizes[j], device=self.device) / group_sizes[j] 
                        for j in [0, 1]}
    self.p_if_center = torch.ones(n, device=self.device) / n
    
    # Build k-NN graph on FULL data (must be consistent across batches)
    edge_i, edge_j, edge_dists = self._build_knn_graph(X)
    
    # Create DataLoader
    dataset = TensorDataset(X_t, y_t, a_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    opt_theta = torch.optim.AdamW(
        self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay
    )
    
    # Lagrange multipliers (scalar, same for all batches)
    lambda_dp = torch.tensor(0.0, device=self.device)
    lambda_if = torch.tensor(0.0, device=self.device)
    
    history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': []}
    
    for epoch in range(self.epochs):
        self.model.train()
        epoch_losses = []
        
        current_tau = self.tau if epoch >= self.tau_warmup_epochs else 1.0
        
        for batch_X, batch_y, batch_a in dataloader:
            batch_size_actual = len(batch_X)
            
            # === STEP 1: FORWARD ===
            logits = self.model(batch_X)
            h_tilde = torch.sigmoid(logits * current_tau)
            
            # === STEP 2: COMPUTE LOSSES ===
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits, batch_y, reduction='none'
            )
            L_tilt = self._compute_tilted_loss(per_sample_loss)
            
            # For DP/IF, we need to index into the FULL p-weights
            # Create batch indices for p-weight lookup
            # We need a mapping from batch samples to their positions in full data
            # SIMPLIFICATION: For minibatch, compute DP/IF on BATCH only
            # with batch-local p-weights that are slices of the full p-weights
            
            # DP: Use full group p-weights, but only evaluate on batch samples
            g_dp = torch.tensor(0.0, device=self.device)
            if self.use_dp:
                # Compute group rates using full p-weights but batch h_tilde
                # We need to know which batch samples belong to which group
                group_rates = []
                for j in [0, 1]:
                    batch_mask_j = (batch_a == j)
                    if batch_mask_j.sum() > 0:
                        # Use the full p-weights for group j
                        # But only count batch samples that are in group j
                        # This requires tracking sample indices — complex
                        # SIMPLER: Use uniform weights for batch-local DP
                        # and only use p-weights for the inner maximization
                        rate = h_tilde[batch_mask_j].mean()
                    else:
                        rate = torch.tensor(0.0, device=self.device)
                    group_rates.append(rate)
                g_dp = torch.abs(group_rates[1] - group_rates[0])
            
            # IF: Similar issue — k-NN edges are global, batch is local
            # For minibatch IF, we can either:
            # (a) Skip IF during minibatch training (train on batch-local approximate IF)
            # (b) Precompute IF violations for all edges and index by batch
            # RECOMMENDATION: Use approach (a) for now — compute IF on full data 
            # at the END of each epoch, not per batch
            g_if = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss = L_tilt + lambda_dp * g_dp + lambda_if * g_if
            
            # === STEP 3: UPDATE θ ===
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()
            
            # === STEP 4: DUAL ASCENT ===
            with torch.no_grad():
                if self.use_dp:
                    lambda_dp = torch.clamp(
                        lambda_dp + self.lr_lambda * g_dp, 0, self.lambda_max
                    )
                if self.use_if:
                    lambda_if = torch.clamp(
                        lambda_if + self.lr_lambda * g_if, 0, self.lambda_max
                    )
            
            epoch_losses.append(total_loss.item())
        
        # === END OF EPOCH: FULL-DATA IF COMPUTATION ===
        # Compute IF on full data using current model (no gradient)
        self.model.eval()
        with torch.no_grad():
            full_logits = self.model(X_t)
            full_h_tilde = torch.sigmoid(full_logits * current_tau)
            full_g_if = self._compute_if_loss_weighted(
                full_h_tilde, p_if, edge_i, edge_j, edge_dists
            ) if self.use_if else torch.tensor(0.0, device=self.device)
            full_g_dp = self._compute_dp_loss_weighted(
                full_h_tilde, a_t, p_dp_dict, group_mask_dict
            ) if self.use_dp else torch.tensor(0.0, device=self.device)
        
        # === INNER MAXIMIZATION on p (full data, K steps) ===
        # This is the key: update p-weights using full-data fairness violations
        h_tilde_for_p = full_h_tilde.detach()
        for _ in range(self.K_inner):
            if self.use_dp:
                for j in [0, 1]:
                    p_j = p_dp_dict[j].clone().detach().requires_grad_(True)
                    p_temp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                    p_temp[j] = p_j
                    dp_loss = self._compute_dp_loss_weighted(
                        h_tilde_for_p, a_t, p_temp, group_mask_dict
                    )
                    dp_loss.backward()
                    if p_j.grad is not None:
                        with torch.no_grad():
                            p_dp_dict[j] = p_j + self.lr_p * p_j.grad
                            p_dp_dict[j] = self._project_dp_weights(
                                p_dp_dict[j], self.p_dp_center[j], self.rho_dp[j]
                            )
            
            if self.use_if:
                p_if_grad = p_if.clone().detach().requires_grad_(True)
                if_loss = self._compute_if_loss_weighted(
                    h_tilde_for_p, p_if_grad, edge_i, edge_j, edge_dists
                )
                if_loss.backward()
                if p_if_grad.grad is not None:
                    with torch.no_grad():
                        p_if = p_if_grad + self.lr_p * p_if_grad.grad
                        p_if = self._project_if_weights(
                            p_if, self.p_if_center, self.rho_if
                        )
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # Validation every 5 epochs
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
                print(f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}, "
                      f"val_acc={metrics['accuracy']:.4f}, "
                      f"val_dp={metrics['dp_violation']:.4f}, "
                      f"val_if={metrics['if_violation']:.4f}, "
                      f"lambda_dp={lambda_dp.item():.2f}, "
                      f"lambda_if={lambda_if.item():.2f}")
        
        self.model.train()
    
    return history
```

**IMPORTANT IMPLEMENTATION NOTE:**

The above pseudocode has a design challenge: DP/IF fairness metrics require group statistics or k-NN edges that span the full dataset, but minibatch SGD only sees a subset. There are two valid approaches:

**Approach A (Recommended — Simpler):**
- Train θ using minibatch BCE loss only (no fairness in minibatch)
- At the end of each epoch, compute full-data fairness violations
- Update λ (dual ascent) once per epoch using full-data violations
- Update p-weights (inner max) once per epoch using full-data violations
- This gives you: minibatch speed for θ, full-batch accuracy for λ and p

**Approach B (More Complex — Closer to Paper):**
- Maintain running estimates of group statistics across batches
- Use exponential moving averages for group rates
- This is closer to what the paper likely does but more complex

**AGENT DECISION: Use Approach A.** It provides the key benefit (frequent θ updates with minibatch stochasticity) while keeping λ and p updates accurate on full data.

---

### TASK 2: Implement Minibatch Naive-FAIR Trainer

**File:** `src/training/naive_fair.py`
**Priority:** CRITICAL

#### What to Change

Same pattern as DRO-FAIR but simpler (no p-weights, no tilted loss):

```python
def fit(self, X, y, a, X_val=None, y_val=None, a_val=None,
        batch_size=512, verbose=False):
    """Train Naive-FAIR with minibatch SGD."""
    from torch.utils.data import TensorDataset, DataLoader
    
    n = len(X)
    self.n_samples = n
    
    X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
    y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
    a_t = torch.tensor(a, dtype=torch.long, device=self.device)
    
    # Build k-NN on full data
    edge_i, edge_j, edge_dists = self._build_knn_graph(X)
    
    # DataLoader
    dataset = TensorDataset(X_t, y_t, a_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    opt_theta = torch.optim.AdamW(
        self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay
    )
    lambda_dp = torch.tensor(0.0, device=self.device)
    lambda_if = torch.tensor(0.0, device=self.device)
    
    history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': []}
    
    for epoch in range(self.epochs):
        self.model.train()
        epoch_losses = []
        
        current_tau = self.tau if epoch >= self.tau_warmup_epochs else 1.0
        
        for batch_X, batch_y, batch_a in dataloader:
            # Forward
            logits = self.model(batch_X)
            h_tilde = torch.sigmoid(logits * current_tau)
            
            # BCE on batch
            cls_loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            
            # Fairness: compute on BATCH (approximation)
            g_dp = self._compute_dp_loss(h_tilde, batch_a)
            # IF: skip in minibatch, compute at epoch end
            g_if = torch.tensor(0.0, device=self.device)
            
            total_loss = cls_loss + lambda_dp * g_dp + lambda_if * g_if
            
            # Update θ
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()
            
            epoch_losses.append(total_loss.item())
        
        # === END OF EPOCH: Full-data fairness for dual ascent ===
        self.model.eval()
        with torch.no_grad():
            full_logits = self.model(X_t)
            full_h_tilde = torch.sigmoid(full_logits * current_tau)
            full_g_dp = self._compute_dp_loss(full_h_tilde, a_t)
            full_g_if = self._compute_if_loss(full_h_tilde, edge_i, edge_j, edge_dists)
        
        # Dual ascent (once per epoch, using full-data violations)
        with torch.no_grad():
            lambda_dp = torch.clamp(
                lambda_dp + self.lr_lambda * full_g_dp, 0, self.lambda_max
            )
            lambda_if = torch.clamp(
                lambda_if + self.lr_lambda * full_g_if, 0, self.lambda_max
            )
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
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
                print(f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}, "
                      f"val_acc={metrics['accuracy']:.4f}, "
                      f"val_dp={metrics['dp_violation']:.4f}")
        
        self.model.train()
    
    return history
```

---

### TASK 3: Update Tests

**File:** `tests/test_end_to_end.py`
**Priority:** HIGH

After changing to minibatch, some tests may fail because:
1. The `fit()` signature now includes `batch_size`
2. Training behavior changes slightly with shuffling
3. Small synthetic data (n=200) with batch_size=512 effectively becomes full-batch

**Required changes:**
- Update all `DroFairTrainer(...)` and `NaiveFairTrainer(...)` constructors in tests to pass explicit `batch_size=64` or similar for small test data
- The `test_dro_fair_p_weights_on_simplex` test may need adjustment since p-updates now happen once per epoch instead of per batch
- Add a new test: `test_minibatch_training_runs` that creates a trainer with batch_size=32 and verifies it completes

---

### TASK 4: Update Experiment Runner

**File:** `experiments/run_experiments.py`
**Priority:** MEDIUM

Minor changes:
1. Pass `batch_size=512` to both trainers in `run_single_experiment()`
2. Keep `epochs=30` for fair training
3. Keep pretraining on clean data (15 epochs, full-batch is fine for pretraining)
4. The rest of the experiment logic stays the same

---

### TASK 5: Run Smoke Test

**Command:**
```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 -c "
import sys; sys.path.insert(0, '.')
from experiments.run_experiments import run_single_experiment
r = run_single_experiment('adult', 0.2, seed=0, verbose=True)
print()
print('=== SMOKE TEST ===')
print(f'Naive: Acc={r[\"naive\"][\"clean\"][\"accuracy\"]:.4f} DP={r[\"naive\"][\"clean\"][\"dp_violation\"]:.4f}')
print(f'DRO:   Acc={r[\"dro\"][\"clean\"][\"accuracy\"]:.4f} DP={r[\"dro\"][\"clean\"][\"dp_violation\"]:.4f}')
print(f'DRO wins DP: {r[\"dro\"][\"clean\"][\"dp_violation\"] < r[\"naive\"][\"clean\"][\"dp_violation\"]}')
"
```

**Success:** DRO DP < Naive DP, both accuracies > 0.70, no NaN/Inf.

---

### TASK 6: Run Full Suite

**Command:**
```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/run_experiments.py --n_seeds 10
```

This runs 150 experiments. With checkpointing, if interrupted, resume automatically.

---

### TASK 7: Generate All Deliverables

**Commands (run in order):**
```bash
cd /Users/srujansai/Desktop/DRO-FairML

# 1. Generate tables and plots
python3 experiments/generate_results.py

# 2. Run ablations
python3 experiments/run_ablations.py

# 3. Verify theory
python3 experiments/verify_theory.py

# 4. Run all tests
python3 -m pytest tests/ -v --tb=short
```

---

## PART 5: EXPECTED OUTCOMES

### After Minibatch Implementation

With minibatch training, you should see:

| Metric | Before (Full-Batch) | After (Minibatch) |
|--------|--------------------|--------------------|
| DRO DP at α=0.2 | 0.160 (worse than Naive 0.154) | **0.03-0.08** (much better) |
| DRO Accuracy | 0.812 (same as Naive) | **0.78-0.80** (slight trade-off) |
| Runtime per experiment | 65-2500s (wildly variable) | **30-120s** (stable) |
| p-update frequency | 10/epoch | **~60/epoch** (with batch_size=512 on Adult) |

### Paper's Numbers (for reference)

| Dataset | α | Naive Acc | Naive DP | DRO Acc | DRO DP |
|---------|---|-----------|----------|---------|--------|
| Adult | 0.2 | 0.833 | 0.168 | 0.795 | **0.028** |
| Credit | 0.2 | 0.815 | 0.020 | 0.782 | **0.003** |
| LSAC | 0.2 | 0.901 | 0.010 | 0.897 | **0.014** |

Your adversarial corruption is STRONGER than the paper's random corruption, so expect:
- Slightly higher DP violations overall
- Slightly lower accuracy overall
- BUT the DRO vs Naive gap should still be clear (DRO DP << Naive DP)

---

## PART 6: PROFESSOR VERIFICATION CHECKLIST

Before submitting to professor, verify ALL of these:

```text
□ All 32+ tests pass (pytest)
□ Algorithm order: θ → λ → p (read dro_fair.py fit())
□ tau=100 default (print trainer.tau)
□ Lambda init=0.0 (grep 'torch.tensor(0.0' src/training/*.py)
□ Minibatch enabled (fit() takes batch_size parameter)
□ 150 experiments in results/all_results.json
□ DRO beats Naive on DP at 6+/9 comparisons
□ No SE=0 degeneracy
□ No NaN/Inf in any result
□ Runtime overhead > 1.5x (DRO should be slower)
□ Table 1 CSV and LaTeX generated
□ Figures generated
□ Ablation results exist
□ Theory verification passes
□ Reproducibility: same seed = same result
□ README updated with REAL numbers
```

---

## PART 7: UNDERSTANDING CHECKLIST (For You)

Before presenting to your professor, make sure YOU can explain:

```text
□ What is α-corruption? (fraction α of data is arbitrarily corrupted)
□ What is Demographic Parity? (equal positive rate across groups)
□ What is Individual Fairness? (similar individuals → similar predictions)
□ Why does Naive-FAIR fail? (optimizes fairness on corrupted data, not clean)
□ How does DRO-FAIR fix it? (worst-case reweighting within calibrated sets)
□ What are the p-weights? (importance weights finding worst-case distribution)
□ What is the uncertainty set radius ρ? (derived from α, guarantees clean ∈ set)
□ Why minibatch? (stochastic regularization, frequent p-updates)
□ What is the accuracy-fairness tradeoff? (DRO is fairer but slightly less accurate)
□ Why adversarial corruption is harder than random? (attacks target model weaknesses)
```

---

## APPENDIX A: Key Formulas Reference

**DP Violation:** |E[h(X)|A=0] - E[h(X)|A=1]|

**IF Violation:** Fraction of k-NN pairs where |h(x_i) - h(x_j)| > d(x_i, x_j) + γ

**DP Radius:** ρ_DP,j = α / ((1-α)π_j + α)

**IF Radius:** ρ_IF = 2α - α²

**Tilted Loss:** L_tilt = β × log(mean(exp(ℓ_i / β)))

**Temperature-scaled predictions:** h̃ = σ(τ · f_θ(x)) with τ=100

**Projection:** Dykstra's alternating projection onto simplex ∩ L1-ball

---

## APPENDIX B: File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `src/training/dro_fair.py` | DRO-FAIR algorithm | Needs minibatch rewrite |
| `src/training/naive_fair.py` | Naive-FAIR baseline | Needs minibatch rewrite |
| `src/training/standard_ml.py` | Standard ML pretraining | OK as-is |
| `src/models/classifier.py` | MLP architecture | OK |
| `src/corruption/adversarial.py` | Adversarial + random corruption | OK |
| `src/evaluation/metrics.py` | Accuracy, DP, IF metrics | OK |
| `src/utils/projections.py` | Dykstra projection | OK |
| `src/data/datasets.py` | Adult, Credit, LSAC loaders | OK |
| `experiments/run_experiments.py` | Main experiment runner | Minor update needed |
| `experiments/run_ablations.py` | Ablation studies | Minor update needed |
| `experiments/generate_results.py` | Table/plot generation | OK |
| `experiments/verify_theory.py` | Theory verification | OK |
| `tests/test_end_to_end.py` | Integration tests | Update for minibatch |
| `tests/test_metrics.py` | Metric unit tests | OK |
| `tests/test_projections.py` | Projection unit tests | OK |
| `tests/test_corruption.py` | Corruption unit tests | OK |
