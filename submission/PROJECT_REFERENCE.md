# DRO-FAIR — Project Reference

A quick-reference card for the project: facts, numbers, file structure, and conventions used in this implementation.

---

## Project Overview

This repository implements **DRO-FAIR** (Distributionally Robust Optimization for Fair Classification) from an ICML submission, extended with multi-modal adversarial corruption (PGD feature attacks, coordinated label flips, minority-targeted attribute flips).

**Key result.** DRO-FAIR achieves statistically significant DP reductions on Credit (up to −92%) and LSAC (up to −100%) under adversarial corruption. Adult exhibits a documented failure mode at high α — an honest empirical limitation discussed in Section 5 of the report.

---

## Authoritative Data Source

`results/all_results.json` is the ground truth. All claims in the report and figures are verified against this file. It contains 150 experiments: 3 datasets × 5 α values × 10 seeds.

---

## Statistical Win Criterion

The report uses **Wilcoxon signed-rank test, p < 0.05** as the official criterion (one-sided, paired n=10).

| Criterion | DP wins | IF wins |
|-----------|---------|---------|
| Wilcoxon p < 0.05 (used in report) | 6/9 | 5/9 |
| Mean-based (reference only)        | 6/9 | 7/9 |

Cells are evaluated at α ∈ {0.1, 0.2, 0.3} across the three datasets.

---

## Headline Numbers

| Dataset | α   | DP reduction | IF reduction | Accuracy drop |
|---------|-----|--------------|--------------|---------------|
| LSAC    | 0.3 | −99.6%       | −100.0%      | −0.1 pp       |
| Credit  | 0.3 | −91.8%       | −96.0%       | −1.9 pp       |
| Adult   | 0.3 | −206% (collapse) | —        | −28 pp        |

---

## Known Failure Mode (Adult)

At α ≥ 0.3, DRO-FAIR's accuracy collapses to roughly 25–40% on 6/10 seeds for Adult. Mechanism: baseline DP on Adult is ~0.17, an order of magnitude larger than Credit/LSAC. Coordinated adversarial label flips amplify this disparity until λ_DP saturates at `λ_max`, causing the inner maximization to push sample weights to extremes and the model to collapse toward constant predictions. This is documented as an empirical limitation, not a code bug.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda_max` | 1.5 | Stability adjustment (paper uses 2.0) |
| `tau_warmup_epochs` | 15 | Stability (paper uses 10) |
| `grad_clip` | 0.5 | Stability (paper uses 1.0) |
| `lambda_lr_decay` | 0.95^epoch | Stability, dampens λ growth |
| `epochs` | 60 | Matches paper §7.1 |
| `K_inner` | 10 | Matches paper §G.4 |
| `lr_theta`, `lr_lambda`, `lr_p` | 1e-3, 5e-3, 5e-3 | Match paper §G.4 |
| `beta` (tilt) | 5.0 | Matches paper §G.5 |
| `tau` | 100 (α≤0.3), 1 (α≥0.4) | Matches paper §G.6 |
| `k` (kNN) | 5 | Matches paper §7.1 |
| `gamma` (IF tolerance) | 0.0 | Matches paper §7.1 |

---

## Algorithm 1 Implementation Checklist

When modifying [src/training/dro_fair.py](../src/training/dro_fair.py), the following invariants must hold:

1. **Step order:** θ → λ → p (NOT p → θ → λ)
2. **Inner gradient:** ∇g only (NOT λ∇g — argmax unchanged, avoids instability)
3. **Bias correction:** `π_clean = (π_obs − α) / (1 − 2α)`, clipped to [0, 1]
4. **L1-ball radius:** `2ρ` (TV → L1 conversion)
5. **Dykstra tail loop:** `max_iter = 500`
6. **τ schedule:** `sigmoid(τ · logits)` with multiplication (not division)

---

## File Structure

```
src/training/dro_fair.py       # Algorithm 1
src/training/naive_fair.py     # Baseline (ρ = 0)
src/training/standard_ml.py    # ERM baseline (used for PGD warm-start)
src/corruption/adversarial.py  # PGD + coordinated flips + RandomCorruptor
src/utils/projections.py       # Simplex, L1-ball, Dykstra
src/evaluation/metrics.py      # DP, IF, accuracy
src/data/datasets.py           # Adult, Credit, LSAC loaders
src/models/classifier.py       # MLP [128, 64], dropout 0.1
```

---

## Coding Conventions

- Use `tau` by **multiplication**: `σ(τ · logits)`, never division.
- Use a **local** `np.random.RandomState(seed)` inside corruption classes, not the global `np.random.seed`.
- Always **clip** the bias-corrected π proportions to `[0, 1]` for numerical safety.
- Validation metrics use `torch.no_grad()`. Training-time fairness constraints **must allow gradient flow** through `g_DP` and `g_IF`.

---

## Build Commands

```bash
# Run the validation suite (Wilcoxon checks)
python3 experiments/validate_results.py

# Verify all four theorems numerically on real data
python3 experiments/verify_theory.py

# Generate all main figures
python3 experiments/generate_figures.py

# Compile the LaTeX report
cd report && tectonic report.tex

# Run the test suite (32 tests)
python3 -m pytest tests/ -v
# If pytest hangs on hypothesis plugin discovery, use:
python3 -m pytest tests/ -p no:hypothesis
```

---

## Submission Package

The `submission/` folder contains the final deliverables. After any code change that affects results:

1. Re-copy `report/report.pdf` → `submission/report.pdf`
2. Verify all 7 figures (`fig1_...png` through `fig7_...png`) are present in `submission/`
3. Verify `submission/src/` matches the canonical `src/` tree
4. Re-run `experiments/validate_results.py` to confirm the win counts still hold
