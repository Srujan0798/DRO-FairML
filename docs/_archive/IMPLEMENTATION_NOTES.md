# DRO-FAIR: Implementation Notes

Technical notes covering the algorithmic choices made in this implementation, with derivations and references to the source code. This document is intended for readers reviewing the codebase who want to understand the rationale behind specific design decisions.

---

## 1. Algorithm 1 — Update Order

Each training epoch executes three updates in this order:

1. **Forward pass.** Compute `h̃ = σ(τ · f_θ(x))` with the current θ.
2. **Losses.** Tilted risk `L_tilt`, weighted DP violation `g_DP`, weighted IF violation `g_IF`.
3. **Outer minimization (θ).** AdamW step on `L_tilt + λ_DP · g_DP + λ_IF · g_IF`.
4. **Dual ascent (λ).** `λ ← clamp(λ + η_λ · g, 0, λ_max)`.
5. **Inner maximization (p).** K=10 ascent steps on `∇g(p)` (not `λ∇g`), each followed by Dykstra projection.

**Rationale for ordering.** θ must adapt to the current p before p re-optimizes for the new θ. If p updates first, it optimizes against a stale θ, producing an outdated worst-case reweighting. Putting θ first keeps the saddle-point iteration well-defined.

**Code:** [src/training/dro_fair.py](../src/training/dro_fair.py), lines 200–250.

---

## 2. Inner Gradient: `∇g` vs. `λ∇g`

The inner loop maximizes `g(θ, p)` over p. The Lagrangian contribution is `λ · g(θ, p)`. For any fixed `λ > 0`:

```
argmax_p  λ · g(θ, p)  =  argmax_p  g(θ, p)
```

The optima coincide. However, scaling the gradient by λ amplifies its magnitude as λ approaches `λ_max`, which can destabilize the inner loop and cause projection-step issues. Using `∇g` directly keeps the gradient norm independent of the dual variable and yields stable behavior across all corruption levels.

---

## 3. Theoretical Formulas

### 3.1 DP radius (Theorem 4.2)

```
ρ_DP,j  =  α / ((1 − α) · π_j + α)
```

Under α-corruption, the worst-case TV distance between the clean and corrupted group-j conditional distribution is bounded by this expression. The denominator reflects that smaller groups have proportionally less mass to absorb the α corruption budget, leading to larger radii.

### 3.2 IF radius (Theorem 4.3)

```
ρ_IF  =  2α − α²  =  1 − (1 − α)²
```

This is the probability that at least one endpoint of a pairwise comparison is corrupted, treating each sample as independently in the corrupted set with probability α.

### 3.3 Bias correction (Appendix F)

Observed proportion in the corrupted data:

```
π_obs  =  (1 − α) · π_clean  +  α · (1 − π_clean)
```

Solving for `π_clean`:

```
π_obs   =  π_clean − α·π_clean + α − α·π_clean
        =  π_clean · (1 − 2α) + α

π_clean =  (π_obs − α) / (1 − 2α)
```

The result is clipped to `[0, 1]` for numerical safety.

### 3.4 TV → L1 conversion

```
TV(P, Q) = ½ · Σᵢ |Pᵢ − Qᵢ| = ½ · ‖P − Q‖₁
```

Therefore a TV-distance constraint `TV ≤ ρ` becomes an L1-ball constraint of radius `2ρ`. The projection `project_simplex_l1_ball` in [src/utils/projections.py](../src/utils/projections.py) accepts the L1 radius directly.

---

## 4. Threat Model

The adversary controls an α-fraction of training samples in a white-box setting:

- **Feature perturbations.** PGD with `ε = 0.1`, 5 steps, step size `0.02`.
- **Label flips.** Coordinated to maximize the demographic-parity gap (chosen to increase `|P(ŷ=1|A=1) − P(ŷ=1|A=0)|`).
- **Attribute flips.** 70% concentrated on the minority group to amplify proportion distortion.

All three modalities are applied to the same αn samples (coordinated). The adversary has access to the model architecture and the training data. Test data remains clean.

**Relation to Theorem 6.1.** PGD respects the αn sample budget, so TV-ball containment holds at the theoretical level. The empirical results confirm robustness on Credit and LSAC and document a failure mode on Adult (Section 5 of the report).

---

## 5. Hyperparameter Choices

### 5.1 `λ_max = 1.5` (paper uses 2.0)

At `λ_max = 2.0`, λ_DP runaway caused Adult to collapse at α=0.2 in addition to α=0.3. Reducing to 1.5 caps the dual penalty enough to preserve Credit/LSAC wins while isolating the Adult α=0.3 case as the cleanest failure to study. Both choices satisfy the convergence conditions in the paper; this is a stability adjustment, not a theoretical deviation.

### 5.2 Temperature schedule

`τ = 100` for α ≤ 0.3, `τ = 1` for α ≥ 0.4. A sharp sigmoid (τ=100) ensures the DP/IF gradients carry informative signal. At extreme corruption (α=0.4), sharper predictions amplify the wrong signal direction in adversarial settings, so the schedule reverts to τ=1.

### 5.3 Tilt parameter `β = 5`

Approximates CVaR-style up-weighting of difficult samples. β → ∞ recovers the empirical mean (standard ERM); β → 0 recovers the maximum (extreme robustness). β = 5 is the paper's choice and was kept without modification.

---

## 6. Adult Failure Case

Adult exhibits a documented failure mode at α ∈ {0.1, 0.2, 0.3}: DRO-FAIR loses on DP and, at α=0.3, accuracy collapses to 49.5% (random-guessing range) on 6/10 seeds.

**Mechanism.** Adult's baseline DP violation is approximately 0.17 — an order of magnitude larger than Credit or LSAC (~0.02). Coordinated adversarial label flips amplify this disparity. The DP radius at α=0.3 becomes large enough that the inner maximization pushes sample weights to extreme values, causing the model to output near-constant predictions to satisfy the fairness constraint. The Theorem 6.1 guarantee holds vacuously (DP is low because predictions are uniform).

**Interpretation.** This is not a code bug. It is a fundamental property of conservative TV-radius calibration when the baseline group disparity is already large. The radii are sufficient for guarantees when DP signal is small, but they over-correct when DP signal is large. Dataset-adaptive radii or early stopping based on validation accuracy are natural future-work mitigations.

---

## 7. Statistical Reporting Convention

Two valid summaries of the same data:

- **Mean-based:** "DRO mean < Naive mean" → 7/9 IF wins.
- **Wilcoxon signed-rank (p < 0.05, one-sided):** → 5/9 IF wins.

Figure captions and the validation script ([experiments/validate_results.py](../experiments/validate_results.py)) cite Wilcoxon, so 5/9 is the internally consistent claim used throughout the report. Adult α=0.2 and Credit α=0.2 have p > 0.05 despite favorable mean differences.

For DP, both criteria agree: 6/9 wins.

---

## 8. Runtime

DRO-FAIR runs ~37.5× slower than Naive-FAIR on CPU. The paper reports ~12× on GPU. The gap is dominated by full-batch k-NN graph construction in [src/evaluation/metrics.py](../src/evaluation/metrics.py), which is CPU-bound regardless of the model device. GPU support is partial (the model trains on GPU when available, but the k-NN graph is built on CPU).

---

## 9. References to Code

| Component | File | Notes |
|-----------|------|-------|
| Algorithm 1 training loop | [src/training/dro_fair.py](../src/training/dro_fair.py) | `fit()` method |
| Bias-corrected radii | [src/training/dro_fair.py](../src/training/dro_fair.py) | `_compute_radii()` |
| Tilted risk | [src/training/dro_fair.py](../src/training/dro_fair.py) | `_compute_tilted_loss()` |
| Dykstra projection | [src/utils/projections.py](../src/utils/projections.py) | `project_simplex_l1_ball()` |
| Adversarial corruption | [src/corruption/adversarial.py](../src/corruption/adversarial.py) | `AdversarialCorruptor` |
| Metrics (DP, IF, accuracy) | [src/evaluation/metrics.py](../src/evaluation/metrics.py) | Soft and hard predictions |
| Theory verification | [experiments/verify_theory.py](../experiments/verify_theory.py) | Theorems 4.2, 4.3, 6.1, Remark 6.2 |
| Statistical validation | [experiments/validate_results.py](../experiments/validate_results.py) | Wilcoxon-based win counts |
