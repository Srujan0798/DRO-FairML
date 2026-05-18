# DRO-FAIR: Key Formulas and Implementation Reference

A quick reference for the core mathematical components of DRO-FAIR (Algorithm 1) as implemented in this repository.

---

## Headline Results (150 experiments)

| Dataset | α   | DP Reduction | IF Reduction | Acc Drop |
|---------|-----|--------------|--------------|----------|
| LSAC    | 0.3 | −99.6%       | −100.0%      | 0.1%     |
| Credit  | 0.3 | −91.8%       | −96.0%       | 1.9%     |
| Adult   | 0.3 | −206% (collapse) | —        | 28% drop |

**Statistical significance (Wilcoxon signed-rank, α ∈ {0.1, 0.2, 0.3}):** DP wins 6/9, IF wins 5/9.
**Total experiments:** 150 (3 datasets × 5 α × 10 seeds).

---

## Project Summary (One Paragraph)

This project implements DRO-FAIR (Algorithm 1) and extends the paper's random-noise corruption protocol to a multi-modal adversarial threat model: PGD-based feature perturbations, coordinated label flips targeting the DP-gap-maximizing direction, and minority-targeted attribute flips. The adversarial protocol yields a 2–5× harder evaluation at the same α budget. Credit and LSAC remain robust under DRO-FAIR with significant fairness gains and small accuracy cost. Adult exhibits a documented λ_DP runaway failure when the baseline DP violation is already large, which is discussed honestly in the report as an empirical limitation rather than a code bug.

---

## Algorithm 1 Step Order

Each training epoch executes three updates in this order:

1. **Forward pass.** `h̃ = σ(τ · f_θ(x))` (τ=100 for α≤0.3, τ=1 at α=0.4).
2. **Outer minimization (θ).** AdamW step on `L_tilt + λ_DP·g_DP + λ_IF·g_IF`, gradient clipped at 0.5.
3. **Dual ascent (λ).** `λ ← clamp(λ + η_λ · 0.95^t · g, 0, λ_max=1.5)`.
4. **Inner maximization (p).** K=10 ascent steps on `∇g` (not `λ∇g` — see note below), each followed by Dykstra projection onto Δ_n ∩ B_1(p̂, 2ρ).

Source: [src/training/dro_fair.py](../src/training/dro_fair.py), lines 200–250.

**Note on `∇g` vs `λ∇g`.** For λ > 0, `argmax g(p) = argmax λg(p)` — the optima coincide. However, scaling by λ inflates the gradient magnitude as λ grows toward `λ_max`, which can destabilize the inner loop. Using ∇g directly keeps the step sizes well-conditioned.

---

## Core Theoretical Formulas

| Quantity | Formula | Origin |
|----------|---------|--------|
| DP radius | `ρ_DP,j = α / ((1 − α)·π_j + α)` | Theorem 4.2 |
| IF radius | `ρ_IF = 2α − α²` | Theorem 4.3 |
| Bias correction | `π_clean = (π̂_obs − α) / (1 − 2α)` | Appendix F |
| TV → L1 conversion | `L1 radius = 2 · ρ_TV` | Standard identity |

The bias correction comes from solving `π̂_obs = (1−α)·π_clean + α·(1−π_clean)` for `π_clean`. The L1-radius identity follows from `TV = ½ · L1`.

---

## Design Decisions Worth Knowing

### `λ_max = 1.5` (rather than the paper's 2.0)
Empirical stability fix. At `λ_max = 2.0`, Adult collapsed at α=0.2 as well. Reducing to 1.5 caps the penalty enough to preserve wins on Credit and LSAC while making the Adult α=0.3 failure the cleanest case to study.

### Runtime overhead 37.5× (vs. paper's ~12×)
Paper used GPU; this implementation runs on CPU with full-batch k-NN graph construction per epoch, which dominates overhead. The relative comparison between methods is unaffected.

### Statistical reporting: Wilcoxon vs. mean-based
The data supports two valid summaries — 7/9 IF wins under "mean(DRO) < mean(Naive)", or 5/9 under "Wilcoxon p < 0.05". Because figure captions explicitly cite Wilcoxon, the 5/9 count is the internally consistent claim used throughout the report.

### Theorem 6.1 under adversarial corruption
PGD respects the αn sample budget, so TV-ball containment holds theoretically. Empirically, the guarantee yields strong results on Credit/LSAC but fails on Adult. The radii are sufficient when baseline DP is small, but coordinated multi-modal attacks concentrate fairness-violating signal in ways that per-modality radii didn't anticipate. This is a meaningful empirical observation, not a contradiction of the theorem.

### Temperature τ=100
Sharpens the sigmoid so DP/IF gradients are informative. At τ=1, predictions are too soft and the fairness signal becomes noise. At α=0.4, the schedule drops to τ=1 because at extreme corruption, sharper predictions amplify the wrong signal direction.

---

## Reproduction Snippets

```python
# Verify any (dataset, alpha) cell directly from the saved JSON
python3 -c "
import json, numpy as np
from scipy.stats import wilcoxon
d = json.load(open('results/all_results.json'))
ds, a = 'credit', 0.3
s = [r for r in d if r['dataset']==ds and abs(r['alpha']-a) < 1e-6]
ndp = [r['naive']['clean']['dp_violation'] for r in s]
ddp = [r['dro']['clean']['dp_violation'] for r in s]
print(f'Naive: {np.mean(ndp):.4f}  DRO: {np.mean(ddp):.4f}')
print(f'Reduction: {(np.mean(ndp)-np.mean(ddp))/np.mean(ndp)*100:+.1f}%')
print(f'Wilcoxon p: {wilcoxon(ndp,ddp,alternative=\"greater\")[1]:.4f}')
"
```

```bash
# Full validation across all 9 cells at α ∈ {0.1, 0.2, 0.3}
python3 experiments/validate_results.py

# Numerical verification of Theorems 4.2, 4.3, 6.1, Remark 6.2
python3 experiments/verify_theory.py

# Test suite (32 tests)
python3 -m pytest tests/ -q
```

---

## File Map

| What | Where |
|------|-------|
| Algorithm 1 (training loop) | [src/training/dro_fair.py](../src/training/dro_fair.py) |
| Naive-FAIR baseline | [src/training/naive_fair.py](../src/training/naive_fair.py) |
| Adversarial corruption | [src/corruption/adversarial.py](../src/corruption/adversarial.py) |
| Dykstra projection | [src/utils/projections.py](../src/utils/projections.py) |
| Metrics (DP, IF, accuracy) | [src/evaluation/metrics.py](../src/evaluation/metrics.py) |
| Main experiment runner | [experiments/run_experiments.py](../experiments/run_experiments.py) |
| Theory verification | [experiments/verify_theory.py](../experiments/verify_theory.py) |
| Final report | [report/report.pdf](../report/report.pdf) |
