# DRO-FAIR-AL: Augmented-Lagrangian constraint enforcement (design)

Date: 2026-08-05. Status: pre-registered BEFORE any results were seen.
Goal-hook context: Prof. Manisha asked for a principled improvement to the
DRO-FAIR *method itself* so that its win over Naive becomes materially
larger, not merely present.

## 1. Diagnosis (grounded in code + prior negative results)

The canonical trainer (`src/training/dro_fair.py`) enforces the DP/IF
constraints only through the linear dual term `λ·g`:

```text
total_loss = L_tilt + λ_dp·g_dp + λ_if·g_if          (line ~343)
λ ← clamp(λ + lr_λ·0.95^epoch · g, 0, λ_max)         (lines ~328, ~353)
```

Two facts make this term nearly inert:

1. **Geometric dual decay.** `lr_λ = 5e-3 · 0.95^epoch` has total mass
   `Σ_t 5e-3·0.95^t ≈ 0.1`. With typical `g ≈ 0.05–0.2`, λ can accumulate
   to only ≈ 0.005–0.02 over the entire 60-epoch run — two orders of
   magnitude below the 1.5 ceiling.
2. **Confirmed empirically.** The pre-registered Arm-A ablation
   (`results/fairness_aggressiveness_summary.md`) found `λ_max: 1.5 → 2.0`
   produced 6/6 byte-identical outputs: the ceiling is never touched, so
   the *accumulation rate*, not the cap, is the binding bottleneck.
   (Measured, single-seed history dump, Adult α=0.2 seed 0, canonical
   settings: max λ_dp over 60 epochs = **0.0119** vs ceiling 1.5 (126×
   below); mean g_dp = 0.180; max penalty λ·g = **0.0029** vs final
   train loss 0.538 — the constraint term is ~0.5% of the loss. Under
   μ=5 the AL constraint gradient μ·g ≈ 0.9, roughly a 75× stronger
   signal than the effective λ ever provides.)

Consequence: the fairness penalty contributes ~1e-3 to a total loss of
~0.4 — the constraint machinery that differentiates DRO from Naive barely
influences θ. Both the λ decay and the conservative λ_max are relics of the
old unstable τ=100 schedule; the instability they guarded against is gone
(τ=1 fixed since the tau-artifact fix).

## 2. Approaches considered

**A. Augmented Lagrangian (chosen).** Add a quadratic constraint penalty:

```text
total_loss = L_tilt + λ_dp·g_dp + (μ/2)·g_dp² + λ_if·g_if + (μ/2)·g_if²
```

Classical constrained-optimization remedy for slow dual ascent
(Hestenes 1969; Bertsekas 1982): the quadratic term supplies a
constraint-violation gradient `μ·g·∇g` immediately, independent of how
slowly λ accumulates. Both `g_dp` (abs) and `g_if` (relu-sum) are
nonnegative by construction, so no `max(g,0)` clamp is needed. Dual update
left untouched (conservative variant; classical AL would use `λ += μ·g`,
but changing one thing at a time keeps attribution clean).

**B. Remove the 0.95 λ-decay / raise lr_λ.** Attacks the same bottleneck
but is a pure hyperparameter change — weaker methodological story, and the
resulting λ trajectory is dataset-coupled and harder to reason about.
Rejected in favor of A, which subsumes its benefit (penalty active from
epoch 0 regardless of λ).

**C. Corruption-aware sample down-weighting** (train against the inverse of
the inner-max adversarial weights). Genuinely novel but a much larger
conceptual change with new failure modes; kept as future work, not the
first lever.

## 3. Implementation

- `DroFairTrainer.__init__` gains `aug_lagrangian_mu: float = 0.0`.
  Default 0.0 is an **exact no-op** (term multiplies to zero); canonical
  behavior and byte-identical reruns preserved.
- One-line change in `fit()`: add `(μ/2)·g²` terms to `total_loss`.
- Thread `aug_lagrangian_mu` through
  `experiments/run_fairness_pgd.run_single_experiment` (recorded in row
  provenance). Naive path untouched.
- Unit tests (`tests/test_aug_lagrangian.py`):
  (a) μ=0 reproduces canonical training bit-for-bit on a fixed seed;
  (b) μ>0 changes the θ gradient in the expected direction (loss increases
  with g, gradient includes μ·g·∇g term);
  (c) provenance field appears in experiment rows.

## 4. Pre-registered experiment (values fixed before results)

- Script: `experiments/run_aug_lagrangian.py` via the shared
  `run_ablation_parallel` driver (respects the machine-wide ablation lock;
  queues behind the currently running jobs).
- Grid: datasets {adult, credit} × attack=dp × α ∈ {0.1, 0.2} ×
  seeds 0–5 × μ ∈ {5.0, 10.0}, DRO only → 48 runs →
  `results/aug_lagrangian.json`.
- Reference arms: canonical DRO and Naive rows from the locked
  `results/canonical_tau1.json` (same seeds/protocol; no reruns needed).
- Rationale for μ values: penalty μ·g² at g≈0.1 gives 0.05–0.1 —
  comparable to L_tilt's scale (~0.4) without dominating it. μ=5 is the
  conservative arm, μ=10 the aggressive arm.
- Analysis: seed-paired Wilcoxon per (dataset, α) cell, one-sided
  H1: canonical-DRO DP > AL-DRO DP (AL improves fairness), plus the
  DRO-vs-Naive margin comparison and accuracy guard.
- **Success criterion:** AL-DRO reduces DP violation vs canonical DRO with
  p<0.05 in ≥2 of 4 cells AND mean accuracy drop vs canonical DRO
  ≤ 0.005. LSAC excluded (degenerate DP regime, documented separately).
- **Honest-negative commitment:** if the criterion fails, the result is
  written up as a negative in results/ and paper Future Work, exactly as
  the λ_max/β/shrinkage negatives were. No post-hoc μ tuning.

## 5. Scope guards

- No writes to `results/canonical_tau1.json` or `results/utkface_*.json`
  (enforced by `_assert_safe_results_path`).
- Paper/report integration only after the experiment completes and the
  criterion is evaluated; framed as "proposed improvement + evidence",
  clearly separated from the locked canonical protocol.
