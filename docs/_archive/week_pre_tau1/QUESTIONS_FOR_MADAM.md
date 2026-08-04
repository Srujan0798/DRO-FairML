# Questions for Madam — Core Confusions & Blockers

## Context
> Mam, we identified and fixed 3 critical bugs in the attack code. Random vs adversarial comparison is done. K=10 validation is complete. The full 270 re-run with fixed attack is running overnight. We want to clarify all core confusions before proceeding further.

---

## 🔴 CRITICAL — Blocking Next Steps

### Q1. What exactly does "redo all experiments" include?
- Just the tabular 270 (Adult/Credit/LSAC)?
- Also UTKFace image experiments?
- Also random vs adversarial baseline?
- How many seeds minimum for publishable results? (We have 3; Wilcoxon p<0.05 needs n≥6)

### Q2. Is K_inner=5 acceptable for CPU feasibility?
- Our K=10 validation shows K=5 and K=10 are virtually identical (diff=0.0000 for DP).
- Paper spec says K_inner=10 mandatory.
- **Can we use K=5 for local development and K=10 only for final server runs?**

### Q3. LSAC α=0 anomaly — DRO still diverges from Naive even with α=0 guard
- Adult: fixed (diff reduced from 0.012 → 0.0005) ✅
- LSAC: still diverges (diff ~0.038, 6× worse).
- **Is this expected for highly imbalanced group structures, or is there another bug?**

### Q4. LSAC DP attack DECREASES DP instead of increasing it
- Adult DP attack: DP increases 3.4× (works correctly)
- LSAC DP attack: DP drops from 0.007 → 0.0004 (almost zeroed out).
- **Is this "DP is hard to attack on LSAC" itself a finding, or is our attack mis-targeted?**

---

## 🟡 THEORETICAL — Radii & Paper Alignment

### Q5. Does the radii formula need fixing for coordinated attacks?
- DRO's `_compute_radii()` assumes uniform corruption: `π_clean = (π̂ − α)/(1 − 2α)`.
- But our attack uses coordinated targeting (70% minority).
- On Adult α=0.2: formula estimates Female clean = 0%, true = 33%.
- **Question:** Is the paper's Theorem 4.2 worst-case bound already calibrated for 100% targeting? Or do we need to derive new closed-form radii for non-uniform corruption?
- **Are we claiming the paper is wrong, or are our bugs causing the mismatch?**

### Q6. Adult shows a two-regime pattern — is this real?
- α=0.0–0.3: DRO makes fairness WORSE (p<0.05)
- α=0.4: DRO makes fairness BETTER (p<0.001)
- **Is this a real research finding (radii mismatch hurts at moderate α, helps at high α), or still a bug artifact?**

### Q7. Is within-group k-NN for IF attack intentional?
- IF attack computes k-NN WITHIN protected groups.
- Training/evaluation computes k-NN over ALL samples.
- **This creates attack↔eval mismatch. Is this by design?**

---

## 🟢 UTKFace & Server

### Q8. UTKFace priority — should we push for server access this week?
- Tabular experiments are nearly complete.
- UTKFace needs GPU (L40S or similar).
- **Should we prioritize tabular completion + theory, or push for server now?**

### Q9. Alpha=0.4 with τ=1 (vs τ=100 for α≤0.3) — intended behavior?
- `get_temperature()` returns τ=100 for α≤0.3, τ=1 for α≥0.4.
- This makes predictions much softer at α=0.4.
- **Is this the intended design?**

---

## 🔵 METHODOLOGY — Clarifications

### Q10. Adult IF attack DECREASES DP — expected or bug?
- IF attack aims to increase individual fairness (IF) but decreases DP on Adult.
- IF and DP are inversely related — maximizing IF naturally reduces DP.
- **Is this correct behavior, or should the IF attack also target DP?**

### Q11. What statistical tests are acceptable?
- With 3 seeds: Wilcoxon p<0.05 is impossible (min p=0.125).
- With 5 seeds: Wilcoxon becomes possible.
- **What is the minimum sample size you expect for claims in the paper?**

### Q12. Should we fix the radii formula for coordinated attacks?
- Deriving new closed-form expressions for non-uniform corruption is non-trivial.
- **Is this within the project scope, or should we document the limitation and move on?**

---

## Summary of What We Need

| # | Question | Why It Blocks Us |
|---|----------|------------------|
| 1 | Scope of "redo all experiments" | Can't plan timeline without clarity |
| 2 | K_inner=5 vs 10 acceptance | Need to know if local K=5 is publishable |
| 3 | LSAC α=0 divergence | Don't know if it's a bug or expected |
| 4 | LSAC DP attack decreases DP | Don't know if attack is wrong or LSAC is special |
| 5 | Radii formula for coordinated attacks | Don't know if we're challenging the paper or fixing our bugs |
| 6 | Two-regime pattern reality check | Don't know if this is a real finding or artifact |
| 7 | Within-group k-NN intentional? | Need to know if attack↔eval mismatch is by design |
| 8 | UTKFace priority | Can't allocate time without knowing priority |
| 9 | τ=1 at α=0.4 intended? | Need confirmation on temperature design |
| 10 | IF attack decreasing DP | Need to know if this is expected inverse relationship |
| 11 | Minimum seeds for stats | Need to know if 3 seeds is exploratory or final |
| 12 | Radii formula scope | Need to know if theory derivation is in scope |
