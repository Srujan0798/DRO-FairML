# When Robust Fairness Fails: An Adversarial Feedback Loop in DRO

*Implementing DRO-FAIR under adversarial corruption, and what we found on the Adult dataset*

---

## The Setup

Fairness-aware classifiers are supposed to treat protected groups equitably. But what happens when an adversary corrupts your training data? **DRO-FAIR** (Distributionally Robust Optimization for Fairness) proposes a clean answer: solve a min-max Lagrangian over a TV-uncertainty set calibrated to the corruption rate α.

I implemented Algorithm 1 from the ICML 2026 submission exactly — corruption-calibrated radii (ρ_DP,j = α/((1−α)π_j + α), ρ_IF = 2α − α²), K=10 inner PGD steps on importance weights, Dykstra projection onto simplex ∩ ℓ₁-ball, the works. Then I replaced the paper's random noise with **adversarial corruption**: PGD feature attacks, label flips coordinated to maximize the DP gap, and minority-targeted attribute flips.

Across 150 experiments (3 datasets × 5 α values × 10 seeds), DRO-FAIR achieved up to **92% DP reduction on Credit** and **100% on LSAC** with p<0.001 — the theory works.

Then I ran Adult.

## The Collapse

At α=0.3 on Adult, DRO-FAIR's accuracy dropped to **49.5%** — barely better than random. The DP violation actually got *worse* than Naive-FAIR. Six of ten seeds collapsed to <45% accuracy. This is a published-paper-grade algorithm, faithfully implemented, with 32 passing unit tests verifying every formula. So what gave?

## The Feedback Loop

Adult has a baseline DP of ~0.17 — **8x larger** than Credit or LSAC. Under coordinated adversarial label flips, the attack specifically amplifies this disparity. Here's what unfolds:

```
Coordinated label flips → DP gap grows
  → DRO inner-max concentrates weights on high-DP samples
    → λ_DP grows during dual ascent
      → Outer min over-penalizes the DP term
        → Group rates forced equal at any cost
          → Model collapses to near-constant prediction
            → Accuracy ≈ 25–40%
```

The TV radii are **conservative by design** — they're sized for worst-case random corruption. When the corruption is adversarial *and* the baseline disparity is already large, the radii are large enough that the inner maximizer can drive λ_DP into a runaway regime before training stabilizes.

## Why This Matters

The paper's Theorem 6.1 proves (ε_DP + ε_IF)-fairness guarantees under random TV-ball corruption. Our empirical finding: **the guarantee does not transfer to adversarial corruption on datasets with high baseline DP.** This isn't a bug — it's the theory's frontier showing itself.

Credit and LSAC, with baseline DP ~0.02, never hit this regime. Their radii × baseline disparity stays in a range where the dual variables converge cleanly. Adult crosses a threshold.

## Lessons

1. **Re-derive your guarantees for your threat model.** Random-noise robustness ≠ adversarial robustness.
2. **Conservative radii cut both ways.** They protect against bad seeds *and* enable runaway dynamics when baseline disparity is high.
3. **Honest failure modes matter.** I could have hidden Adult's collapse by tuning λ_max per-dataset. Reporting it is more useful: it tells the next researcher exactly where the method needs work.

## What's Next

Three concrete fixes I'd try:

- **Dataset-adaptive λ_max**, capped at baseline DP
- **Warm-start λ** at small positive values to avoid the cold-start kick
- **Bootstrap-based per-group α estimation** for tighter radii

Code, results, 150 experiment JSONs, and the full LaTeX report:
**https://github.com/Srujan0798/DRO-FairML**

---

*Srujan Sai · May 2026*
