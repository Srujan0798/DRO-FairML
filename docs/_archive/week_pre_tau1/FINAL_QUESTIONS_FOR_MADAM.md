# Final Questions for Madam — Merged & Organized

> Mam, while we wait for the next meeting, I wanted to consolidate all my questions and confusions so I can proceed clearly without wasting time going in wrong directions. Here are the key blockers:

---

## 🔴 RESULTS NARRATIVE — What Story Can We Tell?

**Q1. Is "DRO is fragile under coordinated fairness attacks" a valid finding?**
- With the corrected attack (DP-targeted PGD), DRO largely ties or underperforms Naive.
- The only clear DRO win is at high corruption (α=0.4) on Adult.
- Is this a valid paper finding, or should we reframe the story?
- **Are we claiming DRO is fragile, or are there still bugs in our evaluation?**

**Q2. Adult shows a two-regime pattern — is this real or still a bug artifact?**
- α=0.0–0.3: DRO makes fairness WORSE (p<0.05)
- α=0.4: DRO makes fairness BETTER (p<0.001)
- Is this a real research finding (radii mismatch hurts at moderate α, helps at high α)?

**Q3. LSAC DP attack DECREASES DP — finding or bug?**
- Adult DP attack: DP increases 3.4× (works correctly)
- LSAC DP attack: DP drops from 0.007 → 0.0004 (almost zeroed out)
- Is "DP is hard to attack on LSAC" itself a finding, or is our attack mis-targeted?

**Q4. LSAC α=0 anomaly — expected or another bug?**
- Adult: fixed by α=0 guard (diff reduced from 0.012 → 0.0005) ✅
- LSAC: still shows DRO/Naive divergence at α=0 (diff ~0.038, 6× worse)
- Is this expected for highly imbalanced group structures, or is there another bug?

---

## 🟡 THEORY & PAPER ALIGNMENT

**Q5. Does the radii formula need fixing for coordinated attacks?**
- DRO's `_compute_radii()` assumes uniform corruption: `π_clean = (π̂ − α)/(1 − 2α)`.
- Our attack uses coordinated targeting (70% minority).
- On Adult α=0.2: formula estimates Female clean = 0%, true = 33%.
- Is the paper's Theorem 4.2 worst-case bound already calibrated for 100% targeting?
- **Are we claiming the paper is wrong, or are our bugs causing the mismatch?**
- Deriving new closed-form radii for non-uniform corruption is non-trivial — is this within scope?

**Q6. Is within-group k-NN for IF attack intentional?**
- IF attack computes k-NN WITHIN protected groups.
- Training/evaluation computes k-NN over ALL samples.
- This creates attack↔eval mismatch. Is this by design?

**Q7. Adult IF attack DECREASES DP — expected inverse relationship or bug?**
- IF attack aims to increase individual fairness (IF) but decreases DP on Adult.
- IF and DP are inversely related — maximizing IF naturally reduces DP.
- Is this correct behavior, or should the IF attack also target DP?

---

## 🟢 METHODOLOGY & EXPERIMENTAL DESIGN

**Q8. What exactly does "redo all experiments" include?**
- Just the tabular 270 (Adult/Credit/LSAC)?
- Also UTKFace image experiments?
- Also random vs adversarial baseline?
- What is the complete scope?

**Q9. How many seeds minimum for publishable results?**
- We currently have 3 seeds per setting.
- Wilcoxon p<0.05 requires at least 6 paired observations (n≥6).
- Should we run 6 seeds now, or is 3 acceptable for the current analysis phase?

**Q10. Is K_inner=5 acceptable for CPU feasibility?**
- Our K=10 validation shows K=5 and K=10 are virtually identical (diff=0.0000 for DP).
- Paper spec says K_inner=10 mandatory.
- Can we use K=5 for local development and K=10 only for final server runs?

**Q11. Presentation format — absolute DP or percentage change?**
- For the adversarial vs random DP comparison, should we report:
  - Absolute DP values (e.g., 0.15 → 0.53)?
  - Or percentage change (e.g., +253%)?
- Wanted to confirm before finalizing figures.

**Q12. Alpha=0.4 with τ=1 (vs τ=100 for α≤0.3) — intended behavior?**
- `get_temperature()` returns τ=100 for α≤0.3, τ=1 for α≥0.4.
- This makes predictions much softer at α=0.4.
- Is this the intended design?

---

## 🔵 UTKFace & PRIORITIES

**Q13. UTKFace priority — push for GPU access this week or finish tabular first?**
- Tabular experiments are nearly complete (270 re-run in progress).
- UTKFace needs GPU (flair2 or similar).
- Should we prioritize tabular completion + theory, or push for server access now?

---

## Summary Table

| # | Category | Question | Why It Blocks |
|---|----------|----------|---------------|
| 1 | Narrative | Is DRO fragility a valid finding? | Can't write paper story without knowing |
| 2 | Narrative | Two-regime pattern real or artifact? | Don't know if we have a real result |
| 3 | Narrative | LSAC DP decrease — finding or bug? | Don't know if attack is wrong |
| 4 | Narrative | LSAC α=0 divergence expected? | Don't know if fixable or inherent |
| 5 | Theory | Radii formula for coordinated attacks? | Don't know if we're challenging the paper |
| 6 | Theory | Within-group k-NN intentional? | Need to know if mismatch is by design |
| 7 | Theory | IF↔DP inverse relationship expected? | Need to know if targeting is correct |
| 8 | Method | Scope of "redo all experiments"? | Can't plan timeline without clarity |
| 9 | Method | How many seeds minimum? | Need to know if 3 seeds is enough |
| 10 | Method | K_inner=5 acceptable? | Need to know if local runs are publishable |
| 11 | Method | Absolute DP or % change in figures? | Need to know before finalizing plots |
| 12 | Method | τ=1 at α=0.4 intended? | Need confirmation on temperature design |
| 13 | Priority | UTKFace now or tabular first? | Can't allocate time without priority |

---

*Please let me know which of these are most urgent, and I'll prioritize those first.*
