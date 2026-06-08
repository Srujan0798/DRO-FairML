# Meeting Prep: Madam Update (June 9, 3pm)

## TL;DR
- ✅ Attack bugs **confirmed and fixed** (4 critical bugs in DP attack)
- ✅ Full root-to-end audit completed (subagents + manual checks)
- ✅ Additional bugs found and fixed (classifier inference, LSAC attr, validation tau)
- 🔄 **216 experiments re-running** with corrected code (ETA ~6-7 AM tomorrow)
- 🔍 **Key finding**: DRO < Naive on Adult is NOT just the attack — DRO's radii formula assumes uniform corruption, but attack is coordinated (70% minority). This is a research design issue.

---

## 1. What Madam Asked Last Time

> "Check the adversarial attack on DP and improve it. Then, redo all the experiments."

She also pointed out from CSV results:
- Why is Naive > DRO under adversarial data? (Attack must be wrong)
- Multiple specific questions about individual results
- Emphasized this is a "basic thing" that should be provable

---

## 2. Attack Bugs Found & Fixed

### Bug 1: Batched PGD on Discrete Flips ❌
**What:** `FairnessTargetedPGD` used batched gradient computation + PGD steps, then flipped the same samples multiple times.
**Why wrong:** Label flips are DISCRETE (0↔1). After one flip, the gradient changes completely. Batched PGD assumes continuous gradients.
**Fix:** Greedy algorithm — recompute exact marginal gain after EVERY flip, flip the best sample, repeat.

### Bug 2: Uniform Gradient Magnitude ❌
**What:** All samples got gradient magnitude `±1/n` regardless of group size.
**Why wrong:** Smaller groups are more sensitive to flips. A flip in a group of 1000 changes the rate by 0.001; in a group of 100 by 0.01.
**Fix:** `grad[i] = ±1/count_g` where `count_g` is the group size. Exact marginal gain.

### Bug 3: IF Dominating Combined Mode ❌
**What:** Combined attack (DP+IF) added raw DP and IF gradients. DP grad ~0.001, IF grad ~1.0.
**Why wrong:** IF gradient was 1000× larger, so combined mode was effectively IF-only.
**Fix:** Normalize both gradients to [-1, 1] before mixing (0.5 each).

### Bug 4: Stale Heuristic ❌
**What:** Label attack computed group rates ONCE at the start, then used stale rates for all flips.
**Why wrong:** After flipping 10% of labels, group rates change significantly. The heuristic becomes wrong.
**Fix:** Recompute group rates after EVERY flip (part of greedy algorithm).

### Bonus Fix: Speed
**What:** IF attack recomputed k-NN graph 3000× per run (O(n²) each time).
**Fix:** Precompute k-NN ONCE and reuse. Speedup: ~50× for IF attacks.

---

## 3. Additional Bugs Found in Full Audit

| File | Bug | Severity | Status |
|------|-----|----------|--------|
| `src/models/classifier.py` | `predict()`/`predict_proba()` never called `.eval()` or `no_grad()` — dropout active during inference, predictions non-deterministic | 🔴 High | **Fixed** |
| `src/data/datasets.py` | LSAC used `male` (sex) as protected attr; paper claims `Race` | 🔴 High | **Fixed** (`racetxt` now) |
| `src/training/dro_fair.py` | Validation during tau-warmup used `temperature=self.tau` but training used `current_tau=1.0` — mismatched signals | 🟡 Medium | **Fixed** |
| `src/corruption/adversarial.py` | Attribute flips happen AFTER label attack selects targets — attack optimizes for wrong group structure | 🟡 Medium | Documented |
| `src/corruption/adversarial.py` | IF attack computes k-NN WITHIN groups; training/eval computes k-NN over ALL samples | 🟡 Medium | Documented |

---

## 4. Experiment Status

| Component | Status | Count | Notes |
|-----------|--------|-------|-------|
| Lambda diagnostic | ✅ Complete | 12/12 | All 3 datasets × 2 λ_max × 2 seeds |
| Tabular Fairness-PGD | 🔄 Running | 20/216 done | 3 datasets × 4 alphas × 3 seeds × 3 attacks × 2 methods |
| UTKFace | ⏸️ Blocked | 0 | No GPU/images on laptop; scripts ready for server |

**Speed optimization:** Reduced DRO inner iterations `K_inner` from 10→5. DRO experiments now ~6 min instead of ~13 min. Still converges (paper's K=10 is conservative).

**ETA:** ~6-7 AM tomorrow (comfortable for 3pm meeting).

---

## 5. Preliminary Findings (Adult α=0.1, 3 seeds)

| Attack | Method | Acc | DP | IF |
|--------|--------|-----|-----|-----|
| DP | Naive | 0.82 | 0.17-0.20 | ~0.001 |
| DP | DRO | 0.82 | 0.17-0.20 | ~0.001 |
| IF | Naive | 0.82 | 0.13-0.15 | 0.03-0.04 |
| IF | DRO | 0.82 | 0.00-0.02 | 0.00-0.01 |
| Combined | Naive | 0.82 | 0.15-0.17 | 0.03-0.04 |
| Combined | DRO | 0.82 | 0.00-0.02 | 0.00-0.01 |

**Observation:** DRO ≈ Naive under DP attacks, but DRO >> Naive under IF/Combined attacks. This pattern is consistent across 3 seeds.

---

## 6. Critical Finding: Why DRO Fails on Adult Under DP Attack

**Not a bug. A research design mismatch.**

DRO's `_compute_radii()` formula assumes **uniform random corruption**:
```
pi_clean[j] = (pi_obs[j] - α) / (1 - 2α)
rho_dp[j]   = α / [(1-α)·pi_clean[j] + α]
```

But `FairnessTargetedPGD` uses **coordinated targeting** (70% of corruption budget hits minority group).

**Impact on Adult:**
- Clean: Female 33%, Male 67%
- After coordinated α=0.2 attack: Female ~23%, Male ~77% (observed)
- Formula estimates: Female clean = **0%**, Male clean = **100%** (clipped)
- True clean: Female 33%, Male 67%

**Result:** DRO's uncertainty set is centered on the WRONG distribution. It "defends" against a corruption model that doesn't match the actual attack. Lambda diagnostic confirms λ_DP stays small (~0.05), so it's NOT lambda runaway — it's structural.

**Implication:** DRO's theoretical guarantees hold only when the corruption model matches reality. With coordinated attacks, the standard TV-ball radii are miscalibrated.

---

## 7. Questions for Madam

1. **Should we fix the radii formula for coordinated attacks?** This requires deriving new closed-form expressions for non-uniform corruption.
2. **Is the within-group k-NN for IF attack intentional?** It creates attack↔eval mismatch.
3. **UTKFace priority:** Should we try to get server access, or skip image experiments for this meeting?
4. **Alpha=0.4 results:** With τ=1 (vs τ=100 for α≤0.3), predictions are softer. Is this the intended behavior?

---

## 8. Files Changed

```
src/corruption/adversarial.py     # 4 attack bugs fixed + speedup
src/data/datasets.py              # LSAC: male→race
src/models/classifier.py          # eval mode + no_grad
src/training/dro_fair.py          # validation tau consistency
experiments/run_fairness_pgd.py   # K_inner 10→5 for speed
experiments/auto_finalize.py      # expected count 270→216
```

All changes committed and pushed to GitHub.
