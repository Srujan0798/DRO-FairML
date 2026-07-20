# Meeting Script: Honest Update for Madam (June 9, 3pm)

## Opening (30 seconds)

> "Madam, I want to be direct about what I found. I went deep into the attack code this week, and I discovered two issues — one I fixed, one I'm still working on."

---

## Issue 1: Feature Attack Was Random Noise (FIXED)

**What I claimed before:** "The attack uses gradient-based PGD on features."

**What was actually happening:** The feature perturbation used `np.random.randn()` — pure Gaussian noise with a sign based on target label. This is NOT PGD. It's barely different from random corruption.

**Code location:** `src/corruption/adversarial.py:532` — `noise = self.rng.randn(X.shape[1])`

**Fix:** Added `_train_surrogate()` that fits a LogisticRegression on `(X, y)` and wraps it as a torch module. `corrupt()` now uses this surrogate for **real gradient-based PGD** with `torch.autograd.grad()` instead of random noise.

**Validation:** 3 seeds, Adult α=0.2, Naive-FAIR

| Condition | Mean DP | vs Random |
|-----------|---------|-----------|
| RandomCorruptor | 0.154 | 1.0× |
| **Old** FairnessTargetedPGD (random noise) | ~0.16 | ~1.0× |
| **New** FairnessTargetedPGD (surrogate PGD) | **0.328** | **2.13×** |

**Verdict:** The attack is now **genuinely adversarial**, not random noise. The 2× improvement is real and meaningful.

---

## Issue 2: Coordinated Targeting Weakens DP Attack (DISCOVERED)

**What I found during validation:**

| Setting | Adult α=0.2 DP |
|---------|---------------|
| `coordinated=True` (70% minority quota) | 0.006 |
| `coordinated=False` (no quota) | 0.313 |

**Why:** The greedy label attack flips labels to maximize group-rate disparity. But `coordinated=True` forces 70% of flips into the minority group even after the gradient says we should flip majority samples. This over-concentrates flips and actually **reduces** final DP.

**Current experiments** use `coordinated=True` (hardcoded in runner). This is why DP attack results at α=0.2 are weak (~0.03) while IF attack results are stronger (~0.15).

**Fix:** Change runner to `coordinated=False` for DP attack. Easy one-line change.

---

## Experiment Status

| Component | Status |
|-----------|--------|
| Lambda diagnostic | ✅ 12/12 complete |
| Tabular re-run | 🔄 30/216 done (new code with real PGD) |
| ETA | ~9 AM tomorrow |

**Current results with fixed code:**
- Adult α=0.2 IF naive: dp=0.144
- Adult α=0.2 IF dro: dp=0.195
- Adult α=0.2 DP naive: dp=0.026 (weak due to coordinated=True)

---

## What I Also Found (Orthogonal)

**DRO radii mismatch:** DRO's `_compute_radii()` assumes uniform random corruption, but our attack is coordinated. On Adult, the formula estimates clean female proportion as **0%** (clipped) when true is **33%**. This misaligns DRO's uncertainty set. Lambda diagnostic confirms λ_DP stays small (~0.05) — it's structural, not runaway.

**This is a research finding, not a bug fix.**

---

## Honest Assessment

| What | Status |
|------|--------|
| Attack uses real gradients now | ✅ Fixed |
| Attack > 3× random noise | ❌ Not achieved (2× is real, 3× was optimistic) |
| Coordinated targeting hurts DP | 🔍 Discovered, easy fix |
| Full tabular experiments | 🔄 Running, results by morning |
| UTKFace | ⏸️ Blocked (no GPU/server) |

---

## Next Steps (This Week)

1. **Switch DP attack to `coordinated=False`** — one-line fix, rerun DP experiments
2. **Complete tabular experiments** — results by tomorrow afternoon
3. **Analyze and generate figures** — auto-finalize script ready
4. **UTKFace** — need server access or skip for this round

---

## If Madam Asks Tough Questions

**Q: "Why did you miss the random noise bug earlier?"**
> "I focused on the label attack logic and found real bugs there. But I didn't trace the feature attack path carefully — `_attack_features_fgsm` had a misleading name. I should have checked what it actually does."

**Q: "Why is DRO worse than Naive?"**
> "Two reasons: (1) DRO's radii formula assumes uniform corruption, but our attack is coordinated — this is a research design mismatch. (2) With the fixed attack, we're seeing DRO ≈ Naive on some configs and DRO < Naive on others. The radii issue explains part of it."

**Q: "When will you have final results?"**
> "Tabular experiments finish by tomorrow morning. I'll run analysis and generate figures by noon. UTKFace needs server access — can we discuss?"

---

*All code changes are committed and pushed to GitHub. Auto-commit runs every 30 min to backup results.*
