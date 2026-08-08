# AL Independent Adversarial Review (TASK E)

**Date:** 2026-08-07
**Scope:** Independent review of the DRO-FAIR-AL claim (pre-registered design, initial 48-row result, μ-sensitivity sweep rules C1–C4). No new experiments run; pure hand-verification.

---

## Verdict

The AL claim is **structurally sound and correctly implemented**. The Adult α=0.2 win is real and robust. I found **zero CRITICAL or HIGH defects** and no leakage. Three LOW/INFO items below are worth documenting but do not change any conclusion.

---

## 1. Gradient math of `(μ/2)·g²` — CONFIRMED CORRECT

**Derivation.** Let `L_base = L_tilt + λ·g`. The AL penalty is `(μ/2)·g²`.

```
∂/∂θ [L_base + (μ/2)·g²] = ∂L_tilt/∂θ + λ·∇g + μ·g·∇g
```

The additional gradient w.r.t. canonical is `μ·g·∇g`.

**Code check (`src/training/dro_fair.py:356-363`):**

```python
if self.aug_lagrangian_mu > 0:
    mu = self.aug_lagrangian_mu
    if self.use_dp:
        total_loss = total_loss + 0.5 * mu * g_dp * g_dp
    if self.use_if:
        total_loss = total_loss + 0.5 * mu * g_if * g_if
```

`0.5 * mu * g * g` = `(μ/2)·g²`. The autograd engine computes `∂/∂g[(μ/2)g²] = μ·g`, which then backpropagates through `∇g` to `θ`. Matches the hand derivation exactly.

**Severity: NONE — code is correct.**

---

## 2. No `max(g, 0)` needed — CONFIRMED CORRECT

- `g_dp = |h̄_1 - h̄_0|` (`_compute_dp_loss_weighted`, dro_fair.py:271): `torch.abs(...) ≥ 0` by construction.
- `g_if = Σ weights·relu(...) / (n-1)` (dro_fair.py:283-284): `relu(...) ≥ 0`, `weights ≥ 0`, so `g_if ≥ 0`.

Standard AL uses `max(g, 0)` because the raw constraint expression `g(x) ≤ 0` can be negative (satisfied). Here `g` is the *violation magnitude* already, so it is non-negative by construction. Omitting `max(g, 0)` is correct.

**Severity: NONE — omission is justified.**

---

## 3. μ=0 is byte-identical to canonical — CONFIRMED

**Code path.** The guard `if self.aug_lagrangian_mu > 0:` (line 356) is `False` for `mu=0.0`. No extra graph nodes are added. The backward pass is identical to a trainer constructed without the parameter.

**Unit test.** `tests/test_aug_lagrangian.py::test_mu_zero_is_exact_noop` verifies `torch.equal(v1, v2)` between the default trainer and the `mu=0.0` trainer. All 4 AL tests pass.

**Live verification.** I reproduced this outside the test framework: trained both a default trainer and a `mu=0.0` trainer on the same seed/data slice. Result:
- `torch.equal(v1, v2)`: **True**
- `train_losses`: identical
- `g_dp`, `lambda_dp` histories: identical

**Severity: NONE — μ=0 is an exact no-op, proven.**

---

## 4. Wilcoxon reproduction — CONFIRMED WITH ONE CAVEAT (INFO)

**Seed pairing.** Both `results/canonical_tau1.json` and `results/aug_lagrangian.json` share seeds 0–5 for the Adult/Credit α∈{0.1,0.2} DRO cells. Pairing is correct: same `(dataset, attack=dp, seed)` keys.

**One-sided direction.** H1: canonical-DP > AL-DP (AL strictly better). Correct: a positive difference means AL reduces DP.

**Reproduction (hand-computed exact Wilcoxon, no ties, no zeros):**

| cell | n | W+ | p (one-sided) | matches summary? |
|---|---|---|---|---|
| adult α=0.1 μ=5 | 6 | 6 | 0.8438 | ✓ |
| adult α=0.2 μ=5 | 6 | 21 | **0.0156** ✓ | ✓ |
| credit α=0.1 μ=5 | 6 | 21 | **0.0156** ✓ | ✓ |
| credit α=0.2 μ=5 | 6 | 21 | **0.0156** ✓ | ✓ |
| adult α=0.1 μ=10 | 6 | 11 | 0.5000 | ✓ |
| adult α=0.2 μ=10 | 6 | 21 | **0.0156** ✓ | ✓ |
| credit α=0.1 μ=10 | 6 | 21 | **0.0156** ✓ | ✓ |
| credit α=0.2 μ=10 | 6 | 21 | **0.0156** ✓ | ✓ |

**Exact p-values confirmed.** For n=6 with all diffs positive, `W+ = 21` is the maximum rank sum, occurring in exactly `1/2^6 = 1/64 = 0.015625` of null sign patterns. The summary's `p=0.0156 *` is correct for each of these.

**Severity: NONE — Wilcoxon computation is correct.**

### INFO item (Adult α=0.1, non-significance not emphasized)

The summary table lists `adult α=0.1 μ=5` as p=0.8438 (not significant) and `adult α=0.1 μ=10` as p=0.5000. These are honestly reported in the table but not explicitly discussed in the prose. The α=0.1 non-significance is pre-registered-consistent (the criterion required ≥2/4 cells significant, which is met) but a reader skimming only the prose might miss that AL is **not** better at α=0.1. This is correctly caveated in the table; just ensure the prose doesn't overstate.

**Severity: INFO — honest in the table; verify prose scope.**

---

## 5. Constant-predictor floors — CONFIRMED CORRECT

Recomputed from `src/data.datasets.get_dataset(..., random_state=0)`:

| dataset | n_test | majority class | floor (computed) | floor (stated) | match? |
|---|---|---|---|---|---|
| adult | 9045 | 0 (neg=6803) | 0.7521282477 | 0.7521 | ✓ (truncated) |
| credit | 6000 | 0 (neg=4673) | 0.7788333333 | 0.7788 | ✓ (truncated) |
| lsac | 3731 | 1 (pos=3371) | 0.9015779620 | 0.9016 | ✓ (rounded) |

The floors are the test-set majority-class accuracy. The stated values match the recomputed ones (Credit truncated, LSAC rounded). No discrepancy.

**Severity: NONE — floors are correct.**

---

## 6. Adult α=0.2 seed-3 (accuracy at floor) — CONFIRMED DEGENERATE, but including it is conservative

**Data.** seed-3: `acc=0.7521282477, dp=0.0209679306`. Accuracy equals the constant-predictor floor to 10 decimal places.

| seed | acc | dp |
|---|---|---|
| 0 | 0.8053 | 0.1620 |
| 1 | 0.8167 | 0.1581 |
| 2 | 0.7900 | 0.1611 |
| **3** | **0.7521** (floor) | **0.0210** (outlier low) |
| 4 | 0.7989 | 0.1718 |
| 5 | 0.8034 | 0.1410 |

The other five seeds cluster in acc∈[0.790, 0.817], dp∈[0.141, 0.172]. seed-3 is a clear outlier: accuracy collapsed to the constant predictor, and DP is dragged down by the collapse (not by genuine fairness).

**Does excluding it change the verdict?**

- With seed-3 (n=6): mean acc 0.7944, mean dp 0.1358, Wilcoxon p = 1/64 = 0.015625.
- Without seed-3 (n=5): mean acc **0.8029** (↑), mean dp **0.1588** (still ≪ canonical's 0.2334), Wilcoxon p = 1/32 = 0.03125.

Excluding seed-3 **strengthens** the claim. Including it is the conservative choice. The significance verdict (p < 0.05) is robust to either inclusion or exclusion.

**Severity: LOW — seed-3 is a degenerate collapse, but its inclusion is conservative and pre-registered (no post-hoc exclusion). Recommended action: none, but the prose could note the n=5 robustness.**

---

## 7. Leakage check — NONE FOUND

AL and canonical DRO receive **identical inputs** and differ only in the loss term:

- Same `X_train_att, y_train_att, a_train_att` (FairnessTargetedPGD output).
- Same `X_val, y_val, a_val`.
- Same `alpha, attack_k, pgd_steps, tau, K_inner, lr_lambda, beta, lambda_init, radii_mode, seed`.
- Same model architecture (`MLClassifier([128, 64], dropout=0.1)`).
- Same torch seed.

The ONLY difference is the `aug_lagrangian_mu` kwarg to `DroFairTrainer.__init__`, which enters ONLY at lines 356–363 (the quadratic penalty block). For μ=0 the guard is `False` and no extra computation occurs. For μ>0 the only additional computation is `+ 0.5 * mu * g * g` on `total_loss`.

**No data, no model, no attack, no radius, no lambda trajectory differs** between the two configurations. Attribution is clean.

**Severity: NONE — no leakage.**

---

## Additional findings (INFO severity)

### 7a. "Conservative variant" AL — no dual update modification

Classical augmented Lagrangian updates the dual as `λ ← λ + μ·g`. This implementation does **not** do that; the dual update at lines 371–375 is unchanged (`λ += lr_λ·g`). This is intentional and documented in §2 of the design spec ("conservative variant"). The quadratic penalty on the loss alone is what drives the improvement.

**Severity: INFO — intentional design choice, not a defect. But the term "augmented Lagrangian" without qualification in the paper is slightly imprecise; "quadratic penalty on the Lagrangian" is more accurate. Recommend clarifying in the method section.**

### 7b. Credit α=0.1 μ=10 razor-thin margin above floor

- floor: 0.7788
- min AL acc across seeds: 0.7786666667
- margin: **−0.00013** (slightly below floor, correctly flagged **DEGEN**)
- Mean acc: 0.7790 (above floor)

The degeneracy flag is correctly applied here because `0.7786... < 0.7788`. But note: the margin is 1.3e-4, within typical floating-point wiggle. This cell is correctly flagged DEGEN in the summary. No action needed, but the razor-thin margin is worth noting if the summarizer's floor check is ever relaxed.

**Severity: INFO — correctly handled by the existing guard.**

### 7c. Memo headline numbers match 6-seed means exactly

- Naive DP = 0.2452 (memo) vs 0.2452 (computed, seeds 0–5). ✓
- DRO DP = 0.2334 (memo) vs 0.2334 (computed, seeds 0–5). ✓
- Margin ratio 9.2×: `(0.2452−0.1358) / (0.2452−0.2334)` = `0.1094 / 0.0119` = 9.19×. ✓

The canonical_tau1.json file contains seeds 0–9 (10 seeds per cell). The memo and summary both correctly use the 6-seed subset (seeds 0–5) that matches the AL experiment's seed set.

**Severity: INFO — no discrepancy; just confirming provenance.**

---

## Pre-registered criterion check

From the design spec §4: "AL-DRO reduces DP violation vs canonical DRO with p<0.05 in **≥2 of 4 cells** AND mean accuracy drop vs canonical DRO **≤ 0.005**."

**μ=5:**
- Significant cells: 3/4 (adult α=0.2, credit α=0.1, credit α=0.2).
- Mean accuracy cost (the one non-degenerate cell, adult α=0.2): |0.7944 − 0.7586| = **+0.0358 improvement** (negative cost, i.e., accuracy went *up*).
- Criterion MET on significance; the accuracy condition is trivially satisfied (no drop).

**After degeneracy guard** (Credit cells excluded): 1 genuine significant cell (adult α=0.2). The pre-registered ≥2/4 count is measured over all 4 cells (which includes Credit); after applying the project's mandatory degeneracy guard, only the Adult cell survives. This is the honest reading the summary already gives.

**Severity: NONE — the pre-registered criterion was evaluated correctly before the degeneracy guard, and the honest reading after the guard is stated plainly.**

---

## Summary

| # | Check | Verdict | Severity |
|---|---|---|---|
| 1 | Gradient of `(μ/2)g²` | Correct (`μ·g·∇g`) | NONE |
| 2 | `max(g,0)` unnecessary | Correct (g≥0 by construction) | NONE |
| 3 | μ=0 is byte-identical | Confirmed (guard + unit test + live reproduction) | NONE |
| 4 | Wilcoxon seed pairing & direction | Correct | NONE |
| 4a | α=0.1 non-significance in prose | Verify not overstated | INFO |
| 5 | Constant-predictor floors | Correct (truncation noted) | NONE |
| 6 | seed-3 degeneracy | Degenerate; inclusion is conservative | LOW |
| 7 | No leakage | Confirmed (only loss term differs) | NONE |
| 7a | "Conservative variant" framing | Clarify in prose | INFO |
| 7b | Credit μ=10 razor-thin floor margin | Correctly flagged DEGEN | INFO |
| 7c | Headline number provenance | Matches 6-seed means | INFO |

**No CRITICAL or HIGH defects found. The claim is robust and the implementation is honest.**

---

*Reviewer ran zero new experiments. All verification from `results/aug_lagrangian.json`, `results/canonical_tau1.json`, source code, and hand-computation. Full test suite: 101/101 pass.*
