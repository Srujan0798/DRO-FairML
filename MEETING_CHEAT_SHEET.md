# Meeting Cheat Sheet — June 9, 3pm

## Status at a Glance
- **Experiments:** 192/270 done (71%). LSAC complete. Adult & Credit in progress.
- **Time:** ~2.5 hours until meeting
- **Key files:** `MEETING_PREP_JUNE_9.md`, `figures/fig11_lambda_diagnostic.pdf`, `figures/partial_results_dp.png`

---

## What to Tell Madam

### 1. Bugs Fixed ✅
- **4 critical attack bugs:** batched PGD on discrete flips, uniform gradient magnitude, IF dominating combined, stale heuristic
- **3 additional bugs:** classifier eval mode, LSAC attr (sex→race), validation tau mismatch
- All fixes committed and experiments re-running with corrected code

### 2. Lambda Diagnostic ✅
- **Complete (12/12).** λ_DP on Adult is ~0.05, stable, bounded.
- Raising cap from 0.5→1.5 produces identical results.
- **Ruling out H3:** DRO failure on Adult is NOT lambda runaway.

### 3. Critical Finding: Radii Mismatch 🔍
- DRO's radii formula assumes **uniform corruption**, but attack uses **coordinated targeting** (70% minority).
- On Adult: formula estimates Female=7.5%, true=32.5% (massive error).
- **This is a research design mismatch, not a code bug.**

### 4. Emerging Results (Partial, 192/270)

| Dataset | Pattern |
|---------|---------|
| **Adult** | DRO consistently WORSE for DP/Combined. **Surprising: DRO helps for IF attack at α=0.2 (-5.3%).** |
| **Credit** | DRO neutral to slightly BETTER. Dramatic win for IF at α=0.2 (-82.6% with n=3, stabilizing to ~-8% with n=4). |
| **LSAC** | Alpha-dependent, attack-dependent. DRO wins at α=0.1, mixed at 0.2-0.3, near-zero/collapse at high α. |

### 5. Honest Limitations
- K_inner=5 (pragmatic for CPU), not paper spec K_inner=10
- 3 seeds only — insufficient for Wilcoxon significance
- Results partial (~71%) at meeting time
- LSAC high variance with race attribute

---

## Key Questions for Madam
1. Should we derive new radii formula for coordinated attacks?
2. Within-group k-NN for IF attack — intentional or mismatch?
3. UTKFace priority — server access this week or skip?
4. K_inner=5 pragmatic choice OK for now?

---

## Files to Reference
- `MEETING_PREP_JUNE_9.md` — full detailed report
- `figures/fig11_lambda_diagnostic.pdf` — lambda trajectory figure
- `figures/partial_results_dp.png` — DRO vs Naive curves
- `experiments/demonstrate_radii_mismatch.py` — concrete numerical demo

*Generated: June 9, ~12:35 PM IST*
