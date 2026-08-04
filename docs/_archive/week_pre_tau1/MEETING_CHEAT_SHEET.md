# Meeting Cheat Sheet — June 9, 3pm

## Status at a Glance
- **Experiments:** 269/270 done (99.6%). One Adult α=0.4 combined DRO run remaining.
- **Time:** ~25 minutes until meeting
- **Key files:** `MEETING_PREP_JUNE_9.md`, `figures/partial_results_dp.png`, `experiments/analyze_lsac_complete.py`

---

## What to Tell Madam

### 1. Bugs Fixed ✅
- **4 critical attack bugs:** batched PGD on discrete flips, uniform gradient magnitude, IF dominating combined, stale heuristic
- **3 additional bugs:** classifier eval mode, LSAC attr (sex→race), validation tau mismatch
- All fixes committed and experiments re-running with corrected code
- **Results:** 270 runs essentially complete with fixed code

### 2. Lambda Diagnostic ✅
- **Complete (12/12).** λ_DP on Adult is ~0.05, stable, bounded.
- Raising cap from 0.5→1.5 produces identical results.
- **Ruling out H3:** DRO failure on Adult is NOT lambda runaway.

### 3. Critical Finding: Radii Mismatch 🔍
- DRO's radii formula assumes **uniform corruption**, but attack uses **coordinated targeting** (70% minority).
- On Adult: formula estimates Female=7.5%, true=32.5% (massive error).
- **This is a research design mismatch, not a code bug.**

### 4. Final Results (269/270, t-tested)

#### Adult — Two-Regime Pattern
- **α=0.0-0.3:** DRO makes attacks **significantly worse** (p<0.05)
  - α=0.2 combined: DRO DP +0.229, p=0.0011 ✅
  - α=0.2 dp: DRO DP +0.175, p=0.0060 ✅
- **α=0.4:** DRO **significantly helps** for DP attack
  - α=0.4 dp: DRO DP -0.027, p=0.0006 ✅ (n=3)

**Narrative flip:** DRO is NOT universally broken on Adult. The radii mismatch hurts at moderate corruption but becomes less harmful at high corruption.

#### Credit — COMPLETE (90/90). No Significant Effects
- All p-values > 0.05 across all alphas and attacks
- DRO is effectively **neutral** on Credit
- Closest: α=0.2 dp with p=0.066 (not significant)

#### LSAC — COMPLETE (90/90). Significant Only at α=0.4
- α=0.1-0.3: **No significant differences** (all p > 0.05)
- α=0.4 significant effects:
  - dp: DRO worse (+0.050), p=0.0019 ✅
  - if: DRO worse (+0.006), p=0.0088 ✅
  - combined: DRO better (-0.009), p=0.0111 ✅

### 5. Honest Limitations
- K_inner=5 (pragmatic for CPU), not paper spec K_inner=10
- 3 seeds only — low power for t-tests
- One Adult run still pending (will complete before meeting)
- UTKFace still blocked (no GPU)

---

## Key Questions for Madam
1. Should we derive new radii formula for coordinated attacks?
2. Within-group k-NN for IF attack — intentional or mismatch?
3. How to interpret Adult two-regime pattern? Is high-α benefit useful?
4. UTKFace priority — server access this week or skip?

---

## Files to Reference
- `MEETING_PREP_JUNE_9.md` — full detailed report with all t-tests
- `figures/partial_results_dp.png` — DRO vs Naive curves
- `experiments/demonstrate_radii_mismatch.py` — concrete numerical demo
- `experiments/analyze_lsac_complete.py` — LSAC t-test script

*Generated: June 9, ~2:35 PM IST*
