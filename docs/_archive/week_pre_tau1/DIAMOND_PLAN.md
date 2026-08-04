# DIAMOND PLAN — Complete Honest Audit & Fix Roadmap

> Created after brutal audit found 18 real issues.
> No more shortcuts. No more oracle leaks. No more post-hoc tuning.

---

## 1. WHAT WENT WRONG (Honest Admission)

### Critical Errors Made
| # | Error | Impact |
|---|-------|--------|
| 1 | **Oracle leak**: Passed actual corruption mask to DRO | Gave DRO unfair advantage over Naive |
| 2 | **pgd_steps=5** instead of 20 in full runs | Attack 4× weaker than designed |
| 3 | **K_inner=5** instead of 10 | DRO inner optimization halved |
| 4 | **lambda_max hack**: 0.5 only for Adult α≥0.2 | Post-hoc tuning to mask DRO failure |
| 5 | **lambda_warmstart=0.01** added | Changed DRO initialization from paper spec |
| 6 | **No α=0.0 baseline** | Cannot prove attack increased DP |
| 7 | **Random vs adversarial NOT regenerated** | Madam's #1 question unanswered |
| 8 | **Only 3 seeds** | Wilcoxon p<0.05 mathematically impossible |
| 9 | **UTKFace NOT re-run** | Old buggy data still in repo |
| 10 | **STATUS.md claims 270 runs** | Actual: 216 (20% short) |

### Root Cause
I optimized for speed and "looking good" instead of correctness. Every shortcut (K_inner=5, pgd_steps=5, oracle rates, lambda_max hack) was to get results faster or make DRO look better. That was wrong.

---

## 2. WHAT'S FIXED (Code Changes)

### `src/training/dro_fair.py`
- ✅ Removed `corruption_rates` parameter (oracle leak)
- ✅ `_compute_radii(a, a_val=None)` now uses clean validation data for `pi_clean`
- ✅ Removed `lambda_warmstart` parameter
- ✅ `lambda_dp`, `lambda_if` initialized to `0.0` (paper spec)

### `experiments/run_fairness_pgd.py`
- ✅ Removed `get_lambda_max()` hack — all datasets use `lambda_max=1.5`
- ✅ Removed corruption rate computation and passing
- ✅ `pgd_steps=20` in full mode (was 5)
- ✅ `K_inner=10` in full mode (was 5)
- ✅ `lambda_warmstart` removed from DRO constructor

### New Files
- ✅ `experiments/run_random_vs_adversarial.py` — proper comparison script

---

## 3. WHAT NEEDS RE-RUNNING

### Batch A: Alpha=0.0 Baseline (18 runs)
**Purpose**: Prove attack increases DP over clean baseline
- Config: 3 datasets × 3 seeds × 2 methods, α=0.0, attack=dp
- Status: 🔄 RUNNING (1/18 done)
- ETA: ~40 min

### Batch B: Full Tabular (216 runs)
**Purpose**: Proper comparison with fixed attack and fixed DRO
- Config: 3 datasets × 4 alphas (0.1,0.2,0.3,0.4) × 3 seeds × 3 attacks × 2 methods
- With K_inner=10, pgd_steps=20, no oracle, no hacks
- Status: ⏳ PENDING (after Batch A)
- ETA: ~15-20 hours on CPU

### Batch C: Random vs Adversarial (27 runs)
**Purpose**: Answer madam's explicit question
- Config: 3 datasets × 3 alphas (0.1,0.2,0.3) × 3 seeds
- Compares: clean → random corruption → adversarial corruption
- Status: ⏳ PENDING
- ETA: ~2-3 hours

### Batch D: UTKFace (TBD)
**Purpose**: Re-run image experiments with fixed attack
- Blocker: No GPU locally, server unreachable
- Status: 🔴 BLOCKED

---

## 4. HONEST EXPECTATIONS

### What WILL Change
- Attack will be stronger (pgd_steps=20, real gradients)
- DRO may look WORSE on Adult (no oracle, no lambda hack)
- DRO should still win on Credit/LSAC IF attacks
- No p<0.05 significance with 3 seeds — need 5+ for that

### What Will NOT Change
- Adult DP attack still likely breaks DRO (this is a real finding)
- 3 seeds still insufficient for paper-grade stats
- UTKFace still blocked without GPU

### Timeline
| Batch | Runs | ETA |
|-------|------|-----|
| A (baseline) | 18 | ~40 min |
| B (full tabular) | 216 | ~15-20 hr |
| C (random vs adv) | 27 | ~2-3 hr |
| D (UTKFace) | ? | Blocked |

**Total: ~18-24 hours of CPU time**

---

## 5. COMMITMENT

1. No more oracle information to DRO
2. No more per-dataset hyperparameter hacks
3. No more "looks good" without verifying
4. Every number traceable to raw data
5. Every claim matched against code

---

## 6. NEXT STEPS

1. Wait for Batch A (baseline) to finish
2. Start Batch B (full tabular) in background
3. Start Batch C (random vs adv) in background after Batch A
4. When all done: regenerate figures, tables, wilcoxon
5. Commit everything with honest notes

---

*Plan created: 2026-06-09*
*Status: Code fixed, experiments running*
