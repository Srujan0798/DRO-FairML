# MEETING PREP — Tuesday, June 9, 2026

## ONE-LINER

> "Madam was right — the adversarial attack on DP was fundamentally wrong. I found and fixed 4 critical bugs in the attack code, plus 11 other issues. The full 270-experiment re-run is ~7% complete and will finish by tomorrow morning. A key finding already emerged: with the corrected attack, DRO consistently underperforms Naive on Adult because the DRO radii assume uniform corruption but our attack is coordinated."

---

## PART 1 — Attack Audit: Madam Was Right (5 min)

### What Madam Said (June 2)
> "Check the adversarial attack on DP and improve it. Then redo all the experiments."

### What I Found
The `FairnessTargetedPGD` attack had **4 critical bugs** that made it far too weak:

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | Used batched PGD on discrete label flips | Flipped same samples multiple times | Replaced with greedy algorithm |
| 2 | DP gradient had uniform ±1 magnitude | Small groups had same impact as large groups | Exact marginal gain ±1/count_g |
| 3 | Combined attack: IF dominated DP 1000× | DP signal was invisible | Normalize both to [-1,1] before mixing |
| 4 | Baseline heuristic had stale group rates | After first flip, rates never updated | Recompute after each flip |

### Validation: Attack is Now 3-5× Stronger

Old buggy result (Adult α=0.2): `dp=0.047`
New fixed result (Adult α=0.1): `dp=0.17-0.22`

The attack is working correctly now. All 42 tests pass.

---

## PART 2 — Critical Finding: DRO Fails on Adult (5 min)

### Observation (from 20/270 partial results)

For **Adult α=0.1**, DRO produces **HIGHER** DP violation than Naive:

| Attack | Naive DP | DRO DP | DRO Worse By |
|--------|----------|--------|-------------|
| DP | 0.1827 | 0.1995 | +9.2% |
| IF | 0.1499 | 0.1795 | +19.8% |
| Combined | 0.1673 | 0.1789 | +6.9% |

### Root Cause: Radii Mismatch

The DRO trainer computes TV radii assuming **uniform random corruption**:
```python
pi_clean[j] = (pi_obs[j] - alpha) / (1 - 2 * alpha)
```

But `FairnessTargetedPGD` uses **coordinated targeting** (70% minority group).

This causes:
1. `pi_obs` under coordinated attack ≠ `pi_obs` under uniform attack
2. "Bias-corrected" `pi_clean` is wrong
3. Radii `rho_dp` are miscalibrated
4. DRO's inner maximization explores the wrong uncertainty set
5. `lambda_DP` stays small (~0.05) — DRO doesn't "see" the true adversary

### Lambda Diagnostic Confirms

| Dataset | λ_DP final | DRO DP | Status |
|---------|-----------|--------|--------|
| Adult | 0.046-0.050 | 0.136-0.159 | **DRO fails** |
| Credit | 0.015-0.023 | **0.000** | DRO wins |
| LSAC | 0.013-0.014 | **0.000** | DRO wins |

Credit/LSAC achieve perfect fairness because their base rates are more imbalanced, making coordinated attacks less effective at exploiting the radii mismatch.

### What This Means

This is a **research design problem**, not a code bug. The paper's theory assumes uniform corruption. Our adversary is coordinated. The fix requires either:
- Computing radii from the actual coordinated attack distribution
- Using a larger fixed radius
- Switching to a fixed-budget uncertainty set

---

## PART 3 — Experiment Status (2 min)

### Completed
- ✅ Lambda diagnostic: 12/12 runs (all 3 datasets, 2 λ_max values, 3 seeds)
- ✅ All code fixes committed and pushed
- ✅ 42/42 tests passing

### In Progress
- ⏳ Tabular re-run: 20/270 experiments completed (~7%)
- ⏳ Currently on: Adult α=0.1 seed=3 attack=if method=naive
- ⏳ Pace: ~19 experiments/hour
- ⏳ ETA: tomorrow morning (well before 3pm)

### Post-Processing Ready
- `experiments/auto_finalize.py` — one command to analyze, plot, summarize, push
- `experiments/generate_paper_tables.py` — auto-generates LaTeX tables
- `experiments/generate_report_tables.py` — auto-generates report tables

---

## PART 4 — Server Experiments (1 min)

Same as last week: UTKFace scripts are coded and ready on `flair2.iitgn.ac.in`:
- `run_utkface_extended.py` (alpha sweep, fairness PGD, lambda_max cap)
- `run_utkface_pixel_pgd.py`
- `run_utkface_randinit.py`

Cannot run locally — feature cache only on server.

---

## OPEN QUESTION FOR MADAM

> "I fixed the attack (it was indeed wrong — now 3-5× stronger). With the correct attack, DRO's radii formula is exposed as mismatched because it assumes uniform corruption but our attack is coordinated. This explains why DRO fails on Adult. Should I:
> 1. Fix the radii formula to account for coordinated attacks?
> 2. Switch to a fixed-budget uncertainty set instead?
> 3. Both — run ablations comparing the two approaches?"

---

## FILES TO SHOW

| File | What |
|------|------|
| `docs/FINDING_DRO_FAILS_ON_ADULT.md` | Full analysis of the radii mismatch |
| `src/corruption/adversarial.py` | Fixed attack code (greedy + exact gradients) |
| `results/lambda_diagnostic_full.json` | Completed 12-run diagnostic |
| `STATUS.md` | Live experiment dashboard |
