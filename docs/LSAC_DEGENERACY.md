# LSAC/DP Degeneracy — Diagnosis (Agent B)

**Date:** 2026-07-20 · **Source:** `results/canonical_tau1.json` (540 rows, τ=1, k_inner=10)

## Observation (verified from canonical data)

For **LSAC / DP attack / DRO** (30 rows: 5 α × 6 seeds):

| α | mean acc_clean | mean dp_clean |
|---|---------------|--------------|
| 0.0 | 0.9023 | 0.1829 |
| 0.1 | 0.9046 | 0.2539 |
| 0.2 | 0.9033 | 0.2230 |
| 0.3 | 0.9032 | 0.2220 |
| 0.4 | 0.9029 | 0.2211 |

Two red flags:

1. **Accuracy is pinned to the constant-predictor baseline** (LSAC majority rate ≈
   **0.9016**) at *every* α. The model is, for practical purposes, the majority-class
   predictor. Any "win" on DP is measured on a degenerate classifier.
2. **DP does not move with corruption** in the expected way: it peaks at α=0.1
   (0.2539) and is *lower* at α=0.2–0.4 (≈0.22). A metric that dips as corruption
   triples is not tracking a robust-vs-fragile effect — it is tracking a collapsed model.

This matches MASTER_DISPATCH.md BLOCKER 2: "LSAC/dp is not simply a negative result — it
is a **degenerate run** that must be diagnosed before it is either reported or excluded."

## Hypothesis (to be confirmed by code inspection)

LSAC is ~90/10 imbalanced on the protected attribute. The DRO radii
`rho_dp[j] = alpha / ((1-alpha)*pi_clean[j] + alpha)` (in `src/training/dro_fair.py`)
blow up on the tiny minority group: as `pi_clean[minority] → 0.1`, the minority radius
grows large, over-weighting a near-empty group in the worst-case reweighting and driving
the classifier to ignore the feature signal and predict the majority class. This is an
**artifact of the radii formula on imbalanced data**, not evidence that "DRO loses on
LSAC."

## Position (what to report)

- **LSAC/DP is reported as a degenerate/diagnostic result, NOT as a DRO win or loss.**
  It must not appear in any "DRO vs Naive" win table alongside Adult/Credit.
- **LSAC/Combined is a genuine clean win** (p=0.0156 at α=0.1/0.3/0.4) and is the
  honest LSAC result to lead with.
- **Before LSAC/DP is usable**, the radii formula must be revisited for imbalanced
  groups (e.g. clip/renormalize radii, or use empirical radii) so the minority group is
  not over-weighted into collapse. This is open work, not yet done.

## Open question carried from the dispatch

- **α=0 anomaly:** with zero corruption DRO and Naive still differ structurally (DRO
  optimizes a tilted risk `β·logsumexp`, decays the dual LR, validates on a different
  schedule) and post-hoc "wins" at α=0 (Adult 6/6, p=0.016). A win at α=0 is not
  robustness — it is a different objective. **Decision needed:** either justify the α=0
  difference explicitly, or exclude α=0 from win counts. (Kuldeep's Q4, raised Jun 9,
  never resolved.)
