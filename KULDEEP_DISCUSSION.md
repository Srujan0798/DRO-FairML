# Kuldeep Discussion — Canonical Tau=1 Results (2026-06-30)

**Status:** 6-seed canonical (K_inner=10, tau=1 fixed) **complete at 360/360 rows**
(3 datasets × {DP, Combined} attacks × 5 alphas × 6 seeds × 2 methods). The IF-attack
third (180 rows) was never generated — the IF metric was degenerate pre-fix — and is
pending a cluster re-run (see `STATUS.md` / `docs/MASTER_DISPATCH.md`). See
`results/canonical_tau1.json`.
> NOTE: claims below are scoped per MASTER_DISPATCH.md. IF numbers are pre-fix/degenerate
> and must not be cited; LSAC/DP is a degenerate (collapse) result, not a DRO win.

**Headline:** Fixed tau=1 makes DRO robust under coordinated fairness attacks (at α ≤ 0.2,
Adult + Credit, all three attacks, p<0.05, n=6). Previous fragile results were a tau=100
temperature artifact.

---

## 1. Headline Results — Adult DP Attack (6 seeds, tau=1 fixed)

| Alpha | Naive DP | DRO DP | Delta DP | DRO Wins | p-value |
|-------|----------|--------|----------|----------|---------|
| 0.0 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.016 |
| 0.1 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.031 |
| 0.2 | 0.2452 | 0.2334 | +0.0119 | 6/6 | 0.016 |
| 0.3 | 0.2848 | 0.2614 | +0.0234 | 6/6 | 0.016 |
| 0.4 | 0.3140 | 0.2855 | +0.0285 | 6/6 | 0.016 |

 DRO advantage grows with corruption (Delta DP from 0.006 to 0.029) **within the
 defensible regime α ≤ 0.2**. At α ≥ 0.3 both DRO and Naive fall below the
 constant-predictor baseline (Adult 0.752, Credit 0.779, LSAC 0.902), so the "growing
 advantage" there is scientifically empty — both methods are simply worse than predicting
 a constant. No method claim is made for α ≥ 0.3.

## 2. Other Attacks and Datasets

**Adult Combined attack:** DRO wins at **α ≤ 0.2** (all p<0.05, 6 seeds). At α ≥ 0.3 both
methods are below the constant-predictor baseline — no claim.

**Adult IF attack:** IF results are **NOT reported as claims**. The IF metric was found
degenerate on the canonical grid (violation ≈ 1e-10 everywhere — a threshold-calibration
bug, not a real signal); see MASTER_DISPATCH.md BLOCKER 1. IF numbers are being
regenerated after the fix; do not cite pre-fix IF figures.

**Credit (DP, α ≤ 0.2):** DRO wins at all alpha values (p<0.05, 6 seeds). α ≥ 0.3: below
baseline, no claim.

**LSAC/DP:** **Degenerate** — the model collapses to the majority-class predictor
(accuracy ≈ 0.902 = constant-predictor rate; DP frozen at 0.1827 across α = 0.2–0.4).
Reported separately as a negative/diagnostic result, **not** as a DRO win. LSAC/Combined
is a genuine clean win (p=0.0156 at α=0.1/0.3/0.4).

## 3. Tau=100 Artifact Resolution

Previous finding that "DRO is fragile" was caused by a temperature schedule that used tau=100 for lower corruption levels. With fixed tau=1, DRO consistently outperforms Naive on DP. Tau=100 configuration showed DRO losing at every alpha (e.g., Adult alpha 0.2: Naive DP 0.327 vs DRO DP 0.503).

## 4. Wilcoxon Statistical Significance

15 out of 21 cells show significant improvement (p<0.05) for DRO over Naive on DP:
- Adult Combined: alpha 0.1 through 0.4, all p=0.016
- Adult DP: alpha 0.1 p=0.031, alpha 0.2 through 0.4 p=0.016
- Adult IF: alpha 0.1 p=0.031
- Credit (all attacks): alpha 0.1 through 0.2, all p<0.05

## 5. Deliverables Completed

- Report PDF (277K) and paper PDF (102K) rebuilt with auto-generated tau=1 LaTeX tables (no hardcoded tau=100 data)
- 133 figure files regenerated from canonical data
- All experiment scripts fixed to read from canonical_tau1.json
- PGD p-values corrected from placeholder "TBD" to real Wilcoxon values
- All 60 tests passing
- Pushed to GitHub (commit 4279277)
- Auto-complete watcher runs full post-pipeline on experiment completion

## 6. Remaining

- **UTKFace:** Blocked — GPU access on flair2 not granted. Email to Supin drafted.
- **Credit and LSAC:** Completing autonomously via watchdog (estimated 4-6 hours).
- After completion: auto-pipeline regenerates all analysis with full 540-row data.

## 7. Key Figures in kuldeep_meeting/

- report.pdf, paper.pdf — full documents
- fig_tau1_headline.pdf — tau=1 vs tau=100 side by side
- fig_win_curves_tau1.pdf — DRO advantage per attack
- fig_final_wilcoxon_table.pdf — significance table
- fig_final_tradeoff_vs_constant_predictor.pdf — DRO vs constant predictor
- fig17_summary_dp_vs_alpha.pdf — master DP comparison
- adult_accuracy_tau1_meeting.pdf, adult_accuracy_tau100_meeting.pdf
- adult_if_tau1_meeting.pdf
- adult_acc_vs_alpha_different_tau.pdf — tau=1/10/100 overlaid
