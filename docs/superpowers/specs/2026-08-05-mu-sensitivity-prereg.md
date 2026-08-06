# TASK C pre-registration — μ sensitivity and the safe operating range

**Written and committed BEFORE the experiment ran.** Date: 2026-08-05.
Parent: `2026-08-05-augmented-lagrangian-design.md`,
`2026-08-05-al-generalisation-prereg.md`.

## Why this is mandatory, not optional

TASK A found that μ=5 produces a **catastrophic failure at α=0.4 on Adult**:
DP 0.0551 (−80.7%, p=0.0156 — the best-looking number in the whole study) with
accuracy **0.3495**, below chance for this label balance. The model is
destroyed while its fairness metric looks superb.

We therefore cannot recommend any μ until we know **where the safe boundary is
as a function of corruption level**. A single "use μ=5" recommendation would be
actively dangerous.

Second open question from TASK A: Credit collapsed at μ=5 *and* μ=10 at α≤0.2.
Is Credit simply intolerant of AL, or does a **smaller** μ give Credit a genuine
non-degenerate win? That distinction decides whether AL is Adult-only.

## Grid

DRO only, 6 seeds, attack=dp, μ=0 reference and μ∈{5,10} already exist.

| block | dataset | α | μ | runs |
|---|---|---|---|---|
| safety frontier | adult | 0.0, 0.2, 0.4 | 0.5, 1, 2, 20 | 72 |
| Credit rescue | credit | 0.2 | 0.5, 1, 2 | 18 |

**90 runs** → `results/mu_sensitivity.json` (new file; nothing locked touched).
Combined with existing μ∈{5,10} rows this yields 5–6 point curves.

## Pre-registered definitions and decision rules

Floors: Adult 0.7521, Credit 0.7788. `SAFE(μ, α, ds)` iff mean accuracy
> floor + 0.005. `EFFECTIVE(μ, α, ds)` iff one-sided Wilcoxon vs canonical DRO
(n=6) gives p < 0.05 with lower DP.

**Rule C1 — safe operating range.** For each α on Adult, report
`μ_max_safe(α)` = the largest tested μ that is SAFE. The headline deliverable
is whether `μ_max_safe` **decreases as α increases**. Pre-registered
expectation (stated before seeing data): it decreases, i.e. heavier corruption
tolerates less constraint pressure.

**Rule C2 — recommendation rule.** The recommended μ is the largest tested μ
that is simultaneously SAFE and EFFECTIVE at that α. If no μ is both, the
recommendation for that α is **"do not use AL"** — stated plainly, not softened.

**Rule C3 — Credit rescue.** Credit is rescued iff some μ ∈ {0.5, 1, 2} is both
SAFE and EFFECTIVE at α=0.2. If yes, AL generalises to a second dataset with a
dataset-specific μ. If no, we state that AL is **Adult-only among the datasets
tested**, which materially limits the claim and must appear in the paper.

**Rule C4 — monotonicity sanity check.** DP should fall monotonically (or near
so) as μ rises, since μ scales constraint pressure. A non-monotone DP curve
would indicate optimisation instability rather than a clean trade-off, and
must be reported as such rather than smoothed over.

## Commitment

Run once. Report the frontier as measured. If the safe range turns out to be
so narrow that AL is impractical, say so — that is a legitimate outcome and is
more useful to the submission than a tuned number.
