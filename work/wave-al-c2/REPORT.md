# TASK C2 — AL × radius compound test

**Verdict: CONFLICT.** The two fixes do not compound and are not redundant —
combined is worse than either single because the extra constraint pressure
collapses the model.

## What I did

Pre-registered before running (`docs/superpowers/specs/2026-08-07-al-radius-compound-prereg.md`),
then completed the 2×2 grid (Adult, DP attack, seeds 0–5) by running the 18
missing cells into `results/al_radius_compound.json` (AL-only at α=0.3 plus the
combined arm at both α). Canonical / AL-only-α=0.2 / radius-only arms were
reused read-only from `canonical_tau1.json`, `mu_sensitivity.json`,
`radius_sensitivity.json`. Grid now 48/48 complete. Degeneracy guard applied:
Adult floor 0.7521 + 0.005 = 0.7571.

## Numbers (α=0.2, the scoped cell)

| arm | DP | acc | verdict |
|---|---|---|---|
| canonical | 0.2334 | 0.7586 | reference |
| radius-only (r=2, μ=0) | 0.2291 | 0.7609 | inert (R=+1.8%) |
| AL-only (r=1, μ=20) | 0.0682 | 0.7783 | genuine Pareto win |
| **combined (r=2, μ=20)** | **0.0139** | **0.7561** | **DEGENERATE** |

## Why conflict

The combined arm's DP (R=+94%) is a collapse artifact: accuracy 0.7561 is at/below
the degeneracy threshold, so the DP ≈ constant-predictor. AL-only is the real win
already; the larger radius destroys the accuracy margin AL preserves. This
disagrees with my pre-registered prediction (REDUNDANT) — the degenerate
counter-hypothesis flagged in the pre-registration is what occurred. α=0.3 is
fully degenerate across all arms (excluded regime), stress-test only.

Full detail: `results/al_radius_compound_summary.md`.
