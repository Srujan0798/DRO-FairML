# TASK C2 pre-registration — does AL compound with the radius fix, or is it redundant?

**Written and committed BEFORE the experiment ran. No criterion changed after.**
Date: 2026-08-07. Parents: `2026-08-05-augmented-lagrangian-design.md`,
`2026-08-05-mu-sensitivity-prereg.md`, the N1 radius ablation
(`results/radius_sensitivity.json`), and `docs/TASKS_AL_VALIDATION.md` §TASK C2.

## What we are testing

Two independent findings both say the canonical DRO-FAIR configuration
under-enforces its own fairness objective:

1. **The AL fix** (DRO-FAIR-AL, TASK A/C): the dual variable λ is starved by
   geometric decay (max λ 0.0119 vs ceiling 1.5), so the linear λ·g penalty is
   ~0.5% of the loss. Fixed by adding the quadratic penalty `(μ/2)g²`.
   TASK C established μ=20 is the safe, effective value at α≤0.2 (Adult:
   DP −70.8% at α=0.2, −81.7% at α=0.0, accuracy held or improved).
2. **The radius finding** (N1 ablation, complete): 11/15 (dataset, α) cells
   prefer `radii_scale=2.0` over canonical 1.0 — the closed-form TV radius is
   systematically too small. A larger radius lets the inner maximization push
   the importance weights further from uniform, so the DP violation `g_dp`
   measured during training is larger and the constraint penalty bites harder.

Both levers act on the **same penalty term** in the DRO Lagrangian,
`L = L_tilt + λ·g + (μ/2)·g²`: AL inflates the coefficient μ; the radius
inflates the measured `g` itself. This is why the two findings are suspected of
being two views of one deficiency.

## Grid

Adult, attack=dp, DRO only, α ∈ {0.2, 0.3}, 6 seeds (0–5), 2×2 design:

| arm | radii_scale | aug_lagrangian_mu | where the data lives |
|---|---|---|---|
| canonical C | 1.0 | 0 | `results/canonical_tau1.json` (exists) |
| AL-only A | 1.0 | 20 | α=0.2: `results/mu_sensitivity.json` (exists); **α=0.3: NEW** |
| radius-only R | 2.0 | 0 | `results/radius_sensitivity.json` (exists) |
| combined B | 2.0 | 20 | **NEW (all 12 cells)** |

Existing rows are reused read-only (never re-run, never overwritten). New rows
go to `results/al_radius_compound.json` — a new file; nothing locked is touched.
18 new runs total (6 AL-only α=0.3 + 12 combined).

## Pre-registered definitions and decision rules (numeric, fixed now)

Floors: Adult 0.7521 + margin 0.005 → `DEGEN_THRESH = 0.7571`.

For each α separately, over the same 6 seeds (seed-paired):

- `R_X = (mean_DP_C − mean_DP_X) / mean_DP_C` — relative DP reduction of arm X
  vs canonical (as in prior tasks).
- `S` = the better single = argmin over {A, R} of mean DP, restricted to
  non-degenerate singles.
- Wilcoxon: one-sided seed-paired `wilcoxon(DP_best, DP_other,
  alternative='greater')` — p is the chance the first arm's DP exceeds the
  second's.

**Rule 1 — degeneracy guard (mandatory).** Any arm whose mean accuracy
`≤ 0.7571` is **DEGENERATE** (collapse to the constant predictor; DP≈0 by
construction is not fairness). Its DP is reported as collapse, never as a win.
If the combined arm B is DEGENERATE, the verdict for that α is **CONFLICT**
regardless of its DP number.

**Rule 2 — COMPOUND.** B is compounding over both singles iff all of:
1. B non-degenerate;
2. one-sided Wilcoxon `p(B < S) < 0.05` AND `mean_DP_B < mean_DP_S − 0.005`;
3. same vs the *other* single (B beats each single individually).

**Rule 3 — REDUNDANT.** B is redundant with the best single iff B is
non-degenerate and B neither beats nor loses to S by the Rule-2/4 thresholds:
one-sided Wilcoxon `p(B < S) ≥ 0.05` AND `p(S < B) ≥ 0.05` AND
`|mean_DP_B − mean_DP_S| < 0.005`. (Combined ≈ best single: the two fixes
correct the same underlying deficiency through different mechanisms. This is a
publishable finding, not a failure.)

**Rule 4 — CONFLICT.** B is conflicting iff B is non-degenerate but is
significantly *worse* than the best single: one-sided Wilcoxon
`p(S < B) < 0.05` AND `mean_DP_S < mean_DP_B − 0.005` — or B is DEGENERATE
(Rule 1).

**Rule 5 — overall verdict.** α=0.2 is the scoped, in-scope cell (TASK C: AL
is only recommended for α≤0.2; α=0.3 canonical is already a marginal/excluded
regime). The overall verdict is: **COMPOUND if α=0.2 is COMPOUND**;
**REDUNDANT if α=0.2 is REDUNDANT**; **CONFLICT if α=0.2 is CONFLICT**.
α=0.3 is reported separately as a stress test (canonical there sits near/below
the floor), never used to rescue or over-claim.

## Pre-registered prediction

**REDUNDANT.** Both levers amplify the *same* term `λ·g + (μ/2)g²` on the same
constraint: AL via the coefficient, radius via the measured `g`. Once the
stronger lever (AL at μ=20) has already driven `g_dp` near zero at α=0.2
(measured DP 0.0682 vs canonical 0.2164), the radius has little residual `g`
left to amplify — so combined should land close to AL-only. The alternative
that would overturn this (→ COMPOUND) is if the larger radius inflates `g`
enough during training to give the quadratic penalty genuine extra headroom
even near convergence. The degenerate counter-hypothesis (→ CONFLICT at α=0.3)
is real: a larger radius plus μ=20 is exactly the combination most likely to
collapse the model, and α=0.3 is already a marginal regime.

## Commitment

Run once. Report whatever comes out. If the verdict disagrees with the
pre-registered prediction, the mismatch is stated plainly and the data decides
— the pre-registered numeric rules are applied mechanically regardless.
