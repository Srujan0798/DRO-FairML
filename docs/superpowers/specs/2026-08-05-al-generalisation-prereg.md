# TASK A pre-registration — does the AL improvement generalise?

**Written and committed BEFORE the experiment ran. No criterion changed after.**
Date: 2026-08-05. Parent design:
`docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md`.

> **Status (2026-08-07): COMPLETE, 42/42.** This ran at μ=5 exactly as
> specified below — that value is correct and unedited here for audit
> purposes. Its α=0.4 finding (μ=5 destroys the model) motivated a follow-up
> sensitivity sweep which found μ=20 is the better value at α≤0.2; see the
> addendum in `2026-08-05-augmented-lagrangian-design.md` and
> `docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`. Results:
> `results/aug_lagrangian_extended_summary.md`.

## What we are testing

DRO-FAIR-AL is proven on exactly one cell (Adult, α=0.2, μ=5): DP
0.2334 → 0.1358, p=0.0156, 6/6 seeds, accuracy 0.7586 → 0.7944. One cell is not
a result. Three questions:

1. **Falsification (the important one).** Does AL help just as much with **no
   corruption at all** (α=0)? AL is a fairness penalty, so it will reduce DP at
   α=0 to *some* degree — that is expected and is not itself a failure. The
   question is **magnitude**. If AL's benefit at α=0 is comparable to its
   benefit under attack, then AL is a **generic fairness regulariser** and
   framing it as corruption-robustness is wrong.
2. Does the win hold at higher corruption (Adult α ∈ {0.3, 0.4})?
3. Does it survive other attacks (Adult α=0.2, attack ∈ {if, combined}) and a
   degenerate dataset (LSAC α ∈ {0.1, 0.2})?

## Pre-registered decision rules (numeric, fixed now)

Let `R(α) = (DP_canonicalDRO − DP_AL) / DP_canonicalDRO` — the **relative** DP
reduction AL buys at that cell. Reference already measured: `R(0.2) = 41.8%`
on Adult.

**Rule 1 — falsification.** The corruption-robustness framing is SUPPORTED only
if `R(0.0) < 0.5 × R(0.2)`, i.e. **R(0.0) < 20.9%**.
If `R(0.0) ≥ 20.9%`, we declare the robustness framing **NOT supported** and
report AL as a general fairness regulariser whose benefit is not corruption-
specific. This is written down precisely so it cannot be softened afterwards.

**Rule 2 — higher corruption.** The win "holds at higher α" if Adult α ∈ {0.3,
0.4} each show p < 0.05 (one-sided Wilcoxon, canonical DRO DP > AL DP, n=6) AND
are non-degenerate (accuracy > floor + 0.005; Adult floor 0.7521).

**Rule 3 — other attacks.** The win "transfers across attacks" if at least one
of {if, combined} at Adult α=0.2 shows p < 0.05 and is non-degenerate.

**Rule 4 — LSAC.** Expected to be DEGENERATE (accuracy at/below 0.9016 floor),
consistent with the documented LSAC/DP collapse. Confirming degeneracy is the
success condition here; a "DP win" on LSAC must be reported as collapse.

**Degeneracy guard applies everywhere.** Floors: Adult 0.7521, Credit 0.7788,
LSAC 0.9016. Any DP improvement at/below floor + 0.005 is model collapse, not
fairness, and is reported as such regardless of p-value.

## Grid

μ=5 only (μ=10 already shown to cost more accuracy for no extra genuine cell).
DRO only; canonical DRO reference rows come read-only from
`results/canonical_tau1.json` (same seeds/protocol, so pairing is valid).

| block | dataset | attack | α | seeds | runs |
|---|---|---|---|---|---|
| falsification | adult | dp | 0.0 | 0–5 | 6 |
| higher-α | adult | dp | 0.3, 0.4 | 0–5 | 12 |
| other attacks | adult | if, combined | 0.2 | 0–5 | 12 |
| degenerate ds | lsac | dp | 0.1, 0.2 | 0–5 | 12 |

**Total 42 runs** → `results/aug_lagrangian_extended.json` (new file; the
existing `results/aug_lagrangian.json` is not appended to, and no locked file is
touched).

## Commitment

Run once. Report whatever comes out. If Rule 1 fails, the headline framing
changes and that is reported prominently in the memo to the advisor rather than
buried — the whole point of running this control first is to find our own
weakness before a reviewer does.
