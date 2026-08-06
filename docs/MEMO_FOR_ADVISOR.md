# Proposed improvement to DRO-FAIR — findings and evidence

**Date:** 2026-08-05 · **Status:** result verified **and stress-tested**;
two decisions for you at the end · **Reproduce:**
`results/aug_lagrangian_summary.md`, `results/aug_lagrangian_extended_summary.md`

You asked whether we could suggest anything to make DRO-FAIR genuinely better
rather than only marginally better than Naive. We can. Here it is, with the
caveats stated as plainly as the result.

> **Headline correction (added after the falsification test).** Our first
> instinct was to present this as a *corruption-robustness* improvement. We ran
> a control at α=0 (no corruption at all) to try to disprove that, and it
> **failed the test**: the fix helps *more* with no corruption than under
> attack. So we are **not** claiming corruption robustness. The correct claim is
> narrower and, we think, cleaner: **DRO-FAIR under-enforces its own fairness
> constraint by construction, and fixing that is a large, general improvement.**
> Details in §5a. We would rather report this ourselves than have it found in
> review.

---

## 1. Why DRO-FAIR was barely beating Naive

The fairness constraints reach the model through one term, `λ·g`. But the dual
learning rate decays geometrically (`lr_λ · 0.95^epoch`), so λ's *total possible*
accumulation over 60 epochs is about 0.1 — against a ceiling of 1.5.

Measured directly (Adult, α=0.2, seed 0, canonical settings):

| quantity | value |
|---|---|
| max λ reached over 60 epochs | **0.0119** (ceiling 1.5 — 126× below) |
| mean constraint violation `g` | 0.180 |
| fairness penalty `λ·g` at its peak | **0.0029** |
| training loss for comparison | 0.538 |

**The fairness term was about 0.5% of the loss.** The machinery that is supposed
to distinguish DRO from Naive was effectively switched off. This also explains an
earlier confusing result: raising `λ_max` from 1.5 to 2.0 produced *byte-identical*
output. The ceiling was never binding — the accumulation *rate* was.

## 2. The fix

A textbook augmented Lagrangian (Hestenes 1969; Bertsekas 1982) — add a quadratic
penalty so the constraint gradient no longer waits on λ:

```text
total_loss = L_tilt + λ_dp·g_dp + (μ/2)·g_dp²  +  λ_if·g_if + (μ/2)·g_if²
```

`μ=0` recovers the current objective exactly, so **every previously reported
result stands unchanged** (unit-tested as bit-identical, and verified against the
locked canonical row on a full run).

## 3. Result — Adult, α=0.2, μ=5, n=6 seeds, seed-paired Wilcoxon

| | Naive | DRO (current) | **DRO-FAIR-AL** |
|---|---|---|---|
| DP violation | 0.2452 | 0.2334 | **0.1358** (p=0.0156) |
| accuracy | — | 0.7586 | **0.7944** |
| margin over Naive | — | +0.0119 | **+0.1094 — 9.2× larger** |

Beats Naive on DP in **6/6 seeds**, and accuracy *rises* (constant-predictor
floor is 0.7521). This is a Pareto improvement, not a fairness/accuracy trade.

## 4. What does **not** work — stated up front

**Credit's results look even better and are not real.** AL drives Credit's DP to
~0.002, but accuracy lands at or below the constant-predictor floor (0.7730–0.7790
vs floor 0.7788). A constant classifier has DP = 0 by definition — that is
collapse, not fairness. It is the same degeneracy already documented for LSAC.
We are **not** claiming Credit. A floor check is now enforced in the analysis code
so this cannot be reported as a win by accident.

Also negative, tested and reported honestly: raising `λ_max` (no-op), sharpening
β (worse), π-shrinkage on LSAC (no-op), empirical radii (negative).

## 5a. Stress test — we tried to disprove our own result (42 further runs)

Criterion written down and committed **before** the data existed
(`docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md`).

**(i) The framing failed its own test.** At α=0 — no corruption whatsoever —
AL reduces DP by **52.6%**, *more* than the 41.8% it achieves under attack, and
amplifies the margin over Naive **12.7×** versus 9.2×. We had pre-committed to
withdrawing the robustness claim if this exceeded 20.9%. It did, so we withdraw
it.

This is not an embarrassment so much as a sharper diagnosis. The defect we found
— λ decaying geometrically and never reaching its ceiling — has **nothing to do
with corruption**. A starved dual is starved identically on clean data. A fix
aimed at it should help everywhere, and it does. The contribution is therefore:
*DRO-FAIR under-enforces its fairness constraint by construction; correcting
that yields a large, general improvement.*

**(ii) A new failure mode we would not otherwise have caught.** At α=0.4, AL's
DP is **0.0551 — an 80.7% reduction, p=0.0156, the best-looking number in the
whole study.** Its accuracy is **0.3495**, below chance for this label balance.
The model is destroyed. Anyone reading the DP column alone would report this as
the headline finding. **μ=5 is not a safe default**; μ must be chosen as a
function of corruption level, which is now a required experiment rather than an
optional one.

**(iii) It does transfer across attacks.** Under the COMBINED attack at α=0.2:
DP 0.1784 → 0.0950 (−46.8%, p=0.0156) with accuracy *improving* 0.7599 → 0.8032.
The IF attack is borderline (accuracy 0.7522, sitting exactly on the 0.7521
floor) and we do not claim it.

**(iv) LSAC collapses,** as predicted — accuracy pinned at exactly 0.9016, the
constant predictor. Reported as collapse, not as a win.

**Net position.** Three genuine, non-degenerate wins: Adult DP α=0, Adult DP
α=0.2, Adult COMBINED α=0.2 — each a 42–53% DP reduction with accuracy held or
improved. Everything else is either an already-excluded regime or an honest
negative.

## 5. A second, independent finding pointing the same way

The radius ablation (180/180 runs) answers Kuldeep's original May-29 question:
the radius that minimises DP **grows with attack strength** (Spearman ρ=+0.668,
p=0.0065). His hypothesis was right.

Notably, **11 of 15 cells prefer radius scale 2.0 over the canonical 1.0** — the
closed-form radius is systematically too *small*.

So two independent lines of evidence say the canonical configuration
**under-enforces its own objective**: once through a starved dual, once through an
undersized uncertainty set. That is a considerably stronger argument than either
finding alone, and a better answer to "why is DRO only slightly better" than any
single hyperparameter story.

---

## 6. Two things needing your decision

**(a) Should the augmented Lagrangian go into the submission, and under which
framing?** The generalisation test is now done (§5a), so this is a real choice
rather than a pending question. Our recommendation: **include it**, framed as a
*fairness-optimisation* fix rather than a robustness one, scoped explicitly to
α ≤ 0.2, and accompanied by the α=0.4 instability as a stated limitation. It is
a principled, one-line change to your Lagrangian with a measured 42–53% DP
reduction behind it and accuracy that holds or improves. If you would rather
keep the submission's scope unchanged, the alternative is to report it as a
Future Work item with the evidence attached — but we think it is strong enough
to stand in the paper.

**(b) A reproducibility gap we found and want to close.**
`results/canonical_tau1.json` was generated before a k-NN metric fix landed, so
current code no longer reproduces it exactly. Scope, verified by re-running:
accuracy reproduces **exactly**, DP shifts by ~1e-7 (no conclusion, p-value, or
win count moves), but the IF column of the DP/COMBINED rows is floating-point
noise (~1e-11) rather than a measurement.

That noise was being fed into a Wilcoxon test, which printed two cells as
significant (p=0.016\*\*\*) beside a 0.0% effect size. Those are now suppressed and
disclosed in the Limitations section.

Re-running the grid under the corrected metric costs roughly one overnight run
(~6 hours) and would close the gap entirely. **We recommend doing it** — the DP
results provably do not move, so there is no risk to the headline claims, and it
turns a disclosed limitation into a non-issue.

---

*All numbers above are reproducible from the committed repository. Nothing in
this memo was tuned after seeing results — the experiment design, μ values, and
success criterion were written down and committed before the data existed
(`docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md`).*
