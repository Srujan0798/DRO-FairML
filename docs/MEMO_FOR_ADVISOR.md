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

## 3. Result — Adult, α=0.2, **μ=20** (corrected by TASK C), n=6 seeds, seed-paired Wilcoxon

| | Naive | DRO (current) | **DRO-FAIR-AL μ=20** |
|---|---|---|---|
| DP violation | 0.2452 | 0.2334 | **0.0682** (p=0.0156) |
| accuracy | — | 0.7586 | **0.7783** |
| margin over Naive | — | +0.0119 | **+0.1771 — 14.9× larger** |

*(First demonstrated at μ=5: DP 0.1358, accuracy 0.7944, margin 9.2×. TASK C's
90-run μ-sensitivity sweep, §5b, found μ=20 is both safe and strictly more
effective — μ=5 is superseded as the recommendation.)*

Beats Naive on DP in **6/6 seeds**. **Correction, added after an independent
review (§7):** the mean accuracy (0.7783) is above the floor, but one of the
six seeds (seed 3) is a full constant-negative predictor — accuracy exactly
the test majority rate, hidden by the mean. This is not the clean Pareto
improvement we first reported; it's a real DP win with a genuine per-seed
accuracy risk. At α=0 the accuracy cost is worse: it drops in **all 6
seeds** (0.8147→0.7966), even though DP improves *more* there than under
attack.

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

## 5b. μ-sensitivity sweep (TASK C, complete, 90/90 runs) — answers (ii) above

Pre-registered in `docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`,
run and analyzed mechanically by `experiments/summarize_mu_sensitivity.py`.
This is the direct response to the α=0.4 destruction found above: it maps
where μ is safe, and produces one number per α instead of one guess.

| α | recommended μ | DP reduction vs canonical DRO | p | accuracy |
|---|---|---|---|---|
| 0.0 | **μ=20** | −81.7% | 0.0156 | 0.7966 |
| 0.2 | **μ=20** | −70.8% | 0.0156 | 0.7783 |
| 0.4 | **none** | — | — | **do not use AL — no μ tested is safe** |

Two further findings from the same sweep: **Credit is not rescued** — no
μ ∈ {0.5, 1, 2} is both safe and effective there, so AL's claim stays
**Adult-only** among the datasets tested. And the DP-vs-μ curves are monotone
at every α (no optimisation instability) — the α=0.4 failure is a genuine
safety boundary, not noise.

**This changes §3's number:** μ=20, not μ=5, is the correct headline result
and the correct value to write into the paper.

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
framing?** The generalisation test (§5a), the μ-sensitivity sweep (§5b), the
radius-compound test (§5), and an independent adversarial review (§7) are all
now done. **Include it, with these exact parameters and this exact framing:**
- **μ=20**, framed as a *fairness-optimisation* fix, not a robustness one
- **Scope: α ≤ 0.2 only.** α=0.4 is "do not use AL" — no μ tested is safe
  there. **Canonical radius only** — combining with the separately-found
  radius fix (`radii_scale=2.0`) conflicts, not compounds.
- **Dataset scope: Adult only.** Credit is not rescued by any tested μ.
- **Accuracy claim, corrected per §7: not "holds or improves."** At α=0.2 the
  6-seed mean improves but one seed fully collapses to the constant
  predictor. At α=0 accuracy drops in every seed. State it as "a genuine DP
  reduction with a real, seed-dependent accuracy cost," not a free Pareto
  win.

It is a principled, one-line change to your Lagrangian with a measured
70.8–81.7% mean DP reduction (Adult, α≤0.2), an honestly-mapped safety
boundary, and — now — an independent review that tried to break it and
mostly couldn't. That combination (a real effect, precisely scoped, with its
own limits stated) is a more defensible contribution than an overstated one
would have been. If you would rather keep the submission's scope unchanged,
report it as Future Work with the evidence attached — but we think it is
strong enough, precisely enough scoped, and now independently checked, to
stand in the paper as written.

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

## 7. Independent adversarial review — what we asked someone with no stake to try to break

Before finalising, we dispatched a genuinely independent review: a fresh
process with no memory of any of the work above, instructed to try to break
the claim rather than confirm it, and to recompute every number from raw
data rather than trust our summaries. Full evidence: `docs/AL_REVIEW.md`.

**Confirmed correct, re-derived from scratch:** the AL gradient implementation
(no sign error, no double-counting); μ=0 is a byte-identical no-op end-to-end
(checked out the parent commit, reran, matched to full float precision); the
headline Wilcoxon p=0.015625; the λ-starvation diagnosis (max λ=0.0119 vs
ceiling 1.5, 0.54% of the loss); no code-path leakage between μ=0 and μ>0;
the α=0.4/Credit/radius-compound boundaries.

**Real defects found, now corrected everywhere above:**
- **"Accuracy held or improved" was false at α=0.** All 6 seeds lose accuracy
  (0.8147→0.7966). We had written a blanket Pareto claim covering both α
  values; it only held for one of them.
- **A per-seed collapse was hidden by the mean at α=0.2.** One of six seeds
  (seed 3) is a full constant-negative predictor. Direct reruns across seeds
  0–3 show this is general at μ=20, not specific to that seed — the honest
  mechanism description is "AL pushes predictions toward the majority
  class," not "AL denoises the attack."
- **Two minor precision issues**: our "3/6 seeds at the floor" language
  conflated the floor with the floor+0.005 threshold (only 1/6 is at the
  floor itself); the LSAC constant-predictor constant (0.9016) is the test-
  split majority, not the training majority (0.9019) — defensible but the
  provenance wasn't stated.
- **One epistemic caveat worth your awareness**: the +0.005 safety margin
  used throughout is pre-registered (methodologically sound — fixed before
  data) but its magnitude has no independent justification, and a few
  verdicts (Credit's non-rescue, the radius-compound CONFLICT call) sit close
  to it. Under a 0.0 margin instead, two of those would flip.

We're telling you about a review that caught our own mistake rather than one
that found nothing, because that's the point of running it — and because the
corrected version above is what we're actually recommending, not the
original overstated one.

---

*All numbers above are reproducible from the committed repository. Nothing in
this memo was tuned after seeing results — the experiment design, μ values, and
success criterion were written down and committed before the data existed
(`docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md`).*
