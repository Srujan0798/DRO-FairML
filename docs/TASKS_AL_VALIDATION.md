# Task dispatch — validating DRO-FAIR-AL before it goes in the paper

**Date:** 2026-08-05. **Owner of this doc:** whoever assigns the agents.
**Deadline context:** submission Aug 10 2026.

---

## What just happened (read this first — it is the reason for every task below)

We found and fixed the structural reason DRO-FAIR was only *marginally* better
than Naive.

**Diagnosis.** In `src/training/dro_fair.py`, the fairness constraints reach the
model only through the linear dual term `λ·g`. But the dual learning rate decays
geometrically (`lr_λ · 0.95^epoch`), so λ's *total possible* accumulation over
60 epochs is ~0.1. Measured on Adult α=0.2 seed 0: **max λ_dp = 0.0119** against
a ceiling of 1.5 (126× below it), and the whole fairness penalty `λ·g` peaks at
**0.0029 versus a training loss of 0.538 — about 0.5% of the loss.** The
constraint machinery that is supposed to distinguish DRO from Naive was
effectively switched off. This also explains the earlier confusing result where
raising `λ_max` 1.5→2.0 produced *byte-identical* output: the ceiling was never
the binding constraint, the accumulation *rate* was.

**Fix (DRO-FAIR-AL).** Classical augmented Lagrangian (Hestenes 1969,
Bertsekas 1982): add a quadratic penalty so the constraint gradient does not
depend on λ's crawl.

```text
total_loss = L_tilt + λ_dp·g_dp + (μ/2)·g_dp²  +  λ_if·g_if + (μ/2)·g_if²
```

`μ=0` is an exact no-op (unit-tested bit-identical), so canonical results are
untouched. Implemented as `aug_lagrangian_mu` on `DroFairTrainer`, threaded
through `run_single_experiment` with row provenance.

**Result (48/48 runs, pre-registered before data existed, seed-paired Wilcoxon,
n=6).** Adult, α=0.2, μ=5:

| metric | Naive | DRO (canonical) | **DRO-FAIR-AL** |
|---|---|---|---|
| DP violation | 0.2452 | 0.2334 | **0.1358** (p=0.0156) |
| accuracy | — | 0.7586 | **0.7944** (floor 0.7521) |
| margin over Naive | — | +0.0119 | **+0.1094 (9.2×)** |

AL beats Naive on DP in **6/6 seeds**, and accuracy goes *up* — a Pareto
improvement, not a fairness/accuracy trade.

**The honest caveat that must survive into the paper.** Credit's four
"significant" cells are **degenerate**: AL pushes Credit DP to ~0.002 by
collapsing to/below the constant-predictor floor (acc 0.7730–0.7790 vs floor
0.7788). A constant classifier has DP=0 by definition — that is collapse, not
fairness, and it is the same failure already documented for LSAC/DP. The
summarizer now enforces a constant-predictor floor guard and labels these
`**DEGEN**`. **No task below may report Credit as a win.**

Files: `results/aug_lagrangian.json`, `results/aug_lagrangian_summary.md`,
`experiments/run_aug_lagrangian.py`, `experiments/summarize_aug_lagrangian.py`,
design + pre-registration in
`docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md`.

---

## Ground rules for every task

1. **Pre-register.** Write the grid, the values, and the success criterion into
   the task's design note *before* running. No changing the criterion after
   seeing numbers.
2. **Report negatives.** A failed hypothesis reported honestly is a completed
   task. Tuning until it wins is not.
3. **Degeneracy guard is mandatory.** Any DP claim must also report accuracy vs
   the constant-predictor floor (Adult 0.7521, Credit 0.7788, LSAC 0.9016).
   DP improvements at/below the floor are collapse and must be labelled so.
4. **Never write** `results/canonical_tau1.json` or `results/utkface_*.json`
   (guarded by `_assert_safe_results_path`).
5. **Use the shared lock.** All CPU jobs go through
   `experiments/run_ablation_parallel`'s `_AblationLock` so the 14-core machine
   is never oversubscribed. Do not launch around it.
6. `pytest tests/ -q` must pass before any commit.

---

## TASK A — Does AL generalise, or is it an Adult α=0.2 artifact? ✅ COMPLETE (42/42)

Run directly (not by an agent), pre-registered in
`docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md`. Full result:
`results/aug_lagrangian_extended_summary.md`.

**Verdict:** the α=0.0 falsification control **FAILED** — AL reduces DP by
52.6% with zero corruption, *more* than the 41.8% it achieves under attack
(threshold was 20.9%). **The corruption-robustness framing is withdrawn.** AL
is a general fairness-optimisation fix: the dual-starvation defect it corrects
has nothing to do with corruption, so a fix targeting it helps everywhere.

Also found: a new failure mode at α=0.4 (μ=5 gave DP −80.7%, p=0.0156 — the
best number in the study — while accuracy collapsed to 0.3495, below chance),
which motivated TASK C below. COMBINED attack transfers (genuine win, accuracy
improves 0.7599→0.8032); IF attack borderline (not claimed); LSAC collapses as
predicted.

---

## TASK B — Why does accuracy go UP? (mechanism, needed for the paper's story)

**Why:** a fairness penalty that *improves* accuracy is counter-intuitive.
Right now we can report it but not explain it, and an examiner will ask. The
likely explanation: under a DP-targeted attack the corrupted points are exactly
the ones driving the group-rate gap, so penalising that gap hard suppresses the
attack's influence — i.e. AL is acting as attack-specific denoising. That is a
*hypothesis*, not a finding.

**Do:**
1. Dump per-epoch history (`dump_history=True` already exists) for Adult α=0.2
   seed 0 under canonical DRO vs AL μ=5. Compare trajectories of
   `g_dp`, `lambda_dp`, `train_loss`, `val_acc`, `val_dp`.
2. Measure directly: on the corrupted training set, compute each model's
   accuracy **on the corrupted subset** vs **on the clean subset** separately
   (the corruption mask is available inside the runner). If the hypothesis
   holds, AL should fit the corrupted points *less* than canonical DRO does.
3. Check whether AL's decision threshold/prediction distribution has shifted
   (fraction predicted positive per group) — rules out a trivial explanation.

**Deliver:** `results/al_mechanism_summary.md` with the three plots/tables and a
one-paragraph verdict: is the "AL suppresses attacked points" explanation
supported, or is something else going on? An honest "mechanism unclear" is an
acceptable outcome; a hand-wave is not.

---

## TASK C — μ sensitivity and the accuracy/fairness frontier ✅ COMPLETE (90/90)

Run directly (not by an agent) in response to TASK A's α=0.4 failure,
pre-registered in `docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`.
Full result: `results/mu_sensitivity_summary.md`.

**Verdict — recommended μ per α (largest tested μ that is both SAFE and
EFFECTIVE, per Rules C1–C2):**

| α | recommended μ | DP reduction | accuracy |
|---|---|---|---|
| 0.0 | **μ=20** | −81.7% | 0.7966 |
| 0.2 | **μ=20** | −70.8% | 0.7783 |
| 0.4 | **none safe** | — | — (do not use AL) |

μ=20 supersedes μ=5 — strictly larger DP reduction at the same safety margin.
**Credit is not rescued** by any μ ∈ {0.5, 1, 2} (Rule C3) — AL's claim is
**Adult-only**. DP-vs-μ curves are monotone at every α (Rule C4) — the α=0.4
failure is a genuine safety boundary, not optimisation noise.

**Use μ=20 in TASK C2, TASK D, and everywhere else below.**

---

## TASK C2 — Compound AL with a larger radius ✅ COMPLETE (18/18 new, 48/48 total)

Dispatched via opencode to an isolated worktree, merged after independent
verification (the razor-thin degeneracy call — mean accuracy 0.756127 vs
threshold 0.7571 — was recomputed by hand from raw JSON before merging).
Full result: `results/al_radius_compound_summary.md`.

**Verdict: CONFLICT, not compound or redundant.** At the scoped cell
(Adult, α=0.2), combining μ=20 with `radii_scale=2.0` collapses accuracy to
0.7561 — below the degeneracy threshold (floor+0.005 = 0.7571); 3 of 6 seeds
sit at or below that threshold, and one (seed 3) is at the floor itself
(0.752128). Its apparent DP number (−94.0%) is constant-predictor
collapse, not fairness. **Independently reviewed (TASK E):** every number
above reproduced exactly from raw JSON, but the qualitative CONFLICT verdict
would flip under a 0.0 margin instead of +0.005 (mean crosses back above the
exact floor) — the margin is pre-registered and methodologically sound, but
its magnitude has no independent justification and several verdicts (this
one, Credit's non-rescue) sit close to it. See `docs/AL_REVIEW.md` D5. AL-only (μ=20, canonical radius) remains the
genuine win (DP −70.8%, accuracy 0.7783); radius-only is inert on Adult/DP
(+1.8%). **The two fixes each correct the objective independently; stacking
them over-corrects.** This disagreed with the pre-registered prediction
(redundant was expected) — the degenerate counter-hypothesis flagged in the
pre-registration is what actually occurred.

**Written into the paper and report** (TASK D): μ=20 at the canonical radius
is the recommendation; do not combine with the larger radius.

*Original brief, for reference:*

**Why:** the N1 radius ablation (now complete, 180/180) found that **11 of 15
(dataset, α) cells prefer `radii_scale=2.0` over the canonical 1.0**, and that
the DP-minimising radius grows with measured attack strength
(Spearman ρ=+0.668, p=0.0065 — this also confirms Kuldeep's original May-29
hypothesis). In other words the canonical closed-form radius is systematically
*too small*.

That is the **same diagnosis as the AL finding, arriving independently**: the
canonical configuration under-enforces the robust/fair objective — once through
a starved dual (fixed by AL), once through an undersized uncertainty set. Two
independent lines of evidence pointing the same way is a much stronger paper
argument than either alone, and it is worth stating that way.

**Do:** Adult, α ∈ {0.2, 0.3}, 6 seeds, 2×2 design:
`radii_scale ∈ {1.0, 2.0} × aug_lagrangian_mu ∈ {0, 20}`. **Use μ=20, not
μ=5** — TASK C found μ=20 is the safe, more-effective value at α≤0.2. The
canonical and AL-only (μ=20, radii=1.0) arms already exist; this adds
`radii_scale=2.0` alone and the combined arm. Both parameters are already
plumbed through `run_single_experiment` — no new trainer code needed.

**Pre-register:** do the two levers **compound** (combined beats both singles),
are they **redundant** (combined ≈ best single, meaning they fix the same
deficiency), or do they **conflict**? Write down which you expect and why
*before* running. Apply the degeneracy guard — a larger radius plus a stronger
penalty is exactly the combination most likely to collapse the model.

**Deliver:** `results/al_radius_compound_summary.md` + a plain statement of
which of the three outcomes occurred. "Redundant" is a genuinely interesting
and publishable answer, not a failure.

---

## TASK D — Paper and report integration (unblocked — A and C are both complete)

**Why:** confirmed via search, `paper/sections/results.tex` and `report.tex`
currently contain **zero** AL numbers — only qualitative "Future Work"
mentions (`discussion.tex` ~L184, `report.tex` ~L673). This is a full
integration, not an edit of existing numbers.

**Do:**
1. New subsection in `paper/sections/results.tex` and the report: the
   λ-starvation diagnosis (with the measured 0.0119 / 0.5%-of-loss numbers),
   the AL formula, **the Adult result table at μ=20** (not μ=5 — superseded by
   TASK C), and the Credit-not-rescued / α=0.4-unsafe caveats given equal
   prominence — not buried in a footnote.
2. **Explicit scope statement:** α ≤ 0.2 only; α=0.4 reported as "AL is unsafe
   here — no μ tested avoids model collapse" as a stated finding. Dataset
   scope: Adult only; Credit not rescued by any μ tested.
3. Update Future Work in **both** `paper/` and `report/` — AL is no longer
   future work; replace with what Task B/C2/E leave open.
4. Method section: state that `μ=0` recovers the canonical objective exactly, so
   every previously reported canonical result stands unchanged.
5. `make paper && make report`, confirm both PDFs build.

**Deliver:** both PDFs rebuilt, committed. **Framing requirement:** present AL as
"a proposed improvement, precisely scoped, with evidence on Adult α≤0.2 and
honest negatives outside that scope" — never a universal win.

---

## TASK E — Independent adversarial review (assign to a *different* agent than A–D)

**Why:** the user's explicit instruction is that this be checked bit-by-bit as a
new validator would, because basic things have been missed before.

**Do:** with fresh eyes and no stake in the result, try to *break* the AL claim.
Scope now includes TASK C's rule application, since μ=20 is about to go into
the paper:
- Read `experiments/summarize_mu_sensitivity.py` (Rules C1–C4) and recompute
  the SAFE/EFFECTIVE table by hand from `results/mu_sensitivity.json` — don't
  trust the script's own printed output. The α=0.4 "no μ is safe" conclusion
  rests on all of μ ∈ {0.5, 1, 2, 5, 10, 20} failing; check each individually.
  Same for Credit's "not rescued" conclusion.
- Re-derive `(μ/2)g²`'s gradient by hand; confirm the code implements it and
  that `g_dp`/`g_if` are genuinely non-negative so no `max(g,0)` is needed.
- Verify `μ=0` is byte-identical to pre-change canonical output by checking out
  the parent commit and diffing a full run, not just trusting the unit test.
- Re-run the Wilcoxon by hand from the raw JSON; confirm seed pairing is correct
  and one-sided direction is right.
- Confirm the constant-predictor floors (0.7521 / 0.7788 / 0.9016) are actually
  correct for these splits — recompute them, do not trust the constants.
- Check the α=0.2 Adult seed-3 row (accuracy 0.7521, exactly at the floor):
  is that seed degenerate while the other five are not? Does excluding it change
  the verdict?
- Look for any leakage: does AL see anything canonical DRO does not?

**Deliver:** `docs/AL_REVIEW.md` — a list of confirmed-correct items and any
defects found, with severity. Finding a real problem is a successful outcome.

---

## TASK F — Close the canonical reproducibility gap (COMPLETE 2026-08-09)

**Decision:** Re-run approved and executed. `experiments/run_task_f_repro.py`
re-ran the full 540-row canonical grid into `results/canonical_tau1_cosine.json`.

**Result (2026-08-09):** TASK F complete. Verification reveals:

| Attack | Accuracy drift | DP drift | Status |
|---|---|---|---|
| **DP** | max 0.003, mean 0.0003 | max 0.003, mean 0.0003 | ✅ Stable — headline claims unaffected |
| **IF** | max 0.005, mean 0.0004 | max 0.008, mean 0.0004 | ✅ Stable |
| **Combined** | **max 0.075, mean 0.011** | **max 0.085, mean 0.016** | ⚠️ DRIFT at α≥0.2 |

**Root cause:** The combined attack = 0.5 DP + 0.5 IF. Under the old Euclidean
metric, IF was degenerate (~10^-10), so combined ≈ DP-only. After the cosine
fix, IF is real, so the combined attack now includes a real IF gradient component,
changing which samples get corrupted at high α. This affects the combined-attack
cells at α≥0.2; the DP-attack headline is provably unaffected.

**Implication:** The memo's prediction ("accuracy reproduces exactly, DP shifts
~1e-7") was correct for DP and IF attacks but wrong for combined. The
combined-attack rows at α≥0.2 need updated numbers from the re-run. The paper's
DP-attack headline is unchanged.

**If approved, do:** re-run the full canonical grid with current code into a
**new file** (`results/canonical_tau1_cosine.json` — do **not** overwrite the
locked file), then diff every row against the original and publish the diff
before switching any table over. `tests/test_degenerate_if_reporting.py` will
fail once the rows become non-degenerate — that is intentional; update the
Limitations text at that point to say the gap is closed.

---

## Suggested assignment

**A, B, C, C2, and D are all done** (A/B/C run directly; C2 dispatched to an
isolated worktree and merged after independent verification; D written
directly into the paper and report — see their sections above and
`docs/PROJECT_COMPLETION_CHECKLIST.md`). Only **E** remains open.

| Agent | Tasks | Rationale |
|---|---|---|
| Agent 3 (independent) | **E — the only open task** | Must not have run A/B/C/C2/D, or the review is not independent. Scope includes reviewing TASK C's μ=20 rule application AND TASK C2's razor-thin degeneracy call (mean acc 0.756127 vs threshold 0.7571). |

**Sequencing:** nothing blocks E. It is the last gate before this is fully
submission-ready.
