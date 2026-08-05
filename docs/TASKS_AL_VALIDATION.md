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

## TASK A — Does AL generalise, or is it an Adult α=0.2 artifact? (highest priority)

**Why:** the whole claim currently rests on one cell. A reviewer will ask this
first, and right now we cannot answer it.

**Do:** extend the AL grid along the axes we have not touched.
- α ∈ {0.0, 0.3, 0.4} on Adult (we only have 0.1, 0.2). α=0.0 is the important
  control: with no corruption, AL should *not* help much — if it produces a big
  win at α=0.0 too, then AL is just a generic fairness regulariser and the
  "corruption-robustness" story is wrong. **This is a falsification test; run it
  first.**
- LSAC at α ∈ {0.1, 0.2} (expect degeneracy — confirm and document it).
- attack ∈ {if, combined} on Adult α=0.2 (currently dp only).

**Grid:** reuse `experiments/run_aug_lagrangian.py` structure, μ=5 only (μ=10
costs more accuracy for no extra genuine cell). 6 seeds. New output file —
do not append to `results/aug_lagrangian.json`.

**Success criterion (pre-register before running):** AL's DP improvement over
canonical DRO holds (p<0.05, non-degenerate) in ≥2 of the new Adult
corruption cells (α ∈ {0.3, 0.4}), AND the α=0.0 control shows a *materially
smaller* effect than the α=0.2 cell. State explicitly what "materially smaller"
means numerically before you look.

**Deliver:** `results/aug_lagrangian_extended{,_summary}.md/json` + one
paragraph stating plainly whether the improvement generalises.

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

## TASK C — μ sensitivity and the accuracy/fairness frontier

**Why:** we tested μ ∈ {5, 10} only. We do not know if μ=5 is near-optimal, nor
where AL starts collapsing models (Credit collapsed at μ=5 already). The paper
needs a defensible statement about how to choose μ.

**Do:** Adult + Credit, α=0.2, 6 seeds, μ ∈ {0.5, 1, 2, 20}. Combined with
existing μ ∈ {5, 10} this gives a 6-point curve. Plot DP and accuracy vs μ with
the constant-predictor floor drawn on the accuracy axis — the point where each
dataset's curve crosses the floor *is* the degeneracy threshold, and that is the
paper-ready figure.

**Deliver:** `results/mu_sensitivity_summary.md` + the figure + a stated
recommendation for choosing μ (e.g. "largest μ whose validation accuracy stays
≥ floor + 0.02"), justified by the curve rather than by the winning number.

---

## TASK D — Paper and report integration (do only after A completes)

**Why:** the paper currently has no AL section, and its Future Work still lists
augmented Lagrangian as an untried idea.

**Do:**
1. New subsection in `paper/sections/results.tex` and the report: the
   λ-starvation diagnosis (with the measured 0.0119 / 0.5%-of-loss numbers),
   the AL formula, the Adult result table, and the Credit degeneracy caveat
   given equal prominence — not buried in a footnote.
2. Update Future Work in **both** `paper/` and `report/` — AL is no longer
   future work; replace with what Task A/B/C left open.
3. Method section: state that `μ=0` recovers the canonical objective exactly, so
   every previously reported canonical result stands unchanged.
4. `make paper && make report`, confirm both PDFs build.

**Deliver:** both PDFs rebuilt, committed. **Framing requirement:** present AL as
"a proposed improvement with evidence on Adult and an honest negative on
Credit/LSAC", never as a universal win.

---

## TASK E — Independent adversarial review (assign to a *different* agent than A–D)

**Why:** the user's explicit instruction is that this be checked bit-by-bit as a
new validator would, because basic things have been missed before.

**Do:** with fresh eyes and no stake in the result, try to *break* the AL claim:
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

## Suggested assignment

| Agent | Tasks | Rationale |
|---|---|---|
| Agent 1 (CPU/experiments) | A, then C | Both are compute grids on the same machinery; A gates the paper claim. |
| Agent 2 (analysis) | B, then D | Mechanism work feeds directly into how D writes it up. |
| Agent 3 (independent) | E | Must not have run A–D, or the review is not independent. |

**Sequencing:** A is blocking for D. B and C can run in parallel with A but
must queue behind the ablation lock. E can start immediately — it reviews what
already exists.
