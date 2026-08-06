# Ready-to-paste agent briefs

Three agents, three lanes, no overlap. Paste the block verbatim into each agent.
Full detail lives in `docs/TASKS_AL_VALIDATION.md`; these are the dispatch messages.

**Assign in this order.** Agent 1 first (it gates the paper), Agent 3 can start
immediately in parallel, Agent 2 any time.

---

## AGENT 1 — "Does the improvement hold up?" (highest priority)

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Read `docs/TASKS_AL_VALIDATION.md` first — the top section explains the
> augmented-Lagrangian (AL) finding you are validating. Then do **TASK A**.
>
> Context: we found DRO-FAIR's fairness constraint was contributing ~0.5% of the
> training loss because the dual variable λ never rises above ~0.012 (ceiling
> 1.5). Adding a quadratic penalty `(μ/2)·g²` fixed it — on Adult α=0.2 the
> margin over Naive grew 9.2× (0.0119 → 0.1094, p=0.0156, 6/6 seeds) with
> accuracy *rising* 0.7586 → 0.7944. But that is ONE cell. Your job is to find
> out whether it generalises or whether we got lucky.
>
> **Run the α=0 falsification control FIRST.** If AL helps just as much with
> zero corruption, then it is a generic fairness regulariser and our
> corruption-robustness framing is wrong. We want to discover that ourselves,
> not in a viva. Write down what "materially smaller effect" means numerically
> BEFORE you look at the output.
>
> Then extend: Adult α ∈ {0.3, 0.4}; LSAC α ∈ {0.1, 0.2} (expect degeneracy —
> confirm it); attack ∈ {if, combined} on Adult α=0.2. μ=5 only. 6 seeds.
> Copy `experiments/run_aug_lagrangian.py` to a new script writing to a NEW
> results file. Do not append to `results/aug_lagrangian.json`.
>
> **Mandatory checks:** every DP claim must also report accuracy against the
> constant-predictor floor (Adult 0.7521, Credit 0.7788, LSAC 0.9016). A DP
> improvement at or below the floor is model collapse, not fairness — label it
> DEGENERATE. This is exactly how Credit's apparent AL "win" turned out fake.
>
> Rules: pre-register grid + success criterion before running; report negatives
> honestly (a clean negative is a completed task, tuning until it wins is not);
> never write `results/canonical_tau1.json` or `results/utkface_*.json`; use the
> shared `_AblationLock`; `pytest tests/ -q` green before any commit.
>
> Deliver: results JSON + summary .md + a one-paragraph plain answer to "does
> the AL improvement generalise beyond Adult α=0.2?"

---

## AGENT 2 — "Why does accuracy go UP?" + paper integration

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Read `docs/TASKS_AL_VALIDATION.md` first. Do **TASK B**, then **TASK D**
> (D only after Agent 1's TASK A finishes — check with the user before starting D).
>
> TASK B: a fairness penalty that *improves* accuracy is counter-intuitive and
> an examiner will ask why. Our hypothesis: under a DP-targeted attack the
> corrupted points are exactly the ones driving the group-rate gap, so
> penalising that gap hard suppresses the attack's influence — i.e. AL acts as
> attack-specific denoising. That is a HYPOTHESIS. Test it:
> 1. `dump_history=True` already exists on `run_single_experiment`. Dump Adult
>    α=0.2 seed 0 for canonical DRO vs AL μ=5; compare `g_dp`, `lambda_dp`,
>    `train_loss`, `val_acc`, `val_dp` trajectories.
> 2. Measure directly: accuracy on the CORRUPTED training subset vs the CLEAN
>    subset, separately, for both models. If the hypothesis holds, AL fits the
>    corrupted points *less*.
> 3. Check the predicted-positive rate per group, to rule out a trivial
>    threshold-shift explanation.
>
> "Mechanism unclear" is an acceptable, honest outcome. A hand-wave is not.
>
> TASK D (after A): write the AL section into `paper/sections/results.tex` and
> the report — the λ-starvation diagnosis with the measured numbers (max λ
> 0.0119 vs ceiling 1.5; penalty 0.0029 vs loss 0.538), the formula, the Adult
> table, and the Credit degeneracy caveat given EQUAL prominence, not a
> footnote. Update Future Work in BOTH paper and report (AL is no longer future
> work). State that μ=0 recovers the canonical objective exactly so all prior
> results stand. `make paper && make report`, both PDFs must build.
>
> Framing requirement: "a proposed improvement with evidence on Adult and an
> honest negative on Credit/LSAC" — never a universal win.

---

## AGENT 3 — Independent adversarial review (must NOT do Agent 1 or 2's work)

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Do **TASK E** in `docs/TASKS_AL_VALIDATION.md`. You are the independent
> reviewer — do not run Agent 1's or Agent 2's tasks, or the review stops being
> independent.
>
> Your job is to BREAK the augmented-Lagrangian claim, not to confirm it.
> Finding a real defect is a successful outcome. Recent history justifies the
> paranoia: a verification pass yesterday found the built PDF was printing
> "0.0% effect, p=0.016***" — a significance star on floating-point noise.
> Assume more of that exists.
>
> Check, from first principles:
> - Re-derive the gradient of `(μ/2)g²` by hand; confirm the code implements it
>   and that `g_dp`/`g_if` are genuinely non-negative so no `max(g,0)` is needed.
> - Verify `μ=0` is byte-identical to pre-change canonical behaviour by checking
>   out the parent commit and diffing a full run — do not trust the unit test.
> - Recompute the Wilcoxon by hand from raw JSON; confirm seed pairing and that
>   the one-sided direction is correct.
> - **Recompute the constant-predictor floors** (0.7521 / 0.7788 / 0.9016) for
>   these exact splits. Do not trust the constants.
> - Adult α=0.2 seed 3 has accuracy 0.7521 — exactly the floor. Is that one seed
>   degenerate while the other five are not? Does excluding it change the verdict?
> - Look for leakage: does AL see anything canonical DRO does not?
>
> Deliver `docs/AL_REVIEW.md`: confirmed-correct items, plus any defects with
> severity. Be blunt.

---

## The one decision that is NOT an agent's — needs Prof. Manisha

**TASK F, the canonical re-run.** `results/canonical_tau1.json` can no longer be
reproduced by current code (it predates a k-NN metric fix). Verified scope:
accuracy reproduces *exactly*, DP shifts ~1e-7 (nothing moves), but the IF column
of DP/COMBINED rows is noise (~1e-11) rather than a measurement. Already
disclosed in Limitations and the bogus significance stars removed.

**Useful nuance:** re-running into a NEW file is completely non-destructive — it
touches nothing locked. So the compute can start tonight regardless; her decision
is only whether the paper *switches* to the new numbers. If you want to save a
day, start the run and decide after.

Cost ~6 hours (540 rows, one overnight). Recommendation: do it. The DP story
provably does not move, and it converts a disclosed limitation into a non-issue.
