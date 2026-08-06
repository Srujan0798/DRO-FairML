# Ready-to-paste agent brief

**Status: A, B, C, C2, and D are all DONE.** Only **TASK E** (independent
adversarial review) remains before this is submission-ready. Full detail in
`docs/TASKS_AL_VALIDATION.md`; this is the one dispatch message left.

---

## COMPLETED — read before starting E

**TASK A** (generalisation, 42/42 runs): the α=0 falsification control
**FAILED** — AL helps *more* with zero corruption (52.6%) than under attack
(41.8%). **Corruption-robustness framing withdrawn**; AL is a general
fairness-optimisation fix. Found a new failure mode at α=0.4: μ=5 gave the
best-looking DP number in the study (−80.7%) while accuracy collapsed to
0.3495, below chance. `results/aug_lagrangian_extended_summary.md`.

**TASK C** (μ-sensitivity, 90/90 runs): μ=20 is the safe, effective value at
α≤0.2 (DP −70.8% to −81.7%, accuracy held/improved). No μ is safe at α=0.4.
Credit not rescued by any μ. `results/mu_sensitivity_summary.md`.

**TASK B** (mechanism, Adult α=0.2 seed 0): AL actively resists the
corrupted training points (corrupted-subset accuracy: canonical 51.3% → AL
17.8–18.0%) while clean-subset accuracy holds or improves — supports
"attack-specific denoising." **Caveat found and written up honestly:** at
μ=20 on this seed, predicted-positive rate collapses to {0.0%, 4.4%} by
group, nearly identical to Adult's constant-negative-predictor accuracy
(0.7522 vs this seed's 0.7643) — the 6-seed mean is safely clear of the
floor, but the margin is thinner on individual seeds than the mean suggests.
`results/al_mechanism_summary.md`.

**TASK C2** (AL × radius compound, 18 new runs, dispatched to an isolated
worktree, merged after independent verification): **CONFLICT, not compound
or redundant.** Combining μ=20 with `radii_scale=2.0` at α=0.2 collapses
accuracy to 0.7561 — right at the degeneracy threshold, 3/6 seeds at or
below the floor itself. Its apparent −94.0% DP number is constant-predictor
collapse, not fairness. μ=20 at the **canonical radius** is the
recommendation; do not combine with the larger radius.
`results/al_radius_compound_summary.md`.

**TASK D** (paper + report integration): both PDFs now contain the AL
section — diagnosis, formula, the μ=20 result table, and all four scope
boundaries (not corruption-specific, α=0.4 unsafe, Credit not rescued,
conflicts with the radius fix) stated as findings, not omissions. Verified
by `pdftotext` that the actual numbers render in both built PDFs, not just
compile. `paper/sections/results.tex` §Augmented-Lagrangian,
`paper/sections/discussion.tex` §Limitations item 5, `report/report.tex`
§Augmented-Lagrangian Constraint Enforcement.

**Use μ=20, Adult, α≤0.2, canonical radius (`radii_scale=1.0`) as the scope
in everything.** μ=5 and any radius/μ combination are superseded.

---

## TASK E — Independent adversarial review (the one open task)

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Read the COMPLETED section above, then `docs/TASKS_AL_VALIDATION.md`'s TASK E
> section. You did not run A/B/C/C2/D — that is what makes this review
> independent. Do not re-derive their conclusions by reading their summaries;
> verify them from raw data.
>
> Your job is to BREAK the claim, not confirm it. Finding a real defect is a
> successful outcome. Recent history justifies the paranoia: a verification
> pass earlier in this project found the built PDF printing a significance
> star on floating-point noise. Assume more exists.
>
> **Specific things to check, in addition to the general scope in
> TASKS_AL_VALIDATION.md:**
> - **TASK C's μ=20 rule application.** Read
>   `experiments/summarize_mu_sensitivity.py` (Rules C1–C4) and recompute the
>   SAFE/EFFECTIVE table by hand from `results/mu_sensitivity.json` — don't
>   trust the script's printed output. The "no μ safe at α=0.4" conclusion
>   rests on μ ∈ {0.5,1,2,5,10,20} all failing; check each individually.
> - **TASK C2's degeneracy call is razor-thin** — mean accuracy 0.756127
>   against threshold 0.7571, a gap of 0.001. Recompute this yourself from
>   `results/al_radius_compound.json` (already done once independently before
>   merging, but a second check by someone with no stake in the result is
>   exactly what this task is for). Check whether the per-seed accuracies
>   (3 of 6 at or below the floor itself) change the read.
> - **TASK B's mechanism caveat**: does the near-zero positive-prediction
>   rate at μ=20 (seed 0) generalize to other seeds, or is it seed-specific?
>   This bears directly on whether "denoising" is the right description of
>   what μ=20 actually does.
> - Re-derive `(μ/2)g²`'s gradient by hand; confirm `g_dp`/`g_if` are
>   genuinely non-negative so no `max(g,0)` is needed.
> - Verify `μ=0` is byte-identical to pre-change canonical behaviour by
>   checking out the parent commit and diffing a full run — don't trust the
>   unit test alone.
> - Recompute the Wilcoxon tests by hand from raw JSON for at least the
>   headline Adult α=0.2 cell; confirm seed pairing and one-sided direction.
> - Recompute the constant-predictor floors (0.7521 / 0.7788 / 0.9016) —
>   don't trust the constants.
> - Look for leakage: does AL see anything canonical DRO does not?
>
> Deliver `docs/AL_REVIEW.md`: confirmed-correct items, plus any defects
> found, with severity. Be blunt.
