# DRO-FAIR-AL — status record (no dispatch remaining)

**All six tasks (A, B, C, C2, D, E) are DONE.** Nothing left to assign. This
file is now a record of what each task found, kept for anyone who needs the
history without re-reading every commit. Full detail in
`docs/TASKS_AL_VALIDATION.md` and `docs/AL_REVIEW.md`. The one remaining
decision (whether AL ships in the submission at this scope, and whether to
re-run the canonical grid) is Prof. Manisha's, in
`docs/MEMO_FOR_ADVISOR.md`.

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
accuracy to 0.7561 — below the degeneracy threshold (0.7571); 3/6 seeds sit
at or below that threshold, one (seed 3) at the exact floor. Its apparent
−94.0% DP number is constant-predictor collapse, not fairness. μ=20 at the
**canonical radius** is the recommendation; do not combine with the larger
radius. `results/al_radius_compound_summary.md`.

**TASK E** (independent adversarial review, `docs/AL_REVIEW.md`): the fresh
reviewer confirmed the core claim — DP reduction, Wilcoxon p=0.0156, μ=0
no-op, λ-starvation diagnosis, no leakage, all α=0.4/Credit/radius-compound
boundaries — and found real framing defects in how it was written up:
- **"Accuracy held or improved / Pareto"** is false at α=0.0: all 6 seeds
  *lose* accuracy (0.8147→0.7966). True only at α=0.2 (aggregate improves,
  but see next point).
- **Seed 3 at μ=20, α=0.2 is a full constant-negative predictor** — its
  accuracy (0.752128) is the test majority rate exactly, hidden by the
  6-seed mean (0.7783). Direct per-seed reruns (seeds 0–3) confirm this
  positive-rate collapse is **general, not seed-0-specific**: "AL pushes
  toward the majority class" is the accurate mechanism description, not
  "AL denoises the attack" (the TASK B framing, now superseded).
- The `+0.005` degeneracy margin is pre-registered but arbitrary; several
  verdicts (Credit's non-rescue, the C2 CONFLICT call) sit close to it.
All corrected in the paper, report, and this file. Full evidence and two
more minor findings in `docs/AL_REVIEW.md`.

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

**TASK E** (independent adversarial review — see the record above): ran to
completion in a fresh, independent `opencode` process with no memory of
A/B/C/C2/D, exactly as the brief below required. Confirmed the core claim,
found the framing defects now folded into this file, the paper, and the
report. Nothing further to dispatch here.
