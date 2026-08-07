# TASK E — Independent adversarial review of the DRO-FAIR-AL claim

You are a fresh reviewer with no history on this project. That is
deliberate: you are being asked to verify, from raw data and code only, a
set of claims that other work (which you did not do and should not read as
authoritative) made about an improvement called "DRO-FAIR-AL". Your job is
to try to BREAK the claim, not confirm it. Finding a real defect is a
successful outcome for this task, not a failure.

Do not treat any existing `results/*_summary.md` file as ground truth --
those are the claims under review. Recompute from the raw JSON and code.

## Context (stated neutrally, so you know what's being claimed)

The project's canonical trainer (`src/training/dro_fair.py`,
`DroFairTrainer`) enforces DP/IF fairness constraints via a dual variable
lambda, updated with geometric decay. The claim is that this decay starves
lambda so badly (never exceeding ~0.012 against a ceiling of 1.5) that the
fairness penalty is negligible (~0.5% of the training loss). An
"augmented-Lagrangian" fix was added: a new constructor parameter
`aug_lagrangian_mu` (default 0.0) on `DroFairTrainer`, which when > 0 adds a
quadratic penalty `(mu/2)*g^2` per constraint to the loss, threaded through
`experiments/run_fairness_pgd.py::run_single_experiment`.

The claim, as it currently stands in `paper/sections/results.tex` (search
for "Augmented-Lagrangian") and `report/report.tex` (same section title):
at `mu=20`, on the Adult dataset, alpha <= 0.2, DP-targeted attack, this
reduces the DP fairness violation by 70.8-81.7% versus the canonical trainer
while holding or improving accuracy (6/6 seeds, Wilcoxon p=0.0156), with
these stated boundaries: (1) it is NOT corruption-specific -- it helps more
with zero corruption than under attack; (2) at alpha=0.4 no mu tested is
safe -- accuracy collapses below chance; (3) Credit is not rescued by any mu
tested; (4) combining mu=20 with a larger uncertainty-set radius
(`radii_scale=2.0`) CONFLICTS -- it collapses accuracy rather than helping.

## What to actually do

1. **Re-derive the gradient by hand.** Open `src/training/dro_fair.py`,
   find where `aug_lagrangian_mu` is used (search for it), and verify by
   hand that the code correctly implements `d/dtheta [(mu/2)*g^2] =
   mu*g*(dg/dtheta)` via autograd (i.e. that the penalty term is added to
   the scalar loss correctly, not e.g. added to the wrong tensor, or with a
   sign error, or double-counted). Confirm `g_dp`/`g_if` (the constraint
   violation terms) are genuinely non-negative quantities, so no `max(g,0)`
   clamp is silently needed but missing.

2. **Verify mu=0 is a true no-op, independently.** Check out the git commit
   that exists immediately before augmented-Lagrangian was introduced (`git
   log --oneline -- src/training/dro_fair.py` to find it -- look for the
   commit message mentioning "augmented-Lagrangian" or "DRO-FAIR-AL" and use
   its parent). In a scratch directory, run the SAME single experiment
   config (Adult, alpha=0.2, seed=0, dp attack, dro method) at that parent
   commit and at current HEAD with `aug_lagrangian_mu=0.0`. Diff `acc_clean`,
   `dp_clean`, `if_clean` -- they must match to full float precision. Do not
   trust the existing unit test (`tests/test_aug_lagrangian.py`) as
   sufficient; run this yourself end-to-end.

3. **Recompute the headline Wilcoxon test by hand.** Pick the Adult,
   alpha=0.2, DP-attack cell. Pull the 6 canonical-DRO `dp_clean` values
   (seeds 0-5) from `results/canonical_tau1.json` (filter for
   `tau==1.0, method=='dro', attack=='dp', alpha==0.2, seed<6,
   corruptor_type=='adversarial'`) and the 6 AL `dp_clean` values at mu=20
   from `results/mu_sensitivity.json` (filter `dataset=='adult',
   alpha==0.2, aug_lagrangian_mu==20.0`). Confirm the pairing is by SEED,
   not by list position (a documented historical bug class in this project
   -- see `tests/test_wilcoxon_seed_pairing.py` for why this matters).
   Recompute the one-sided Wilcoxon signed-rank p-value
   (`scipy.stats.wilcoxon(dro_vals, al_vals, alternative='greater')`)
   yourself and confirm it matches the claimed p=0.0156.

4. **Recompute the constant-predictor floors from scratch.** The claim uses
   Adult=0.7521, Credit=0.7788, LSAC=0.9016 as accuracy floors below which a
   DP "win" is model collapse rather than fairness. Load each dataset via
   `src/data/datasets.py::get_dataset` and compute the actual majority-class
   baseline accuracy on the training labels yourself. Do the numbers match?

5. **Stress-test the razor-thin degeneracy call in TASK C2.** In
   `results/al_radius_compound.json`, filter for `dataset=='adult',
   alpha==0.2, radii_scale==2.0, aug_lagrangian_mu==20.0` (6 seeds). Compute
   the mean accuracy yourself. The claim is this mean (0.756127) sits just
   below a degeneracy threshold of 0.7571 (floor 0.7521 + 0.005 margin), and
   that 3 of the 6 individual seeds are at or below the floor itself.
   Verify both numbers independently. Then ask: is a 0.005 margin the right
   threshold at all, or arbitrary? Does the qualitative conclusion
   (collapse, not a fairness win) survive if you use a stricter or looser
   margin (try 0.01 and 0.0)?

6. **Check the mu=20 mechanism caveat for seed-dependence.** TASK B found
   that at mu=20, Adult, alpha=0.2, SEED 0 specifically, the predicted-
   positive rate collapses to near-zero for both protected groups (0.0%,
   4.4%), close to a constant-negative predictor. This was reported as a
   caveat, not verified across other seeds. Pick 2-3 other seeds (e.g. 1,
   2, 3), train canonical DRO and AL mu=20 yourself following the pattern in
   `experiments/run_al_mechanism.py`, and check whether this near-collapse
   predicted-positive-rate pattern is seed-0-specific or general. This
   directly affects whether "AL denoises the attack" is the right mechanism
   description or whether "AL pushes toward the majority class" is more
   accurate.

7. **Look for leakage.** Does the AL training path see anything the
   canonical DRO path does not (e.g. different data splits, different
   random seeding order, an extra pass over validation data)? Read
   `experiments/run_fairness_pgd.py::run_single_experiment` end to end and
   confirm the only code-path difference between mu=0 and mu>0 is the
   penalty term itself.

8. **Sanity-check one specific seed row.** Confirm whether any Adult
   alpha=0.2 canonical-DRO seed has accuracy suspiciously close to
   0.7521 (the floor) -- if so, is that seed itself borderline-degenerate,
   and does excluding it change any of the paper's stated conclusions?

## Deliverable

Write `docs/AL_REVIEW.md` in this worktree (repo root `docs/`, not inside
`work/`): a list of items you CONFIRMED as correct, and a list of any
DEFECTS found, each with a severity (critical / moderate / minor) and the
exact evidence (file, line, computed numbers) that shows the defect. Be
blunt -- a clean bill of health is only credible if you can show you
actually tried to break it.

Before finishing: run `pytest tests/ -q` from the repo root and report
whether it's green. Commit your changes (including `docs/AL_REVIEW.md` and
any scratch scripts you wrote, under `experiments/` if reusable) with a
clear message. Do NOT merge into main yourself -- leave the branch for
review. Write a short final report to `work/wave-al-e/REPORT.md` (under
250 words) summarizing your verdict and pointing to `docs/AL_REVIEW.md` for
detail.
