# Ready-to-paste agent briefs

Three agents, three lanes, no overlap. Paste the block verbatim into each agent.
Full detail lives in `docs/TASKS_AL_VALIDATION.md`; these are the dispatch messages.

**Status: TASK A and TASK C are DONE** (run directly, not by an agent — the
machine was idle). Agent 1's original assignment (TASK A) is complete, so
Agent 1 is reassigned to **TASK C2** below. Agent 2 and Agent 3 keep their
original assignments, updated with TASK C's corrected μ value.

**Assign in this order.** Agent 2 first (paper integration is now unblocked),
Agent 3 in parallel, Agent 1 (TASK C2) any time.

---

## COMPLETED — read before assigning anything

**TASK A** (generalisation, 42/42 runs) found the AL improvement is **real but
narrower than first thought**:
- The α=0 falsification control FAILED: AL helps *more* with zero corruption
  (52.6% DP reduction) than under attack (41.8%). **The corruption-robustness
  framing is withdrawn.** AL is a general fairness-optimisation fix — the
  dual-starvation defect it corrects has nothing to do with corruption.
- Found a new failure mode: at α=0.4, AL (μ=5) gave the best-looking DP number
  in the whole study (−80.7%, p=0.0156) while accuracy collapsed to 0.3495 —
  below chance. The model was destroyed while its fairness metric looked superb.
- COMBINED attack transfers (genuine win, accuracy improves); IF attack is
  borderline (not claimed); LSAC collapses as predicted.
- Full detail: `results/aug_lagrangian_extended_summary.md`.

**TASK C** (μ-sensitivity, 90/90 runs), run in direct response to the α=0.4
failure, found the safe/effective operating range:

| α | recommended μ | DP reduction | accuracy |
|---|---|---|---|
| 0.0 | **μ=20** | −81.7% | 0.7966 |
| 0.2 | **μ=20** | −70.8% | 0.7783 |
| 0.4 | **none safe** | — | — |

**μ=20 supersedes μ=5 as the recommendation** — larger DP reduction, same
safety margin. Credit is **not rescued** by any tested μ (0.5, 1, 2) — AL's
claim is **Adult-only**. DP-vs-μ curves are monotone at every α (clean
trade-off, not instability). Full detail: `results/mu_sensitivity_summary.md`.

**Use μ=20 and the α≤0.2 / Adult-only scope in everything below.** μ=5 was
the original discovery value and is cited historically, but is no longer the
number to build on.

---

## AGENT 1 — TASK C2: does AL compound with the radius finding?

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Read `docs/TASKS_AL_VALIDATION.md` first, then the "COMPLETED" section at the
> top of `docs/AGENT_BRIEFS.md` for full context on TASK A and TASK C, which are
> both done. Then do **TASK C2**.
>
> Context: independently of the AL work, the N1 radius ablation found that
> 11/15 (dataset, α) cells prefer `radii_scale=2.0` over the canonical 1.0 —
> the closed-form radius is systematically too small. That is the same
> diagnosis as AL's finding (starved dual), arriving through a completely
> different mechanism (undersized uncertainty set). Your job: find out whether
> the two fixes compound, are redundant, or conflict.
>
> **Use μ=20, not μ=5** — TASK C found μ=20 is the safe, more-effective value
> at α≤0.2 (see the COMPLETED section above for the full table).
>
> Run: Adult, α ∈ {0.2, 0.3}, 6 seeds, 2×2 design:
> `radii_scale ∈ {1.0, 2.0} × aug_lagrangian_mu ∈ {0, 20}`. The canonical and
> AL-only (μ=20, radii=1.0) arms already exist; add `radii_scale=2.0` alone and
> the combined arm. Write to a NEW results file, do not append to any existing
> ablation file.
>
> **Pre-register before running:** do you expect compounding (combined beats
> both singles), redundancy (combined ≈ best single), or conflict? Write it
> down first. Apply the degeneracy guard (floor 0.7521 for Adult) — a larger
> radius plus a stronger penalty is exactly the combination most likely to
> collapse the model, and α=0.3 is already a marginal regime.
>
> Deliver: results JSON + `results/al_radius_compound_summary.md` stating which
> of the three outcomes occurred. "Redundant" is a genuinely interesting,
> publishable answer, not a failure. `pytest tests/ -q` green before any commit.

---

## AGENT 2 — TASK B: mechanism, then TASK D: paper integration (unblocked now)

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Read `docs/TASKS_AL_VALIDATION.md` first, then the "COMPLETED" section at the
> top of `docs/AGENT_BRIEFS.md`. Do **TASK B**, then **TASK D** — both are now
> unblocked (TASK A and TASK C, which gated them, are done).
>
> TASK B: a fairness penalty that *improves* accuracy is counter-intuitive and
> an examiner will ask why. Our hypothesis: under a DP-targeted attack the
> corrupted points are exactly the ones driving the group-rate gap, so
> penalising that gap hard suppresses the attack's influence — i.e. AL acts as
> attack-specific denoising. That is a HYPOTHESIS. Test it:
> 1. `dump_history=True` already exists on `run_single_experiment`. Dump Adult
>    α=0.2 seed 0 for canonical DRO vs AL. **Run this at both μ=5 and μ=20** —
>    μ=20 is the value going into the paper, but μ=5 is the value the original
>    hypothesis was formed against, so check the mechanism holds at both.
>    Compare `g_dp`, `lambda_dp`, `train_loss`, `val_acc`, `val_dp` trajectories.
> 2. Measure directly: accuracy on the CORRUPTED training subset vs the CLEAN
>    subset, separately, for both models. If the hypothesis holds, AL fits the
>    corrupted points *less*.
> 3. Check the predicted-positive rate per group, to rule out a trivial
>    threshold-shift explanation.
>
> "Mechanism unclear" is an acceptable, honest outcome. A hand-wave is not.
>
> TASK D: write the AL section into `paper/sections/results.tex` and the
> report. Confirmed via search: neither currently contains any AL numbers —
> only qualitative "Future Work" mentions (`discussion.tex` ~L184,
> `report.tex` ~L673). This is a real integration, not an edit of existing
> numbers. Include:
> - The λ-starvation diagnosis with measured numbers (max λ 0.0119 vs ceiling
>   1.5; penalty 0.0029 vs loss 0.538)
> - The formula, `μ=0` recovers the canonical objective exactly (so all prior
>   results stand — state this explicitly)
> - **The Adult table at μ=20** (DP −70.8% to −81.7%, accuracy held or
>   improved) — not μ=5, which is superseded
> - **Explicit scope statement: α ≤ 0.2 only.** α=0.4 is reported as "AL is
>   unsafe here — no μ tested avoids model collapse," stated as a finding, not
>   hidden in a footnote
> - **Explicit dataset scope: Adult only.** Credit is not rescued by any μ
>   tested (0.5, 1, 2, 5, 10, 20) — state this directly
>
> Remove AL from Future Work in both documents (it is no longer future work).
> `make paper && make report`, both PDFs must build.
>
> Framing requirement throughout: "a proposed improvement, precisely scoped,
> with evidence on Adult α≤0.2 and honest negatives outside that scope" —
> never a universal win.

---

## AGENT 3 — Independent adversarial review (must NOT do Agent 1 or 2's work)

> You are working in the DRO-FairML repo at `/Users/srujansai/Desktop/DRO-FairML`.
> Do **TASK E** in `docs/TASKS_AL_VALIDATION.md`. You are the independent
> reviewer — do not run Agent 1's or Agent 2's tasks, or the review stops being
> independent.
>
> Your job is to BREAK the augmented-Lagrangian claim, not to confirm it.
> Finding a real defect is a successful outcome. Recent history justifies the
> paranoia: a verification pass found the built PDF printing "0.0% effect,
> p=0.016***" — a significance star on floating-point noise. Assume more exists.
>
> Your scope now includes **TASK C's rule application**, since μ=20 is the
> number about to be written into the paper and hasn't been adversarially
> checked yet:
> - Read `experiments/summarize_mu_sensitivity.py` (Rules C1–C4) and
>   `docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`. Recompute the
>   SAFE/EFFECTIVE table by hand from `results/mu_sensitivity.json` — do not
>   trust the script's own output.
> - The α=0.4 "no μ is safe" conclusion rests on μ ∈ {0.5, 1, 2, 5, 10, 20}
>   all failing. Check each one individually rather than trusting the summary.
> - Credit's "not rescued" conclusion: same — recompute from raw data.
>
> Original scope, unchanged:
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
