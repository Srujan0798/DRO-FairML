# Copy-paste prompts for agents

Repo: `/Users/srujansai/Desktop/DRO-FairML`. Full context in `docs/MASTER_DISPATCH.md`.

**Order:** A + B + F can start now (A and B both edit `src/training/dro_fair.py` — serialize or coordinate). C waits for A and B. D waits for C. E anytime.

---

## AGENT A — Fix the IF metric (BLOCKING)

```
You are working in /Users/srujansai/Desktop/DRO-FairML, a research repo comparing a
DRO fairness method against a Naive baseline under adversarial corruption. Read
docs/MASTER_DISPATCH.md first for full context.

YOUR TASK: the Individual Fairness (IF) metric is broken and every IF result in the
project is meaningless. Fix it.

THE EVIDENCE (verify this yourself before starting):
  python3 -c "
  import json; d=json.load(open('results/canonical_tau1.json'))
  print(len(d), max(abs(r['if_clean']) for r in d))"
Prints 540 and 4.66e-10. The if_clean field is identically zero in all 540 canonical
rows (adult max 4.7e-10, credit 9.9e-12, lsac exactly 0.0). Every IF p-value in every
table is noise on floating-point dust.

WHAT TO DO:
1. Diagnose src/evaluation/metrics.py:59-118 (compute_if_violation). It returns
   relu(|h_i - h_j| - d_ij - gamma) over a k-NN graph. Instrument it on a single Adult
   run: print the distributions of |h_i - h_j|, d_ij, and gamma. Find out why the relu
   saturates to zero for every pair. Likely causes: gamma too large, d_ij on the wrong
   scale (raw feature distance vs [0,1] sigmoid predictions), or predictions clustered
   too tightly. Report which it is before fixing.
2. Fix the calibration so IF produces real signal that responds to the IF-targeted
   attack. Add a regression test: IF > 1e-4 on a deliberately unfair predictor, ~0 on
   a constant predictor.
3. Fix the confirmed attack/eval k-NN mismatch. The attack builds neighbours WITHIN
   protected groups (src/corruption/adversarial.py:293-308, loops `for g in [0,1]`)
   while training (src/training/dro_fair.py:167, src/training/naive_fair.py:46) and
   eval (src/evaluation/metrics.py:88) build them over ALL samples. Also
   compute_if_violation accepts an `a` argument at :59 and never uses it. Make attack
   and eval use the same graph.
4. The IF attack gradient (adversarial.py:355-357) is a bare label agree/disagree count
   that ignores d_ij and gamma entirely, so it optimizes a different quantity than the
   one measured. Align it with the actual objective.
5. Re-run ONLY the IF-attack third of the canonical grid: 180 rows = 3 datasets x 5
   alphas x 6 seeds x 2 methods. Use experiments/run_canonical.py. Config must match
   exactly: tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0,
   coordinated=False. DO NOT re-run the DP or Combined rows — they are unaffected.

CONSTRAINTS:
- Run `python3 -m pytest tests/ -q` before and after. 60 tests pass now; keep it there.
- Do not touch results/canonical_tau1.json DP or Combined rows.
- If the fix changes what IF means, say so plainly. Do not tune until numbers look good.
- Report what you found, what you changed, and the new IF numbers with before/after.
```

---

## AGENT B — Diagnose the LSAC degeneracy (BLOCKING)

```
You are working in /Users/srujansai/Desktop/DRO-FairML, a research repo comparing a
DRO fairness method against a Naive baseline under adversarial corruption. Read
docs/MASTER_DISPATCH.md first for full context.

YOUR TASK: on the LSAC dataset under the DP attack, DRO loses to Naive at every
corruption level, 0/6 seeds, p=1.0. Find out why. This has never been reported.

THE EVIDENCE (verify yourself):
  lsac/dp  alpha=0.0  naive DP 0.1447  dro DP 0.1829  0/6 wins
  lsac/dp  alpha=0.1  naive DP 0.2201  dro DP 0.2539  0/6 wins
  lsac/dp  alpha=0.2  naive DP 0.1827  dro DP 0.2230  0/6 wins
  lsac/dp  alpha=0.3  naive DP 0.1827  dro DP 0.2220  0/6 wins
  lsac/dp  alpha=0.4  naive DP 0.1827  dro DP 0.2211  0/6 wins
Two red flags: DP is BIT-IDENTICAL (0.1827) at alpha 0.2/0.3/0.4 — a metric that does
not move as corruption doubles. And accuracy sits at 0.902-0.903 against LSAC's
constant-predictor baseline of 0.9016 at every alpha. The model appears to have
collapsed to the constant predictor.

WHAT TO DO:
1. Explain the frozen DP and the pinned accuracy. First hypothesis to test: LSAC is
   ~90/10 class-imbalanced; the model collapses to the majority class; DRO's radii
   rho_dp[j] = alpha/((1-alpha)*pi_clean[j] + alpha) at src/training/dro_fair.py:114-115
   blow up on the tiny minority group and over-weight it into collapse. Test it, don't
   assume it.
2. Determine whether DRO losing on LSAC/dp is (a) a genuine negative result about
   imbalanced groups, or (b) an artifact of collapse. REPORT HONESTLY EITHER WAY. Do
   not tune hyperparameters until DRO wins — a real negative result is a valid finding
   and is more valuable than a manufactured win.
3. Fix a provenance lie. _compute_radii at src/training/dro_fair.py:97 prefers `a_val`
   whenever it is non-None, and experiments/run_fairness_pgd.py:102 and :117 ALWAYS
   pass it. So the uniform closed form (pi_hat - alpha)/(1-2alpha) at :103 NEVER
   EXECUTES, and the renormalization at :107-110 sits inside an unreachable branch. All
   540 canonical rows are labelled radii_mode="uniform" but actually used clean
   validation proportions. Either fix the dispatch or relabel the provenance — the
   current label is false.
4. Resolve the alpha=0 question. With zero corruption DRO and Naive should coincide,
   and they do not (Adult 0.1491 vs 0.1426, counted as a 6/6 "win", p=0.016). The
   alpha=0 guard at dro_fair.py:330 only stops the inner p-loop advancing the RNG. The
   trainers still differ structurally: DRO optimizes a tilted risk (beta*logsumexp,
   dro_fair.py:302), Naive optimizes plain BCE mean (naive_fair.py:136); DRO decays the
   dual LR; they validate on different schedules. A "win" at alpha=0 is a different
   objective, not robustness. Either justify counting it or recommend excluding alpha=0
   from all win counts. (This is reviewer question Q4, open since Jun 9.)

CONSTRAINTS:
- `python3 -m pytest tests/ -q` must stay at 60 passing.
- Note: Agent A is also editing src/training/dro_fair.py (IF/k-NN paths). You own radii
  and the alpha=0 question. Coordinate before writing.
- Deliverable is a written explanation, not a green number.
```

---

## AGENT C — Regenerate every downstream artifact (needs A + B)

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
first for full context.

YOUR TASK: the 540-row canonical run completed on 2026-07-02, but every CSV, LaTeX
table, figure and PDF in this repo was built from a 307/540 partial snapshot on Jun 30.
Regenerate the entire downstream chain from results/canonical_tau1.json.

WHAT TO DO:
1. Repoint experiments/generate_report_tables.py at results/canonical_tau1.json. It
   currently reads results/tau1_summary.csv, a stale intermediate where LSAC only
   reaches alpha=0.1 and n_seeds is mixed 1-6. This is why NO LaTeX table in the repo
   contains a single LSAC row — verify with: grep -c lsac report/sections/*.tex
   paper/auto_generated/*.tex  (returns 0 everywhere).
2. Remove the silent-fallback landmines. Each of these can make a script quietly use
   wrong data:
   - experiments/analyze_tau1.py:131-136 prefers a deleted K_inner=5 backup file over
     the canonical K_inner=10 file whenever it has more rows
   - experiments/compute_canonical_wilcoxon.py:49-61 silently falls back to
     tau_ablation_tau1.json
   - results/fairness_pgd_results.json is contaminated (270 pre-provenance rows, all
     tau=None) and is the MOST-READ results file in experiments/ at 13 call sites
   Make every loader fail loudly instead of falling back. Assert provenance
   (tau==1.0, k_inner==10) on load.
3. Regenerate in order: Wilcoxon across all 45 cells
   (experiments/compute_canonical_wilcoxon.py), summary CSVs, all figures, then both
   PDFs via `tectonic` (report/report.tex and paper/main.tex).
4. Compute the constant-predictor baseline PER DATASET from the actual data and delete
   the hardcoded CONSTANT_PREDICTOR_ACC = 0.752 from all four generators that carry it:
   experiments/plot_high_alpha_tau.py:40, generate_final_figures.py:44,
   plot_lambda_heatmap_highalpha.py:34, generate_all_deliverables.py:81. It is Adult's
   majority rate and is silently wrong for Credit (0.7788) and LSAC (0.9016). It is also
   baked into caption strings in generate_all_deliverables.py and
   generate_final_meeting_figure.py.
5. Move every results/* file with mtime older than 2026-07-02 18:35 into
   results/stale_archived/.

VERIFY BEFORE REPORTING DONE:
- Every table contains LSAC rows
- Wilcoxon covers all 45 cells at n=6 (no cells at n=2 or n=3)
- No figure or table traces back to a file other than results/canonical_tau1.json
- Both PDFs rebuild without error
```

---

## AGENT D — Rewrite every claim to match the data (needs C)

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
first — it contains the verified numbers you must write to.

YOUR TASK: the docs and paper make claims the completed data does not support. Fix
every claim so it is traceable to results/canonical_tau1.json.

WHAT TO DO:
1. Delete "DRO wins on DP at every alpha" everywhere (README.md, STATUS.md,
   KULDEEP_DISCUSSION.md, MASTER_PLAN.md, paper/, report/). Replace with the verified
   scoped claim:
     "At alpha <= 0.2, DRO-FAIR achieves lower demographic parity violation than
      Naive-FAIR on Adult and Credit under all three attacks (p<0.05, n=6 seeds).
      LSAC under the DP attack is degenerate and is reported separately. At
      alpha >= 0.3 both methods fall below the constant-predictor accuracy baseline
      on every dataset, and no method claim is made in that regime."
2. Strip every IF claim pending Agent A's fix. The IF metric is identically zero in
   all 540 rows, so all current IF numbers and IF plots are meaningless.
3. Fix paper/auto_generated/key_findings.tex — it still hardcodes 3-seed claims
   ("Wilcoxon pending n=6", "wins 2/3,3/3,3/3,3/3") and its own header admits it was
   hand-patched. n=6 has been complete for weeks.
4. Re-derive or explicitly label as historical the hardcoded tau=100 comparison numbers
   at report/report.tex:441 and :463, and paper/sections/results.tex:23-24 and :54. One
   of them cites "tau1_summary.csv row 66-67" — a row index into a file that has since
   been regenerated, so the citation is meaningless.
5. Resolve docs/UTKFACE_RESULTS.md. It claims a 23,705-image 5-seed GPU run. The rest
   of the repo says GPU access on flair2 was never granted and only a 2-row CPU smoke
   test exists. Note that experiments/run_utkface.py:52-105 silently substitutes
   _make_synthetic_utkface (random Gaussians) when the real dataset is missing.
   Establish whether those numbers are real, simulated, or synthetic. If synthetic, say
   so in the file title and delete every figure derived from it.
6. Draft an honest correction note for the collaborator (Kuldeep) covering three things:
   - the IF plots sent on Jun 30 (adult_if_*_meeting.pdf) plotted DP data under an IF
     label, and the quoted "IF = 0.0195 vs 0.0177" came from the DP column
   - LSAC was reported as "pending" but has been complete since Jul 2 and is negative
   - the alpha >= 0.3 regime is below the constant-predictor baseline
   Lead with the correction, not with new figures. Adult and Credit at alpha <= 0.2 are
   a solid defensible result on their own. On Jun 30 he asked: "After drafting the
   reply, could you please verify all the claims? Sometimes AI tends to make claims just
   to make the results appear correct." Meet that request properly.

CONSTRAINT: every number you write must be reproducible from results/canonical_tau1.json.
If you cannot trace it, delete it.
```

---

## AGENT E — Finish or formally drop the ablations

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
first for context.

YOUR TASK: five ablation studies are incomplete or incoherent. For each, either finish
it or formally drop it with a written reason. Do not leave any in limbo.

1. TAU ABLATION — currently not publishable. No LSAC in ANY tau file; tau=5 and tau=20
   are Adult-only; k_inner is MIXED WITHIN FILES (tau_ablation_tau1.json has 109 rows
   with k_inner=None and 15 with k_inner=10; tau=10 and tau=100 are ~50/50). Comparing
   across tau therefore confounds tau with k_inner. Either re-run clean at k_inner=10
   across all datasets, or restrict the published scope to Adult and state that limit
   explicitly.

2. LAMBDA GRID — 26 of 720 configs done (3.6%), crashed. Check lambda_comprehensive.log:
   config [25/720] at lambda_init=1.0 took 64308 seconds (17.9 HOURS) versus ~300-1500s
   for its neighbours. Diagnose that pathology FIRST — a naive restart will hang again.
   Then scope down to a feasible grid.

3. KNN ABLATION — k=10 complete (144 rows); k=5 short by 12 rows, k=15 short by 24.
   Backfill the 36 missing rows. Cheap, and it closes reviewer question Q6.

4. EMPIRICAL RADII (Q5) — 29 of 270 rows, Adult only, and the alpha=0 rows are exact
   no-ops (delta 0.000000). Finish for Adult at minimum, or drop with a written reason.

5. RANDOM VS ADVERSARIAL — 27 rows dated Jun 9, predates the canonical protocol. Re-run
   under canonical config.

CANONICAL CONFIG for any re-run: tau=1.0, k_inner=10, epochs=60, pgd_steps=20,
lambda_init=0.0, coordinated=False, 6 seeds.

Report a table of what you finished, what you dropped, and why.
```

---

## AGENT F — Repo consolidation (safe, parallel, touches no science)

```
You are working in /Users/srujansai/Desktop/DRO-FairML. Read docs/MASTER_DISPATCH.md
first. Your work must not change any scientific result — only structure.

DELETE (all verified duplicates or orphans):
  kuldeep_meeting/            6.8MB, gitignored, byte-identical copies of figures/+results/
  kuldeep_meeting.zip         superseded snapshot of the above
  "What changed for α≥0.ini"  byte-duplicate of KULDEEP_REPLY.md, misnamed, non-ASCII
  KULDEEP_REPLY.md            ephemeral chat reply, content preserved in the chat export
  docs/CHAT_HISTORY_MAY_JUNE.md   third copy of the same chat log
  knn_ablation.log            0 bytes
  These 5 figure stems (.pdf and .png each) — no generating script, zero references:
    figures/fig2_dp_reduction_heatmap_complete
    figures/fig9_fairness_pgd_curves_complete
    figures/fig_complete_3x3_results
    figures/fig_final_constant_predictor_acc_complete
    figures/tradeoff_accuracy_dp

UNTRACK (git rm --cached, add to .gitignore):
  paper/ICML_submission.pdf   2.9MB, dated May 4, largest tracked file, encodes the
                              retracted "DRO is fragile" conclusion
  submission/                 STALE FORK — 7 of 14 src/ files have diverged from live
                              src/, still on tau=100 defaults, and ships a report.pdf
                              asserting the retracted finding. Anyone opening this
                              directory first gets the wrong science. Move to
                              docs/_archive/submission_may2026/ with a SUPERSEDED.md.
  logs/*.pid                  4 dead June process IDs committed to version control

MOVE:
  "gChat Conversation.md"       -> docs/chat/gchat_raw_export.md
  GOOGLE_CHAT_CONVERSATION.md   -> prepend as summary to the above, then delete
  MASTER_PLAN.md, docs/project_management/ (14 dead files), docs/archive/
                                -> merge all into a single docs/_archive/
  the 12 orchestration scripts misfiled under logs/ (canonical_watcher.py,
  agent_data_refresher.py, quick_poll_loop.sh, etc.) -> scripts/

KILL THE STEPPED-TAU ZOMBIE (this is the bug that cost the project a month):
  `return 1.0 if alpha >= 0.4 else 100.0` is STILL LIVE in 6 files:
    experiments/run_ablations.py:30
    experiments/run_utkface.py:49, run_utkface_randinit.py:52,
      run_utkface_extended.py:41, run_utkface_pixel_pgd.py:47
    submission/run_experiments.py:33
  Plus hardcoded tau=100.0 in run_lambda_diagnostic.py:56 and
  run_lambda_diagnostic_full.py:52, and a stepped variant in
  run_random_vs_adversarial.py:27.
  get_temperature is defined NINE TIMES across the repo. Consolidate to one shared
  import. Also delete the stale comment at run_experiments.py:79 that still says
  "Paper §G.6: tau=100 for alpha<=0.3" directly above a function now returning 1.0.

FIX:
  README.md broken links to HANDOFF.md and SERVER_RUNBOOK.md (both moved into docs/)
  Makefile dead targets: `monitor` prints "Monitor script removed"; `review` points at
    docs/REVIEW_CHECKLIST.md which now lives in docs/_archive/
  Add Makefile `paper` and `report` targets invoking tectonic, so the build is discoverable
  Add data/download_data.sh — A FRESH CLONE CURRENTLY CANNOT REPRODUCE ANYTHING
  src/corruption/__init__.py omits FairnessTargetedPGD — the canonical attack — from its
    imports and __all__, so every runner works around it with a full-path import

DELETE DEAD CODE:
  src/corruption/adversarial.py:384-420  _select_targets (never called)
  src/corruption/adversarial.py:540-557  _attack_features_fgsm (never called)
  src/corruption/image_pgd.py            never imported anywhere
  src/training/__init__.py:8-58          get_run_config (exported, zero callers)
  src/training/dro_fair.py:193-205       _project_dp_weights and _project_if_weights
                                         are byte-identical

CONSTRAINT: run `python3 -m pytest tests/ -q` after every stage. 60 tests pass now.
If any deletion breaks a test or an import, stop and report rather than patching around it.
```
