# MASTER PLAN v2 — Complete DRO-FairML (detailed agent assignments)

> Orchestrator: Srujan. Distribute the four briefs (§4–§7) to four agents.
> Paste §0 (constraints) **and** §1 (coordination protocol) into EVERY agent.
> §1 is non-negotiable — ignoring it already cost us a 4-way collision and a
> contaminated dataset. Read §2 (status) and §3 (canonical spec) before starting.

---

## §0. GLOBAL HARD CONSTRAINTS (paste into every agent)
1. **Corruption is ALWAYS adversarial** (`FairnessTargetedPGD`). `RandomCorruptor` is a baseline comparator only — never the method.
2. **Paper spec, mandatory:** `epochs=60`, `K_inner=10`, step order θ→λ→p, inner-max ascends `∇g` (not `λ∇g`), dual λ init 0.0 (the `lambda_init` param is ONLY for the Q1 ablation), `lambda_max=1.5` all datasets.
3. **No oracle leak:** DRO knows only `α` (and the *known attack structure* for empirical radii) — never the true per-sample corruption mask.
4. **Evidence before claims.** Run it, show output. No "should work." (We have been burned by overclaiming repeatedly.)
5. **Private repo, professor only.** No publicity.
6. **Verified headline:** fixed **tau=1** makes DRO beat Naive on DP at every α on Adult (wins 2/3,3/3,3/3,3/3 seeds), accuracy equal-or-better. The old "DRO is fragile / two-regime" story was a `tau=100` artifact. Whole narrative is now built on tau=1.
7. Commit with clear messages, `Co-Authored-By: Claude <noreply@anthropic.com>`, work on `main`.

---

## §1. COORDINATION PROTOCOL (the fix for the chaos — MANDATORY)
We have 4 agents + orchestrator on one machine and one repo. Two failures already happened: (a) **4 processes wrote the same JSON** and froze it at 109/270; (b) **mixed `k_inner` (5 vs 10)** silently contaminated results because rows don't record their config.

Rules:
1. **Agent A is the SOLE launcher of any `experiments/run_*.py`.** B, C, D, and the orchestrator NEVER start a training run. If you need data, ask A.
2. **One process per results file.** Before launching anything, `ps aux | grep run_` — if it's alive, do not start another writer of the same file.
3. **No blanket `pkill python`.** Kill by specific PID only. A `pkill` nukes every agent's work.
4. **Every result row MUST record its full config** (see §3) — `k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps, n_seeds_planned`. A run without provenance is unusable.
5. **File ownership is exclusive:** A=`experiments/`+`results/`+`logs/`; B=`src/`; C=`figures/`+`results/*.csv`+analysis scripts; D=`report/`+`paper/`+`docs/`+top-level `*.md`. Do not edit another agent's files; if you must, post in the group first.
6. **Currently running (DO NOT relaunch — adopt these):** `run_tau_ablation.py --tau 1 --k_inner 10` (writing `results/tau_ablation_tau1.json`, preliminary) and `run_lambda_lr_grid.py` (writing `results/lambda_lr_grid.json`). Let them finish; do not start duplicates.

---

## §2. STATUS — where each Kuldeep question stands
Meeting today is **with Kuldeep** (Madam: "meet without me, I'll take updates from Kuldeep later"). So it's a technical working session: present Adult results, discuss improvements.

| Q | Topic | Status | Owner of remaining work |
|---|---|---|---|
| Q1 | λ_init × lr tuning to tighten DP | grid RUNNING (Adult, tau=1) | A run, C analyze |
| Q3 | LSAC → use IF attack (inherent low DP) | narrative decided | D writes |
| Q5 | empirical radii (attack known) | code DONE (`radii_mode='empirical'`) — needs EXPERIMENTS + theory writeup | A run uniform-vs-empirical, B prove math, D appendix |
| Q6 | IF k-NN ablation k∈{5,10,15} | Adult DONE (k-insensitive) | A extend to Credit/LSAC, C table |
| Q12 | fix tau for all α; ablate tau | Adult DONE (tau=1 wins) — **the headline** | A extend to Credit/LSAC, C figure |
| Q9 | seeds for significance | moving 3→6 (3 can't reach p<0.05) | A run 6 seeds, C Wilcoxon |
| Q2/Q4 | two-regime / α=0 anomaly | resolved on Adult+Credit at tau=1; LSAC TBD | A data, C confirm |
| Q7 | IF attack lowers DP (inverse) | characterize | C analyze, D explain |
| Q13 | UTKFace | BLOCKED (flair2 GPU/SSL; email supin.gopi) | A/Srujan chase access |

**Verified finding (Adult, tau=1):** DRO < Naive DP at every α; advantage grows with α; accuracy equal-or-better. Adversarial attack raises DP ~31× more than random noise. IF attack insensitive to k.

**Known debts:** `tau_ablation_tau1.json` is preliminary (mixed k_inner). Result rows lack full config provenance. No clean canonical dataset yet. Only ≤3 seeds so far.

---

## §3. THE CANONICAL DATASET (definition — Agent A's primary deliverable)
The publishable result set. Everything else (figures, report) derives from it.
- **File:** `results/canonical_tau1.json` (FRESH — do not reuse the contaminated `tau_ablation_tau1.json`).
- **Grid:** 3 datasets × α∈{0.0,0.1,0.2,0.3,0.4} × {dp,if,combined} × {naive,dro} × seeds 0–5 (**6 seeds**) = 540 rows.
- **Fixed config:** `tau=1`, `K_inner=10`, `epochs=60`, `pgd_steps=20`, `coordinated=False`, `lambda_max=1.5`, `radii_mode='uniform'`.
- **Every row records:** all of the above + `dp_clean, if_clean, acc_clean, total_time`.
- **Empirical-radii companion:** `results/canonical_tau1_empirical.json` — same grid but DRO with `radii_mode='empirical'` (Q5). Lets C show uniform-vs-empirical.
- **Run via ONE sequential master script** (no parallelism → no collision), incremental save + resume, commit every ~30 min.

---

## §4. AGENT A — Experiments & Data (sole run owner; owns `experiments/`, `results/`, `logs/`)
**Mission:** deliver the canonical dataset + all ablations, with full provenance, no collisions.

**Tasks (in order):**
1. **Add config provenance** to the result dict in `run_single_experiment` / ablation scripts: record `k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps`. (Coordinate with B if a `src` helper is cleaner — but the runner is yours.)
2. **Write `experiments/run_canonical.py`** = one sequential driver producing `results/canonical_tau1.json` per §3 (6 seeds, k_inner=10, tau=1). Resume-safe. This SUPERSEDES `tau_ablation_tau1.json`.
3. **Empirical-radii companion** `results/canonical_tau1_empirical.json` (DRO `radii_mode='empirical'`), same grid. Only after B confirms the empirical math + "src frozen."
4. **Finish the λ grid** (`lambda_lr_grid.json`); if time, add α∈{0.1,0.4}.
5. **Extend IF k-NN ablation to Credit + LSAC** (`run_knn_ablation.py --k 5/10/15 --datasets adult credit lsac`).
6. **UTKFace (Q13):** attempt flair2 access (email supin.gopi); if up, run the same canonical config on UTKFace (binary protected attr = gender; trainer is binary-only — never pass 5-class race).
7. Commit `results/` incrementally. Keep exactly one writer per file.

**Acceptance:** `canonical_tau1.json` (540 rows, full provenance) + `canonical_tau1_empirical.json` + complete grid + 3-dataset k-NN, all pushed. Report any coverage holes.
**Depends on:** B "src frozen" before canonical runs (else mixed code). Until then, the in-flight preliminary runs are fine for today's Kuldeep meeting.

---

## §5. AGENT B — Core code & theory (sole `src/` owner)
**Mission:** finish the code so the canonical run is correct, prove the Q5 math, freeze.

**Tasks:**
1. **Empirical radii (Q5) — validate.** `radii_mode='empirical'` exists; write a unit test: synthesize a known coordinated 70%-minority attack, confirm `_empirical_pi_clean` recovers the true clean proportions within tolerance. Paste output.
2. **Config provenance helper** (support A): make it trivial for the runner to capture `{k_inner,tau,radii_mode,lambda_init,coordinated,pgd_steps}`. Small, in `src` or a shared util.
3. **Remaining audit fixes:** (a) validation τ consistency — in-fit validation should use the epoch's `current_tau`, not always `self.tau`; (b) `compute_dp_violation` for >2 groups (assert or document); (c) `run_fairness_pgd.py` docstring lie (no clean-train/retrain). [classifier `eval()` fix already DONE ✓]
4. **Theory writeup for Q5:** clean derivation of the empirical π_clean inversion (the `pi_clean[min]=pi_obs[min]+0.4α` etc.) → give to D for the appendix.
5. **Run the full test suite, paste output.** Then post **"src frozen"** so A can launch the canonical runs.

**Acceptance:** empirical-radii test passes; audit fixes verified with output; tests green; "src frozen" posted; pushed.
**Depends on:** nothing. **Blocks A's canonical runs.** Do this FAST and FIRST.

---

## §6. AGENT C — Analysis, figures, stats (owns `figures/`, `results/*.csv`, analysis scripts)
**Mission:** turn the canonical data into the paper's figures/tables + significance.

**Tasks:**
1. **Headline figure — "two-regime resolved":** DRO−Naive DP vs α at tau=100 (loses) vs tau=1 (wins), side by side. This is the story.
2. **Win-curves** (tau=1): DRO−Naive DP vs α per attack, all 3 datasets (advantage grows with α).
3. **n=6 Wilcoxon** significance table from `canonical_tau1.json`; mark cells with p<0.05 (now achievable). Save `results/canonical_wilcoxon.csv`.
4. **Uniform vs empirical radii (Q5):** compare `canonical_tau1.json` vs `canonical_tau1_empirical.json` — does empirical tighten DP? Figure + table.
5. **k-NN ablation table** (3 datasets) and **λ_init×lr heatmap** (best config).
6. **Random-vs-adversarial** bar chart, **absolute DP** values (not %).
7. Computer Modern fonts, error bars, no shading. Every figure regenerable from a committed script. PDF+PNG.

**Acceptance:** all figures + CSVs from CANONICAL data (not preliminary), each script-regenerable, pushed.
**Depends on:** A's canonical data. **Can start NOW** on preliminary Adult data, then re-point scripts to `canonical_tau1.json` when ready.

---

## §7. AGENT D — Report / paper / docs (owns `report/`, `paper/`, `docs/`, top-level `*.md`)
**Mission:** the written deliverables — paper, today's Kuldeep doc, clean repo.

**Tasks:**
1. **Today's Kuldeep discussion doc** (`KULDEEP_DISCUSSION.md`): concise — tau=1 result, ablations, the improvements to explore (empirical radii, λ tuning, 6-seed significance, UTKFace). This is for the meeting NOW; use current Adult numbers, flag what's still landing.
2. **Finalize report + paper** around tau=1 + empirical radii + ablations. **Every number pulled from C's CSVs — zero hand-typed stats.** Add Q5 derivation (from B) to the appendix.
3. **LSAC framing (Q3):** present via IF attack; explain inherent low DP.
4. **Q7 explanation:** IF↔DP inverse relationship, why IF attack can lower DP.
5. **Build the PDF clean** (`[hidelinks]`, `float`); paste the build log; no broken refs.
6. **Repo polish:** top-level = README, HANDOFF, MASTER_PLAN, MEETING_SCRIPT, KULDEEP_DISCUSSION, SERVER_RUNBOOK only; everything else in `docs/_archive/`. Update HANDOFF to current state.

**Acceptance:** Kuldeep doc ready for today; report/paper build clean with traceable numbers; repo tidy; pushed.
**Depends on:** C (figures/tables) for the final report; can scaffold prose + write the Kuldeep doc NOW.

---

## §8. SEQUENCING (dependency graph)
```
B (src + Q5 math + provenance)  ──"src frozen"──►  A (canonical 6-seed runs)  ──►  C (figures + n=6 stats)  ──►  D (final report)
        │                                                                                   ▲
        └── D writes Kuldeep doc + scaffolds report NOW ───────────────────────────────────┘
        └── C builds scripts on preliminary Adult data NOW, re-points to canonical later ───┘
```
- **Today / immediate:** B finishes & freezes; D writes the Kuldeep meeting doc; C builds figure scripts on preliminary data; A keeps the two in-flight runs alive and adds provenance.
- **After "src frozen":** A launches the canonical 540-run + empirical companion (sequential, single process). This is the long pole (~hours) — start ASAP once B is done.
- **As canonical lands:** C regenerates everything; D finalizes.

## §9. DEFINITION OF DONE (the whole project)
- [ ] `canonical_tau1.json` (3 datasets, 6 seeds, k_inner=10, tau=1, full provenance) + empirical companion.
- [ ] Every Kuldeep Q (1,3,5,6,7,9,12) answered with data; Q2/Q4 (two-regime/α=0) confirmed resolved at tau=1 on all 3 datasets.
- [ ] n=6 Wilcoxon significance table; report which cells reach p<0.05.
- [ ] All figures regenerated from canonical data; uniform-vs-empirical radii comparison done.
- [ ] Report + paper build clean, every number traced to a CSV, Q5 derivation in appendix.
- [ ] UTKFace: either run on flair2, or documented blocked with the exact access steps.
- [ ] Repo tidy; HANDOFF current; no contaminated/duplicate result files.
```
```

---

## §10. REMAINING WORK TO 100/100 (current snapshot — assign these)

> State as of now: code complete (tau=1, K=10, DP-PGD, α=0 guard, empirical radii,
> classifier eval fix, provenance, 60 tests claimed green). Adult ablations done.
> Report/paper updated to tau=1. The GAPS below are what stand between us and done.
> §1 coordination rules still apply: **Agent A is the only one who launches runs**,
> always `--k_inner 10`, every row carries provenance, never blanket-`pkill`.

### Currently running (do not double-launch)
- Grid `run_lambda_lr_grid.py` (12/72) — full CPU now. **BUG: resume not matching the 12 done rows (0 SKIP lines, recomputing from [1/72]); fix the key match — likely seed int/float or λ_init 0 vs 0.0.**
- Canonical `run_canonical.py` PAUSED at 57/540 (resume-safe). Chain watcher PID 23515 auto-resumes it when the grid finishes.

### AGENT A — Experiments (sole launcher; `experiments/`, `results/`, `logs/`)
1. **Fix grid resume** (key-type mismatch), let grid finish 72/72. Then commit `lambda_lr_grid.json`.
2. **Finish the canonical 540** (`canonical_tau1.json`): Credit + LSAC + seeds 0–5, k_inner=10, tau=1, full provenance. This is the publishable foundation. (~long; chain watcher already resumes it post-grid.)
3. **Empirical-radii companion** `canonical_tau1_empirical.json` (DRO `radii_mode='empirical'`, same grid) for the Q5 comparison.
4. **Extend IF k-NN ablation** (k 5/10/15) to Credit + LSAC if not already (Adult done).
5. **UTKFace (Q13):** email supin.gopi for flair2; if access works, run canonical config (binary protected attr=gender). Else keep the documented block + local smoke.

### AGENT B — Code/theory (sole `src/`) — mostly done; remaining:
1. **Confirm the 60 tests actually pass** (paste `pytest` output) and that the empirical-radii unit test is real.
2. Finish any leftover audit fixes (validation-τ consistency, `compute_dp_violation` >2 groups) if not already in; paste evidence.
3. Keep "src frozen" — no edits during A's canonical run.

### AGENT C — Analysis/figures/stats (`figures/`, `results/*.csv`)
1. **n=6 Wilcoxon** from the FINISHED canonical (`canonical_wilcoxon.csv`) — report which cells hit p<0.05.
2. **Regenerate ALL figures from FINAL canonical** (not the preliminary/contaminated `tau_ablation_tau1.json`): two-regime headline, win-curves (3 datasets), random-vs-adv (absolute DP), k-NN table, λ-grid heatmap, uniform-vs-empirical radii.
3. **Grid analysis (Q1):** confirm/refute the preliminary signal that `lr_lambda=0.001` tightens DP (was DP 0.16 vs 0.24 at α=0.2, n=3 — needs full grid + more seeds; could be noise).
4. **High-α figure:** use `results/high_alpha_tau_analysis.txt` — acc vs α across tau with the constant-predictor baseline line, showing tau isn't the lever and α≤0.2 is the meaningful regime.

### AGENT D — Report/paper/docs (`report/`, `paper/`, `docs/`, top-level `*.md`)
1. **Fold in the high-α finding** (from `analyze_high_alpha.py`): narrative = "DRO wins DP at every α; meaningful accuracy regime is α≤0.2; at α≥0.3 both methods degrade below the constant predictor but DRO still holds lower DP; tau is not the lever, λ/lr is the next knob." Honest framing.
2. **Re-point every number to the FINAL canonical CSVs** once C regenerates them (the report currently cites preliminary/Adult-3-seed numbers).
3. Q5 appendix (empirical radii derivation), LSAC-IF framing (Q3), Q7 inverse effect — confirm present.
4. **Rebuild PDFs, paste build logs**; keep root minimal.

### ORCHESTRATOR / QA GATE (verify, don't trust "done")
- Actually run the test suite, build both PDFs, and spot-check ~5 report numbers against their CSV rows before declaring done. The "100% complete" claims must be evidence-backed.

### DEFINITION OF 100/100
- [ ] Grid 72/72 (resume fixed) + Q1 conclusion. [ ] Canonical 540/540 + empirical companion.
- [ ] n=6 Wilcoxon with p<0.05 cells identified. [ ] All figures from FINAL canonical.
- [ ] Report/paper numbers all trace to final CSVs; PDFs build clean (verified, not claimed).
- [ ] High-α + tau-not-the-lever + λ-lever story written. [ ] UTKFace run or documented-blocked.
- [ ] Tests pass (verified). [ ] Repo clean, HANDOFF current.
