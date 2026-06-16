# MASTER PLAN — Complete the DRO-FairML project (distribute to 3–4 agents)

> Orchestrator notes for Srujan. Hand each brief below to one agent. They are
> separated by **file ownership** so agents do not collide. Dependencies are
> stated. Global hard constraints apply to EVERY agent — paste §0 into each.

---

## §0. GLOBAL HARD CONSTRAINTS (paste into every agent brief)

1. **Corruption is ALWAYS adversarial** (`FairnessTargetedPGD`). Never use `RandomCorruptor` as the method — it exists only as a baseline comparator. This is a professor-mandated design choice.
2. **Paper-spec values are mandatory:** `epochs=60`, `K_inner=10`. Algorithm-1 step order `θ → λ → p`. Inner max ascends `∇g` (not `λ∇g`). Dual `λ` init = 0.0 by default (the new `lambda_init` param exists ONLY for the Q1 ablation; production stays 0.0). `lambda_max=1.5` for all datasets (no per-dataset hacks).
3. **No oracle leak:** DRO only knows `α`, never the true per-group corruption mask/rates.
4. **Never claim something works without running it** and showing output. Evidence before assertions. The user has been burned repeatedly by overclaiming.
5. **This repo is private to the professor.** No publicity, no blog/README-for-show/Pages.
6. **The headline finding (already verified, Adult):** fixed **tau=1** makes DRO beat Naive on DP at every α; the old "DRO is fragile" story was a `tau=100` artifact. Kuldeep endorsed fixing tau for all α. The project's new narrative is built on tau=1.
7. Commit with clear messages; end commits with `Co-Authored-By: Claude <noreply@anthropic.com>`. Work on `main` unless told otherwise.

---

## §1. CURRENT STATE (as of handoff)

- **RUNNING now (do not kill):**
  - `experiments/run_tau_ablation.py --tau 1 --datasets adult credit lsac` → `results/tau_ablation_tau1.json` (Adult 72 rows done; adding Adult α=0, Credit, LSAC → target 270). Log: `logs/tau1_full_rerun.log`. **This is the canonical tau=1 dataset.**
  - `experiments/run_lambda_lr_grid.py` → `results/lambda_lr_grid.json` (Adult, λ_init×lr grid, 72 runs). Log: `logs/lambda_lr_grid.log`.
- **DONE:** Adult tau ablation (tau 1/10/100), Adult IF k-NN ablation (k 5/10/15), 270-run fixed-attack grid at old stepped tau (`results/fairness_pgd_results.json`, `results/fairness_pgd_wilcoxon.csv` — these reflect OLD tau=100 and show DRO losing; they will be SUPERSEDED).
- **Three bugs already fixed (committed):** K_inner=10, DP-targeted feature PGD, α=0 inner-max guard.
- **Docs:** `HANDOFF.md` (full history), `ADULT_RESULTS_FOR_KULDEEP.md` (current results), `SERVER_RUNBOOK.md` (UTKFace GPU), `README.md`. Stale docs archived in `docs/_archive/week_pre_tau1/`.

---

## §2. AGENT A — Experiments & Data (owns `experiments/*.py`, `results/*.json`, `logs/`)

**Mission:** produce the complete, canonical tau=1 result set across all datasets and finish all ablations. Does NOT edit `src/`.

**Tasks (in order):**
1. **Monitor + keep alive** the two running jobs (§1). If a process dies, relaunch with the same command (both support resume/skip-done).
2. When the tau=1 full re-run (`results/tau_ablation_tau1.json`) hits 270 rows, verify coverage: 3 datasets × 5 α (0.0–0.4) × 3 attacks × 2 methods × 3 seeds. Report any holes.
3. **Add seeds for significance.** With 3 seeds Wilcoxon p<0.05 is impossible (min p=0.125). Run **3 more seeds** (seeds 3,4,5) for the tau=1 grid so n=6 → significance is achievable. Command pattern:
   `python3 experiments/run_tau_ablation.py --tau 1 --datasets adult credit lsac --n_seeds 6` (it resumes; only the new seeds run). ⚠️ confirm the script's seed loop covers 0..5.
4. **Extend the IF k-NN ablation to Credit + LSAC** (currently Adult only): `python3 experiments/run_knn_ablation.py --k 5` (and 10, 15) `--datasets adult credit lsac`.
5. **Finish the λ_init×lr grid** and, if time, extend to α∈{0.1,0.4} for completeness.
6. Keep `results/` committed incrementally (every ~30 min) so progress is never lost.

**Acceptance:** `results/tau_ablation_tau1.json` complete at n=6 seeds for all 3 datasets; k-NN ablation covers 3 datasets; grid complete. All pushed.

**Depends on:** Agent B's `src/` changes should be FINAL before the canonical n=6 re-run (otherwise results mix code versions). Coordinate: do the n=6 re-run only after B signals "src frozen."

---

## §3. AGENT B — Core code & theory (owns `src/`)

**Mission:** implement Kuldeep's Q5 (empirical radii) and fix the outstanding audit bugs. Touch ONLY `src/`. Announce "src frozen" when done so Agent A can run the canonical set.

**Tasks:**
1. **Q5 — empirical radii calibration** in `src/training/dro_fair.py::_compute_radii`. Kuldeep: "this is empirical not theoretical; if the attack is known we can use the approximation according to the attack." Implement an **empirical mode** that estimates clean group proportions `π_clean` from the observed post-attack data given the known coordinated (70%-minority) attack, instead of the uniform-corruption closed form `π_clean=(π̂−α)/(1−2α)`. Keep the closed form as default; add a flag `radii_mode='uniform'|'empirical'`. Do NOT pass the true corruption mask (no oracle) — derive from α + attack structure only.
2. **Audit bug — inference mode (HIGH):** `src/models/classifier.py` `predict`/`predict_proba` do not call `.eval()`/`torch.no_grad()`, so Dropout is active at inference. Fix with `self.eval()` + `torch.no_grad()`. Verify training path unaffected.
3. **Audit bug — validation τ inconsistency:** during the 15 warmup epochs, training uses τ=1 but in-fit validation calls `compute_metrics_torch(..., temperature=self.tau)`. Make validation use the same `current_tau` as the epoch. (Lower priority — note if risky.)
4. **Audit bug — docstring lie** in `experiments/run_fairness_pgd.py` lines 5–7 (claims clean-train + retrain; there is none). Fix the docstring. (This file is shared with A — coordinate, it's a 3-line comment.)
5. **`compute_dp_violation`** silently handles only 2 groups — add an assert/clear handling for >2 (matters for UTKFace race).
6. Add/verify a unit test for each fix. Run the existing test suite, paste output.

**Acceptance:** all fixes verified with output; `radii_mode='empirical'` runs end-to-end on Adult; tests pass; pushed. Then announce "src frozen."

**Depends on:** nothing. Blocks Agent A's canonical re-run.

---

## §4. AGENT C — Analysis, figures, tables (owns `experiments/analyze_*.py`, `experiments/generate_*figures*.py`, `figures/`, `results/*.csv`)

**Mission:** turn the tau=1 data into the figures/tables the report and meeting need. Consumes Agent A's JSON; does NOT run training or edit `src/`.

**Tasks:**
1. **tau ablation figure:** DRO vs Naive DP across α, one panel per fixed tau (1/10/100), showing DRO wins at tau=1 and loses at tau=100. Headline figure.
2. **Adult win-curve:** DP(Naive)−DP(DRO) vs α at tau=1 for dp/combined/if attacks (the "advantage grows with α" story).
3. **k-NN ablation table:** IF & DP by k∈{5,10,15} × dataset — show insensitivity to k.
4. **λ_init×lr heatmap:** DP (and acc) over the grid at tau=1 — identify best config.
5. **Wilcoxon at n=6** (once Agent A delivers 6 seeds): regenerate `results/fairness_pgd_wilcoxon.csv` equivalent for the tau=1 data; report which cells reach p<0.05.
6. **Random-vs-adversarial** figure using absolute DP values (Kuldeep/Madam preference, not %): `results/random_vs_adversarial_new.json` already shows adversarial ≫ random (3–31× on Adult/Credit) — make it a clean bar chart.
7. Use Computer Modern fonts, error bars, no shading (the user has strong figure-quality preferences — high-end, authentic, not "AI-generated looking").

**Acceptance:** all figures in `figures/` (PDF+PNG), summary CSVs in `results/`, each regenerable from a script. Pushed.

**Depends on:** Agent A (data). Can start NOW on Adult (tau ablation, k-NN, grid already have data).

---

## §5. AGENT D — Report / paper / repo polish (owns `report/`, `paper/`, `docs/`, top-level `*.md`)

**Mission:** rewrite the results narrative around tau=1 and make the repo submission-clean. Consumes Agent C's figures/tables; does NOT touch `src/` or `results/`.

**Tasks:**
1. **Rewrite results narrative:** old `report/report.tex` + `paper/sections/results.tex` are full of hardcoded OLD (tau=100) numbers showing DRO losing. Replace with the tau=1 story: "DRO consistently reduces DP under adversarial attack once temperature is fixed; advantage grows with α." Pull every number from Agent C's regenerated CSVs — no hand-typed stats.
2. **Add the ablation sections:** tau ablation (why tau=1), IF k-NN ablation (k-invariance), λ/lr grid (Q1), random-vs-adversarial (attack ≫ noise), empirical-radii note (Q5).
3. **LSAC framing (Q3):** present LSAC via the IF attack (it has inherent low DP, so DP attack can't raise it — not a bug).
4. **Verify the PDF builds** with no hyperlink boxes (`\usepackage[hidelinks]{hyperref}`), `\usepackage{float}`. Paste build output.
5. **Repo polish:** ensure top-level has only README, HANDOFF, ADULT_RESULTS_FOR_KULDEEP, SERVER_RUNBOOK, MASTER_PLAN, MEETING_TODAY. Update README to point to HANDOFF as the entry point. Confirm `docs/_archive/` holds the rest.

**Acceptance:** `report/` PDF builds clean and every number traces to a CSV; repo tidy; pushed.

**Depends on:** Agent C (figures/tables). Can start the prose scaffolding now.

---

## §6. UTKFace (Task 2 from Madam, May 19) — BLOCKED, assign opportunistically

Needs the flair2 GPU (email supin.gopi for the account; previous attempts hit SSL/connection issues — see `SERVER_RUNBOOK.md`). Once access works: run the same fixed-attack tau=1 experiment on UTKFace (binary protected attr = gender; the trainer is binary-only — do NOT pass 5-class race). Lower priority than the tabular completion; give to whichever agent frees up, or to you to chase access.

---

## §7. SUGGESTED ASSIGNMENT (3–4 agents)
- **Agent 1 = A** (experiments/data) — start immediately, owns the machine's runs.
- **Agent 2 = B** (src code + Q5 radii + bug fixes) — start immediately, independent.
- **Agent 3 = C** (figures/analysis) — start on Adult now, expand as A delivers.
- **Agent 4 = D** (report/paper/cleanup) — scaffold now, finalize after C.
- UTKFace (§6) → whoever frees up / you chase GPU access.

**Critical ordering:** B finishes `src/` → A runs canonical n=6 set → C regenerates figures/CSVs → D writes report. Until B is done, A's in-flight tau=1 (n=3) runs are fine for the meeting; just don't treat them as final-canonical.
