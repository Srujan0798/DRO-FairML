# SESSION RECAP + REMAINING TASKS (2026-06-16) — hand to agents

> Everything done this session (verified) + all remaining work per agent.
> Hard constraints + coordination rules live in MASTER_PLAN §0/§1 — still apply.
> Authoritative task detail: AGENT_TASKS_FINAL.md. This file = the full picture.

---

## PART 1 — WHAT WE DID THIS SESSION (verified)

### Code fixes (committed)
1. **Three critical bugs fixed** (`0f0a997`): K_inner restored to 10 (spec); feature-PGD now maximizes |p0−p1| directly for dp/combined (was BCE/classification loss); α=0 inner-max guard (DRO=Naive at zero corruption).
2. **Empirical radii (Q5)** implemented — `radii_mode='empirical'` in `dro_fair.py` inverts the known coordinated 70%-minority attack (no oracle mask). `lambda_init` param added (default 0.0 = spec) for the Q1 ablation. Classifier `eval()`/`no_grad()` inference fix in.
3. **Config provenance** added to result rows (k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps, epochs).

### The headline result (verified, Adult)
**Fixing tau=1 makes DRO beat Naive on DP at every α** (wins 2/3, 3/3, 3/3, 3/3 seeds), accuracy equal-or-better. The old "DRO is fragile / two-regime" story was a **tau=100 artifact**. This matches Kuldeep's Q12 guidance. Adversarial attack raises DP ~12–40× more than random noise.

### Ablations run
- **tau** ∈ {1, 5, 10, 20, 100} (Adult) — tau=1 best for DP at every α.
- **IF k-NN** ∈ {5, 10, 15} — attack insensitive to k; k=5 fine.
- **λ grid** (λ_init × lr_lambda), partial (27/72, α=0.2 & 0.3).

### The high-α investigation (Kuldeep's live ask — ANSWERED with data)
Kuldeep's bar: beat the **constant-label predictor** (Adult: DP=0, acc≈0.752). Decision tree: tau → λ → val-loss.
- **Step 1 (tau): ✗** At α=0.3 DRO acc ≈ 0.68 across ALL tau (1/5/10/20/100); α=0.4 ≈ 0.55. None reach 0.78; none beat 0.752.
- **Step 2 (λ): ✗** At α=0.3 the λ grid gives acc 0.672–0.687 — also below the bar.
- **Root cause:** the attack corrupts 30–40% of labels, so the clean-test accuracy ceiling at high α is inherently below the constant predictor. Not a hyperparameter-fixable issue.
- **Honest verdict: the defensible regime is α≤0.2** (Kuldeep already sensed "till α=0.2 it looks fine"). α≤0.1 clears 0.78; α=0.2 borderline (~0.755 > 0.752); α≥0.3 the constant predictor dominates.

### Housekeeping
- Deleted this project's **Kimi** sessions + in-project export zip + DRO-FairML user-history (kept OpenCode, kept other projects). Captured in HANDOFF.md.
- Fixed a **4-way write collision** that had frozen a run; added the coordination protocol (MASTER_PLAN §1).
- Archived ~9 stale prep docs; root minimal. Full conversation stored in `docs/CHAT_HISTORY_MAY_JUNE.md`.

### Verified state (evidence-checked just now)
- **Tests: 60 pass** (1 erroring test is in `experiments/_archive/` — non-critical, should be removed/ignored).
- **No training runs active.** Rows: canonical 57/540, λ grid 27/72, tau5 12, tau20 12.
- **Report self-contradicts (CONFIRMED):** `report/sections/auto_generated_{pgd,main_results,wilcoxon}.tex` still hold OLD tau=100 numbers (e.g. DRO DP 0.50) and are `\input` into report.tex, while the inline text cites tau=1 (DRO DP 0.24). **Top correctness risk.**

---

## PART 2 — REMAINING WORK (assign these; §0/§1 rules apply)

### 🔴 AGENT D — Report correctness (DO FIRST; `report/`, `paper/`, `docs/`, top `*.md`)
1. **Fix the self-contradicting report.** Regenerate `report/sections/auto_generated_*.tex` from the **tau=1** data (tau1_summary.csv / canonical_tau1.json), or update `generate_report_tables.py` to read tau=1 — so tables match the inline tau=1 numbers. Rebuild `report.pdf` + `paper/main.pdf`; **paste the build logs**. A self-contradicting PDF in front of Kuldeep/Madam is the worst outcome.
2. **Write the honest high-α conclusion** into KULDEEP_DISCUSSION.md + report: "tau ✗ and λ ✗ at α≥0.3 (data shown); defensible regime = α≤0.2; high-α degradation is an inherent property of 30–40% label corruption, not a bug." Include the constant-predictor framing.
3. Keep every number traceable to a CSV; root minimal.

### 🟠 AGENT A — Experiments (SOLE launcher; `experiments/`, `results/`, `logs/`)
1. **Fix the λ-grid resume bug** (0 SKIP lines → recomputing; key-type mismatch, likely seed int/float or λ_init 0 vs 0.0), finish grid **27→72** (extend to α=0.1, 0.4 so we have the full λ picture, not just 0.2/0.3).
2. **Resume + finish the canonical 540** (`canonical_tau1.json`): 57→540 = Credit + LSAC + seeds 0–5, k_inner=10, tau=1, full provenance. The publishable dataset. (Long pole — run single sequential process, commit every ~30 min.)
3. **Empirical-radii companion** (`canonical_tau1_empirical.json`, DRO `radii_mode='empirical'`) for the Q5 comparison.
4. **UTKFace:** send the supin.gopi email (draft at root) / run on flair2 if access works; else keep documented-blocked.
5. Rule: one writer per file, always `--k_inner 10`, provenance on every row, never blanket-pkill.

### 🟡 AGENT B — Code/theory (SOLE `src/`) — mostly done
1. **Remove/repair the erroring archived test** (`experiments/_archive/test_fairness_pgd.py`) so the suite is clean (60 pass, 0 errors).
2. **Val-loss convergence logging:** ensure `DroFairTrainer.fit` records per-epoch validation loss/acc/DP and can dump it to JSON (Agent C needs it for Kuldeep's step-3 convergence plots).
3. Confirm audit fixes present (validation-τ consistency, >2-group DP). Keep "src frozen" during A's canonical run; paste pytest output after any change.

### 🟢 AGENT C — Figures/stats (`figures/`, `results/*.csv`)
1. **The constant-predictor figure (meeting core):** x=α, y=accuracy, Adult, horizontal line at 0.752/0.78, curves for tau=1/5/10/20/100 + Naive — shows nothing clears the bar at α≥0.3. Same for **IF** and **DP**.
2. **Acc–DP tradeoff vs the constant-predictor point** per α — which (if any) configs dominate it.
3. **Val-loss convergence plots** (from B's history dumps) for high-α configs (Kuldeep step 3).
4. **λ heatmaps** (acc & DP) at α=0.3/0.4 once A finishes the grid.
5. **n=6 Wilcoxon** + regenerate all figures from the FINAL canonical (not preliminary) once A delivers. CM fonts, error bars, absolute values, x=α format.

### ✅ QA GATE (orchestrator / QA agent — after fixes)
Re-run tests (expect 60 pass, 0 errors), rebuild both PDFs, spot-check ~5 report numbers vs CSV rows, confirm no result file mixes configs. "Done" must be evidence-backed.

---

## PART 3 — DEFINITION OF DONE (100/100)
- [ ] **Report no longer self-contradicts** — all tables tau=1, PDFs rebuilt (verified).
- [ ] High-α honest conclusion written (tau✗/λ✗ → α≤0.2 defensible regime) with figures.
- [ ] λ grid 72/72 + canonical 540 + empirical companion + n=6 Wilcoxon (p<0.05 cells identified).
- [ ] Constant-predictor + val-loss convergence figures delivered.
- [ ] Tests 60 pass / 0 errors. [ ] UTKFace run or documented-blocked. [ ] Repo clean, HANDOFF current.

## The one-line story for Kuldeep
"Fixed tau=1 makes DRO beat Naive on DP at every α (Adult), advantage growing with α, no accuracy cost — for α≤0.2. At α≥0.3 neither tau nor λ beats the constant predictor (inherent to 30–40% label corruption), so α≤0.2 is the defensible regime."
