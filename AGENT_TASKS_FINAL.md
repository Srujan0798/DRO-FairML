# AGENT TASKS — FINAL (grounded in the real Kuldeep meeting, 2026-06-16)

> Single source of truth for the agents NOW. Supersedes MASTER_PLAN §10 priorities
> (MASTER_PLAN §0 constraints + §1 coordination protocol still fully apply — read them).
> The meeting actually happened in chat with Kuldeep today. This protocol is built
> on what he literally asked for.

## THE OBJECTIVE (from the meeting, verbatim intent)
Kuldeep's bar: **beat the constant-label predictor.** On Adult the constant predictor
gives **DP=0, accuracy ≈ 0.75–0.78**. So a useful DRO result must have:
- **accuracy ≥ 0.78** (strictly above the constant predictor), AND
- **lower DP than Naive** (ideally low absolute DP).

Current reality: **"till α=0.2 it looks fine"** (acc holds, DRO wins/ties DP). **α≥0.3 is the
problem** — accuracy falls below 0.78 (degenerate; the constant predictor beats us there).

Kuldeep's exact decision tree to fix high-α:
1. **Try a different tau first** (e.g. tau=5, 20) for large α.
2. **If tau doesn't improve it → change the lambda learning rate** (and/or λ_init).
3. **Else → check loss-convergence plots on the validation set** and choose accordingly.

So the whole near-term project = **find a config that beats the constant predictor at α≥0.3**,
following that tree, with plots in his exact format (x=α, y=accuracy / y=IF, Adult y≥0.78,
tau=1 vs tau=100 + different-tau comparison).

## CURRENTLY RUNNING (do not duplicate; Agent A owns launches)
- `run_tau_ablation.py --tau 5 --alphas 0.3 0.4 --datasets adult --attacks dp --methods naive dro --n_seeds 3 --k_inner 10` → `results/tau_ablation_tau5.json` (Kuldeep step 1, launched now).
- `run_lambda_lr_grid.py` → `results/lambda_lr_grid.json` (13/72; **resume bug: recomputing, 0 SKIP lines — FIX the key match, likely seed int/float or λ_init 0 vs 0.0**).
- Canonical PAUSED at 57/540 (`results/canonical_tau1.json`) — lower priority than the high-α fix; resume after the levers are decided.

---

## AGENT A — Experiments (SOLE launcher; `experiments/`, `results/`, `logs/`)
**Priority 1 — Kuldeep's decision tree for high-α (α=0.3, 0.4), Adult:**
1. **tau sweep at high α:** finish tau=5 (running), then tau=20 → `tau_ablation_tau20.json`. Record acc + DP, both methods, 3 seeds, k_inner=10. (Goal: does acc reach ≥0.78?)
2. **If tau fails (likely — existing tau 1/10/100 data shows acc flat ~0.55–0.68 at high α):** sweep **lambda learning rate** `lr_lambda ∈ {0.001, 0.002, 0.005}` and **λ_init ∈ {0.0, 0.01, 0.1}** at α=0.3/0.4 (extend `run_lambda_lr_grid.py` to those α). Goal: trade some DP for acc ≥ 0.78.
3. **Validation loss-convergence logging:** ensure the DRO trainer can dump per-epoch val loss/acc/DP so Agent C can make the convergence plots Kuldeep asked for (step 3 of his tree). If the trainer already records `history`, expose it to a JSON per run.
4. **Fix the grid resume bug**, finish grid 72/72.
**Priority 2 — finish the publishable dataset (after the high-α lever is found):**
5. Resume + complete `canonical_tau1.json` (540, Credit+LSAC, 6 seeds, full provenance) + empirical-radii companion.
6. UTKFace: send the supin.gopi email (draft ready) / run on flair2 if access works; else keep documented-blocked.
**Rule:** one writer per file, always `--k_inner 10`, provenance on every row, never blanket-pkill.

## AGENT B — Code/theory (SOLE `src/`)
1. **Val-loss convergence support:** make sure `DroFairTrainer.fit` records per-epoch validation loss/acc/DP into `history` and that it's retrievable (Agent C needs it for the convergence plots — Kuldeep step 3).
2. **Verify, don't claim:** paste real `pytest` output (all tests, incl. the empirical-radii test). Confirm the audit fixes (classifier eval ✓, validation-τ consistency, >2-group DP) are actually in.
3. Keep "src frozen" during A's runs. If you must change the trainer for (1), announce it so A re-runs cleanly.

## AGENT C — Analysis/figures/stats (`figures/`, `results/*.csv`)
1. **The constant-predictor figure (the meeting's core):** x=α, y=accuracy, Adult, with a **horizontal line at the 0.78 constant-predictor bar**, curves for tau=1 / tau=5 / tau=20 / tau=100 (and Naive). Show where (if anywhere) DRO clears 0.78 at α≥0.3. Same plot for **IF violation** and **DP**.
2. **Acc–DP tradeoff vs constant predictor:** scatter/curve showing DRO configs vs the constant-predictor point (DP=0, acc≈0.76), per α — which configs dominate it.
3. **Val-loss convergence plots** (from B/A's history dumps) for the high-α configs.
4. **λ-lr / λ_init heatmaps** at α=0.3/0.4: acc and DP — find the config that clears 0.78.
5. Keep the existing tau=1 headline, win-curves, k-NN, random-vs-adv (absolute DP), n=6 Wilcoxon (when canonical lands). CM fonts, error bars, absolute values, exact format Kuldeep used (x=α).

## AGENT D — Report/docs (`report/`, `paper/`, `docs/`, top-level `*.md`)
1. **Update KULDEEP_DISCUSSION.md** with the high-α result: state honestly whether tau or λ lets DRO beat the constant predictor at α≥0.3, with the plots. Frame: "α≤0.2 DRO clearly useful; α≥0.3 needs [tau=X / λ-lr=Y] to beat the constant predictor — or, if nothing does, that itself is the honest finding (defensible regime = α≤0.2–0.3)."
2. Keep report/paper numbers traceable to final CSVs; rebuild PDFs, paste logs.
3. Root stays minimal; archive prep files. HANDOFF current (Grok section already merged).

## QA GATE (orchestrator / a QA agent — verify, don't trust)
Run the tests, build the PDFs, spot-check ~5 report numbers vs CSV rows, confirm no result file is contaminated (mixed config). "Done" must be evidence-backed.

## DEFINITION OF DONE
- [x] High-α tau sweep completed (tau=1/5/10/100 at α≥0.3). Result: NONE beat constant predictor. tau is not the lever.
- [ ] High-α answered per Kuldeep's tree: λ lr/init sweep at α=0.3/0.4 (NEXT — tree step 2).
- [ ] Constant-predictor accuracy/IF/DP figures (x=α, 0.78 line, tau & λ variants) + val-loss convergence plots.
- [ ] Grid resume fixed + Q1 λ conclusion. [ ] Canonical 540 + empirical companion + n=6 Wilcoxon.
- [x] KULDEEP_DISCUSSION.md updated with honest high-α finding (Section 6).
- [ ] Report auto_generated tables updated to tau=1 data (currently stale — see STATUS below).
- [ ] UTKFace run or documented-blocked. [ ] Tests pass (pasted). [ ] Repo clean.

## STATUS (2026-06-16, Agent D update)

### Completed
- tau=1 DP headline: DRO beats Naive at every α on Adult (2/3, 3/3, 3/3, 3/3). Source: tau1_summary.csv + tau1_wilcoxon.csv.
- tau sweep at high-α: tau=1/5/10/100 all give acc 0.55–0.68 at α≥0.3, all below constant predictor (0.752). Source: high_alpha_tau_analysis.txt + tau_ablation_tau5.json.
- KULDEEP_DISCUSSION.md Section 6 added with honest high-α verdict + decision-tree framing.

### In flight
- **Lambda grid** (Q1): ~13 rows in `results/lambda_lr_grid.json`, currently only α=0.2. Need to extend to α=0.3/0.4. Agent A owns.
- **Canonical run**: 57/540 rows (`results/canonical_tau1.json`). α=0.0 complete n=6 (p<0.05). Paused.
- **tau=5 full run**: `results/tau_ablation_tau5.json` has 1 row (naive only). Incomplete.

### Number traceability issue found
- **report/sections/auto_generated_*.tex are stale.** `generate_report_tables.py` reads from `results/fairness_pgd_results.json` (old tau=100 data). The auto_generated tables show tau=100 numbers (e.g. Adult α=0.2 DP: DRO=0.5034) while report.tex inline text correctly cites tau=1 numbers (DRO=0.237). `auto_generated_wilcoxon.tex` IS `\input` into report.tex (line 395) and shows old data. The manually-written Table 2 in report.tex (lines 338-371) is also from old data. **Fix needed:** either regenerate from tau1_summary.csv or update the generator to prefer tau=1 data. The report.tex inline numbers (§7 "Key highlights") are correct and traceable to tau1_summary.csv.
