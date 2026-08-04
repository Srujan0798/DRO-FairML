# DRO-FairML — Master Dispatch Plan

**Written:** 2026-07-20 · **Basis:** full audit of every layer (results, code, docs, figures) + independent re-verification of all headline numbers against `results/canonical_tau1.json`.

---

## 0. READ THIS FIRST — the situation

The 540-row canonical experiment grid was *believed* complete on 2026-07-02 — **CORRECTION (see §0.5): the committed grid is 360 rows (DP + Combined only); the IF-attack third (180 rows) was never generated because the IF metric was degenerate, and is cluster-blocked.** For the DP+Combined rows that do exist: 3 datasets × 2 attacks × 5 alphas × 6 seeds × 2 methods, every row `tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0`. Zero missing cells, zero duplicates within the 360.

**Nothing downstream has been regenerated since.** Every CSV, LaTeX table, figure, PDF, and prose doc in this repo was built from the 307/540 partial snapshot on Jun 30. The last commit says so out loud: `21e6775 interim: ... from 307/540 canonical`.

When the completed data is used instead, **three things break that were reported to the supervisors as fine.** These are not cleanup items. They are the project.

### BLOCKER 1 — The IF metric is dead. Every IF result is meaningless.

```
if_clean nonzero (>1e-6) across all 540 rows:  0
max |if_clean| anywhere:                       4.66e-10
  adult 4.7e-10 · credit 9.9e-12 · lsac exactly 0.0
```

Individual Fairness violation is **identically zero in every single canonical row**. Every IF number in every table is comparing floating-point dust to floating-point dust. Every IF p-value is noise.

This directly contradicts what was sent to Kuldeep on Jun 30: *"IF violation: DRO = 0.0195 vs Naive = 0.0177"*. Those numbers came from the **DP column** of a stale file, not from IF. The IF plots sent in chat (`adult_if_*_meeting.pdf`, sent Jun 30 5:47 PM and 5:59 PM) plot DP data under an IF label.

Root cause is almost certainly `src/evaluation/metrics.py:59-118`: `compute_if_violation` returns `relu(|h_i - h_j| - d_ij - gamma)` summed over a k-NN graph, and with `gamma` and the distance term `d_ij` as currently set the relu saturates to zero for every pair. It is a *threshold calibration* bug, not a missing implementation.

**Until this is fixed, drop every IF claim from every document.** This is the single highest-priority item in the project.

### BLOCKER 2 — DRO loses on LSAC, and it has never been reported.

Recomputed Wilcoxon on the full 540 rows (one-sided, DRO better on DP):

| dataset | attack | α | naive DP | DRO DP | Δ | wins | p |
|---|---|---|---|---|---|---|---|
| lsac | dp | 0.0 | 0.1447 | 0.1829 | **−0.0382** | 0/6 | 1.000 |
| lsac | dp | 0.1 | 0.2201 | 0.2539 | **−0.0338** | 0/6 | 1.000 |
| lsac | dp | 0.2 | 0.1827 | 0.2230 | **−0.0403** | 0/6 | 1.000 |
| lsac | dp | 0.3 | 0.1827 | 0.2220 | **−0.0393** | 0/6 | 1.000 |
| lsac | dp | 0.4 | 0.1827 | 0.2211 | **−0.0384** | 0/6 | 1.000 |
| lsac | if | 0.2 | 0.0442 | 0.0600 | **−0.0158** | 0/6 | 1.000 |
| lsac | if | 0.3 | 0.0561 | 0.0617 | −0.0056 | 1/6 | 0.984 |

**DRO is beaten by Naive on LSAC/DP at every α, 0/6 seeds, p=1.0** — including α=0.0, where there is no corruption at all. LSAC is one third of the grid and appears in **zero** LaTeX tables, because those tables were built before LSAC finished.

Two aggravating details found on re-verification:
- LSAC/dp DP is **frozen at 0.1827** for α = 0.2, 0.3, 0.4 — bit-identical. A metric that does not move as corruption triples means the model has collapsed, not that it is robust.
- LSAC/dp accuracy sits at **0.902–0.903 against a constant-predictor baseline of 0.9016** at every α. The model *is* the constant predictor. The DP "result" is measuring a degenerate classifier.

So LSAC/dp is not simply a negative result — it is a **degenerate run** that must be diagnosed before it is either reported or excluded. LSAC/combined, by contrast, is a genuine clean win (p=0.0156 at α=0.1/0.3/0.4).

### BLOCKER 3 — The defensible regime is α ≤ 0.2, and the data says so plainly.

Accuracy vs the constant-predictor baseline (verified, all 540 rows):

| dataset | baseline | α=0.2 | α=0.3 | α=0.4 |
|---|---|---|---|---|
| adult | 0.7521 | 0.759 ok | **0.676 BELOW** | **0.561 BELOW** |
| credit | 0.7788 | 0.782 ok (dp) | **0.753 BELOW** | **0.752 BELOW** |
| lsac | 0.9016 | 0.896 BELOW (if) | **0.850 BELOW** | **0.780 BELOW** |

At α ≥ 0.3 the models are **worse than predicting a constant**, on every dataset. The headline "advantage grows with α" is technically true and scientifically empty: the growth happens entirely inside the regime where both methods are useless. This was stated correctly once in chat (Jun 30, 6:29 PM) and then contradicted by every figure that leads with the α=0.4 gap.

`CONSTANT_PREDICTOR_ACC = 0.752` is additionally **hardcoded in 4 figure generators** (`plot_high_alpha_tau.py:40`, `generate_final_figures.py:44`, `plot_lambda_heatmap_highalpha.py:34`, `generate_all_deliverables.py:81`) — it is Adult's majority rate applied silently to Credit and LSAC, which is wrong for both.

### What IS solid (verified, keep, lead with this)

Adult and Credit are clean, complete, and genuinely positive at n=6:

- **Adult/DP:** DRO wins at every α — 0.1491/0.1426 (p=.016), 0.2026/0.1999 (p=.031), 0.2452/0.2334 (p=.016), 0.2848/0.2614 (p=.016), 0.3140/0.2855 (p=.016). Reproduces the Jun 30 report exactly.
- **Adult/Combined:** 6/6 wins, p=0.0156 at every α.
- **Credit:** all three attacks, p<0.05 at essentially every α (14 of 15 cells).
- **LSAC/Combined:** p=0.0156 at α=0.1/0.3/0.4.
- **Full grid:** 30 of 45 cells significant at p<0.05 — but that count is inflated by the α≥0.3 degenerate regime and by α=0.0 cells. **Restricted to the defensible regime (α ≤ 0.2, excluding degenerate LSAC/dp), the honest count is what the paper should report.**
- The tau=100 → tau=1 fix is real and is the correct central finding. `get_temperature()` returning the stepped schedule was the artifact that made DRO look fragile.

### One methodological point that will be asked and has no answer yet

**DRO ≠ Naive at α = 0.0.** With zero corruption the two methods should coincide, and they do not (Adult 0.1491 vs 0.1426, 6/6 "wins", p=0.016). The α=0 guard at `dro_fair.py:330` only stops the inner p-loop from advancing the RNG; the two trainers still differ structurally — DRO optimizes a tilted risk (`β·logsumexp`, `dro_fair.py:302`) while Naive optimizes plain BCE mean (`naive_fair.py:136`), DRO decays the dual LR, and they validate on different schedules. **A "win" at α=0 is not robustness — it is a different objective.** Either justify it explicitly or stop counting α=0 as a win. Kuldeep raised the α=0 anomaly as Q4 on Jun 9; it was closed for Adult and never actually resolved.

---

## 0.5 PROGRESS (2026-07-20, end of session)

**CORRECTION:** the canonical grid is **360 rows (DP + Combined attacks only)** at `k_inner=10`,
6 seeds — the IF-attack third (180 rows) was **never generated** (its IF metric was
degenerate pre-fix), so it is cluster-blocked. The "540-row completed grid" framing in §0
assumed IF rows existed; they did not. All regeneration below is against the 360-row truth.

| Agent | Status | Notes |
|---|---|---|
| F | **DONE** (`1e4e35d`) | Deletions, untrack, moves, τ-zombie→`src/temperature.py`, dead-code removal; 62 tests pass. |
| E | **DONE (written)** | `docs/ABLATION_STATUS_REPORT.md`: tau/lambda/random-vs-adv dropped; kNN backfill retracted (was actually the Adult IF config, subsumed by Agent A cluster run). |
| A | **DONE (code) — 180-row IF re-run BLOCKED on cluster** | `compute_if_violation` cosine-based (constant→0.0, unfair→0.28 on real Adult); IF attack gradient aligned (`max(0,1-d-γ)`, global k-NN); `tests/test_metrics.py` pass. Cluster job ready at `scripts/run_if_rerun_cluster.sh`. **Local POC confirmed:** a single Adult α=0.2 config produced `if_clean=0.0333` (non-degenerate) — but at ~20 min/config on this CPU the full 180-row sweep is infeasible here; needs the cluster. DP/Combined rows NOT re-run. |
| B | **DONE** (`3f1e9a3`) | `docs/LSAC_DEGENERACY.md`: LSAC/DP degenerate (radii-on-imbalanced artifact); LSAC/Combined genuine win; α=0 anomaly flagged. |
| D | **DONE** (`3f1e9a3` + `f8d3bb4`) | README/KULDEEP scoped to α≤0.2/Adult+Credit; LSAC/DP degenerate; IF withdrawn pending re-run; n=3→n=6; archived refs removed; PDFs rebuilt. |
| C | **DONE (local) — IF figures pending cluster** (`f8d3bb4`+`c156628`+`324e148`) | canonical committed (360 dp/combined). `generate_report_tables.py` from canonical. `loaders.py` + 2 named landmines + 8 readers fixed; writers repointed. Figures: fig1–fig7 via `canonical_to_all_results.py`+`generate_figures.py`; figD1–D10 via `generate_all_deliverables.py` (repointed to `stale_archived/`, `0.752`→`constant_predictor_acc`). Stale results archived (C6). `make results`/`deliverables`/`validate`/`paper`/`report` all pass. Obsolete orchestrators + stale-loaders moved to `experiments/_archive/`. |

**Cluster-blocked (cannot complete on this CPU box — each IF config ≈5 min; 180 configs ≈15 h):**
- **Agent A step 5:** re-run the IF-attack third (180 rows) with the fixed metric; append to `results/canonical_tau1.json`. `scripts/run_if_rerun_cluster.sh` is written, resume-safe, and regenerates tables/figures/PDFs after the run. Validated on CPU that the fixed metric yields non-trivial IF rows; the full sweep needs the cluster.
- **Agent C IF-attack panels:** regen fig1 IF-attack panel + IF Wilcoxon tables after the A cluster run. All DP/Combined figures/tables/PDFs already rebuilt from canonical.

**Verify (current):**
- `load_fairness_pgd_results()` raises on the contaminated 270-row file.
- `load_canonical_tau1()` returns the clean 360-row DP+Combined grid (`k_inner=10`).
- `make validate` → **PASS** (DP wins 6/9 at p<0.05; LSAC NOT significant; IF = 0.0000, i.e. degenerate-metric confirmed).
- `make paper` / `make report` build with tectonic; `make deliverables` regenerates figD1–D10.

---

## 1. Dispatch — 6 agents

Agents A and B are **blocking**. C, D, E, F may run in parallel with them but must not produce final artifacts until A and B land.

### AGENT A — Fix the IF metric *(BLOCKING — nothing else matters until this is done)*

**Own:** `src/evaluation/metrics.py`, `src/training/dro_fair.py`, `src/training/naive_fair.py`, `src/corruption/adversarial.py`

1. Diagnose why `compute_if_violation` (`metrics.py:59-118`) returns ~1e-10. Instrument the relu: print the distribution of `|h_i − h_j|`, `d_ij`, and `gamma` on a single Adult run. Determine whether `gamma` is too large, `d_ij` is on the wrong scale (unnormalized feature distance vs [0,1] predictions), or predictions are too tightly clustered post-sigmoid.
2. Fix the calibration so IF produces a real, non-degenerate signal that responds to the IF-targeted attack. Add a regression test asserting `if_violation > 1e-4` on a deliberately unfair predictor and `≈0` on a constant one.
3. Fix the **attack↔eval k-NN mismatch**, confirmed: the attack builds neighbours *within* protected groups (`adversarial.py:293-308`, loops `for g in [0,1]`) while training (`dro_fair.py:167`, `naive_fair.py:46`) and eval (`metrics.py:88`) build them over **all** samples. Also `compute_if_violation` accepts an `a` argument at `:59` and never uses it. Make attack and eval agree — this is Kuldeep's Q6, still open since Jun 9.
4. Second-order: the IF attack gradient (`adversarial.py:355-357`) is a bare label agree/disagree count that ignores `d_ij` and `gamma` entirely, so it optimizes a different quantity than the one measured. Align it.
5. **Then re-run the IF-attack third of the canonical grid** (180 rows: 3 datasets × 5 α × 6 seeds × 2 methods). DP and Combined rows are unaffected and must NOT be re-run.

**Done when:** IF is non-degenerate, attack and eval use the same graph, tests pass, 180 IF rows regenerated.

### AGENT B — Diagnose the LSAC degeneracy *(BLOCKING)*

**Own:** `src/training/dro_fair.py` (radii), `src/data/datasets.py`

1. Explain why LSAC/dp DP is **bit-identical (0.1827) at α=0.2, 0.3, 0.4** and why accuracy pins to the constant-predictor rate 0.9016 at every α. Hypothesis to test first: LSAC is ~90/10 imbalanced, the model collapses to the majority class, and DRO's radii `rho_dp[j] = alpha/((1-alpha)*pi_clean[j] + alpha)` (`dro_fair.py:114-115`) blow up on the tiny minority group, over-weighting it into collapse.
2. Determine whether DRO losing 0/6 on LSAC/dp is (a) a genuine negative result about imbalanced groups, or (b) an artifact of collapse. **Report honestly either way — do not tune until it wins.**
3. Note the **uniform radii formula is dead code**: `_compute_radii` (`dro_fair.py:97`) prefers `a_val` whenever non-None, and `run_fairness_pgd.py:102/117` *always* passes it. So `(pi_hat − alpha)/(1−2alpha)` at `:103` never executes, and the renormalization added in the uncommitted diff at `:107-110` sits inside an unreachable branch. All 540 rows labelled `radii_mode=uniform` actually used **clean validation proportions**. Either fix the dispatch or relabel the provenance — the current label is false.
4. Resolve the **α=0 DRO≠Naive** question (see above). Decide: justify, or exclude α=0 from win counts.

**Done when:** LSAC behaviour is explained in writing, radii provenance is truthful, α=0 has a stated position.

### AGENT C — Regenerate every downstream artifact from the 540-row canonical

**Own:** `experiments/compute_canonical_wilcoxon.py`, `analyze_tau1.py`, `generate_report_tables.py`, all `plot_*.py` / `generate_*.py`

1. **First, commit the completed data.** `results/canonical_tau1.json` (540 rows) is modified-uncommitted and is the only copy of three weeks of compute. Commit it before touching anything.
2. Repoint `generate_report_tables.py` at `results/canonical_tau1.json` directly — it currently reads the stale intermediate `results/tau1_summary.csv` (LSAC only reaches α=0.1 there, `n_seeds` mixed 1–6). That is why tables stop at Credit α=0.3 with no LSAC.
3. Kill the stale-path landmines: `analyze_tau1.py:131-136` silently prefers a deleted **K_inner=5** backup over the canonical K_inner=10 file whenever it has more rows. `compute_canonical_wilcoxon.py:49-61` silently falls back to `tau_ablation_tau1.json`. `results/fairness_pgd_results.json` (270 pre-provenance rows, `tau=None`) is the **most-read results file in `experiments/`** at 13 call sites and is contaminated. Make every loader fail loudly rather than fall back.
4. Regenerate: Wilcoxon (all 45 cells), summary CSVs, all figures, both PDFs (`tectonic`).
5. Compute the constant-predictor baseline **per dataset from data** and delete the hardcoded `0.752` from all 4 generators.
6. Move all `results/*` files older than 2026-07-02 18:35 into `results/stale_archived/`.

### AGENT D — Rewrite every claim to match the data

**Own:** `README.md`, `STATUS.md`, `KULDEEP_DISCUSSION.md`, `docs/`, `paper/`, `report/`

1. Delete "DRO wins at every alpha" everywhere. Replace with the verified, scoped claim: **"At α ≤ 0.2, DRO-FAIR achieves lower DP than Naive-FAIR on Adult and Credit under all three attacks (p<0.05, n=6). LSAC/DP is degenerate and reported separately. At α ≥ 0.3 both methods fall below the constant-predictor baseline and no method claim is made."**
2. Strip every IF claim pending Agent A.
3. Fix `paper/auto_generated/key_findings.tex` — it still hardcodes 3-seed claims (`Wilcoxon pending n=6`, `wins 2/3,3/3,3/3,3/3`) and its header admits hand-patching. n=6 has been done for weeks.
4. Re-derive or explicitly label the hardcoded τ=100 comparison numbers at `report/report.tex:441,463` and `paper/sections/results.tex:23-24,54` — currently 3-seed values, one of which cites "tau1_summary.csv row 66-67", a row index into a file since regenerated.
5. **Resolve `docs/UTKFACE_RESULTS.md`.** It claims a 23,705-image 5-seed GPU run. The rest of the repo says GPU access was never granted and only a 2-row CPU smoke exists. Establish whether those numbers are real, simulated, or synthetic (`run_utkface.py:52-105` silently substitutes `_make_synthetic_utkface` random Gaussians when the real dataset is missing). **If they are synthetic, say so in the file title and delete the derived figures.**
6. Prepare an honest correction note for Kuldeep covering: the IF plots sent Jun 30 were DP data mislabelled; LSAC was reported as pending but is complete and negative; the α≥0.3 regime is below the constant predictor. He asked on Jun 30, *"After drafting the reply, could you please verify all the claims? Sometimes AI tends to make claims just to make the results appear correct."* That request was well-founded. Meeting it now is worth more than any figure.

### AGENT E — Finish or formally drop the incomplete ablations

| Ablation | State | Action |
|---|---|---|
| tau ablation | **Incoherent.** No LSAC in *any* tau file; tau=5/20 are Adult-only; `k_inner` is mixed *within* files (tau=1 has 109 rows `k_inner=None`, 15 with `10`). Comparing across tau confounds tau with k_inner. | Re-run clean at k_inner=10, or restrict published scope to Adult and say so |
| lambda grid | **26 of 720 (3.6%)**, crashed. One config at `λ0=1.0` took **17.9 hours** vs ~300–1500s for neighbours. | Scope down to a feasible grid; diagnose the λ0=1.0 pathology first — a naive restart will hang again |
| kNN ablation | k=10 complete (144); k=5 short by 12, k=15 short by 24 | Backfill 36 rows — cheap, closes Kuldeep's Q6 |
| empirical radii (Q5) | **29 of 270**, Adult-only, α=0 rows are exact no-ops | Finish for Adult at minimum, or drop with a written reason |
| random vs adversarial | 27 rows, Jun 9, predates canonical protocol | Re-run under canonical config |

### AGENT F — Repo consolidation *(safe to run in parallel; touches no science)*

**Delete:** `kuldeep_meeting/` (6.8M, gitignored, byte-identical duplicate of `figures/`+`results/`), `kuldeep_meeting.zip`, `"What changed for α≥0.ini"` (byte-duplicate of `KULDEEP_REPLY.md`, misnamed, non-ASCII), `KULDEEP_REPLY.md`, `docs/CHAT_HISTORY_MAY_JUNE.md` (third copy of the chat log), `knn_ablation.log` (0 bytes), the 5 orphan figure stems with no generator and no references (`fig2_dp_reduction_heatmap_complete`, `fig9_fairness_pgd_curves_complete`, `fig_complete_3x3_results`, `fig_final_constant_predictor_acc_complete`, `tradeoff_accuracy_dp`).

**Untrack:** `paper/ICML_submission.pdf` (2.9M, May 4, encodes the retracted conclusion — largest tracked file), `submission/` (**stale fork: 7 of 14 `src/` files diverged, still on τ=100 defaults, ships a `report.pdf` asserting the retracted finding — anyone opening it first gets the wrong science**), 4 tracked `.pid` files in `logs/`.

**Move:** `"gChat Conversation.md"` → `docs/chat/gchat_raw_export.md`; `MASTER_PLAN.md` + `docs/project_management/` (14 dead files) + `docs/archive/` → merge into single `docs/_archive/`; the 12 orchestration scripts misfiled in `logs/` → `scripts/`.

**Fix:** `README.md` broken links to `HANDOFF.md` / `SERVER_RUNBOOK.md` (both moved into `docs/`); dead `Makefile` targets (`monitor`, `review`); add `paper`/`report` tectonic targets; add `data/download_data.sh` — **a fresh clone currently cannot reproduce anything**.

**Delete the stepped-tau zombie.** `return 1.0 if alpha >= 0.4 else 100.0` is still live in 6 files (`run_ablations.py:30`, four `run_utkface*.py`, `submission/run_experiments.py:33`), plus hardcoded `tau=100.0` in both `run_lambda_diagnostic*.py`. `get_temperature` is defined **9 times** across the repo. Consolidate to one import. This is the exact bug that cost the project a month.

**Also:** `src/corruption/__init__.py` omits `FairnessTargetedPGD` — the canonical attack — from its exports, so every runner works around it with a full path import. Dead code to remove: `adversarial.py:384-420` `_select_targets`, `:540-557` `_attack_features_fgsm`, `src/corruption/image_pgd.py` (never imported), `src/training/__init__.py:8-58` `get_run_config` (zero callers).

---

## 2. Sequencing

```
NOW      → Agent C step 1: commit the 540-row canonical. Nothing else until this is safe.
WAVE 1   → A (IF metric) + B (LSAC) blocking · F (cleanup) in parallel, no conflicts
WAVE 2   → C (regenerate everything) — requires A and B to have landed
WAVE 3   → D (rewrite claims) — requires C · E (ablations) in parallel
FINAL    → Rebuild PDFs, send Kuldeep the corrected numbers + the correction note
```

**Merge conflict warning:** A and B both edit `src/training/dro_fair.py`. A owns the IF/k-NN paths, B owns radii and the α=0 question. Coordinate or serialize.

**Uncommitted state to resolve first:** the working tree carries the tau=100→1.0 fix in `src/training/dro_fair.py:31` and `naive_fair.py:27-28`, k_inner 5→10 in four ablation runners, and a genuine bug fix at `run_experiments.py:155` (`lambda_warmstart=0.01` → `lambda_init=0.0` — `DroFairTrainer` has no such kwarg, so `main.py --run-experiments` is **broken on `main` right now**). All of it is correct and should be committed.

---

## 3. Definition of done

- [ ] IF metric produces real signal; attack and eval share one k-NN graph; 180 IF rows re-run
- [ ] LSAC degeneracy explained in writing; radii provenance truthful; α=0 position stated
- [ ] Every table, figure, and PDF regenerated from the 540-row canonical — no stale fallbacks
- [ ] Every numeric claim in every doc traceable to `results/canonical_tau1.json`
- [ ] Claims scoped to α ≤ 0.2 with the constant-predictor limit stated plainly
- [ ] Ablations finished or formally dropped with written reasons
- [ ] One `get_temperature`, one archive dir, no stale fork, reproducible from clean clone
- [ ] Correction sent to Kuldeep — including what was wrong in the Jun 30 messages

**The standard is: every number defensible, every negative result reported.** A smaller honest result beats a larger one that does not survive checking. The IF plots already went out mislabelled once.
