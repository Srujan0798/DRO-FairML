# Paper / Report Finalization Checklist (Agent K)

> **CURRENT (2026-08-04 post-540):** Grid is **540/540 complete**. IF is **MIXED** (not a clean sweep).  
> Adult/DP α=0.1 is **5/6**. LSAC/DP degenerate. Paper/report prose largely scrubbed of  
> “pending/cluster/360” IF language — remaining “pending” is only UTKFace full grid + Q5 empirical JSON.  
> **Do not claim UTKFace** until multi-attack REAL cells exist.  
> Sections below retain PREP inventory as an audit trail; treat open IF-pending items as **mostly closed**.

**Date:** 2026-08-04 · **Phase:** POST-H (IF third landed; MIXED narrative)  
**Canonical source of truth:** `results/canonical_tau1.json` (540 rows).  
**IF Wilcoxon:** `results/if_wilcoxon_summary.txt`. Meeting: `docs/MEETING_2026-08-04.md`.

---

## 1. Current narrative structure and what must change when IF completes

### Paper (`paper/main.tex` + `paper/sections/*`)

| Section | Current role | Change after IF (Agent H) |
|---|---|---|
| **Abstract** (`main.tex` L31–32) | FairnessTargetedPGD + τ=1 flips Adult story; DP+Combined wins; UTKFace withdrawn. Claims DRO beats Naive **“at every α”** with **advantage growing in α**. | Scope to **α ≤ 0.2** (defensible regime). Add **one honest sentence** on real IF (win/loss/mixed per dataset). Drop or qualify “every α / growing”. |
| **Intro** (`introduction.tex`) | Gap = gradient fairness attacks; contributions include k-NN ablation; **“IF-attack results are pending a cluster re-run”** (L15–16). | Replace pending with real IF contribution bullet; keep τ=1 as central finding. |
| **Attack design** | DP / IF / Combined PGD formulas; k=5. | No structural change; may note IF attack was evaluated end-to-end. |
| **Setup** | Adult/Credit/LSAC; UTKFace blocked; n=6; τ=1. | If Agent M lands real UTKFace: update dataset table; else keep honest “no real run”. |
| **Results** (`results.tex`) | Headline DP+Combined Adult/Credit α≤0.2; τ ablation table; adversarial≫random; **IF pending** throughout; LSAC/DP degenerate; UTKFace withdrawn. | (a) New IF subsection with numbers from H; (b) **input** auto tables (currently unused); (c) fix 3-seed win fractions; (d) Q7 coupling: fill or keep qualitative if mixed. |
| **Discussion** | τ mechanism; adversarial vs random; empirical radii; UTKFace future. | Add IF mechanism / where IF diverges from DP (esp. if LSAC/IF or Adult/IF mixed as partial data suggested). |
| **Related work** | 1 short paragraph. | Optional polish; not blocked on IF. |
| **Conclusion** | Mirrors results; **IF pending**; UTKFace withdrawn. | Same IF sentence as abstract; 360→**540** rows when true. |
| **Appendix files** (`appendix_q1_lambda.tex`, `appendix_q5_empirical.tex`) | Exist on disk **but are not `\input` in `main.tex`**. | Either wire into appendix or leave out of submission; do not claim they are in the paper if unused. |

### Report (`report/report.tex`)

Longer ICML-implementation report: method theorems → setup → **main results (auto-input)** → Wilcoxon → figures → discussion/limitations → theory table → runtime → ablation → Week-2 fairness PGD → conclusion → Q5 appendix.

| Block | Change after IF |
|---|---|
| Abstract L117–118 | Replace “IF-attack results are pending a cluster re-run…” with real IF summary. |
| Setup L329–331 | 360 rows → 540; drop “IF third pending”. |
| Main results + Wilcoxon auto tables | Regen from full grid; IF columns must not be `0.0000 ± 0.0000`. |
| Fig captions (esp. fig7 L412–413) | “IF comparisons pending” → real win rates; fix α∈{0.1,0.2,0.3} framing if defensible is α≤0.2. |
| Discussion § IF k-NN, Q7, limitations L563–570, conclusion L760–770, footer L794 | All pending / 360-row language. |
| Key findings bullets L730–733 | Replace IF-pending bullet with real IF cells (honest losses OK). |

### Fixed narrative (do not spin)

Per Agent K plan + `docs/KULDEEP_CORRECTION.md` + `docs/LSAC_DEGENERACY.md`:

1. FairnessTargetedPGD (DP / IF / Combined).  
2. **τ=1** is the central finding; **τ=100 stepped schedule** was the artifact (show as ablation).  
3. DRO robust on **Adult & Credit at α≤0.2** under attacks with data; n=6, Wilcoxon p<0.05 for DP+Combined today; **extend to IF only if H’s p-values support it**.  
4. **First real IF results** from H — report mixed/negative cells plainly.  
5. Honest negatives: **LSAC/DP degeneracy** (own subsection); **α≥0.3** below constant predictor.  
6. **UTKFace:** no real results unless Agent M completes a real run (never synthetic as real).

---

## 2. Every hardcoded number with file:line (esp. τ=100 / 3-seed leftovers)

### Priority: MASTER_DISPATCH / Agent K flagged lines

| File:line | Issue | Action |
|---|---|---|
| `report/report.tex:441–443` | Prose: α=0.2 stepped-schedule **Naive 0.327, DRO 0.503** (τ=100 artifact) | Keep only if clearly labeled **pre-fix ablation**; do not present as canonical τ=1. Prefer auto or citation to `results/stale_archived` / tau ablation JSON. |
| `report/report.tex:464–466` | Mini-table: τ=1 **0.2480 / 0.2371**, **“DRO wins (3/3)”**; τ=100 **0.3271 / 0.5030** | **3/3 = 3-seed leftover.** Replace win counts with **n=6 (e.g. 6/6)** from canonical Wilcoxon; align τ=1 DP with auto table (**0.2452 / 0.2334** Adult DP α=0.2), not 3-seed tau-ablation means. |
| `paper/sections/results.tex:22–23` | Prose: Naive **0.327147** vs DRO **0.503047** | OK as historical τ=100 example if labeled; cross-check source; round consistently with table. |
| `paper/sections/results.tex:48–62` | Full `tab:tau-comparison`: win columns **2/3, 3/3, 0/3** throughout; τ∈{1,10,100} DP values (e.g. L52–54 **0.2480/0.2371**, **0.3271/0.5030**) | **Critical 3-seed leftover.** Regen from n=6 or mark table “3-seed historical ablation” and **do not** use as main n=6 evidence. τ=1 rows disagree with 6-seed auto tables. |
| `paper/auto_generated/key_findings.tex:6–9` | Claims **“all three attacks”** while IF pending; **360 rows** + pending text | Regen after 540; macros currently **unused** by `main.tex` (dead). |

### Additional hardcoded / stale numeric issues (inventory)

| File:line | Content | Severity |
|---|---|---|
| `paper/main.tex:32` | “at every α” + “advantage growing in α” (conflicts α≥0.3 caveat) | **High** (abstract overclaim) |
| `paper/sections/results.tex:72–75` | Adv vs random: +0.18 vs +0.001; +0.38 vs +0.02; 12–40× | Medium — verify vs `random_vs_adversarial` JSON; keep if traced |
| `paper/sections/results.tex:84` | k-NN DP within ±0.003 | Medium — from knn ablation |
| `paper/sections/results.tex:102–108` | LSAC acc≈0.902, majority 0.9016, 0/6, p=0.0156 | OK if matches canonical (does for DP/Combined) |
| `paper/sections/results.tex:110,127` | **360 rows** | Update → 540 after H |
| `paper/sections/discussion.tex:32` | Adult α=0.4 acc ≈0.55 | OK-ish; prefer auto |
| `paper/sections/conclusion.tex:1` | Full numeric claim block + IF pending | Rewrite after H |
| `paper/sections/appendix_q1_lambda.tex:11,27–30` | “3 distinct initializations”; best DP/acc grid numbers; 0.752 ceiling | Medium — 3 seeds in λ grid by design; not main claim |
| `paper/auto_generated/tabular_results.tex:6–35` | All **IF = 0.000** (pre-metric-fix / no IF attack rows) | **High** after H — must regen |
| `paper/auto_generated/wilcoxon.tex:6–35` | All **ΔIF% = 0.0%** | **High** after H |
| `report/report.tex:109–118` | Abstract numbers + IF pending | High after H |
| `report/report.tex:154` | Effect at α=0.2: random +0.01 vs adv +0.03–0.05 (3–5×) | **Conflicts** results body 12–40× — reconcile |
| `report/report.tex:225–229` | TV radii table (formulaic) | OK if recomputed from formula |
| `report/report.tex:330–331,767,794` | **360 rows** | → 540 |
| `report/report.tex:412–413` | “6/9 DP comparisons at α∈{0.1,0.2,0.3}” | May be stale fig framing vs α≤0.2 policy |
| `report/report.tex:419–424` | Hand bullets: 0.245/0.233, 0.020/0.018, 0.196/0.178, 0.014/0.012 | Prefer cite auto table only |
| `report/report.tex:476–478` | Same adv≫random as paper | Medium |
| `report/report.tex:497` | λ grid prelim DRO DP = 0.237 | 3-seed-ish / partial run |
| `report/report.tex:506` | acc≈0.752 | Majority baseline OK |
| `report/report.tex:636–637` | Runtime 10.6±9.1 vs 397.6±258.5, 37.5× | Hand; verify `runtimes.json` if shipping |
| `report/report.tex:660–665` | Ablation table Acc/DP/**IF** numbers | Likely **pre-τ=1 / old protocol** — do not treat as canonical τ=1 IF |
| `report/report.tex:670–671` | “≈83%”, “≈0.175” prose vs table 0.822 / 0.1034 | Internal inconsistency |
| `report/report.tex:703` | **“270 experiments, 5 seeds”** vs caption “6 seeds” | **Stale header** |
| `report/report.tex:753–754` | Adult α=0.2 0.245 vs 0.233 | OK if matches auto |
| `report/sections/auto_generated_main_results.tex:6–20` | **IF 0.0000 ± 0.0000** every row | **High** after H |
| `report/sections/auto_generated_wilcoxon.tex` | IF Δ 0.0% (DP-attack slice only; no IF-attack rows) | Regen + extend for IF attack |
| `report/sections/auto_generated_pgd.tex` | DP/Combined only; good for those attacks | Add IF attack block after H |

### Count of hardcoded issues found

| Category | Approx. count |
|---|---|
| **3-seed win fractions** (`2/3`,`3/3`,`0/3`) in paper tau table | **12** cells (`results.tex` L48–62) + **1** in report L464 |
| **τ=100 / stepped-schedule DP numbers** (0.327*, 0.503*, etc.) | **≥6** locations (paper L22–23, L54; report L443, L466; related table rows) |
| **τ=1 numbers that mismatch 6-seed canonical** (0.2480/0.2371 vs 0.2452/0.2334) | **≥3** (paper table L52; report L464; prose bullets) |
| **IF degenerate zeros in auto tables** | **30** paper tabular IF cells + **15** report main-results IF cells + Wilcoxon IF columns |
| **“IF pending / cluster re-run” prose** | **~20+** paper+report locations |
| **“360 rows” stale grid size** | **6** tex locations |
| **Abstract / framing overclaims** | abstract “every α”; key_findings “all three attacks”; fig7 α≤0.3; report L703 5 seeds |
| **Orphan / unused auto artifacts** | `paper/auto_generated/*` not `\input` by `main.tex`; key_findings macros never used |
| **Hand ablation / runtime tables** | report L660–665, L636–637 |

**Conservative total of distinct hardcoded / stale issues to clear before Aug 10: ~80+ line-level items** (including every IF=0 cell as one issue class, or **~45 file:line clusters** if grouping table rows).  
**Priority clusters Agent K must clear first: 13** (flagged MASTER_DISPATCH lines + abstract + 360-row + IF-pending blocks + wire/regen auto tables).

---

## 3. Auto-generated vs hand-written table paths

### Auto-generated (from `experiments/generate_report_tables.py` ← `results/canonical_tau1.json`)

| Path | Consumed by? | Status pre-H |
|---|---|---|
| `paper/auto_generated/tabular_results.tex` | **Nowhere** (`main.tex` does not `\input`) | Orphan; IF=0 |
| `paper/auto_generated/wilcoxon.tex` | **Nowhere** | Orphan; IF Δ=0 |
| `paper/auto_generated/key_findings.tex` | **Nowhere** | Orphan macros; “all three attacks” premature |
| `report/sections/auto_generated_main_results.tex` | `report/report.tex:346` | Live; IF zeros |
| `report/sections/auto_generated_wilcoxon.tex` | `report/report.tex:372` | Live; DP-focused |
| `report/sections/auto_generated_pgd.tex` | `report/report.tex:710` | Live; DP+Combined |

**Also related (not always regenerated by the same script):**  
`results/table1_latex.tex`, `results/summary_stats.csv`, Wilcoxon CSVs under `results/` / `results/stale_archived/`.

### Hand-written tables (edit carefully; prefer regen or delete)

| Location | Content |
|---|---|
| `paper/sections/results.tex` L31–65 | `tab:tau-comparison` (τ ablation) |
| `paper/sections/experimental_setup.tex` L2–20 | Dataset sizes |
| `paper/sections/appendix_q1_lambda.tex` L19–32 | λ grid summary |
| `report/report.tex` L139–156 | Adversarial vs random components |
| `report/report.tex` L209–231 | TV radii |
| `report/report.tex` L268–295 | Hyperparameters |
| `report/report.tex` L302–322 | Datasets + baseline DP |
| `report/report.tex` L459–468 | Mini τ table (3-seed issue) |
| `report/report.tex` L584–618 | Theory verification |
| `report/report.tex` L625–638 | Runtime |
| `report/report.tex` L650–666 | Ablation (likely stale protocol) |

**K action:** After H, regenerate all auto paths; **wire paper** to `\input{auto_generated/tabular_results.tex}` and wilcoxon (or embed via generator); eliminate duplicate hand numbers that restate the same cells.

---

## 4. Figure inventory

### Embedded in report only (paper has **zero** `\includegraphics`)

| Figure | Path | Quality / notes | Regen? |
|---|---|---|---|
| Main results | `figures/fig1_main_results.pdf` | Used report L351; Jul 20; style OK | **Yes** after 540 (`generate_figures.py` / `make results`) |
| DP reduction heatmap | `figures/fig2_dp_reduction_heatmap.pdf` | Report L376 | Yes |
| Significance matrix | `figures/fig4_significance_matrix.pdf` | Report L385 | Yes |
| Acc–fairness tradeoff | `figures/fig5_accuracy_fairness_tradeoff.pdf` | Report L400; caption still highlights α=0.3 outlier | Yes + caption |
| Win-rate summary | `figures/fig7_summary_win_rates.pdf` | Report L410; **IF pending** in caption | Yes |
| (Not embedded) seed stability | `figures/fig6_seed_stability.pdf` | Exists | Optional |
| (Not embedded) clean vs corrupted | `figures/fig3_robustness_clean_vs_corrupted.pdf` | Exists | Optional |

### Headline / meeting figures (publication-ish; regenerate for IF)

| Figure | Notes |
|---|---|
| `figures/fig_tau1_headline.pdf` | τ=1 headline |
| `figures/fig_win_curves_tau1.png` / `fig_acc_win_curves_tau1.pdf` | Win curves |
| `figures/fig_final_wilcoxon_table.pdf` / `figD10_final_wilcoxon_table.pdf` | Wilcoxon visual |
| `figures/main_results.pdf` | Alternate main plot |
| `figures/adult_*_if_attack_tau1_meeting.pdf` | **Likely mislabelled / pre-fix IF** per Kuldeep correction — **do not ship until H confirms real IF** |
| `figures/adult_if_tau100_meeting.pdf` | Withdrawn story support only |

### Ablation / diagnostic (keep appendix or archive)

| Set | Generator hint | Notes |
|---|---|---|
| `figC1`–`figC5`, `figC_uniform_vs_emp` | `analyze_tau1.py`, `analyze_lambda_grid.py` | Useful ablation |
| `figD1`–`figD10`, `fig_final_*`, `fig_high_alpha_*` | Agent C manifest Jun 17 | Constant-predictor story; some partial λ grid |
| `fig8_*`, `sensitivity_*`, `summary_dashboard_may29` | Older | Audit before use |
| `figures/stale_archived/` | — | Do not cite |

### Quality verdict

| Tier | Items |
|---|---|
| **Publication-ready after regen + caption fix** | fig1, fig2, fig4, fig5, fig7, fig_tau1_headline, win curves, wilcoxon table figs |
| **Need regen from generators once 540 rows exist** | All of the above + any IF-specific panels; `make results` / `make deliverables` / `generate_figures.py` |
| **Do not use as IF evidence until H** | `adult_if_*_meeting*`, `fig_high_alpha_tau_if`, `figD3` IF lines if sourced from degenerate metric |
| **Paper gap** | Paper PDF has **no figures** — K should add 2–4 key figures for submission quality |

Manifest reference: `figures/FINAL_FIGURES_MANIFEST.txt` (dated 2026-06-17; partially stale relative to Jul 20 fig1–7).

---

## 5. Abstract / intro / related / method / results / discussion / limitations / conclusion coherence

| Issue | Detail |
|---|---|
| **Abstract vs results scope** | Abstract: DRO wins **at every α**, advantage **grows with α**. Body: method claims only **α≤0.2**; α≥0.3 below constant predictor. **Fix abstract.** |
| **Abstract vs Kuldeep correction §3** | “Advantage grows with α” is exactly the framing called empty for α≥0.3. |
| **“All three attacks” premature** | `key_findings.tex` and some headlines imply IF already verified; body says pending. |
| **360 vs 540** | Entire pipeline still talks 360 (DP+Combined only). |
| **Paper never shows main numeric table** | Only hand τ-comparison table; auto tabular_results orphaned → reader cannot see Adult/Credit/LSAC grid in the paper PDF. |
| **Related work too thin** | Single paragraph; methods-heavy paper needs slightly more placement vs Solans / Hashimoto / Madry. |
| **Limitations incomplete in paper** | Report has enumerate (binary A, full-batch, runtime, IF@τ, n=6, UTKFace). Paper discussion lacks a crisp **Limitations** subsection listing LSAC/DP, α≥0.3, IF status, UTKFace. |
| **LSAC story split** | Results correctly mark LSAC/DP degenerate + Combined win; ensure abstract/conclusion always pair them (Combined win without DP degeneracy is spin). |
| **Report internal contradictions** | (1) Adv strength 3–5× (intro table) vs 12–40× (results); (2) § “270 exp, 5 seeds” vs n=6; (3) ablation prose vs ablation table; (4) footer “32 unit tests” vs plan “62 passing”. |
| **UTKFace** | Consistently withdrawn in paper/report — **keep** until M delivers real data. Never upgrade synthetic. |
| **α=0 wins** | LSAC_DEGENERACY open Q: wins at α=0 are objective differences, not robustness — paper does not discuss; optional honesty note. |
| **Cluster-re-run language** | Obsolete: IF now runs **locally** (Agent H / `run_if_parallel.py`). Replace “cluster re-run” with “canonical grid regeneration” after H. |
| **Date / venue** | May 2026 on title; align with Aug 10 submission if needed. |

---

## 6. Exact placeholders to fill after `results/if_wilcoxon_summary.txt` appears

When H writes the summary, fill **only** from that file + regenerated auto tables — never invent.

### Paper

| Location | Current placeholder | Fill with |
|---|---|---|
| `main.tex` abstract | No IF sentence / overclaims every-α | 1 sentence: IF attack result summary (Adult/Credit/LSAC; α≤0.2; p if sig) |
| `introduction.tex:15–16` | “IF-attack results are pending a cluster re-run” | Real IF contribution |
| `results.tex:12–15` | IF verification pending / metric ~1e-10 history | Keep metric-fix history as footnote; **lead with live IF numbers** |
| `results.tex:78–86` | k-NN + pending IF re-run | Optional: confirm k still irrelevant under real IF metric |
| `results.tex:111–112` | IF on LSAC pending | LSAC/IF cells from summary |
| `results.tex:114–120` | Q7 coupling pending / claims withdrawn | Qualitative coupling + numeric if available |
| `results.tex:126–127` | 360 rows; IF third pending | **540 rows**; all three attacks present |
| `conclusion.tex:1` | IF pending clause | Same IF sentence as abstract |
| `auto_generated/*` | Regen entirely | Ensure paper `\input`s them |

### Report

| Location | Placeholder |
|---|---|
| Abstract L117–118 | IF pending |
| L329–331 | 360 rows; IF pending |
| L369–370 | IF third pending |
| L413 | IF comparisons pending re-run |
| L487–489 | IF evaluation pending |
| L510, L521–526 | Q7 pending |
| L563–570 | IF metric / grid pending |
| L730–733 | IF attack is pending |
| L760–761, L769–770 | IF rows pending |
| L783 | Future work IF third |
| L794 | 360 rows footer |
| Auto tables IF columns | Must become non-zero where attack=if and metric healthy |

### Checklist after fill

- [ ] `grep -n pending paper/ report/ --include='*.tex'` → only true future work (e.g. multi-group), not IF grid  
- [ ] No `0.0000 ± 0.0000` IF for IF-attack rows  
- [ ] No “cluster re-run” for IF  
- [ ] Every IF p-value matches `if_wilcoxon_summary.txt`  
- [ ] Mixed/negative cells stated in abstract or limitations if material  

---

## 7. Build status

| Target | Command | Result (2026-08-04 PREP) |
|---|---|---|
| Paper | `make paper` → `tectonic -X compile paper/main.tex` | **FAIL** — tectonic cannot fetch format bundle (`https://relay.fullyjustified.net/default_bundle_v33.tar.index.gz` **timeout**); `could not open format file latex` |
| Report | `make report` → `tectonic -X compile report/report.tex` | **FAIL** — same network/bundle error |
| Fallbacks | `pdflatex` / `latexmk` / TeX Live | **Not installed** on this machine |
| Stale PDFs present | `paper/main.pdf`, `report/report.pdf` | Dated **2026-07-21** — pre-full-IF; do not treat as final |

**No LaTeX content error was reached** — build is infrastructure-blocked, not (yet) document-blocked. After network/cache fix, re-run both; if compile errors appear, fix minimally.

**Suggested unblock:** restore tectonic cache / offline bundle, or install BasicTeX (`pdflatex`) and add a Makefile fallback.

---

## 8. Post-H execution order for Agent K (when numbers exist)

1. Confirm 540 rows + non-degenerate IF; read `results/if_wilcoxon_summary.txt`.  
2. Run table/figure regen (`generate_report_tables.py`, `make results` / deliverables).  
3. Replace all §6 placeholders; fix abstract scope.  
4. Remove 3-seed win fractions; either regen τ table at n=6 or label historical.  
5. Wire paper to auto tables + 2–4 figures.  
6. Limitations subsection in paper.  
7. `make paper && make report`; fix build env if needed.  
8. One-paragraph “what changed” note for submission.

---

## 9. Explicit non-claims (guardrails)

- **Do not invent IF numbers** before `if_wilcoxon_summary.txt`.  
- **Do not claim UTKFace real results** (synthetic smoke only unless Agent M finishes).  
- **Do not** put LSAC/DP in a “DRO wins” win-rate without degeneracy callout (`docs/LSAC_DEGENERACY.md`).  
- **Do not** lead with α≥0.3 gaps (`docs/KULDEEP_CORRECTION.md` §3).

---

*End of PREP checklist. Optional template: `paper/sections/_IF_PLACEHOLDER_NOTES.md`.*
