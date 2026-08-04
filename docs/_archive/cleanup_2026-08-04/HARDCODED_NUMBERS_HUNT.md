# Hardcoded Scientific Numbers Hunt

**Scope:** `paper/`, `report/`, `docs/` (non-archive focus + key active docs),
`experiments/generate*.py`, `experiments/*summary*.py`.

**Patterns:** `0.752`, `0.779`, `0.902`, `0.0195`, `0.0000`, `tau=100` / `tau = 100`,
`n=3` / `n = 3`, `pending`, `cluster-blocked`, `IF pending`, `synthetic`,
hardcoded means in TeX tables.

**Severity legend**

| Severity | Meaning |
|----------|---------|
| **must-fix before ship** | Shipped PDF/paper claim or table number is stale, wrong seed count, or disagrees with `results/canonical_tau1.json` |
| **ok baseline** | Documented majority-class / constant-predictor rate, historical contrast, or intentional narrative (pending/synthetic caveats) |
| **auto-generated** | Produced by generators from JSON; re-run `make results` / `generate_report_tables.py` to refresh |

**Canonical truth (for comparison):** Adult α=0.2 DP means from auto tables —
Naive DP **0.2452**, DRO DP **0.2334** (n=6). Preliminary n=3 ablation used
Naive **0.2480**, DRO **0.2371**.

---

## Summary counts

| Severity | Hits (`docs/hardcoded_hits.txt`) |
|----------|--------------------------------:|
| must-fix before ship | **77** |
| ok baseline | **49** |
| auto-generated | **22** |
| **Total catalogued** | **148** |

*(Repeated IF `0.0000` table rows and repeated “pending cluster” prose are each listed once per line in the machine file. Narrative grouping in sections below.)*

---

## MUST-FIX BEFORE SHIP

### A. Paper: tau-comparison table = n=3 means, caption claims n=6

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `paper/sections/results.tex:36` | `Mean ± SE over 6 seeds. The τ=1 values are from canonical_tau1.json` | Caption claims 6 seeds + canonical source |
| `paper/sections/results.tex:48` | `0.1 & 1 & 0.2068 & **0.2046** & … & 2/3` | Win fraction **2/3** is n=3; canonical n=6 DP at α=0.1 is ~0.2026/0.1999 |
| `paper/sections/results.tex:52` | `0.2 & 1 & 0.2480 & **0.2371** & … & 3/3` | Hardcoded n=3 ablation means; **≠** canonical 0.2452/0.2334 |
| `paper/sections/results.tex:54` | `0.2 & 100 & 0.3271 & 0.5030` | Historical tau=100 OK if labeled; still mixed into same “6 seed” table |
| `paper/sections/results.tex:56-62` | `0.3…0.4` tau=1 rows with **3/3** | Same n=3/n=6 mismatch |

**Action:** Regenerate tau table from real sources, or retitle as “preliminary n=3 ablation (tau_ablation_*)” and drop “6 seeds / canonical” claim. Prefer auto-input.

### B. Paper abstract overclaim vs body

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `paper/main.tex:32` | `DRO-FAIR beat Naive-FAIR on DP at every α under the DP-targeted and combined attacks` | Body correctly scopes method claim to **α≤0.2**; “every α” overstates (α≥0.3 below constant predictor) |

### C. Report: same n=3 tau mini-table + stale ablation/runtime

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `report/report.tex:464-466` | `1 & 0.2480 & **0.2371** & DRO wins (3/3)` | Hardcoded n=3 DP means in shipped report body |
| `report/report.tex:443` | `Naive 0.327, DRO 0.503` | tau=100 contrast (historical OK if not presented as current); ensure not used as headline |
| `report/report.tex:497` | `show DRO DP = 0.237` | Hardcoded n=3-era mean; canonical α=0.2 DRO DP is **0.2334** |
| `report/report.tex:635-636` | `Naive 10.6±9.1 … DRO 397.6±258.5 … 37.5×` | Hardcoded runtime table; provenance unclear |
| `report/report.tex:652` | caption `6 seeds` on ablation | Claims n=6 |
| `report/report.tex:660-664` | `Standard ML 0.822… DRO Joint 0.788…` full ablation matrix | Hand-written scientific means; not auto-generated from canonical grid; era unknown (IF values nonzero → not current degenerate IF) |
| `report/report.tex:670-671` | `≈83%` / `≈0.175` | Hand prose not tied to current JSON |

### D. Paper lambda appendix: hardcoded grid summary + 0.752 bar

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `paper/sections/appendix_q1_lambda.tex:11` | `Random Seeds: 3 distinct initializations` | Explicit **n=3** in paper appendix |
| `paper/sections/appendix_q1_lambda.tex:25-30` | `Configs ≥ 0.752` / `0.7707` / `0.2225` / ranges | Hardcoded means table not wired to `lambda_grid_comprehensive.json` |
| `paper/sections/appendix_q1_lambda.tex:36` | `falls below the 0.752 constant-predictor ceiling` | Adult-only constant used as paper fact (value OK for Adult; should cite loaders / majority rate) |

### E. Degenerate IF `0.0000` still bolded as “wins” in tables

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `report/sections/auto_generated_main_results.tex:6-20` | every row `0.0000 ± 0.0000` IF, often `\textbf` on DRO | Degenerate pre-fix IF metric; **must not ship as real IF result** |
| `paper/auto_generated/tabular_results.tex:6-35` | `$0.000$` IF columns | Same; auto-gen from current grid where IF≈0 |
| `paper/auto_generated/wilcoxon.tex:6-35` | `ΔIF% 0.0%` all rows | Same degeneracy |
| `report/sections/auto_generated_wilcoxon.tex:6-20` | `ΔIF% 0.0%` | Same |

**Action:** Drop IF columns until IF-attack re-run, or mark “metric degenerate / not evaluated”.

### F. Active docs that still sell n=3 as current protocol

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `docs/FAIRNESS_PGD_RESULTS.md:16` | `Protocol: 3 seeds per condition (n=6 in progress)` | Stale; n=6 canonical has been done |
| `docs/FAIRNESS_PGD_RESULTS.md:24-31` | full headline table with **3/3**, **2/3**, 0.248/0.237 | n=3 numbers presented as “headline result” |
| `docs/FAIRNESS_PGD_RESULTS.md:64-71` | `n=3 cannot achieve p<0.05` / `n=6 IN PROGRESS` | False status vs committed 360-row grid |
| `docs/TAU1_ABLATION_SUMMARY.md:11-54` | entire tables with seeds=**3**, 2/3, 0.2480/0.2371 | Fine as ablation archive **if** paper/report stop quoting it as n=6; currently paper does |
| `docs/PROJECT_FLOW.md:9` | `acc ≥ 0.78 stable` at α≤0.2 | Conflicts with Adult α=0.2 acc ~0.75–0.76 (borderline vs 0.752); soft claim risk |

### G. Placeholder / pending language still in ship surfaces

| File:line | Snippet | Why must-fix |
|-----------|---------|--------------|
| `paper/sections/results.tex:12` | `IF-attack verification is pending a cluster re-run` | Honest but blocks “complete” narrative; ship only with clear limitation |
| `paper/sections/results.tex:79,85,111,114-127` | multiple `pending` / IF coupling pending | Same |
| `paper/sections/introduction.tex:16` | `pending a cluster re-run` | Same |
| `paper/sections/conclusion.tex:1` | `IF-attack results are pending a cluster re-run` | Same |
| `paper/auto_generated/key_findings.tex:8` | `IF-attack rows pending cluster re-run` | Macros used by paper/report |
| `report/report.tex:117,331,369,413,487-488,510,521-522,570,730-735,760,769-770,783-784` | `pending a cluster` / `synthetic smoke` / Q7 pending | Must stay consistent; do not claim full 540-row IF grid |

### H. Fabricated IF number `0.0195` (documented lie — do not reintroduce)

| File:line | Snippet | Severity note |
|-----------|---------|---------------|
| `docs/KULDEEP_CORRECTION.md:12` | `IF violation: DRO = 0.0195 vs Naive = 0.0177` | **ok baseline as correction** of past error — listed so greps don’t re-ship the quote as a result |
| `docs/MASTER_DISPATCH.md:25` | same quote | Historical audit only |
| `docs/AGENT_PROMPTS*.md` | same | Agent instructions |

**Rule:** Never put `0.0195` in paper/report tables.

---

## OK BASELINE (intentional / documented)

| File:line | Snippet | Notes |
|-----------|---------|-------|
| `experiments/loaders.py:26-30` | `_CONSTANT_PREDICTOR_FALLBACK`: adult **0.7521**, credit **0.7788**, lsac **0.9016** | Documented majority rates; preferred over single 0.752 |
| `experiments/loaders.py:68-81` | `constant_predictor_acc` computes from data, fallback to above | Correct design |
| `paper/sections/results.tex:22-23` | Naive **0.327147** vs DRO **0.503047** (stepped schedule) | Explicit historical contrast |
| `paper/sections/results.tex:102-103` | `≈0.902` / `0.9016` majority rate (LSAC) | Correct constant-predictor framing |
| `paper/sections/results.tex:108` | `p=0.0156` LSAC/Combined | Matches Wilcoxon n=6 min |
| `paper/sections/experimental_setup.tex:17,27` | synthetic smoke / UTKFace not run | Honest |
| `paper/sections/discussion.tex:52-54` | synthetic smoke / tau=100-stepped withdrawn | Honest |
| `paper/main.tex:32` | `stepped τ=100` artifact narrative | OK as history (fix “every α” separately) |
| `report/report.tex:225-229` | TV radii table `0.0000…0.6689` | Theory illustration from π formula, not experiment means |
| `report/report.tex:506` | `acc≈0.752` Adult constant bar | Adult majority OK |
| `report/report.tex:513-514` | `≈0.902` / `0.9016` LSAC | OK |
| `docs/KULDEEP_CORRECTION.md` | full correction note with 0.7521/0.7788/0.9016 and 0.0195 retraction | Gold standard honesty doc |
| `docs/LSAC_DEGENERACY.md` | 0.9023 / 0.1829 tables | Diagnostic doc from data |
| `docs/FINAL_COMPLETION_PLAN.md:27` | Adult 0.752 / Credit 0.779 / LSAC 0.902 | Rounded baselines |
| `docs/FINDING_DRO_FAILS_ON_ADULT.md` | tau=100 failure narrative | Historical finding doc |
| `docs/MASTER_DISPATCH.md:76` | tau=100→1 is correct central finding | Process doc |
| `docs/AGENT_PROMPTS.md:146` | hunt CONSTANT_PREDICTOR_ACC=0.752 | Process; note generators now call `constant_predictor_acc` |
| `experiments/generate_report_tables.py:188` | comment `tau=100 numbers again` | Guardrail comment only |
| `experiments/meeting_summary.py:114` | labels `tau=1/10/100` for ablation groups | Loads data; not hardcoding means |
| `experiments/generate_all_deliverables.py:83` | `CONSTANT_PREDICTOR_ACC = constant_predictor_acc('adult')` | Dynamic (Adult-scoped plots) |
| `experiments/generate_final_figures.py:46` | same | Dynamic |

---

## AUTO-GENERATED (refresh from JSON; do not hand-edit)

| File:line | Snippet | Notes |
|-----------|---------|-------|
| `report/sections/auto_generated_main_results.tex:1-21` | Acc/DP/IF means ± SE | From `generate_report_tables.py` + canonical |
| `report/sections/auto_generated_pgd.tex:1-36` | DP reductions + p | Same |
| `report/sections/auto_generated_wilcoxon.tex:1-21` | ΔDP% / p | Same |
| `paper/auto_generated/tabular_results.tex:1-36` | Acc/DP/IF | Same; IF zeros are gen from data |
| `paper/auto_generated/wilcoxon.tex:1-36` | full wilcoxon | Same |
| `paper/auto_generated/key_findings.tex:1-10` | macros n=6, pending IF | Generator output; still has pending string |
| `report/sections/auto_generated_main_results.tex:15` | Credit α=0.4 Acc DRO **0.752** | Real measured acc coinciding with Adult constant; not hardcode |
| `report/sections/auto_generated_main_results.tex:16-20` | Lsac Acc **0.902** | Degenerate majority pin — real measurement |
| `paper/auto_generated/tabular_results.tex:20,26-31` | 0.752 / 0.902 cells | Same |

**Caveat:** auto-generated ≠ ship-safe. IF `0.0000` and bold “better” IF need generator logic change (severity must-fix above).

---

## EXPERIMENTS GENERATE / SUMMARY SCRIPTS

| File:line | Snippet | Severity |
|-----------|---------|----------|
| `experiments/generate_all_deliverables.py:170,222,228` | string literals `horiz 0.752`, caption `0.752` | **must-fix** cosmetic: plot uses dynamic ACC but titles hardcode 0.752 |
| `experiments/generate_all_deliverables.py:320,340,344,350` | labels `0.752` while using `CONSTANT_PREDICTOR_ACC` | **must-fix** string drift risk if Adult rate ≠ 0.752 |
| `experiments/generate_all_deliverables.py:526,586` | title `acc ≥ 0.752` | **must-fix** same |
| `experiments/generate_all_deliverables.py:83,216-217` | `constant_predictor_acc('adult')` for axhline | **ok baseline** (value path correct) |
| `experiments/generate_final_figures.py:46,208-209,254-256,287` | uses dynamic constant | **ok baseline** |
| `experiments/generate_report_tables.py:24-48` | means from rows | **auto-generated** logic |
| `experiments/generate_report_tables.py:188` | tau=100 comment | **ok baseline** |
| `experiments/generate_high_alpha_summary.py` | no pattern hits | clean |
| `experiments/generate_meeting_table.py` | no pattern hits | clean |
| `experiments/generate_sensitivity_analysis.py` | no pattern hits | clean |
| `experiments/generate_summary_dashboard.py` | no pattern hits | clean |
| `experiments/meeting_summary.py:114` | tau labels 1/10/100 | **ok baseline** |
| `experiments/summarize_tau1.py` | tau ablation summarizer | **ok baseline** (reads JSON) |
| `experiments/generate_report_tables.py:25` | docstring still says `540-row grid` | **must-fix** docs drift (grid is 360 DP+Combined) |

---

## HARDCODED MEANS IN TEX TABLES (checklist)

| Location | Auto? | Matches n=6 canonical? |
|----------|-------|------------------------|
| `report/sections/auto_generated_*.tex` | Yes | Yes for Acc/DP; IF degenerate |
| `paper/auto_generated/*.tex` | Yes | Same |
| `paper/sections/results.tex` tab:tau-comparison | **No** | **No** (n=3 means + 2/3,3/3) |
| `report/report.tex` tau mini-table ~464 | **No** | **No** |
| `report/report.tex` ablation ~660 | **No** | Unknown / likely stale |
| `report/report.tex` runtime ~635 | **No** | N/A |
| `report/report.tex` radii ~225 | Hand theory | N/A (formula table) |
| `paper/sections/appendix_q1_lambda.tex` | **No** | n=3 grid search |

---

## Priority fix order (ship)

1. **Retitle or regenerate** `paper/sections/results.tex` tau-comparison table (n=3 vs “6 seeds”).
2. **Align prose means** in report (`0.237`, `0.2480`) with canonical (`0.2334`, `0.2452`) or label as n=3 ablation.
3. **Strip or grey-out IF `0.0000` columns** in auto generators until IF re-run.
4. **Abstract:** change “every α” → “α≤0.2 (defensible regime)”.
5. **Ablation + runtime tables** in `report/report.tex`: regenerate or mark preliminary.
6. **Update** `docs/FAIRNESS_PGD_RESULTS.md` status (or archive) so it cannot be copied into paper.
7. **Replace string `0.752` titles** in `generate_all_deliverables.py` with `f"{CONSTANT_PREDICTOR_ACC}"`.

---

## Out of scope note

`docs/_archive/**` contains many more n=3 / 0.752 / pending hits; treated as historical.
`experiments/_archive/**` excluded except where cited.

Machine-readable companion: [`docs/hardcoded_hits.txt`](hardcoded_hits.txt).
