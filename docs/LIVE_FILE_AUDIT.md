# LIVE FILE AUDIT (brutal) — 2026-08-04

**Working tree files (no .git / pycache):** 167
**Canonical:** `results/canonical_tau1.json` = **540** rows (dp/if/combined = 180 each).

## Status legend
| Tag | Meaning |
|-----|---------|
| LIVE_* | Current, aligned with 540 / meeting story |
| STALE_FIG | Old mtime or incomplete inputs — do not present as 540-fresh |
| REFERENCE | Planning history; STATUS.md wins conflicts |
| LIVE_MIXED | File is live but contains historical sections |
| DATA / CI / META / RUNTIME | Support |

## Counts by tag
| Tag | n |
|-----|--:|
| LIVE_CODE | 43 |
| REFERENCE | 15 |
| LIVE_PAPER | 14 |
| STALE_FIG | 12 |
| LIVE_FIG_REPORT | 11 |
| DOCS_CHECK | 10 |
| LIVE | 8 |
| LIVE_ROOT | 7 |
| DATA | 7 |
| LIVE_DOCS | 6 |
| OPTIONAL_FIG | 5 |
| FIG | 4 |
| RUNTIME | 4 |
| LIVE_REPORT | 4 |
| LIVE_OPS | 4 |
| META | 3 |
| LIVE_FIG_540 | 3 |
| LIVE_PDF | 2 |
| CI | 1 |
| LIVE_MIXED | 1 |
| LIVE_NESTED | 1 |
| LIVE_CANONICAL | 1 |
| LIVE_PARTIAL | 1 |

## Broken / stale path references (must fix or ignore as history)
- docs/ARCHIVE_POLICY.md → mentions `docs/_archive/` (path may be gone)
- docs/ARCHIVE_POLICY.md → mentions `experiments/_archive/` (path may be gone)
- docs/ARCHIVE_POLICY.md → mentions `scripts/_archive/` (path may be gone)
- docs/CLEAN_TREE.md → mentions `docs/_archive/` (path may be gone)
- docs/CLEAN_TREE.md → mentions `experiments/_archive/` (path may be gone)
- docs/CLEAN_TREE.md → mentions `scripts/_archive/` (path may be gone)
- docs/FINAL_COMPLETION_PLAN.md → mentions `docs/_archive/` (path may be gone)
- docs/FINAL_COMPLETION_PLAN.md → mentions `experiments/_archive/` (path may be gone)
- docs/FINAL_COMPLETION_PLAN.md → mentions `docs/MASTER_DISPATCH` (path may be gone)
- docs/FINAL_COMPLETION_PLAN.md → mentions `monitor_if_then_regen` (path may be gone)
- docs/REPO_AUDIT.md → mentions `docs/_archive/` (path may be gone)
- docs/REPO_AUDIT.md → mentions `experiments/_archive/` (path may be gone)
- docs/REPO_AUDIT.md → mentions `docs/MASTER_DISPATCH` (path may be gone)
- docs/REPO_AUDIT.md → mentions `KULDEEP_DISCUSSION.md` (path may be gone)
- docs/REPO_AUDIT.md → mentions `monitor_if_then_regen` (path may be gone)
- docs/REPO_CONSOLIDATION_PLAN.md → mentions `docs/_archive/` (path may be gone)
- docs/REPO_CONSOLIDATION_PLAN.md → mentions `experiments/_archive/` (path may be gone)
- docs/REPO_CONSOLIDATION_PLAN.md → mentions `scripts/_archive/` (path may be gone)
- docs/UTKFACE_STATUS.md → mentions `docs/_archive/` (path may be gone)
- docs/VERIFICATION_REPORT.md → mentions `docs/_archive/` (path may be gone)
- docs/VERIFICATION_REPORT.md → mentions `KULDEEP_DISCUSSION.md` (path may be gone)
- docs/reference/ARCHIVE_POLICY.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/ARCHIVE_POLICY.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/ARCHIVE_POLICY.md → mentions `scripts/_archive/` (path may be gone)
- docs/reference/CLEAN_TREE.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/CLEAN_TREE.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/CLEAN_TREE.md → mentions `scripts/_archive/` (path may be gone)
- docs/reference/FINAL_COMPLETION_PLAN.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/FINAL_COMPLETION_PLAN.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/FINAL_COMPLETION_PLAN.md → mentions `docs/MASTER_DISPATCH` (path may be gone)
- docs/reference/FINAL_COMPLETION_PLAN.md → mentions `monitor_if_then_regen` (path may be gone)
- docs/reference/LOOP_STATUS.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/LOOP_STATUS.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/LOOP_STATUS.md → mentions `scripts/_archive/` (path may be gone)
- docs/reference/REPO_AUDIT.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/REPO_AUDIT.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/REPO_AUDIT.md → mentions `docs/MASTER_DISPATCH` (path may be gone)
- docs/reference/REPO_AUDIT.md → mentions `KULDEEP_DISCUSSION.md` (path may be gone)
- docs/reference/REPO_AUDIT.md → mentions `monitor_if_then_regen` (path may be gone)
- docs/reference/REPO_CONSOLIDATION_PLAN.md → mentions `docs/_archive/` (path may be gone)
- docs/reference/REPO_CONSOLIDATION_PLAN.md → mentions `experiments/_archive/` (path may be gone)
- docs/reference/REPO_CONSOLIDATION_PLAN.md → mentions `scripts/_archive/` (path may be gone)
- scripts/agent_h_finalize.sh → mentions `monitor_if_then_regen` (path may be gone)

## Alignment verdict (critical path)
| Item | Verdict |
|------|---------|
| STATUS.md Adult/DP 5/6 | **ALIGNED** |
| MEETING brief | **ALIGNED** |
| KULDEEP_CORRECTION high-α Naive acc | **ALIGNED** |
| paper results.tex constant predictor scope | **ALIGNED** (Adult+Credit) |
| fig_tau1_headline / fig_final_wilcoxon | **ALIGNED** (2026-08-04, wins [6,5,6,6,6]) |
| fig_final_* Jul 2 | **STALE** vs 540 — prefer figD* |
| figD5–D7 | **STALE** (no individual histories) |
| Makefile `review` | **FIXED** (no dead `_archive` paths) |
| README docs/_archive link | **FIXED** |
| results/all_results.json (90 nested) | **LIVE nested helper** not raw 540 |
| UTKFace | **PARTIAL REAL** — no paper claim |

## Every file
| Status | Path | mtime | Bytes | Note |
|--------|------|-------|------:|------|
| `META` | `.claude/settings.local.json` | 2026-07-20 | 10173 | Tooling |
| `META` | `.gitattributes` | 2026-05-18 | 214 | Tooling |
| `CI` | `.github/workflows/tests.yml` | 2026-05-18 | 807 | CI config |
| `META` | `.gitignore` | 2026-08-04 | 1598 | Tooling |
| `LIVE_ROOT` | `LICENSE` | 2026-05-16 | 1067 | Project root |
| `LIVE_ROOT` | `Makefile` | 2026-08-04 | 4652 | Project root |
| `LIVE_ROOT` | `README.md` | 2026-08-04 | 6113 | Project root |
| `LIVE_ROOT` | `STATUS.md` | 2026-08-04 | 5950 | Project root |
| `LIVE` | `configs/default.yaml` | 2026-08-04 | 671 | tau=1 defaults |
| `DATA` | `data/download_data.sh` | 2026-08-04 | 5548 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/README.md` | 2026-08-04 | 499 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/adult.data` | 2026-05-12 | 3974305 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/adult.test` | 2026-05-12 | 2003153 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/default_of_credit_card_clients.xls` | 2026-05-12 | 5539328 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/lsac.csv` | 2026-05-12 | 993161 | Tabular public + utkface features REAL |
| `DATA` | `data/raw/utkface_features.npz` | 2026-08-04 | 43596085 | Tabular public + utkface features REAL |
| `DOCS_CHECK` | `docs/ARCHIVE_POLICY.md` | 2026-08-04 | 601 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/CLEAN_TREE.md` | 2026-08-04 | 3934 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/FAIRNESS_PGD_DESIGN.md` | 2026-08-04 | 9442 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/FINAL_COMPLETION_PLAN.md` | 2026-08-04 | 16052 | May be planning or live — verify vs STATUS |
| `LIVE_DOCS` | `docs/INDEX.md` | 2026-08-04 | 1153 | Current claims/meeting |
| `LIVE_DOCS` | `docs/KEY_FORMULAS.md` | 2026-05-18 | 6062 | Current claims/meeting |
| `LIVE_DOCS` | `docs/KULDEEP_CORRECTION.md` | 2026-08-04 | 5131 | Current claims/meeting |
| `DOCS_CHECK` | `docs/LOOP_STATUS.md` | 2026-08-04 | 970 | May be planning or live — verify vs STATUS |
| `LIVE_DOCS` | `docs/LSAC_DEGENERACY.md` | 2026-07-20 | 2772 | Current claims/meeting |
| `LIVE_DOCS` | `docs/MEETING_2026-08-04.md` | 2026-08-04 | 16225 | Current claims/meeting |
| `DOCS_CHECK` | `docs/PAPER_FINALIZATION_CHECKLIST.md` | 2026-08-04 | 22010 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/REPO_AUDIT.md` | 2026-08-04 | 24203 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/REPO_CONSOLIDATION_PLAN.md` | 2026-08-04 | 12652 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/SERVER_RUNBOOK.md` | 2026-08-04 | 13607 | May be planning or live — verify vs STATUS |
| `DOCS_CHECK` | `docs/UTKFACE_PIPELINE.md` | 2026-08-04 | 11369 | May be planning or live — verify vs STATUS |
| `LIVE_DOCS` | `docs/UTKFACE_STATUS.md` | 2026-08-04 | 5456 | Current claims/meeting |
| `LIVE_MIXED` | `docs/VERIFICATION_REPORT.md` | 2026-08-04 | 24827 | Agent V MATCH at end; body still has historical 424-row snapshot — cite Agent V section only |
| `REFERENCE` | `docs/reference/ABLATION_STATUS_REPORT.md` | 2026-07-20 | 9329 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/ARCHIVE_POLICY.md` | 2026-08-04 | 705 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/CLEAN_TREE.md` | 2026-08-04 | 5085 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/COORDINATION.md` | 2026-08-04 | 2940 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/FAIRNESS_PGD_DESIGN.md` | 2026-05-20 | 9442 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/FINAL_COMPLETION_PLAN.md` | 2026-08-04 | 15879 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/LOOP_STATUS.md` | 2026-08-04 | 1376 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/PAPER_FINALIZATION_CHECKLIST.md` | 2026-08-04 | 21742 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/Q5_derivation.md` | 2026-06-16 | 4476 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/REPO_AUDIT.md` | 2026-08-04 | 24203 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/REPO_CONSOLIDATION_PLAN.md` | 2026-08-04 | 12652 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/REPO_LAYOUT.md` | 2026-08-04 | 3176 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/SERVER_RUNBOOK.md` | 2026-06-16 | 13607 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/TAU1_ABLATION_SUMMARY.md` | 2026-06-16 | 4644 | Planning/design history — STATUS overrides if conflict |
| `REFERENCE` | `docs/reference/UTKFACE_PIPELINE.md` | 2026-05-20 | 11369 | Planning/design history — STATUS overrides if conflict |
| `LIVE` | `experiments/README.md` | 2026-08-04 | 1886 | experiments map |
| `LIVE_CODE` | `experiments/canonical_to_all_results.py` | 2026-08-04 | 2269 | Active runner/generator |
| `LIVE_CODE` | `experiments/compute_canonical_wilcoxon.py` | 2026-08-04 | 6285 | Active runner/generator |
| `LIVE_CODE` | `experiments/generate_all_deliverables.py` | 2026-08-04 | 27275 | Active runner/generator |
| `LIVE_CODE` | `experiments/generate_report_tables.py` | 2026-07-20 | 12488 | Active runner/generator |
| `LIVE_CODE` | `experiments/generate_results.py` | 2026-08-04 | 13970 | Active runner/generator |
| `LIVE_CODE` | `experiments/loaders.py` | 2026-08-04 | 4045 | Active runner/generator |
| `LIVE_CODE` | `experiments/meeting_summary.py` | 2026-08-04 | 10203 | Active runner/generator |
| `LIVE_CODE` | `experiments/plot_meeting_figs_540.py` | 2026-08-04 | 5496 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_canonical.py` | 2026-06-16 | 7352 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_canonical_empirical.py` | 2026-06-30 | 5975 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_experiments.py` | 2026-07-20 | 11969 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_fairness_pgd.py` | 2026-07-20 | 9189 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_if_parallel.py` | 2026-08-04 | 3484 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_utkface.py` | 2026-08-04 | 14712 | Active runner/generator |
| `LIVE_CODE` | `experiments/run_utkface_server.py` | 2026-08-04 | 7931 | Active runner/generator |
| `LIVE_CODE` | `experiments/validate_results.py` | 2026-08-04 | 6216 | Active runner/generator |
| `LIVE_CODE` | `experiments/verify_theory.py` | 2026-05-14 | 10158 | Active runner/generator |
| `FIG` | `figures/FINAL_FIGURES_MANIFEST.txt` | 2026-06-17 | 3351 | See figures/README.md |
| `FIG` | `figures/README.md` | 2026-08-04 | 929 | See figures/README.md |
| `LIVE_FIG_REPORT` | `figures/fig1_main_results.pdf` | 2026-08-04 | 48833 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/fig2_dp_reduction_heatmap.pdf` | 2026-08-04 | 43258 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/fig4_significance_matrix.pdf` | 2026-08-04 | 46815 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/fig5_accuracy_fairness_tradeoff.pdf` | 2026-08-04 | 31575 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/fig7_summary_win_rates.pdf` | 2026-08-04 | 39720 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_540` | `figures/figD10_final_wilcoxon_table.pdf` | 2026-08-04 | 28178 | Regenerated 2026-08-04 from 540 |
| `LIVE_FIG_REPORT` | `figures/figD1_constant_predictor_accuracy.pdf` | 2026-08-04 | 50398 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/figD2_constant_predictor_dp.pdf` | 2026-08-04 | 28447 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/figD3_constant_predictor_if.pdf` | 2026-08-04 | 28895 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/figD4_tradeoff_vs_constant_predictor.pdf` | 2026-08-04 | 39327 | Report-linked; regen Aug 4 for some |
| `STALE_FIG` | `figures/figD5_convergence_loss.pdf` | 2026-07-20 | 24115 | Jul 20 — needs results/individual/ (missing); do not claim 540-fresh |
| `STALE_FIG` | `figures/figD6_convergence_acc.pdf` | 2026-07-20 | 25329 | Jul 20 — needs results/individual/ (missing); do not claim 540-fresh |
| `STALE_FIG` | `figures/figD7_convergence_dp.pdf` | 2026-07-20 | 24766 | Jul 20 — needs results/individual/ (missing); do not claim 540-fresh |
| `FIG` | `figures/figD8_lambda_heatmap_acc_alpha0_3.pdf` | 2026-08-04 | 32439 | See figures/README.md |
| `FIG` | `figures/figD9_lambda_heatmap_acc_alpha0_4.pdf` | 2026-08-04 | 32193 | See figures/README.md |
| `STALE_FIG` | `figures/fig_acc_win_curves_tau1.pdf` | 2026-07-02 | 35003 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_constant_predictor_acc.pdf` | 2026-07-02 | 21114 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_constant_predictor_dp.pdf` | 2026-07-02 | 18646 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_constant_predictor_if.pdf` | 2026-07-02 | 18479 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_lambda_heatmap_acc_0.3.pdf` | 2026-07-02 | 28773 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_lambda_heatmap_acc_0.4.pdf` | 2026-07-02 | 28713 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_lambda_heatmap_dp_0.3.pdf` | 2026-07-02 | 28486 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_lambda_heatmap_dp_0.4.pdf` | 2026-07-02 | 29352 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `STALE_FIG` | `figures/fig_final_tradeoff_vs_constant_predictor.pdf` | 2026-07-02 | 26748 | Jul 2 pre-540 stamp — prefer figD* / plot_meeting_figs_540 |
| `LIVE_FIG_540` | `figures/fig_final_wilcoxon_table.pdf` | 2026-08-04 | 28178 | Regenerated 2026-08-04 from 540 |
| `OPTIONAL_FIG` | `figures/fig_high_alpha_tau.pdf` | 2026-08-04 | 37541 | Supplementary |
| `OPTIONAL_FIG` | `figures/fig_high_alpha_tau_acc.pdf` | 2026-08-04 | 26397 | Supplementary |
| `OPTIONAL_FIG` | `figures/fig_high_alpha_tau_dp.pdf` | 2026-08-04 | 25115 | Supplementary |
| `OPTIONAL_FIG` | `figures/fig_high_alpha_tau_if.pdf` | 2026-08-04 | 24361 | Supplementary |
| `LIVE_FIG_540` | `figures/fig_tau1_headline.pdf` | 2026-08-04 | 27591 | Regenerated 2026-08-04 from 540 |
| `OPTIONAL_FIG` | `figures/fig_win_curves_tau1.png` | 2026-08-04 | 187263 | Supplementary |
| `LIVE_FIG_REPORT` | `figures/main_results.pdf` | 2026-08-04 | 22816 | Report-linked; regen Aug 4 for some |
| `LIVE_FIG_REPORT` | `figures/test_time_eval.pdf` | 2026-08-04 | 17253 | Report-linked; regen Aug 4 for some |
| `RUNTIME` | `logs/README.md` | 2026-08-04 | 195 | gitignored logs |
| `RUNTIME` | `logs/utkface_canonical.pid` | 2026-08-04 | 6 | gitignored logs |
| `RUNTIME` | `logs/utkface_canonical_run.log` | 2026-08-04 | 523 | gitignored logs |
| `RUNTIME` | `logs/utkface_watchdog.log` | 2026-08-04 | 103 | gitignored logs |
| `LIVE_ROOT` | `main.py` | 2026-08-04 | 6450 | Project root |
| `LIVE_PAPER` | `paper/auto_generated/key_findings.tex` | 2026-08-04 | 1495 | Paper sources |
| `LIVE_PAPER` | `paper/auto_generated/tabular_results.tex` | 2026-08-04 | 5180 | Paper sources |
| `LIVE_PAPER` | `paper/auto_generated/wilcoxon.tex` | 2026-08-04 | 2844 | Paper sources |
| `LIVE_PDF` | `paper/main.pdf` | 2026-08-04 | 110093 | Rebuilt 2026-08-04 |
| `LIVE_PAPER` | `paper/main.tex` | 2026-08-04 | 2931 | Paper sources |
| `LIVE_PAPER` | `paper/references.bib` | 2026-05-31 | 1561 | Paper sources |
| `LIVE_PAPER` | `paper/sections/appendix_q1_lambda.tex` | 2026-06-18 | 2173 | Paper sources |
| `LIVE_PAPER` | `paper/sections/appendix_q5_empirical.tex` | 2026-07-20 | 2111 | Paper sources |
| `LIVE_PAPER` | `paper/sections/attack_design.tex` | 2026-07-20 | 1495 | Paper sources |
| `LIVE_PAPER` | `paper/sections/conclusion.tex` | 2026-08-04 | 1538 | Paper sources |
| `LIVE_PAPER` | `paper/sections/discussion.tex` | 2026-08-04 | 3083 | Paper sources |
| `LIVE_PAPER` | `paper/sections/experimental_setup.tex` | 2026-08-04 | 2207 | Paper sources |
| `LIVE_PAPER` | `paper/sections/introduction.tex` | 2026-08-04 | 1870 | Paper sources |
| `LIVE_PAPER` | `paper/sections/related_work.tex` | 2026-05-31 | 515 | Paper sources |
| `LIVE_PAPER` | `paper/sections/results.tex` | 2026-08-04 | 8401 | Paper sources |
| `LIVE_PDF` | `report/report.pdf` | 2026-08-04 | 286585 | Rebuilt 2026-08-04 |
| `LIVE_REPORT` | `report/report.tex` | 2026-08-04 | 46554 | Report sources |
| `LIVE_REPORT` | `report/sections/auto_generated_main_results.tex` | 2026-08-04 | 2659 | Report sources |
| `LIVE_REPORT` | `report/sections/auto_generated_pgd.tex` | 2026-08-04 | 2776 | Report sources |
| `LIVE_REPORT` | `report/sections/auto_generated_wilcoxon.tex` | 2026-08-04 | 970 | Report sources |
| `LIVE_ROOT` | `requirements.txt` | 2026-08-04 | 865 | Project root |
| `LIVE_NESTED` | `results/all_results.json` | 2026-07-20 | 67492 | Nested view for make results; 90 cells (means path) — not raw 540 flat |
| `LIVE_CANONICAL` | `results/canonical_tau1.json` | 2026-08-04 | 231802 | 540 rows — only scientific SSOT for tabular claims |
| `LIVE` | `results/canonical_wilcoxon.csv` | 2026-08-04 | 4923 | Wilcoxon derived from 540 |
| `LIVE` | `results/canonical_wilcoxon.md` | 2026-08-04 | 4918 | Wilcoxon derived from 540 |
| `LIVE` | `results/if_wilcoxon_summary.txt` | 2026-08-04 | 1338 | Real IF Wilcoxon from 540 |
| `LIVE` | `results/summary_stats.csv` | 2026-08-04 | 3984 | Tables derived from 540 |
| `LIVE` | `results/table1_latex.tex` | 2026-08-04 | 1282 | Tables derived from 540 |
| `LIVE` | `results/table1_results.csv` | 2026-08-04 | 3487 | Tables derived from 540 |
| `LIVE_PARTIAL` | `results/utkface_canonical.json` | 2026-08-04 | 48324 | REAL UTKFace progress only — no paper claim |
| `LIVE_OPS` | `scripts/agent_h_finalize.sh` | 2026-08-04 | 7789 | Operational helper |
| `LIVE_OPS` | `scripts/deploy_utkface_flair2.sh` | 2026-08-04 | 3692 | Operational helper |
| `LIVE_OPS` | `scripts/extract_utkface_features.py` | 2026-08-04 | 7189 | Operational helper |
| `LIVE_OPS` | `scripts/flair2_ssh_config_snippet.txt` | 2026-08-04 | 360 | Operational helper |
| `LIVE_ROOT` | `setup.py` | 2026-05-18 | 1354 | Project root |
| `LIVE_CODE` | `src/corruption/__init__.py` | 2026-07-20 | 162 | Core library |
| `LIVE_CODE` | `src/corruption/adversarial.py` | 2026-07-20 | 27668 | Core library |
| `LIVE_CODE` | `src/data/__init__.py` | 2026-05-11 | 139 | Core library |
| `LIVE_CODE` | `src/data/datasets.py` | 2026-08-04 | 11463 | Core library |
| `LIVE_CODE` | `src/evaluation/__init__.py` | 2026-05-12 | 162 | Core library |
| `LIVE_CODE` | `src/evaluation/metrics.py` | 2026-07-20 | 6250 | Core library |
| `LIVE_CODE` | `src/models/__init__.py` | 2026-05-12 | 67 | Core library |
| `LIVE_CODE` | `src/models/classifier.py` | 2026-06-09 | 1340 | Core library |
| `LIVE_CODE` | `src/models/cnn_classifier.py` | 2026-06-16 | 1912 | Core library |
| `LIVE_CODE` | `src/temperature.py` | 2026-08-04 | 773 | Core library |
| `LIVE_CODE` | `src/training/__init__.py` | 2026-07-20 | 192 | Core library |
| `LIVE_CODE` | `src/training/dro_fair.py` | 2026-07-20 | 19835 | Core library |
| `LIVE_CODE` | `src/training/naive_fair.py` | 2026-07-02 | 7604 | Core library |
| `LIVE_CODE` | `src/training/standard_ml.py` | 2026-05-12 | 2586 | Core library |
| `LIVE_CODE` | `src/utils/__init__.py` | 2026-05-12 | 88 | Core library |
| `LIVE_CODE` | `src/utils/projections.py` | 2026-05-13 | 3424 | Core library |
| `LIVE_CODE` | `tests/__init__.py` | 2026-05-12 | 0 | Tests |
| `LIVE_CODE` | `tests/conftest.py` | 2026-05-18 | 862 | Tests |
| `LIVE_CODE` | `tests/test_cnn_classifier.py` | 2026-06-16 | 2507 | Tests |
| `LIVE_CODE` | `tests/test_corruption.py` | 2026-05-12 | 2883 | Tests |
| `LIVE_CODE` | `tests/test_end_to_end.py` | 2026-06-16 | 16827 | Tests |
| `LIVE_CODE` | `tests/test_fairness_pgd.py` | 2026-05-29 | 6288 | Tests |
| `LIVE_CODE` | `tests/test_greedy_attack_superiority.py` | 2026-06-08 | 2641 | Tests |
| `LIVE_CODE` | `tests/test_metrics.py` | 2026-07-20 | 3449 | Tests |
| `LIVE_CODE` | `tests/test_projections.py` | 2026-05-12 | 3486 | Tests |
| `LIVE_CODE` | `tests/test_radii_calibration.py` | 2026-06-16 | 6484 | Tests |

## Actions taken in this audit pass
1. Fixed Makefile `review` targets (removed dead `docs/_archive/*` paths).
2. Fixed README layout links (no `_archive` claim; REPO_LAYOUT under reference/).
3. Fixed `src/temperature.py` / `key_findings.tex` comments pointing at deleted MASTER_DISPATCH.
4. Superseded banners on FINAL_COMPLETION_PLAN + PAPER_FINALIZATION_CHECKLIST.
5. figures/README.md now labels Jul-2 vs Aug-4 figures honestly.
6. This audit file is the inventory of the **current** tree only.

## Still optional cleanup
- Delete or regenerate Jul-2 `fig_final_*` and Jul-20 `figD5–D7` if you want zero stale PDFs.
- Collapse duplicate docs at `docs/*.md` that also live under `docs/reference/`.
- Rewrite VERIFICATION_REPORT body so historical 424-row tables are clearly appendix-only.
