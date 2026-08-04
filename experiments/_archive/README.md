# experiments/_archive — historical experiment scripts

**Purpose:** Near-duplicate runners, one-off analysis, and superseded figure generators moved here during Stage 3 of repo consolidation (`docs/REPO_CONSOLIDATION_PLAN.md`). Prefer restore via `git mv` / `mv` back to `experiments/` if needed; do not re-wire Makefile to these paths.

**Active pipeline (do not archive):**  
`run_canonical.py`, `run_if_parallel.py`, `run_fairness_pgd.py`, `loaders.py`,  
`compute_canonical_wilcoxon.py`, `generate_report_tables.py`, `generate_results.py`,  
`generate_final_figures.py`, `generate_all_deliverables.py`, `validate_results.py`,  
`canonical_to_all_results.py` (nested bridge for `make results`),  
`run_utkface.py`, `run_utkface_server.py`, `meeting_summary.py`, `verify_theory.py`,  
`run_experiments.py` (still referenced by `Makefile` / `main.py` legacy targets),  
`run_canonical_empirical.py`, headline `plot_*` still used for regeneration.

---

## Inventory (one line each)

### Already archived before Stage 3 (orchestrators / smoke)

| Script | Role |
|--------|------|
| `agent_a_monitor.sh` | Agent A lambda-priority status monitor |
| `aggregate_all_results.py` | Merge scattered result JSON shards |
| `analyze_fairness_pgd.py` | Early fairness-PGD result dump analysis |
| `auto_finalize.py` | Auto finalize after sweeps |
| `auto_generate_deliverables.py` | Old deliverables orchestrator |
| `auto_runner.sh` | Shell auto-runner loop |
| `check_progress.py` | Progress counter for long runs |
| `check_server.sh` / `check_server_progress.py` | Server health / progress checks |
| `generate_final_meeting_figure.py` | One-off meeting figure |
| `generate_paper_tables.py` | Superseded by `generate_report_tables.py` |
| `preliminary_analysis.py` | Early exploratory analysis |
| `quick_attack_check.py` | Quick attack smoke |
| `quick_radii_fix_validation.py` | Radii fix smoke validation |
| `quick_test.py` | Quick local smoke test |
| `run_all_server_parallel.sh` | Server parallel launch wrapper |
| `run_everything.sh` | Kitchen-sink shell runner |
| `setup_server.sh` | Server environment setup |
| `test_dp_attack_bug.py` | DP attack regression probe |
| `validate_attack_strength.py` | Attack strength checks |
| `verify_server_scripts.py` | Server script presence checks |

### Stage 3 Batch A — UTKFace extras / out-of-scope setup

| Script | Role |
|--------|------|
| `run_utkface_extended.py` | Extended UTKFace modes (alpha_sweep / fairness_pgd / lambda_max) |
| `run_utkface_pixel_pgd.py` | Pixel-space PGD (H2) runner |
| `run_utkface_randinit.py` | Random-init backbone (H1) runner |
| `analyze_utkface.py` | UTKFace result plotting one-off |
| `analyze_utkface_stats.py` | UTKFace JSON stats one-off |
| `analyze_dro_failure.py` | Synthetic-era image DRO failure narrative |
| `generate_fig10.py` | UTKFace curves (reads utkface_results.json) |
| `setup_celeba.py` | CelebA prep (out of submission scope) |
| `setup_fairface.py` | FairFace prep (out of submission scope) |

### Stage 3 Batch B — Superseded / broken figure generators

| Script | Role |
|--------|------|
| `generate_all_figures.py` | “ALL” figs incl. UTKFace; called raising `load_fairness_pgd_results` |
| `generate_pdf_report.py` | Standalone matplotlib PDF report (superseded by TeX) |
| `generate_summary_dashboard.py` | Meeting dashboard one-off |
| `generate_sensitivity_analysis.py` | Sensitivity fig; used raising loader |
| `generate_meeting_table.py` | Meeting table; used raising loader |
| `generate_high_alpha_summary.py` | high_alpha CSV from tau ablations |
| `plot_partial_results.py` | Partial PGD plots; calls raising loader |
| `plot_high_alpha_tau.py` | High-α + τ curves (ablation era) |
| `plot_random_vs_adversarial.py` | Random vs PGD ablation plot |
| `plot_uniform_vs_empirical.py` | Uniform vs empirical radii plot |
| `plot_lambda_diagnostic.py` | Lambda trajectory diagnostic plot |
| `plot_lambda_grid_heatmap.py` | Lambda grid heatmap |
| `plot_lambda_heatmap_highalpha.py` | High-α lambda heatmap |
| `plot_convergence.py` | Training convergence curves |

### Stage 3 Batch C — Analysis / diagnostic one-offs

| Script | Role |
|--------|------|
| `analyze_partial_results.py` | Partial PGD analysis |
| `analyze_high_alpha.py` | High-α ablation conclusions |
| `analyze_lambda_grid.py` | Lambda grid heatmap analysis |
| `analyze_lsac_complete.py` | LSAC 90-row analysis (stale path) |
| `analyze_tau1.py` | Master τ=1 story + many figures (superseded by wilcoxon/tables/final figs) |
| `analyze_results.py` | Generic advanced analysis of nested results |
| `diagnostics.py` | Lambda trajectory / group-rate Stream B diagnostics |
| `dro_radii_diagnostic.py` | DRO radii diagnostic |
| `demonstrate_radii_mismatch.py` | Radii mismatch demo |
| `summarize_tau1.py` | Markdown tables for τ=1 |
| `summarize_random_vs_adv.py` | Markdown for random vs adv ablation |

### Stage 3 Batch C — Ablation / ad-hoc runners + obsolete worker

| Script | Role |
|--------|------|
| `run_ablations.py` | Generic ablation driver |
| `run_robust.py` | Merge individual runs → nested all_results |
| `run_tau_ablation.py` | Fixed-τ grid (ablation dropped for claims) |
| `run_knn_ablation.py` | k-NN ablation |
| `run_lambda_diagnostic.py` / `run_lambda_diagnostic_full.py` | λ diagnostics |
| `run_lambda_grid_comprehensive.py` / `run_lambda_lr_grid.py` | λ grids |
| `run_random_vs_adversarial.py` | Random vs PGD runner |
| `run_parallel_batch.py` | Ad-hoc parallel batch helper |
| `run_k10_targeted.py` | k_inner=10 targeted one-off |
| `run_single_adult.py` | Single Adult config one-off |
| `run_all_server_experiments.sh` | Server experiment shell wrapper |
| `_run_if_chunk.py` | Obsolete IF chunk worker (logic inlined in `run_if_parallel.py`) |

---

## Intentionally kept in `experiments/` (Stage 3)

- Canonical tabular: `run_canonical.py`, `run_if_parallel.py`, `run_fairness_pgd.py`, `loaders.py`
- Stats/tables: `compute_canonical_wilcoxon.py`, `generate_report_tables.py`, `validate_results.py`, `meeting_summary.py`
- Figures (active Makefile path): `generate_all_deliverables.py`, `generate_results.py`, `canonical_to_all_results.py`
- Headline plots archived here (regenerate from `_archive/` if needed): `plot_*`, `generate_final_figures.py`
- UTKFace entry: `run_utkface.py`, `run_utkface_server.py`
- Legacy Makefile/`main.py` hooks: `run_experiments.py`, `verify_theory.py`
- Appendix companion: `run_canonical_empirical.py`

### Later cleanup (2026-08-04 loop)

| Script | Role |
|--------|------|
| `generate_latex_extras.py` | Runtime/ablation LaTeX extras; unused by Makefile / paper auto_generated path |
| `generate_figures.py` | Legacy fig1–fig7 from nested `all_results`; superseded by `generate_results.py` / final figs |
| `generate_fig8_matrix.py` | Attack–defense matrix; not a Makefile target |

*Last updated: 2026-08-04 professional tree cleanup (scripts/results/docs).*

## 2026-08-04 batch (tau_ablation-dependent / superseded)
- plot_tau1_headline.py, plot_acc_by_attack.py, plot_if_by_attack.py, plot_acc_win_curves.py, plot_win_curves_tau1.py — require archived tau_ablation_*.json
- generate_final_figures.py — superseded by generate_all_deliverables.py (canonical-only)
