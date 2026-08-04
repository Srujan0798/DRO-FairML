# Figures (LIVE after 2026-08-04 cleanup)

## Meeting / claim figures (from 540-row canonical)
| File | Role |
|------|------|
| `fig_tau1_headline.pdf` | Adult DP vs α; wins **[6,5,6,6,6]** |
| `fig_final_wilcoxon_table.pdf` | DP-attack win matrix (same as `figD10_*`) |
| `figD1`–`figD4` | Constant-predictor + tradeoff |
| `figD8`–`figD9` | Lambda heatmaps (if present) |
| `fig1`, `fig2`, `fig4`, `fig5`, `fig7`, `main_results`, `test_time_eval` | Report includes |

## Removed on purpose
Pre-540 Jul-2 `fig_final_*` (except wilcoxon), Jul-20 convergence `figD5–7` (no per-run histories), old win-curves, high-α extras.

## Regen
```bash
python3 experiments/plot_meeting_figs_540.py
PYTHONPATH=. python3 experiments/generate_all_deliverables.py   # may skip missing optional inputs
```
